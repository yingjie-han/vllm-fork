# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
XPU Grouped GEMM expert implementation for pre-permuted inputs.

Used with XPUDeepSymmPrepareFinalize which provides pre-permuted
hidden_states and rows_per_expert via ExpertTokensMetadata. Calls
cutlass_grouped_gemm_interface directly without internal remap/gather.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kFp8DynamicTensorSym,
    kInt4Static,
)
from vllm.platforms import current_platform


def _dequantize_to_bf16(
    x: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize quantized activations back to BF16.

    Handles per-tensor (scalar/1D scale), block (2D scale with
    block_size columns), and MX (e8m0fnu scale) formats.
    """
    out_dtype = torch.bfloat16
    if scale.numel() == 1:
        return x.to(out_dtype) * scale.to(out_dtype)
    elif scale.dtype == torch.float8_e8m0fnu:
        scale_f = (
            scale.view(torch.uint8).to(torch.int32) - 127
        ).exp2().to(out_dtype)
        block_size = x.shape[-1] // scale.shape[-1]
        scale_expanded = scale_f.repeat_interleave(block_size, dim=-1)
        return x.to(out_dtype) * scale_expanded[..., : x.shape[-1]]
    elif scale.ndim == 2 and scale.shape[0] == x.shape[0]:
        block_size = x.shape[-1] // scale.shape[-1]
        scale_expanded = scale.to(out_dtype).repeat_interleave(
            block_size, dim=-1
        )
        return x.to(out_dtype) * scale_expanded[..., : x.shape[-1]]
    else:
        return x.to(out_dtype) * scale.to(out_dtype)


class XPUGroupedGemmExperts(mk.FusedMoEExpertsModular):
    """
    Expert kernel for pre-permuted inputs using cutlass_grouped_gemm.

    Expects hidden_states already in expert-grouped layout with
    rows_per_expert provided via expert_tokens_meta. Does NOT perform
    internal permutation or gather — those are handled by the
    PrepareFinalize (e.g. XPUDeepSymmPrepareFinalize).
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        self.is_fp8 = False
        self.is_int4 = False
        self.is_mxfp4 = False

    @property
    def expects_unquantized_inputs(self) -> bool:
        # This tree's grouped GEMM takes no activation scale, so a quantized
        # dispatch would only have to be dequantized again before gemm1.
        return True

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # Input is pre-permuted: (total_rows * topk, hidden_size).
        # Treat it as M = total_rows * topk with topk=1 for workspace
        # allocation since permutation is already done.
        # bf16/fp8 weights are [E, K, N]; int4/mxfp4 pack K, giving
        # [E, N, K // pack].
        assert len(w1.shape) == 3 and len(w2.shape) == 3
        E = w1.shape[0]
        N = w1.shape[-2] if (self.is_int4 or self.is_mxfp4) else w1.shape[-1]
        K = a1.size(-1)
        M = a1.size(0)
        topk = 1
        return E, M, N, K, topk

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (None, None),
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
            (kInt4Static, None),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # M is already total_rows * topk (pre-permuted), topk=1 from
        # moe_problem_size override.
        inter_size = N // 2
        is_relu2_no_mul = (activation == MoEActivation.RELU2_NO_MUL)
        inter_size_scale = 2 if is_relu2_no_mul else 1

        workspace13 = (M, 2 * inter_size)
        workspace2 = (M, inter_size * inter_size_scale)
        output = (M, K)
        return (workspace13, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        assert expert_tokens_meta is not None, (
            "XPUGroupedGemmExperts requires expert_tokens_meta with "
            "rows_per_expert from PrepareFinalize"
        )
        rows_per_expert = expert_tokens_meta.expert_num_tokens

        num_experts = self.moe_config.num_local_experts
        num_moe_inputs = hidden_states.size(0)

        # This tree's cutlass_grouped_gemm_interface takes no activation
        # scale, so quantized activations must be dequantized first.
        if a1q_scale is not None:
            hidden_states = _dequantize_to_bf16(hidden_states, a1q_scale)

        hidden_size = hidden_states.size(-1)

        if self.is_int4 or self.is_mxfp4:
            # Packed weights are [E, 2*inter_size, hidden_size // pack].
            inter_size = w1.shape[-2] // 2
        else:
            # XPU weight layout must be [E, K, N] (transposed by
            # prepare_fp8_moe_layer_for_xpu during weight loading).
            assert w1.shape[1] == hidden_size, (
                f"XPUGroupedGemmExperts expects weights in [E, K, N] layout "
                f"(K={hidden_size}), but got w1.shape={list(w1.shape)}. "
                f"Ensure prepare_fp8_moe_layer_for_xpu ran during weight loading."
            )
            inter_size = w1.shape[-1] // 2
        is_relu2_no_mul = (activation == MoEActivation.RELU2_NO_MUL)
        inter_size_scale = 2 if is_relu2_no_mul else 1

        # gemm1: hidden_states @ w13 -> workspace13
        gemm1_output = workspace13[:num_moe_inputs, :2 * inter_size]
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=hidden_states,
            ptr_B=w1,
            ptr_scales=self.w1_scale,
            ptr_bias=self.w1_bias,
            ptr_D=gemm1_output,
            rows_per_expert=rows_per_expert,
            N=2 * inter_size,
            K=hidden_size,
            num_experts=num_experts,
            is_B_int4=self.is_int4,
            is_B_mxfp4=self.is_mxfp4,
        )

        # activation
        act_output = workspace2[:num_moe_inputs, :inter_size * inter_size_scale]
        apply_moe_activation(activation, act_output, gemm1_output)

        # gemm2: act_output @ w2 -> output
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=act_output,
            ptr_B=w2,
            ptr_scales=self.w2_scale,
            ptr_bias=self.w2_bias,
            ptr_D=output,
            rows_per_expert=rows_per_expert,
            N=hidden_size,
            K=inter_size * inter_size_scale,
            num_experts=num_experts,
            is_B_int4=self.is_int4,
            is_B_mxfp4=self.is_mxfp4,
        )
