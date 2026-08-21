# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import ClassVar

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .BlockScaledMMLinearKernel import FP8BlockParams, Fp8BlockScaledMMLinearKernel
from .ScaledMMLinearKernel import FP8ScaledMMLinearKernel, FP8ScaledMMLinearLayerConfig


class XPUFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFP8ScaledMM only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key not in {kFp8StaticChannelSym, kFp8StaticTensorSym}:
            return (
                False,
                "XPUFP8ScaledMM only support per-channel and per-tensor quantization",
            )
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUFP8ScaledMM only support FP8 weight dtype"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # fp8_gemm_w8a16 expects weight in [in, out] layout.
        # Transpose if weight is still in [out, in] layout.
        # For square matrices, use contiguity as tie-breaker:
        # checkpoint weights are contiguous, .t() views are not.
        weight = layer.weight
        out_features, in_features = self.config.weight_shape

        if weight.shape == (out_features, in_features) and (
            in_features != out_features or weight.is_contiguous()
        ):
            replace_parameter(layer, "weight", weight.data.t())
        # else: already in [in, out] layout — no-op

        weight_scale = layer.weight_scale.t().contiguous()
        replace_parameter(layer, "weight_scale", weight_scale.data)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        return torch.ops._xpu_C.fp8_gemm_w8a16(x, weight, weight_scale, bias)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        pass


class XPUFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    # oneDNN fp8_gemm wants the block scale transposed relative to the
    # checkpoint layout, but `weight_scale_inv` on the layer stays in checkpoint
    # layout because later consumers re-read it against the [out, in] weight
    # (MLA's get_and_maybe_dequant_weights, xpu_sparse's wo_a dequant). So the
    # transposed copy lives under its own name and is swapped in on the way to
    # apply_block_scaled_mm.
    TRANSPOSED_SCALE: ClassVar[str] = "weight_scale_xpu_t"

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFp8BlockScaledMM only support on XPU"
        return True, None

    def _get_layer_params(self, layer: torch.nn.Module, **kwargs) -> FP8BlockParams:
        params = super()._get_layer_params(layer, **kwargs)
        scale_t = getattr(layer, self.TRANSPOSED_SCALE, None)
        if scale_t is not None:
            if params.weight_scale_inv is not None:
                params.weight_scale_inv = scale_t
            else:
                params.weight_scale = scale_t
        return params

    def process_weights_after_loading(self, layer: torch.nn.Module):
        super().process_weights_after_loading(layer)
        scale_attr = (
            "weight_scale_inv" if hasattr(layer, "weight_scale_inv") else "weight_scale"
        )
        scale = getattr(layer, scale_attr)
        # Models with scale_fmt=ue8m0 (e.g. DeepSeek-V4) store weight scales
        # as float8_e8m0fnu. The oneDNN fp8_gemm kernel dispatches to its
        # "block quant" path only when NEITHER scale is e8m0:
        #
        #   is_block_quant = (m1_sc != e8m0) && (m2_sc != e8m0) && ...
        #
        # Since activation scales are always float32 (use_ue8m0=False on XPU,
        # DeepGEMM requires Hopper/Blackwell), an e8m0 weight scale causes
        # is_block_quant=false and falls into the wrong per-channel path,
        # producing NaN. Converting e8m0→float32 here at load time (one-time,
        # negligible overhead for small scale tensors) ensures the kernel sees
        # matching dtypes and correctly enters the block-quant path with the
        # actual group_size derived from scale tensor shapes.
        if scale.dtype == torch.float8_e8m0fnu:
            scale = scale.to(torch.float32)
            replace_parameter(layer, scale_attr, scale.data)
        layer.register_buffer(
            self.TRANSPOSED_SCALE, scale.data.t().contiguous(), persistent=False
        )

        # oneDNN can only build the block-quant matmul when the output dim is a
        # whole number of scale blocks, and DeepSeek/GLM fused_qkv_a_proj is not
        # (2048 + 512 + 64 = 2624). The checkpoint scale already has a row for
        # the ragged block, so zero-padding the weight is enough; apply_weights
        # slices the padding back off.
        weight = layer.weight
        block_n = self.weight_group_shape[0]
        pad = (-weight.shape[0]) % block_n
        if pad:
            padded = weight.data.new_zeros((weight.shape[0] + pad, weight.shape[1]))
            padded[: weight.shape[0]] = weight.data
            replace_parameter(layer, "weight", padded)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Bias is applied here rather than by the base class so that it still
        # matches the unpadded output width.
        output = super().apply_weights(layer, x, bias=None, **kwargs)
        out_features = self.config.weight_shape[0]
        if output.shape[-1] != out_features:
            output = output[..., :out_features].contiguous()
        if bias is not None:
            output = output + bias
        return output

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # Weight is [N, K]. Use .t() to create a [K, N] view without copying.
        return torch.ops._xpu_C.fp8_gemm(
            A,
            B.t(),
            self.config.out_dtype,
            As,
            Bs,
            torch.Tensor(),
        )
