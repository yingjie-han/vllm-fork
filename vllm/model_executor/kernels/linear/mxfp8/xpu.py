# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    xpu_mxfp8_quantize as quant_mxfp8,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class XPUMxFp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM on XPU."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUMxFp8 only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Checkpoint scale is [N, K//32] (one E8M0 scale per 32 elements).
        # oneDNN fp8_gemm requires contiguous [K//32, N] layout. Store the
        # transposed contiguous buffer as a .t() view so that:
        #   - dequantization consumers still see the checkpoint shape
        #   - apply_weights recovers the oneDNN layout via .t() at zero cost
        weight_scale = layer.weight_scale.view(torch.float8_e8m0fnu)
        scale_kn = weight_scale.data.t().contiguous()
        replace_parameter(layer, "weight_scale", scale_kn.t())

<<<<<<< HEAD
        # Weight is stored as [N, K] and .t()'d to [K, N] in apply. Default keeps
        # it K-contiguous ("ba"); when forced, repack to N-contiguous ("ab")
        # while preserving the [N, K] shape.
        force_ab = envs.VLLM_XPU_FORCE_AB_LAYOUT_WEIGHT and not getattr(
            layer, "is_bmm", False
        )
        if force_ab:
            weight_kn = layer.weight.data.t().contiguous().t()
=======
        # Weight loads as [N, K] (K-contiguous, "ba"). When forced, store a
        # physically K-major [K, N] contiguous buffer ("ab") so apply_weights
        # can feed oneDNN without relying on a transposed view. The bmm path
        # keeps the [N, K] layout since it builds its own batched buffers.
        force_ab = envs.VLLM_XPU_MXFP8_FORCE_AB_LAYOUT and not getattr(
            layer, "is_bmm", False
        )
        if force_ab:
            weight_kn = layer.weight.data.t().contiguous()
>>>>>>> 44a42177a2 ([XPU] Add env var to force MXFP8 weight into ab layout)
            replace_parameter(layer, "weight", weight_kn)

        if getattr(layer, "is_bmm", False):
            self._prepare_bmm_params(layer, scale_kn)

    def _prepare_bmm_params(
        self, layer: torch.nn.Module, scale_kn: torch.Tensor
    ) -> None:
        """Precompute batched weight and scale for grouped fp8_bmm (e.g. wo_a).

        Splits scale [K//32, N_total] into [G, K//32, N_per_group] and weight
        [N_total, K] into contiguous [G, K, N_per_group] for batch GEMM.
        """
        batch = layer.bmm_batch_size
        k_blocks, n_blocks = scale_kn.shape
        layer.bmm_scale = (
            scale_kn.reshape(k_blocks, batch, n_blocks // batch)
            .permute(1, 0, 2)
            .contiguous()
        )
        w = layer.weight
        n_total, k = w.shape
        layer.bmm_weight = (
            w.reshape(batch, n_total // batch, k).permute(0, 2, 1).contiguous()
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dtype = x.dtype
        x_fp8, x_scale = quant_mxfp8(x)
        # Weight is [N, K]; .t() gives a [K, N] view without copying. When
        # VLLM_XPU_MXFP8_FORCE_AB_LAYOUT is set, weight is already stored as a
        # contiguous [K, N] buffer, so it is passed directly.
        # Scale is stored as [N, K//32]; .t() recovers the contiguous
        # [K//32, N] buffer that oneDNN expects.
        weight = layer.weight
        if not envs.VLLM_XPU_MXFP8_FORCE_AB_LAYOUT:
            weight = weight.t()
        return torch.ops._xpu_C.fp8_gemm(
            x_fp8,
            weight,
            out_dtype,
            x_scale,
            layer.weight_scale.t(),
            bias,
        )
