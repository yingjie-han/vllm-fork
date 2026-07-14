# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU dispatcher for DeepSeekV4 ``dequantize_and_gather_k_cache``.

Prefers the SYCL kernel provided by ``vllm-xpu-kernels``
(``torch.ops._C_cache_ops.dequantize_and_gather_k_cache``) and falls back to
the shared Triton implementation when the SYCL op is unavailable or reports
that the requested configuration is unsupported.

The dispatch can be forced with the ``VLLM_XPU_DEQUANT_GATHER_IMPL`` env var
(values: ``sycl`` (default) or ``triton``). Set
``VLLM_XPU_DEQUANT_GATHER_IMPL_STRICT=1`` to disable the Triton fallback and
surface SYCL errors instead.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops import (
    dequantize_and_gather_k_cache as _dequantize_and_gather_k_cache_triton,
)

logger = init_logger(__name__)


def _sycl_op_available() -> bool:
    """Return True when the SYCL dequant-gather op is registered."""
    try:
        # Importing registers the ops under ``torch.ops._C_cache_ops``.
        import vllm_xpu_kernels._C  # noqa: F401
    except ImportError:
        return False
    cache_ops = getattr(torch.ops, "_C_cache_ops", None)
    return cache_ops is not None and hasattr(
        cache_ops, "dequantize_and_gather_k_cache"
    )


def _is_sycl_unsupported_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "dequantize_and_gather_k_cache" in msg and (
        "unsupported" in msg or "not implemented" in msg
    )


def _dequantize_and_gather_k_cache_sycl(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """Call the SYCL kernel, adapting tensors to its expected layout.

    The SYCL op requires a 2D ``uint8`` cache ``[num_blocks, block_bytes]``,
    ``int32`` index tensors and a ``bfloat16`` output. The paged cache is
    allocated as ``[num_blocks, block_size, 584]`` (uint8); flattening the
    trailing dims yields the ``[num_blocks, block_bytes]`` view without a copy.
    """
    k_cache_2d = k_cache.reshape(k_cache.shape[0], -1)
    if k_cache_2d.dtype != torch.uint8:
        k_cache_2d = k_cache_2d.view(torch.uint8)
    seq_lens = seq_lens.to(torch.int32)
    block_table = block_table.to(torch.int32)
    if gather_lens is not None:
        gather_lens = gather_lens.to(torch.int32)

    torch.ops._C_cache_ops.dequantize_and_gather_k_cache(
        out,
        k_cache_2d,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
    )


def dequantize_and_gather_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """Gather and dequantize FP8 K from the paged cache on XPU.

    Args:
        out: ``[num_reqs, max_num_tokens, 512]`` bf16 output buffer.
        k_cache: paged K cache ``[num_blocks, block_size, 584]`` uint8.
        seq_lens: ``[num_reqs]`` per-request sequence lengths.
        gather_lens: optional ``[num_reqs]`` per-request gather lengths.
        block_table: ``[num_reqs, max_blocks_per_seq]`` block indices.
        block_size: paged cache block size.
        offset: token offset into ``out`` to start writing at.
    """
    impl = os.getenv("VLLM_XPU_DEQUANT_GATHER_IMPL", "sycl").strip().lower()
    if impl == "sycl" and _sycl_op_available():
        logger.info_once("XPU dequantize_and_gather_k_cache path: sycl")
        strict = os.getenv("VLLM_XPU_DEQUANT_GATHER_IMPL_STRICT", "0") == "1"
        try:
            _dequantize_and_gather_k_cache_sycl(
                out,
                k_cache,
                seq_lens,
                gather_lens,
                block_table,
                block_size,
                offset,
            )
            return
        except RuntimeError as exc:
            if strict or not _is_sycl_unsupported_error(exc):
                raise
            logger.warning_once(
                "SYCL dequantize_and_gather_k_cache failed or unsupported; "
                "falling back to Triton (set "
                "VLLM_XPU_DEQUANT_GATHER_IMPL_STRICT=1 to disable fallback)."
            )

    logger.info_once("XPU dequantize_and_gather_k_cache path: triton")
    _dequantize_and_gather_k_cache_triton(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
    )
