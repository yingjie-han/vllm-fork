# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sink-capable XPU sparse MLA backend for HY V4.

HY V4 adds a per-head learnable attention sink on top of sparse MLA. The
platform default XPU sparse backend
(:class:`vllm.v1.attention.backends.mla.xpu_mla_sparse.XPUMLASparseBackend`)
uses a Triton BF16 kernel that has no notion of ``attn_sink`` and rejects any
FP8 KV cache dtype, so both the sink bias and the FP8 layout would be
unavailable.

Deepklox (see https://github.com/intel-innersource/applications.ai.gpu.deepklox)
provides two native XPU kernels that mirror the NVIDIA FlashMLA sparse
interfaces and accept ``attn_sink``:

* ``flash_mla_sparse_fwd``   — BF16 sparse forward (backs BF16 KV cache).
* ``flash_mla_with_kvcache`` — FP8 sparse decode over the DSV3.2 656-byte
  packed KV cache (backs ``fp8_ds_mla``). HY V4's q head_dim (kv_lora_rank=512
  + qk_rope_head_dim=64 = 576) matches the DSV3.2 layout the deepklox kernel
  supports.

For the FP8 path this module intentionally uses the same "mixed batch"
approach as the NVIDIA impl when the per-rank head count is small: every
token goes through the FP8 decode kernel, so no FP8→BF16 upconversion of KV
is needed and the backend has no CUDA-only dependency
(``cp_gather_and_upconvert_fp8_kv_cache``). Deepklox's decode kernel requires
``seq_len_q == 1``, so we treat each token as its own batch of length 1
instead of the NVIDIA impl's ``batch=1, seq=T`` reshape.
"""

from typing import TYPE_CHECKING, ClassVar, Optional

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deepklox
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseBackend,
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
)
from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

# DSV3.2 packed KV cache layout: 512B fp8 NoPE + 16B fp32 scales + 128B bf16
# RoPE. HY V4 shares this layout because its q head_dim is also 576.
_FP8_DS_MLA_BYTES_PER_TOKEN = 656


def _compute_fp8_decode_padded_heads(num_heads: int) -> int:
    """Pad h_q to the FP8 decode kernel's supported multiples (64 or 128).

    Matches ``FlashMLASparseImpl._compute_fp8_decode_padded_heads`` so a HY V4
    layer scheduled onto XPU sees the same padding decision as on NVIDIA.
    """
    return 64 if num_heads <= 64 else 128


class HYV4XPUMLASparseImpl(XPUMLASparseImpl):
    """XPU sparse MLA impl that applies HY V4's per-head learnable sink.

    The sink enters as the ``sinks`` impl kwarg of
    :class:`vllm.model_executor.layers.attention.MLAAttention` and is folded
    into the softmax denominator by the underlying kernel:
    ``out *= exp(lse) / (exp(lse) + exp(sink))``.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Optional["Indexer"] = None,
        **mla_args,
    ) -> None:
        # ``XPUMLASparseImpl`` takes explicit keyword arguments only, so the
        # sink has to be removed before the base class sees ``mla_args``.
        sinks: torch.Tensor | None = mla_args.pop("sinks", None)
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            topk_indices_buffer=topk_indices_buffer,
            indexer=indexer,
            **mla_args,
        )
        self._validate_sinks(sinks, num_heads)
        self.sinks = sinks
        self._use_deepklox = current_platform.is_xpu() and has_deepklox()
        self.use_fp8_kv_cache = kv_cache_dtype == "fp8_ds_mla"
        self.fp8_decode_padded_heads = _compute_fp8_decode_padded_heads(num_heads)

        if self.use_fp8_kv_cache and not self._use_deepklox:
            # No BF16-KV fallback exists for the fp8_ds_mla layout on XPU, so
            # failing early gives a clearer error than an opaque kernel abort.
            raise RuntimeError(
                "HYV4 XPU sparse MLA requires deepklox to serve the "
                "fp8_ds_mla KV cache layout; deepklox is not importable. "
                "Install deepklox or launch with a BF16 KV cache."
            )
        if sinks is not None and not self._use_deepklox:
            logger.warning_once(
                "HYV4 XPU sparse MLA received a learnable sink but deepklox "
                "is unavailable; the sink bias will not be applied."
            )

    @staticmethod
    def _validate_sinks(sinks: torch.Tensor | None, num_heads: int) -> None:
        """Reject sink tensors the sparse-MLA kernels cannot consume.

        Args:
            sinks: Candidate sink tensor, or None when the layer has no sink.
            num_heads: Local (TP-sharded) query head count of this layer.

        Raises:
            ValueError: If the dtype is not float32 or the shape is not
                ``(num_heads,)``.
        """
        if sinks is None:
            return
        if sinks.dtype != torch.float32:
            raise ValueError(
                "HYV4 XPU sparse MLA sinks must have dtype torch.float32, but "
                f"got {sinks.dtype}."
            )
        if sinks.ndim != 1 or sinks.shape[0] != num_heads:
            raise ValueError(
                "HYV4 XPU sparse MLA sinks must have shape "
                f"({num_heads},), but got {tuple(sinks.shape)}."
            )

    def _sinks_for_query(
        self,
        q: torch.Tensor,
        head_dim: int = 1,
        kernel_heads: int | None = None,
    ) -> torch.Tensor | None:
        """Return the sink laid out for the kernel's query head count.

        Args:
            q: Query tensor before any head padding.
            head_dim: Axis of ``q`` holding the query heads (1 for the BF16
                path, 2 for the FP8 ``(T, 1, H, D)`` layout).
            kernel_heads: Head count the kernel is invoked with. When larger
                than the query head count, the sink is padded with ``-inf``
                (a no-op sink) for the padded lanes. ``None`` disables
                padding.

        Returns:
            The sink tensor optionally padded to ``kernel_heads`` or ``None``
            when the layer has no sink.

        Raises:
            ValueError: If the sink and query head layouts disagree, or if
                they live on different devices.
        """
        sinks = self.sinks
        if sinks is None:
            return None
        query_heads = q.shape[head_dim]
        if sinks.shape[0] != query_heads:
            raise ValueError(
                "HYV4 XPU sparse MLA sink head count must match the runtime "
                f"query layout: sinks={sinks.shape[0]}, "
                f"query_heads={query_heads}."
            )
        if sinks.device != q.device:
            raise ValueError(
                "HYV4 XPU sparse MLA sinks and query must be on the same "
                f"device, but got sinks={sinks.device}, query={q.device}."
            )
        if kernel_heads is None or kernel_heads == query_heads:
            return sinks
        if kernel_heads < query_heads:
            raise ValueError(
                "HYV4 XPU sparse kernel head count cannot be smaller than "
                f"the runtime query layout: query_heads={query_heads}, "
                f"kernel_heads={kernel_heads}."
            )
        padded_sinks = sinks.new_full((kernel_heads,), float("-inf"))
        padded_sinks[:query_heads] = sinks
        return padded_sinks

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,  # [num_tokens, num_heads, dim_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [num_blocks, block_size, dim_qk]
        topk_indices: torch.Tensor,  # [num_tokens, topk]
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        # ``flash_mla_sparse_fwd`` expects kv with an explicit h_kv axis and
        # indices reshaped to ``[s_q, h_kv, topk]``, matching what the Triton
        # reference kernel already takes.
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        topk_indices = topk_indices.view(num_tokens, 1, -1)

        if self._use_deepklox:
            from deepklox import flash_mla_sparse_fwd as _dklox_flash_mla_sparse_fwd

            attn_sink = self._sinks_for_query(q, head_dim=1)
            output, _, _ = _dklox_flash_mla_sparse_fwd(
                q,
                kv_c_and_k_pe_cache,
                topk_indices,
                sm_scale=self.softmax_scale,
                d_v=512,
                attn_sink=attn_sink,
                topk_length=None,
                return_softmax_lse=False,
            )
        else:
            output, _, _ = triton_bf16_mla_sparse_interface(
                q,
                kv_c_and_k_pe_cache,
                topk_indices,
                sm_scale=self.softmax_scale,
            )

        return output[:, : self.num_heads, :]

    def _forward_fp8_kv(
        self,
        q: torch.Tensor,  # [num_tokens, num_heads, dim_qk=576]
        kv_c_and_k_pe_cache: torch.Tensor,  # [num_blocks, block_size, 656] uint8
        topk_indices: torch.Tensor,  # [num_tokens, topk] global slot ids
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        """FP8 sparse decode over the DSV3.2 656-byte packed KV cache.

        Deepklox's ``flash_mla_with_kvcache`` requires ``seq_len_q == 1`` per
        batch, so every token becomes its own batch — the mixed-batch idea
        from the NVIDIA impl, reshaped as ``(T, 1, H, D)`` instead of
        ``(1, T, H, D)``.
        """
        from deepklox import flash_mla_with_kvcache as _deepklox_flash_mla_with_kvcache

        num_tokens = q.shape[0]
        actual_num_heads = q.shape[1]
        padded_num_heads = self.fp8_decode_padded_heads

        # Pad h_q to the kernel-supported multiple. Zeroed padding (not
        # new_empty) prevents NaNs in the padded lanes from leaking through
        # the shared topk head reduction.
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                "Padding num_heads from %d to %d for FP8 sparse decode kernel",
                actual_num_heads,
                padded_num_heads,
            )
            q_padded = q.new_zeros(
                (num_tokens, padded_num_heads, q.shape[2]), dtype=q.dtype
            )
            q_padded[:, :actual_num_heads, :] = q
            q = q_padded

        # Add batch and seq_len_q dims: (T, H_pad, 576) -> (T, 1, H_pad, 576).
        q_batched = q.unsqueeze(1)
        # (T, topk) -> (T, 1, topk); indices are already global paged slots.
        topk_indices_batched = topk_indices.view(num_tokens, 1, -1)
        # (num_blocks, block_size, 656) uint8 -> (num_blocks, block_size, 1, 656).
        k_cache = kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2)

        attn_sink = self._sinks_for_query(
            q_batched, head_dim=2, kernel_heads=padded_num_heads
        )

        out, _ = _deepklox_flash_mla_with_kvcache(
            q=q_batched,
            k_cache=k_cache,
            # block_table and cache_seqlens are ignored in sparse mode.
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            softmax_scale=self.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_indices_batched,
            attn_sink=attn_sink,
        )

        # Kernel output shape: (T, 1, H_pad, 512). Drop the seq axis and the
        # padded heads to restore (T, H_actual, 512).
        out = out.squeeze(1)
        return out[:, : self.num_heads, :]

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: XPUMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # BF16 path stays on the parent's forward_mqa (which calls our
        # overridden _forward_bf16_kv); only the FP8 path needs a distinct
        # forward, because the parent hard-rejects quantized KV.
        if not self.use_fp8_kv_cache:
            return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)

        assert is_quantized_kv_cache(self.kv_cache_dtype)

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # Per-request slots -> global paged slots. Shared with the BF16 path
        # in the parent so decode and prefill still share the same index math.
        topk_indices_global = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
        )

        attn_out = self._forward_fp8_kv(
            q, kv_c_and_k_pe_cache, topk_indices_global, attn_metadata
        )
        return attn_out, None


class HYV4XPUMLASparseBackend(XPUMLASparseBackend):
    """XPU sparse MLA backend with attention-sink + FP8 KV support for HY V4.

    Keeps the parent's name (``XPU_MLA_SPARSE``), metadata and builder; only
    the impl class, the sink capability and the supported KV cache dtypes
    differ. Named ``HYV4XPUMLASparseBackend`` for parity with the NVIDIA
    counterpart in :mod:`vllm.models.hy_v4.nvidia`.
    """

    # Extend the parent's dtype set with the DSV3.2 packed FP8 layout that
    # deepklox's ``flash_mla_with_kvcache`` decodes natively on Xe3P. ``fp8``
    # is the CLI alias for ``fp8_ds_mla``.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",
    ]

    @staticmethod
    def get_impl_cls() -> type[HYV4XPUMLASparseImpl]:
        return HYV4XPUMLASparseImpl

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # DSV3.2 packed layout: 512B fp8 NoPE + 16B fp32 scales + 128B bf16
        # RoPE, laid out contiguously as uint8. See deepklox's
        # ``flash_mla_with_kvcache`` docstring for the exact byte layout.
        if cache_dtype_str == "fp8_ds_mla":
            return (num_blocks, block_size, _FP8_DS_MLA_BYTES_PER_TOKEN)
        return XPUMLASparseBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
        )