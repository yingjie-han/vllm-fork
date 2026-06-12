# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU DeepSeek-V4 attention subclasses.

Provides two concrete attention classes for XPU:

* ``DeepseekV4XPUAttention`` – Triton-based baseline (always available).
  Decode uses ``xpu_sparse_decode_fp8``; prefill uses
  ``triton_bf16_mla_sparse_interface``.

* ``DeepseekV4XPUFlashMLAAttention`` – optimised path using xattention
  (``flash_attn`` package).  Active when ``flash_attn`` is importable.
  Decode uses ``flash_mla_with_kvcache``; prefill uses
  ``flash_mla_sparse_fwd``.  Both backends share the same block-segregated
  fp8_ds_mla KV cache layout so no format conversion is needed.

``get_deepseek_v4_xpu_attn_cls()`` returns the best available class.
"""

from typing import TYPE_CHECKING, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.models.deepseek_v4.xpu.xpu_sparse_decode_fp8 import (
    xpu_sparse_decode_fp8,
)
from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

# ---------------------------------------------------------------------------
# Optional xattention (flash_attn) backend
# ---------------------------------------------------------------------------
try:
    from flash_attn.flash_attn_interface_xpu import (
        flash_mla_sparse_fwd as _flash_mla_sparse_fwd,
        flash_mla_with_kvcache as _flash_mla_with_kvcache,
    )
    _XATTN_AVAILABLE = True
except ImportError:
    _XATTN_AVAILABLE = False


class DeepseekV4XPUSparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "XPU_V4_MLA_SPARSE"


class DeepseekV4XPUAttention(DeepseekV4Attention):
    """XPU sparse MLA attention layer for DeepSeek V4."""

    backend_cls = DeepseekV4XPUSparseBackend
    use_flashmla_fp8_layout = True

    def __init__(self, *args, **kwargs) -> None:
        # torch.cuda.Event() raises RuntimeError on XPU ("dummy base class").
        # The Base and DeepseekV4Indexer both create cuda Events in __init__, so
        # we temporarily redirect torch.cuda.Event → torch.xpu.Event.
        _orig_event = torch.cuda.Event
        torch.cuda.Event = torch.xpu.Event  # type: ignore[misc]
        try:
            super().__init__(*args, **kwargs)
        finally:
            torch.cuda.Event = _orig_event  # type: ignore[misc]

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        from typing import cast

        if not isinstance(attn_metadata, dict):
            # Profile run: no-op, just return q (no padding needed on XPU).
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
            xpu_qnorm_rope_kv_fp8_insert,
        )

        xpu_qnorm_rope_kv_fp8_insert(
            q,
            kv,
            self.swa_cache_layer.kv_cache,
            swa_metadata.slot_mapping,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )
        return q

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # XPU uses BF16 reference wo_a path (same as ROCm).
        from vllm.models.deepseek_v4.amd.rocm import rocm_inv_rope_einsum

        z = rocm_inv_rope_einsum(
            self.rotary_emb,
            o,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        return self.wo_b(z.flatten(1))

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: reserve workspace, skip actual kernels.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        assert swa_indices is not None and swa_lens is not None
        self._run_decode_attn(
            q, kv_cache, swa_indices, swa_lens, topk_indices, topk_lens,
            swa_only, output,
        )

    def _run_decode_attn(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        topk_indices: torch.Tensor | None,
        topk_lens: torch.Tensor | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        """Dispatch the decode attention kernel (Triton FP8 path)."""
        xpu_sparse_decode_fp8(
            q=q,
            kv_cache=kv_cache,
            swa_kv_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            attn_sink=self.attn_sink,
            softmax_scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            out=output,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        chunk_size_const = self.PREFILL_CHUNK_SIZE
        num_chunks = (num_prefills + chunk_size_const - 1) // chunk_size_const

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((chunk_size_const, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size_const
            chunk_end = min(chunk_start + chunk_size_const, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )

            self._run_prefill_attn_chunk(
                q_chunk=q[query_start:query_end],
                kv_chunk=kv[:chunk_size],
                combined_indices=combined_indices,
                combined_lens=combined_lens,
                output_chunk=output[query_start:query_end],
            )

    def _run_prefill_attn_chunk(
        self,
        q_chunk: torch.Tensor,
        kv_chunk: torch.Tensor,
        combined_indices: torch.Tensor,
        combined_lens: torch.Tensor,
        output_chunk: torch.Tensor,
    ) -> None:
        """Dispatch one prefill chunk's attention kernel (Triton BF16 path)."""
        kv_ws = kv_chunk.reshape(-1, 1, q_chunk.shape[-1])
        out, _, _ = triton_bf16_mla_sparse_interface(
            q=q_chunk,
            kv=kv_ws,
            indices=combined_indices.unsqueeze(1),
            sm_scale=self.scale,
            d_v=q_chunk.shape[-1],
            block_dpe=0,
        )
        output_chunk[:] = out


class DeepseekV4XPUFlashMLAAttention(DeepseekV4XPUAttention):
    """XPU sparse MLA attention using xattention (flash_attn) kernels.

    Overrides only the two kernel-dispatch hooks; all shared setup logic
    (topk index computation, KV gather, chunk loop, etc.) lives in the base
    class ``DeepseekV4XPUAttention``.

    Active when the ``flash_attn`` package (xattention) is importable.
    """

    def _run_decode_attn(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        topk_indices: torch.Tensor | None,
        topk_lens: torch.Tensor | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        """xattention FP8 sparse decode kernel."""
        # q: (num_decode_tokens, h_q, d_qk) → (b, s_q=1, h_q, d_qk)
        # kv caches: (nb, bs, 584) → (nb, bs, 1, 584)
        _flash_mla_with_kvcache(
            q=q.unsqueeze(1),
            k_cache=self.swa_cache_layer.kv_cache.unsqueeze(-2),
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            tile_scheduler_metadata=None,
            num_splits=None,
            softmax_scale=self.scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=swa_indices,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache.unsqueeze(-2) if kv_cache is not None else None,
            extra_indices_in_kvcache=topk_indices,
            topk_length=swa_lens,
            extra_topk_length=topk_lens,
            out=output.unsqueeze(1),
        )

    def _run_prefill_attn_chunk(
        self,
        q_chunk: torch.Tensor,
        kv_chunk: torch.Tensor,
        combined_indices: torch.Tensor,
        combined_lens: torch.Tensor,
        output_chunk: torch.Tensor,
    ) -> None:
        """xattention BF16 sparse prefill kernel."""
        _flash_mla_sparse_fwd(
            q=q_chunk,
            kv=kv_chunk.view(-1, 1, q_chunk.shape[-1]),
            indices=combined_indices.unsqueeze(1),
            sm_scale=self.scale,
            attn_sink=self.attn_sink,
            topk_length=combined_lens,
            out=output_chunk,
        )


def get_deepseek_v4_xpu_attn_cls() -> type[DeepseekV4XPUAttention]:
    """Return the best available XPU attention class.

    Uses ``DeepseekV4XPUFlashMLAAttention`` (xattention kernels) when the
    ``flash_attn`` package is importable, otherwise falls back to the
    Triton-based ``DeepseekV4XPUAttention``.
    """
    if _XATTN_AVAILABLE:
        return DeepseekV4XPUFlashMLAAttention
    return DeepseekV4XPUAttention
