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

The two kernels are not interchangeable, so the KV cache dtype decides how the
batch is served:

* **BF16 cache** — only the prefill kernel can read it, so the whole batch runs
  on it. Correct, but decode tokens pay the prefill kernel's tiling.
* **``fp8_ds_mla`` cache** (what ``auto`` resolves to) — the batch splits on
  ``num_decode_tokens``, which vLLM sorts to the front:

  * decode tokens go to the FP8 decode kernel, one "request" per token.
    Deepklox's decode kernel requires ``seq_len_q == 1``, and sparse MLA
    scores each token against its own top-k, so per-token batching is exact
    and sidesteps the uniform-decode-length constraint a ``batch=1, seq=T``
    reshape would add;
  * prefill tokens have their pages gathered and up-converted into a shared
    BF16 workspace by ``ops.cp_gather_and_upconvert_fp8_kv_cache`` and then
    run on the prefill kernel, chunked to fit the workspace.

Running prefill tokens through the decode kernel instead is simpler, but a
prefill token does not have a full top-k - token ``i`` has
``min(i + 1, index_topk)`` valid slots - and the prefill kernel exploits that
far better. On Arc B60 at ``topk=2048`` the split is 1.2-4.5x faster depending
on tensor-parallel size and prefill length, while the up-convert itself costs
0.010-0.013 ms. See the pull request for the full measurements.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Optional

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deepklox
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionLayer, CommonAttentionMetadata
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    get_prefill_workspace_size,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseBackend,
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
    XPUMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_prefill_chunks
from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

# DSV3.2 packed KV cache layout: 512B fp8 NoPE + 16B fp32 scales + 128B bf16
# RoPE. HY V4 shares this layout because its q head_dim is also 576.
_FP8_DS_MLA_BYTES_PER_TOKEN = 656
# Full MLA head dim the BF16 prefill kernel and the up-convert workspace use.
_DS_MLA_HEAD_DIM = 576


@dataclass
class HYV4XPUMLASparseMetadata(XPUMLASparseMetadata):
    """`XPUMLASparseMetadata` plus the FP8 prefill up-convert plan."""

    @dataclass
    class PrefillChunk:
        """One workspace-sized group of prefill requests.

        The BF16 workspace is a fixed size, so prefill requests are grouped
        into chunks whose total context length fits in it and processed one
        chunk at a time.
        """

        # Slice of the MQA token axis covered by this chunk.
        tokens_slice: slice
        # Block table rows of this chunk's requests, [chunk_reqs, max_blocks].
        block_table: torch.Tensor
        # Per-request start offset inside the *chunk-local* workspace.
        workspace_starts: torch.Tensor
        # Number of workspace tokens this chunk gathers.
        chunk_tot_seqlen: int

    @dataclass
    class Prefill:
        # -1 for decode tokens, prefill request index otherwise. [num_tokens]
        request_ids: torch.Tensor
        # Chunk-local workspace start per prefill request. [num_prefills]
        workspace_starts: torch.Tensor
        chunks: list["HYV4XPUMLASparseMetadata.PrefillChunk"] = field(
            default_factory=list
        )

    fp8_prefill: "HYV4XPUMLASparseMetadata.Prefill | None" = None


class HYV4XPUMLASparseMetadataBuilder(XPUMLASparseMetadataBuilder):
    """Adds the FP8 prefill up-convert plan to the XPU sparse metadata."""

    metadata_cls: ClassVar[type[XPUMLASparseMetadata]] = HYV4XPUMLASparseMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.use_fp8_kv_cache = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        self.max_prefill_workspace_tokens = get_prefill_workspace_size(
            vllm_config.model_config.max_model_len
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> HYV4XPUMLASparseMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        assert isinstance(metadata, HYV4XPUMLASparseMetadata)

        if self.use_fp8_kv_cache and metadata.num_prefills > 0:
            metadata.fp8_prefill = self._build_prefill_plan(
                common_attn_metadata, metadata
            )
        return metadata

    def _build_prefill_plan(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        metadata: HYV4XPUMLASparseMetadata,
    ) -> HYV4XPUMLASparseMetadata.Prefill:
        """Plan the FP8 -> BF16 gather for this batch's prefill requests.

        vLLM sorts decode requests to the front of the batch, so the prefill
        requests are ``[num_decodes, num_reqs)`` and their tokens are
        ``[num_decode_tokens, num_actual_tokens)``.

        Args:
            common_attn_metadata: Batch metadata from the model runner.
            metadata: The metadata built by the parent builder.

        Returns:
            The prefill plan consumed by `HYV4XPUMLASparseImpl`.
        """
        num_decodes = metadata.num_decodes
        num_prefills = metadata.num_prefills
        num_tokens = metadata.num_actual_tokens

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        # The upper bound is exact for prefill rows, so this needs no D2H sync.
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        assert seq_lens_cpu is not None
        prefill_seq_lens_cpu = seq_lens_cpu[num_decodes : num_decodes + num_prefills]

        # -1 marks a decode token, so a single index-remap call can serve both
        # phases: decode tokens map to global cache slots, prefill tokens to
        # workspace offsets.
        request_ids = torch.full(
            (num_tokens,), -1, dtype=torch.int32, device=self.device
        )
        for req_idx in range(num_prefills):
            global_req_idx = num_decodes + req_idx
            request_ids[
                query_start_loc_cpu[global_req_idx] : query_start_loc_cpu[
                    global_req_idx + 1
                ]
            ] = req_idx

        workspace_starts_cpu = torch.zeros(
            num_prefills, dtype=torch.int32, pin_memory=True
        )
        workspace_starts_cpu[1:] = torch.cumsum(prefill_seq_lens_cpu[:-1], dim=0)
        workspace_starts = torch.empty(
            num_prefills, dtype=torch.int32, device=self.device
        )

        chunks: list[HYV4XPUMLASparseMetadata.PrefillChunk] = []
        for chunk_start, chunk_end in split_prefill_chunks(
            prefill_seq_lens_cpu, self.max_prefill_workspace_tokens
        ):
            # Rebase this chunk's starts to 0 so they index the chunk-local
            # workspace rather than the whole batch.
            offset = workspace_starts_cpu[chunk_start].item()
            workspace_starts_cpu[chunk_start:chunk_end] -= offset

            token_start = query_start_loc_cpu[num_decodes + chunk_start].item()
            token_end = query_start_loc_cpu[num_decodes + chunk_end].item()
            chunks.append(
                HYV4XPUMLASparseMetadata.PrefillChunk(
                    tokens_slice=slice(token_start, token_end),
                    block_table=common_attn_metadata.block_table_tensor[
                        num_decodes + chunk_start : num_decodes + chunk_end
                    ],
                    workspace_starts=workspace_starts[chunk_start:chunk_end],
                    chunk_tot_seqlen=int(
                        prefill_seq_lens_cpu[chunk_start:chunk_end].sum()
                    ),
                )
            )

        # Copy once, after every chunk has rebased its slice in place.
        workspace_starts.copy_(workspace_starts_cpu, non_blocking=True)

        return HYV4XPUMLASparseMetadata.Prefill(
            request_ids=request_ids,
            workspace_starts=workspace_starts,
            chunks=chunks,
        )


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

        # BF16 workspace the FP8 prefill pages are up-converted into, shared
        # across layers: only one layer runs at a time and the gather rewrites
        # every row it uses.
        #
        # Reserved here, at model load, and deliberately not on first use. The
        # workspace manager is sized by memory profiling and then locked by
        # capture_model(), so a later first-use allocation would both escape
        # the KV cache budget and trip the "workspace is locked" assertion.
        # ``__init__`` also runs under set_current_vllm_config, which the
        # forward pass does not. Mirrors FlashMLASparseImpl.__init__.
        self.prefill_bf16_workspace: torch.Tensor | None = None
        if self.use_fp8_kv_cache:
            vllm_config = get_current_vllm_config()
            assert vllm_config is not None and vllm_config.model_config is not None
            workspace_shape = (
                get_prefill_workspace_size(vllm_config.model_config.max_model_len),
                _DS_MLA_HEAD_DIM,
            )
            (self.prefill_bf16_workspace,) = (
                current_workspace_manager().get_simultaneous(
                    (workspace_shape, torch.bfloat16),
                )
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

    def _sparse_prefill(
        self,
        q: torch.Tensor,  # [num_tokens, num_heads, dim_qk]
        kv: torch.Tensor,  # [s_kv, dim_qk] or [s_kv, 1, dim_qk], bf16
        topk_indices: torch.Tensor,  # [num_tokens, topk]
        topk_length: torch.Tensor | None = None,  # [num_tokens]
    ) -> torch.Tensor:
        """Run the deepklox BF16 sparse prefill kernel over an unpaged KV table.

        Args:
            q: Queries for the tokens being served.
            kv: Contiguous BF16 KV rows the indices address.
            topk_indices: Per-token indices into ``kv``.
            topk_length: Per-token count of valid indices, or None when the
                padding is already marked with -1.

        Returns:
            The attention output, ``[num_tokens, num_heads, 512]``.
        """
        from deepklox import flash_mla_sparse_fwd as _dklox_flash_mla_sparse_fwd

        num_tokens = q.shape[0]
        attn_sink = self._sinks_for_query(q, head_dim=1)
        output, _, _ = _dklox_flash_mla_sparse_fwd(
            q,
            kv.view(-1, 1, kv.shape[-1]),
            topk_indices.view(num_tokens, 1, -1),
            sm_scale=self.softmax_scale,
            d_v=512,
            attn_sink=attn_sink,
            topk_length=topk_length,
            return_softmax_lse=False,
        )
        return output

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,  # [num_tokens, num_heads, dim_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [num_blocks, block_size, dim_qk]
        topk_indices: torch.Tensor,  # [num_tokens, topk]
        topk_length: torch.Tensor | None,  # [num_tokens]
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        if self._use_deepklox:
            output = self._sparse_prefill(
                q, kv_c_and_k_pe_cache, topk_indices, topk_length
            )
        else:
            # ``triton_bf16_mla_sparse_interface`` expects kv with an explicit
            # h_kv axis and indices reshaped to ``[s_q, h_kv, topk]``.
            output, _, _ = triton_bf16_mla_sparse_interface(
                q,
                kv_c_and_k_pe_cache.view(-1, 1, kv_c_and_k_pe_cache.shape[-1]),
                topk_indices.view(q.shape[0], 1, -1),
                sm_scale=self.softmax_scale,
            )

        return output[:, : self.num_heads, :]

    def _sparse_decode(
        self,
        q: torch.Tensor,  # [num_tokens, num_heads, dim_qk=576]
        kv_c_and_k_pe_cache: torch.Tensor,  # [num_blocks, block_size, 656] uint8
        topk_indices: torch.Tensor,  # [num_tokens, topk] global slot ids
    ) -> torch.Tensor:
        """Run the deepklox packed-FP8 sparse decode kernel.

        The kernel takes one query token per request, so each token is driven
        as its own request (``b = num_tokens, s_q = 1``). Sparse MLA scores
        every token against its own top-k, so that is exact and it sidesteps
        the uniform-decode-length constraint a ``[b, s_q]`` reshape would add.

        Args:
            q: Queries for the decode tokens.
            kv_c_and_k_pe_cache: The paged fp8_ds_mla cache.
            topk_indices: Global cache slot ids per token.

        Returns:
            The attention output, ``[num_tokens, num_heads, 512]``.
        """
        from deepklox import flash_mla_with_kvcache as _deepklox_flash_mla_with_kvcache

        num_tokens = q.shape[0]

        # Add batch and seq_len_q dims: (T, H, 576) -> (T, 1, H, 576).
        q_batched = q.unsqueeze(1)
        # (T, topk) -> (T, 1, topk); indices are already global paged slots.
        topk_indices_batched = topk_indices.view(num_tokens, 1, -1)
        # (num_blocks, block_size, 656) uint8 -> (num_blocks, block_size, 1, 656).
        # vLLM allocates the fp8_ds_mla cache as uint8; the kernel takes the
        # same bytes but requires the page typed float8_e4m3fn. Both are one
        # byte, so the re-view does not change the shape.
        k_cache = kv_c_and_k_pe_cache.view(torch.float8_e4m3fn).unsqueeze(-2)

        attn_sink = self._sinks_for_query(q_batched, head_dim=2)

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

        # Kernel output shape: (T, 1, H, 512). Drop the seq axis.
        return out.squeeze(1)

    def _forward_fp8_kv(
        self,
        q: torch.Tensor,  # [num_tokens, num_heads, dim_qk=576]
        kv_c_and_k_pe_cache: torch.Tensor,  # [num_blocks, block_size, 656] uint8
        topk_indices: torch.Tensor,  # [num_tokens, topk]
        topk_length: torch.Tensor,  # [num_tokens]
        attn_metadata: HYV4XPUMLASparseMetadata,
    ) -> torch.Tensor:
        """Serve an fp8_ds_mla batch, decode and prefill on their own kernels.

        vLLM sorts decode requests to the front, so the batch splits at
        ``num_decode_tokens``:

        * decode tokens go straight to the packed-FP8 decode kernel;
        * prefill tokens have their pages gathered and up-converted into a
          shared BF16 workspace and run on the BF16 prefill kernel, chunked
          when they do not all fit.

        Args:
            q: Queries for the whole MQA batch.
            kv_c_and_k_pe_cache: The paged fp8_ds_mla cache.
            topk_indices: Global cache slots for decode tokens, chunk-local
                workspace offsets for prefill tokens.
            topk_length: Per-token count of valid indices.
            attn_metadata: Carries the prefill up-convert plan.

        Returns:
            The attention output, ``[num_tokens, num_heads, 512]``.
        """
        num_mqa_tokens = q.shape[0]
        num_decode_tokens = min(attn_metadata.num_decode_tokens, num_mqa_tokens)
        num_prefill_tokens = num_mqa_tokens - num_decode_tokens

        # The chunk token slices index the *whole* batch, so a partial batch
        # that still carries prefill tokens would scatter the output into the
        # wrong rows. HY V4 forces every token onto sparse MQA, so only the
        # decode-only subset or the full batch can arrive here.
        assert num_prefill_tokens == 0 or num_mqa_tokens == (
            attn_metadata.num_actual_tokens
        ), (
            "fp8_ds_mla sparse MLA expects either the decode subset or the "
            f"full batch, got {num_mqa_tokens} of "
            f"{attn_metadata.num_actual_tokens} tokens."
        )

        if num_prefill_tokens == 0:
            return self._sparse_decode(q, kv_c_and_k_pe_cache, topk_indices)[
                :, : self.num_heads, :
            ]

        attn_out = q.new_empty((num_mqa_tokens, q.shape[1], 512))
        if num_decode_tokens > 0:
            attn_out[:num_decode_tokens] = self._sparse_decode(
                q[:num_decode_tokens],
                kv_c_and_k_pe_cache,
                topk_indices[:num_decode_tokens],
            )

        prefill = attn_metadata.fp8_prefill
        assert prefill is not None, (
            "fp8_ds_mla prefill tokens require the prefill up-convert plan; "
            "HYV4XPUMLASparseMetadataBuilder did not build one."
        )
        workspace = self.prefill_bf16_workspace
        assert workspace is not None
        for chunk in prefill.chunks:
            chunk_workspace = workspace[: chunk.chunk_tot_seqlen]
            ops.cp_gather_and_upconvert_fp8_kv_cache(
                kv_c_and_k_pe_cache,
                chunk_workspace,
                chunk.block_table,
                chunk.workspace_starts,
                len(chunk.block_table),
            )
            attn_out[chunk.tokens_slice] = self._sparse_prefill(
                q[chunk.tokens_slice],
                chunk_workspace,
                topk_indices[chunk.tokens_slice],
                topk_length[chunk.tokens_slice],
            )

        return attn_out[:, : self.num_heads, :]

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

        assert isinstance(attn_metadata, HYV4XPUMLASparseMetadata)
        num_actual_toks = q.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        num_decode_tokens = min(attn_metadata.num_decode_tokens, num_actual_toks)
        prefill = attn_metadata.fp8_prefill
        if num_actual_toks > num_decode_tokens:
            assert prefill is not None, (
                "fp8_ds_mla prefill tokens require the prefill up-convert "
                "plan; HYV4XPUMLASparseMetadataBuilder did not build one."
            )
        else:
            prefill = None

        # One remap for the whole batch: decode tokens become global cache
        # slots, prefill tokens become chunk-local workspace offsets (the
        # chunk rebasing is already baked into ``workspace_starts``).
        # ``return_valid_counts`` gives the topk_length the prefill kernel
        # wants, since the workspace rows it addresses are dense.
        topk_indices_global, topk_length = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
            HAS_PREFILL_WORKSPACE=prefill is not None,
            prefill_workspace_request_ids=(
                None if prefill is None else prefill.request_ids[:num_actual_toks]
            ),
            prefill_workspace_starts=(
                None if prefill is None else prefill.workspace_starts
            ),
            return_valid_counts=True,
        )

        attn_out = self._forward_fp8_kv(
            q,
            kv_c_and_k_pe_cache,
            topk_indices_global,
            topk_length,
            attn_metadata,
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

    @staticmethod
    def get_metadata_cls() -> type[HYV4XPUMLASparseMetadata]:
        return HYV4XPUMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type[HYV4XPUMLASparseMetadataBuilder]:
        return HYV4XPUMLASparseMetadataBuilder

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def canonicalize_kv_cache_dtype(cls, kv_cache_dtype: CacheDType) -> CacheDType:
        """Map the unspecified/generic FP8 dtypes onto ``fp8_ds_mla``.

        Only the packed-FP8 ``fp8_ds_mla`` layout has a sparse *decode* kernel
        on XPU, so it is also what ``auto`` resolves to: a BF16 cache would put
        every decode token back on the prefill kernel. Asking for ``bfloat16``
        or ``float16`` explicitly still gets an unquantized cache.

        Args:
            kv_cache_dtype: The dtype the cache config asked for.

        Returns:
            The dtype this backend will actually allocate.
        """
        if kv_cache_dtype in ("auto", "fp8", "fp8_e4m3"):
            return "fp8_ds_mla"
        return kv_cache_dtype

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
