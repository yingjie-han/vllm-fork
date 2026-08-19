# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    get_mla_dims,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.xpu_mla_sparse import (
    DS_MLA_ENTRY_BYTES,
    triton_bf16_mla_sparse_interface,
    triton_concat_and_cache_ds_mla,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer
logger = init_logger(__name__)

# Optional xattention sparse MLA kernels; the Triton kernel is the BF16
# fallback. There is no Triton fallback for the fp8 path: `sparse_decode_fwd`
# is the only kernel that reads the packed fp8_ds_mla layout.
try:
    from xattention import flash_mla_sparse_fwd as _xattn_flash_mla_sparse_fwd
    from xattention import flash_mla_with_kvcache as _xattn_flash_mla_with_kvcache

    _XATTN_SPARSE_AVAILABLE = True
except ImportError:
    _XATTN_SPARSE_AVAILABLE = False


class XPUMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]

    @staticmethod
    def get_name() -> str:
        return "XPU_MLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type["XPUMLASparseMetadata"]:
        return XPUMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["XPUMLASparseMetadataBuilder"]:
        return XPUMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["XPUMLASparseImpl"]:
        return XPUMLASparseImpl

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "fp8_ds_mla":
            # DeepSeek's segmented fp8 layout; see the module-level note in
            # vllm/v1/attention/ops/xpu_mla_sparse.py.
            return (num_blocks, block_size, DS_MLA_ENTRY_BYTES)
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]


@dataclass
class XPUMLASparseMetadata(AttentionMetadata):
    """Mirrors the field set of the shared sparse-MLA metadata.

    ``MLAAttention.forward_impl`` reads the decode/prefill split and the
    prefill fields off any sparse metadata, so this must stay in sync with
    ``SparseMLACommonMetadataBuilder``'s output even though the XPU impl
    routes every token through the top-k MQA path.
    """

    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    seq_lens: torch.Tensor | None = None

    block_size: int = 1
    topk_tokens: int = 2048
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    prefill_max_seq_len: int = 0
    # XPU has no dense-MHA prefill backend, so prefill metadata is never built.
    prefill: None = None


@dataclass
class XPUMLASparseMetadataBuilder(AttentionMetadataBuilder[XPUMLASparseMetadata]):
    # Every tensor the kernels read is either shape-derived or held in a
    # persistent buffer, and the forward has no host syncs or data-dependent
    # branches, so uniform-query-length batches are graph-capturable.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.topk_tokens_tensor = torch.tensor(
            [self.topk_tokens], device=device, dtype=torch.int32
        )
        self.max_model_len_tensor = torch.tensor(
            [self.model_config.max_model_len], device=device, dtype=torch.int32
        )
        # this is ignored by `flash_mla_with_kvcache` if indices not None
        self.dummy_block_table = torch.empty(
            (1, 1), dtype=torch.int32, device=self.device
        )

        self.req_id_per_token_buffer = torch.empty(
            (max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> XPUMLASparseMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # Zero-fill for cudagraphs
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )

        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]

        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold or 1,
        )

        metadata = XPUMLASparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=req_id_per_token,
            seq_lens=common_attn_metadata.seq_lens,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
        )
        return metadata


class XPUMLASparseImpl(MLAAttentionImpl[XPUMLASparseMetadata]):
    is_sparse = True
    # There is no dense-MHA prefill kernel on XPU, so the shared MLA forward
    # must route every token through the top-k MQA path.
    supports_dense_mha_prefill = False

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
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.softmax_scale = scale
        # The indexer carries the shared buffer for normal layers and tests;
        # the explicitly-passed buffer covers backbone skip layers, whose
        # indexer is not constructed (see deepseek_v2.py).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )
        self.use_xattention = _XATTN_SPARSE_AVAILABLE
        self.use_fp8_kv_cache = kv_cache_dtype == "fp8_ds_mla"
        if self.use_fp8_kv_cache and not _XATTN_SPARSE_AVAILABLE:
            raise NotImplementedError(
                "The fp8_ds_mla KV cache on XPU requires the xattention sparse "
                "decode kernel; the Triton fallback is BF16-only. Re-run with "
                "--kv-cache-dtype auto or install xattention."
            )
        if self.use_xattention:
            logger.info_once(
                "Using xattention sparse MLA kernels on XPU (%s KV cache).",
                "fp8_ds_mla" if self.use_fp8_kv_cache else "BF16",
            )
        else:
            logger.info_once(
                "xattention is not available; falling back to the Triton BF16 "
                "sparse MLA kernel on XPU."
            )

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        if kv_cache_dtype != "fp8_ds_mla":
            super().do_kv_cache_update(
                kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
            )
            return
        # The XPU C++ concat_and_cache_mla only writes the flat layout, so pack
        # the 656-byte segmented entries with the Triton kernel instead.
        triton_concat_and_cache_ds_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
        )

    def _forward_fp8_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, block_size, 656] uint8
        topk_indices: torch.Tensor,  # [sq, topk]
        topk_length: torch.Tensor | None,  # [sq]
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        # `sparse_decode_fwd` requires s_q == 1 and takes one `topk_length` per
        # batch entry. Mapping the flat varlen batch to (b=tokens, s_q=1) makes
        # that per-batch length exactly the per-token length we already have.
        output = torch.zeros(
            (num_tokens, 1, q.shape[1], self.kv_lora_rank),
            dtype=q.dtype,
            device=q.device,
        )
        _xattn_flash_mla_with_kvcache(
            q=q.unsqueeze(1),
            k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),
            block_table=None,
            cache_seqlens=None,
            head_dim_v=self.kv_lora_rank,
            softmax_scale=self.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_indices.view(num_tokens, 1, -1),
            topk_length=topk_length,
            out=output,
        )
        return output.squeeze(1)[:, : self.num_heads, :]

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]
        topk_indices: torch.Tensor,  # [sq, topk]
        topk_length: torch.Tensor | None,  # [sq]
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        topk_indices = topk_indices.view(num_tokens, 1, -1)

        if self.use_xattention:
            # Zero-init (as the Triton kernel does): rows whose `topk_length`
            # is 0 are left untouched by the xattention kernel, and empty rows
            # must read back as zeros rather than uninitialized memory.
            output = torch.zeros(
                (num_tokens, q.shape[1], self.kv_lora_rank),
                dtype=q.dtype,
                device=q.device,
            )
            _xattn_flash_mla_sparse_fwd(
                q=q,
                kv=kv_c_and_k_pe_cache,
                indices=topk_indices,
                sm_scale=self.softmax_scale,
                d_v=self.kv_lora_rank,
                topk_length=topk_length,
                out=output,
            )
        else:
            output, _, _ = triton_bf16_mla_sparse_interface(
                q,
                kv_c_and_k_pe_cache,
                topk_indices,
                sm_scale=self.softmax_scale,
                d_v=self.kv_lora_rank,
            )

        return output[:, : self.num_heads, :]

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: XPUMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        # Concatenate q if it's a tuple (ql_nope, q_pe)
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        # The xattention kernels can skip the -1 padding tail of each row when
        # given the per-token valid count, which the remap kernel produces for
        # free; the Triton kernel masks the padding itself.
        remapped = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
            return_valid_counts=self.use_xattention,
        )
        if self.use_xattention:
            topk_indices_global, topk_length = remapped
        else:
            topk_indices_global, topk_length = remapped, None

        forward_fn = (
            self._forward_fp8_kv if self.use_fp8_kv_cache else self._forward_bf16_kv
        )
        attn_out = forward_fn(
            q, kv_c_and_k_pe_cache, topk_indices_global, topk_length, attn_metadata
        )

        return attn_out, None
