# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import LOG2E, LOGE2, tl, triton


@triton.jit
def _bf16_mla_sparse_kernel(
    q_buffer,
    k_buffer,
    v_buffer,
    indices_ptr,
    out_ptr,
    softmax_lse_ptr,
    max_logits_ptr,
    seq_q,
    seq_kv,
    h_q,
    dim_qk,
    dim_v,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_head,
    stride_v_token,
    stride_v_head,
    stride_out_token,
    stride_out_head,
    stride_lse,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    kv_group_num: tl.constexpr,
    index_topk: tl.constexpr,
    BLOCK_H: tl.constexpr,  # block size for num heads
    BLOCK_M: tl.constexpr,  # block size for num tokens
    BLOCK_N: tl.constexpr,  # block size for indices
    BLOCK_DV: tl.constexpr,  # block size for dim_v
    BLOCK_DMODEL: tl.constexpr,  # block size for dim_nope
    BLOCK_DPE: tl.constexpr,  # block size for positional embedding
    LOGE2: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < h_q)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)

    off_q = cur_q * stride_q_token + cur_head[:, None] * stride_q_head + offs_d[None, :]
    mask_dmodel = offs_d < BLOCK_DMODEL
    q = tl.load(
        q_buffer + off_q, mask=(mask_h[:, None]) & (mask_dmodel[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        off_qpe = (
            cur_q * stride_q_token
            + cur_head[:, None] * stride_q_head
            + offs_dpe[None, :]
        )
        # assume dim_qk == BLOCK_DMODEL + BLOCK_DPE
        mask_dpe = offs_dpe < dim_qk
        qpe = tl.load(
            q_buffer + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
        )

    # Use a finite sentinel rather than -inf: if every key in a chunk is
    # masked (e.g. leading -1 padding in `indices`) an -inf running max gives
    # re_scale = exp2(-inf - -inf) = NaN, permanently poisoning acc / e_sum.
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_indice in range(0, index_topk, BLOCK_N):
        offs_indice = start_indice + tl.arange(0, BLOCK_N)
        mask_indice = offs_indice < index_topk
        indices = tl.load(
            indices_ptr
            + (
                cur_q * stride_indices_token
                + cur_kv_head_id * stride_indices_head
                + offs_indice
            ),
            mask=mask_indice,
            other=-1,
        )

        mask_kv = (indices >= 0) & (indices < seq_kv)
        mask_kv_d = mask_dmodel
        offs_k = (
            indices[None, :] * stride_k_token
            + cur_kv_head_id * stride_k_head
            + offs_d[:, None]
        )

        # q_nope @ k_nope
        k = tl.load(
            k_buffer + offs_k, mask=(mask_kv[None, :]) & (mask_kv_d[:, None]), other=0.0
        )
        qk = tl.dot(q, k.to(q.dtype))

        if BLOCK_DPE > 0:
            # q_rope @ k_rope
            offs_kpe = (
                indices[None, :] * stride_k_token
                + cur_kv_head_id * stride_k_head
                + offs_dpe[:, None]
            )
            mask_k_dpe = offs_dpe < dim_qk
            kpe = tl.load(
                k_buffer + offs_kpe,
                mask=(mask_kv[None, :]) & (mask_k_dpe[:, None]),
                other=0.0,
            )
            qk += tl.dot(qpe, kpe.to(q.dtype))

        # apply scaling
        qk *= sm_scale
        qk = tl.where((mask_h[:, None]) & (mask_kv[None, :]), qk, -1.0e30)

        # load v
        mask_v_d = offs_dv < dim_v
        offs_v = (
            indices[:, None] * stride_v_token
            + cur_kv_head_id * stride_v_head
            + offs_dv[None, :]
        )
        v = tl.load(
            v_buffer + offs_v, mask=(mask_kv[:, None]) & (mask_v_d[None, :]), other=0.0
        )

        # online softmax
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]

        # score @ v
        acc += tl.dot(p.to(v.dtype), v)

        # update global sum and max
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # rescaling
    acc /= e_sum[:, None]

    max_logits = e_max * LOGE2
    # calculate lse
    lse = max_logits + tl.log2(e_sum) * LOGE2

    # write output
    offs_o = (
        cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :]
    )
    mask_out_d = offs_dv < dim_v
    tl.store(
        out_ptr + offs_o,
        acc.to(tl.bfloat16),
        mask=(mask_h[:, None]) & (mask_out_d[None, :]),
    )

    offs_lse = cur_q * stride_lse + cur_head
    tl.store(softmax_lse_ptr + offs_lse, lse, mask=mask_h)
    tl.store(max_logits_ptr + offs_lse, max_logits, mask=mask_h)


# reference implementation of bf16 sparse prefill kernel
def triton_bf16_mla_sparse_interface(
    q: torch.Tensor,  # [num_tokens, num_heads_q, dim_qk]
    kv: torch.Tensor,  # [num_tokens, num_heads_kv, dim_qk]
    indices: torch.Tensor,  # [num_tokens, num_heads_kv, topk]
    sm_scale: float,
    d_v: int = 512,
    block_dpe: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    out : [num_tokens, num_heads_q, d_v]
    max_logits : [num_tokens, num_heads_q]
    lse : logsumexp, [num_tokens, num_heads_q]

    Args:
        block_dpe: Size of positional embedding portion of dim_qk.
            Set to 0 when q/kv contain only the nope latent (e.g. DSv4
            prefill where RoPE is not split out).
    """
    num_tokens, num_heads_q, dim_qk = q.shape
    _, num_heads_kv, _ = kv.shape
    assert dim_qk == kv.shape[2], "q and kv have different head dimensions"

    # for deepseek v3.2, index topk should be 2048
    _, _, index_topk = indices.shape

    BLOCK_H = 16
    BLOCK_DPE = block_dpe
    BLOCK_DMODEL = dim_qk - BLOCK_DPE
    BLOCK_M = 32
    BLOCK_N = 16
    BLOCK_DV = 512
    assert d_v == BLOCK_DV, "only support d_v = 512"

    assert dim_qk == BLOCK_DMODEL + BLOCK_DPE, (
        "dim_qk does not match BLOCK_DMODEL + BLOCK_DPE"
    )
    assert num_heads_kv == 1, "only support kv head = 1 for now"
    assert index_topk % BLOCK_N == 0, "index_topk must be multiple of BLOCK_N"

    sm_scale *= LOG2E

    kv_group_num = num_heads_q // num_heads_kv
    grid = (
        num_tokens,
        triton.cdiv(num_heads_q, min(BLOCK_H, kv_group_num)),
    )

    out = torch.zeros((num_tokens, num_heads_q, d_v), dtype=q.dtype, device=q.device)
    softmax_lse = torch.zeros(
        (num_tokens, num_heads_q), dtype=torch.float32, device=q.device
    )
    max_logits = torch.zeros(
        (num_tokens, num_heads_q), dtype=torch.float32, device=q.device
    )

    k = kv
    v = kv[..., :d_v]

    _bf16_mla_sparse_kernel[grid](
        q_buffer=q,
        k_buffer=k,
        v_buffer=v,
        indices_ptr=indices,
        out_ptr=out,
        softmax_lse_ptr=softmax_lse,
        max_logits_ptr=max_logits,
        seq_q=num_tokens,
        seq_kv=kv.shape[0],
        h_q=num_heads_q,
        dim_qk=dim_qk,
        dim_v=d_v,
        stride_q_token=q.stride(0),
        stride_q_head=q.stride(1),
        stride_k_token=k.stride(0),
        stride_k_head=k.stride(1),
        stride_v_token=v.stride(0),
        stride_v_head=v.stride(1),
        stride_out_token=out.stride(0),
        stride_out_head=out.stride(1),
        stride_lse=softmax_lse.stride(0),
        stride_indices_token=indices.stride(0),
        stride_indices_head=indices.stride(1),
        sm_scale=sm_scale,
        kv_group_num=kv_group_num,
        index_topk=index_topk,
        BLOCK_H=BLOCK_H,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DV=BLOCK_DV,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        LOGE2=LOGE2,
    )

    return out, max_logits, softmax_lse


# fp8_ds_mla packed layout, per token (656 bytes, kv_lora_rank=512, pe_dim=64):
#   bytes [0, 512)    : 512 float8_e4m3 NoPE values
#   bytes [512, 528)  : 4 float32 scales, one per 128-element NoPE tile
#   bytes [528, 656)  : 64 bfloat16 RoPE values (kept unquantized)
# This mirrors `concat_and_cache_ds_mla_kernel` in
# csrc/libtorch_stable/cache_kernels.cu, which is CUDA-only.
DS_MLA_TILE_DIM = 128
DS_MLA_ENTRY_BYTES = 656


@triton.jit
def _concat_and_cache_ds_mla_kernel(
    kv_c_ptr,
    k_pe_ptr,
    cache_fp8_ptr,
    cache_f32_ptr,
    cache_bf16_ptr,
    slot_mapping_ptr,
    kv_c_stride,
    k_pe_stride,
    block_stride,
    entry_stride,
    KV_DIM: tl.constexpr,
    PE_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    TILE_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    # Padded tokens carry slot -1 and must not be written.
    if slot_idx < 0:
        return

    block_idx = slot_idx // BLOCK_SIZE
    block_off = slot_idx % BLOCK_SIZE
    # Strides are byte offsets: the caller passes the uint8 view's strides.
    byte_base = block_idx * block_stride + block_off * entry_stride

    kv_off = tl.arange(0, KV_DIM)
    kv_c = tl.load(kv_c_ptr + token_idx * kv_c_stride + kv_off).to(tl.float32)

    kv_2d = tl.reshape(kv_c, (NUM_TILES, TILE_DIM))
    tile_amax = tl.max(tl.abs(kv_2d), axis=1, keep_dims=True)
    # scale = amax / 448 (fp8 e4m3 max), floored to FLT_MIN, matching the
    # reference CUDA kernel.
    tile_scale = tl.maximum(tile_amax * (1.0 / 448.0), 1.1754944e-38)

    kv_fp8 = tl.reshape((kv_2d / tile_scale).to(tl.float8e4nv), (KV_DIM,))
    tl.store(cache_fp8_ptr + byte_base + kv_off, kv_fp8)

    tile_off = tl.arange(0, NUM_TILES)
    tl.store(
        cache_f32_ptr + byte_base // 4 + KV_DIM // 4 + tile_off,
        tl.reshape(tile_scale, (NUM_TILES,)),
    )

    pe_off = tl.arange(0, PE_DIM)
    k_pe = tl.load(k_pe_ptr + token_idx * k_pe_stride + pe_off)
    tl.store(
        cache_bf16_ptr + byte_base // 2 + (KV_DIM // 2 + 8) + pe_off,
        k_pe.to(tl.bfloat16),
    )


def triton_concat_and_cache_ds_mla(
    kv_c: torch.Tensor,  # [num_tokens, kv_lora_rank]
    k_pe: torch.Tensor,  # [num_tokens, pe_dim]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, 656] uint8
    slot_mapping: torch.Tensor,  # [num_tokens] int64
) -> None:
    """Pack latent KV into the 656-byte ``fp8_ds_mla`` cache layout.

    Triton port of the CUDA-only ``concat_and_cache_ds_mla`` kernel, so that
    non-CUDA platforms can use DeepSeek's segmented fp8 KV cache format.

    Args:
        kv_c: Normed latent KV, 16-bit.
        k_pe: RoPE'd positional part, 16-bit.
        kv_cache: uint8 paged cache with a 656-byte entry stride.
        slot_mapping: Destination slot per token; negative entries are skipped.
    """
    kv_lora_rank = kv_c.shape[1]
    pe_dim = k_pe.shape[1]
    block_size = kv_cache.shape[1]
    assert kv_lora_rank == 512, "fp8_ds_mla requires kv_lora_rank == 512"
    assert pe_dim == 64, "fp8_ds_mla requires pe_dim == 64"
    assert kv_c.element_size() == 2 and k_pe.element_size() == 2
    assert kv_cache.dtype == torch.uint8
    assert kv_cache.shape[2] == DS_MLA_ENTRY_BYTES

    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return

    # The three views alias the same buffer; each store uses the element size
    # matching the field it writes.
    cache_fp8 = kv_cache.view(torch.float8_e4m3fn)
    cache_f32 = kv_cache.view(torch.float32)
    cache_bf16 = kv_cache.view(torch.bfloat16)

    _concat_and_cache_ds_mla_kernel[(num_tokens,)](
        kv_c,
        k_pe,
        cache_fp8,
        cache_f32,
        cache_bf16,
        slot_mapping,
        kv_c.stride(0),
        k_pe.stride(0),
        kv_cache.stride(0),
        kv_cache.stride(1),
        KV_DIM=kv_lora_rank,
        PE_DIM=pe_dim,
        BLOCK_SIZE=block_size,
        NUM_TILES=kv_lora_rank // DS_MLA_TILE_DIM,
        TILE_DIM=DS_MLA_TILE_DIM,
    )
