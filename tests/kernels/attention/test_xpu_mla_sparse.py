# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.ops.xpu_mla_sparse import (
    DS_MLA_ENTRY_BYTES,
    triton_bf16_mla_sparse_interface,
    triton_concat_and_cache_ds_mla,
)


# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L7
def _merge_two_lse(
    lse0: torch.Tensor, lse1: torch.Tensor | None, s_q: int, h_q: int
) -> torch.Tensor:
    if lse1 is None:
        return lse0
    else:
        return torch.logsumexp(
            torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
            dim=0,
        )


# Adapted from https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L19
def reference_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
    - o: [s_q, h_q, dv]
    - o_fp32: [s_q, h_q, dv]
    - max_logits: [s_q, h_q]
    - lse: [s_q, h_q]
    """
    s_q, h_q, d_qk = q.shape
    s_kv, _, _ = kv.shape
    _, _, topk = indices.shape

    indices = indices.clone().squeeze(1)
    if topk_length is not None:
        mask = torch.arange(topk, device=topk_length.device).unsqueeze(0).broadcast_to(
            s_q, topk
        ) >= topk_length.unsqueeze(1)  # [s_q, topk]
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= s_kv)  # [s_q, topk]
    indices[invalid_mask] = 0

    q = q.float()
    gathered_kv = (
        kv.index_select(dim=0, index=indices.flatten()).reshape(s_q, topk, d_qk).float()
    )  # [s_q, topk, d_qk]
    P = q @ gathered_kv.transpose(1, 2)  # [s_q, h_q, topk]
    P *= sm_scale
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)  # [s_q, h_q]
    max_logits = P.max(dim=-1).values  # [s_q, h_q]

    lse_for_o = _merge_two_lse(orig_lse, attn_sink, s_q, h_q)
    if not torch.is_inference_mode_enabled():
        lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float(
        "+inf"
    )  # So that corresponding O will be 0
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., :d_v]  # [s_q, h_q, dv]

    lonely_q_mask = orig_lse == float("-inf")  # [s_q, h_q]
    orig_lse[lonely_q_mask] = float("+inf")
    return (out.to(kv.dtype), out, max_logits, orig_lse)


@pytest.mark.parametrize("device_str", ["xpu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
def test_bf16_triton_sparse_mla(device_str, dtype):
    device = torch.device(device_str)
    s_q = 1
    s_kv = 256
    h_q = 64  # kernel expects multiple of 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128

    torch.random.manual_seed(1234)

    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device)
    indices = torch.full((s_q, h_kv, topk), -1, dtype=torch.int32, device=device)
    for t in range(s_q):
        for h in range(h_kv):
            i_i = torch.randperm(max(1, t))[:topk]
            indices[t, h, : len(i_i)] = i_i

    sm_scale = d_qk**-0.5

    out, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v
    )
    assert out.shape == (s_q, h_q, d_v)
    assert max_logits.shape == (s_q, h_q)
    assert lse.shape == (s_q, h_q)

    ref_out, ref_out_fp32, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, d_v
    )
    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)
    assert torch.allclose(max_logits, ref_max_logits, atol=1e-3, rtol=1e-3)
    assert torch.allclose(lse, ref_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("device_str", ["xpu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
def test_bf16_triton_sparse_mla_masked_chunks(device_str, dtype):
    """Rows whose leading BLOCK_N index entries are all masked must not NaN.

    Regression test: with an -inf running max, a fully-masked leading chunk
    produced re_scale = exp2(-inf - -inf) = NaN, permanently poisoning the
    accumulator even though valid keys followed in later chunks.
    """
    device = torch.device(device_str)
    s_q = 3
    s_kv = 256
    h_q = 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128  # 8 chunks of BLOCK_N=16

    torch.random.manual_seed(1234)

    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device)
    indices = torch.full((s_q, h_kv, topk), -1, dtype=torch.int32, device=device)
    # row 0: valid keys only in chunks 1-2 -> leading AND trailing masked chunks
    indices[0, 0, 16:48] = torch.arange(32, dtype=torch.int32, device=device)
    # row 1: fully valid
    indices[1, 0, :] = torch.arange(topk, dtype=torch.int32, device=device)
    # row 2: no valid key at all

    sm_scale = d_qk**-0.5

    out, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v
    )
    assert out.isfinite().all()

    ref_out, _, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, d_v
    )
    assert torch.allclose(out[:2], ref_out[:2], atol=1e-2, rtol=1e-2)
    assert torch.allclose(max_logits[:2], ref_max_logits[:2], atol=1e-3, rtol=1e-3)
    assert torch.allclose(lse[:2], ref_lse[:2], atol=1e-3, rtol=1e-3)
    # A row with no valid key yields zeros (the reference's convention); its
    # lse/max_logits are large-negative finite rather than the reference's
    # +inf/-inf placeholders, so only the output is compared here.
    assert torch.allclose(out[2], torch.zeros_like(out[2]))


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required")
def test_xattention_sparse_mla_matches_triton():
    """The xattention kernel used by XPUMLASparseImpl must match the Triton one.

    ``XPUMLASparseImpl`` prefers the xattention kernel and feeds it the
    per-token valid count as ``topk_length``, so this covers ragged rows
    (including an empty one) that the Triton fallback masks via -1 indices.
    """
    xattention = pytest.importorskip("xattention")

    device = torch.device("xpu")
    dtype = torch.bfloat16
    s_q, s_kv, h_q, d_qk, d_v, topk = 5, 4096, 128, 576, 512, 2048

    torch.random.manual_seed(1234)
    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    kv = torch.randn((s_kv, 1, d_qk), dtype=dtype, device=device)
    indices = torch.full((s_q, 1, topk), -1, dtype=torch.int32, device=device)
    valid_lens = [topk, topk // 2, 7, 0, 1]
    for t, n in enumerate(valid_lens):
        if n:
            indices[t, 0, :n] = torch.randperm(s_kv, device=device)[:n].to(torch.int32)
    topk_length = torch.tensor(valid_lens, dtype=torch.int32, device=device)
    sm_scale = d_qk**-0.5

    # Zero-init like the impl does: the kernel leaves rows with
    # ``topk_length == 0`` untouched.
    out_xattn = torch.zeros((s_q, h_q, d_v), dtype=dtype, device=device)
    xattention.flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        topk_length=topk_length,
        out=out_xattn,
    )
    out_triton, _, _ = triton_bf16_mla_sparse_interface(q, kv, indices, sm_scale, d_v)

    assert out_xattn.isfinite().all()
    assert torch.allclose(out_xattn, out_triton, atol=1e-2, rtol=1e-2)


def _unpack_ds_mla(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the packed 656-byte entries back to (nope, rope)."""
    flat = cache.view(-1, DS_MLA_ENTRY_BYTES)
    nope = flat[:, :512].view(torch.float8_e4m3fn).to(torch.float32)
    scales = flat[:, 512:528].view(torch.float32)
    rope = flat[:, 528:].view(torch.bfloat16)
    return (nope.view(-1, 4, 128) * scales.view(-1, 4, 1)).reshape(-1, 512), rope


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required")
def test_concat_and_cache_ds_mla_roundtrip():
    """Packing then dequantizing must recover the inputs within fp8 error.

    Guards the byte offsets of the three fp8_ds_mla segments (NoPE, per-tile
    scales, RoPE): a wrong offset still "works" but silently corrupts KV.
    """
    device = torch.device("xpu")
    num_blocks, block_size = 4, 16
    torch.random.manual_seed(0)

    kv_c = torch.randn((10, 512), dtype=torch.bfloat16, device=device) * 3
    k_pe = torch.randn((10, 64), dtype=torch.bfloat16, device=device)
    cache = torch.full(
        (num_blocks, block_size, DS_MLA_ENTRY_BYTES),
        0xEE,
        dtype=torch.uint8,
        device=device,
    )
    before = cache.clone()
    slots = torch.arange(10, dtype=torch.int64, device=device)
    # A padded token must be skipped rather than written to slot -1.
    slots[3] = -1

    triton_concat_and_cache_ds_mla(kv_c, k_pe, cache, slots)

    deq, rope = _unpack_ds_mla(cache)
    for i, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        ref = kv_c[i].float()
        rel = ((deq[slot] - ref).abs().amax() / ref.abs().amax()).item()
        # e4m3 with a per-128-element scale keeps well under 5% relative error.
        assert rel < 0.05, f"token {i}: rel err {rel}"
        assert torch.equal(rope[slot], k_pe[i])

    # The slot for the padded token must be untouched.
    assert torch.equal(cache[0, 3], before[0, 3])


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required")
def test_xattention_fp8_sparse_matches_bf16():
    """xattention must read the Triton-packed fp8_ds_mla cache correctly.

    Compares the fp8 decode kernel against the bf16 kernel fed the *same*
    values (dequantized from the packed cache), so any mismatch is a layout
    misinterpretation rather than quantization error.
    """
    xattention = pytest.importorskip("xattention")

    device = torch.device("xpu")
    dtype = torch.bfloat16
    num_blocks, block_size, h_q, topk = 64, 64, 64, 128
    s_kv = num_blocks * block_size
    s_q, d_qk, d_v = 5, 576, 512

    torch.random.manual_seed(7)
    kv_c = torch.randn((s_kv, 512), dtype=dtype, device=device)
    k_pe = torch.randn((s_kv, 64), dtype=dtype, device=device)
    cache = torch.zeros(
        (num_blocks, block_size, DS_MLA_ENTRY_BYTES), dtype=torch.uint8, device=device
    )
    triton_concat_and_cache_ds_mla(
        kv_c, k_pe, cache, torch.arange(s_kv, dtype=torch.int64, device=device)
    )

    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    indices = torch.stack(
        [torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]
    ).to(torch.int32)
    topk_length = torch.tensor([topk, topk, 7, 0, 1], dtype=torch.int32, device=device)
    sm_scale = d_qk**-0.5

    deq, rope = _unpack_ds_mla(cache)
    kv_bf16 = torch.cat([deq.to(dtype), rope], dim=-1).view(s_kv, 1, d_qk)
    ref = torch.zeros((s_q, h_q, d_v), dtype=dtype, device=device)
    xattention.flash_mla_sparse_fwd(
        q=q,
        kv=kv_bf16,
        indices=indices.view(s_q, 1, topk),
        sm_scale=sm_scale,
        d_v=d_v,
        topk_length=topk_length,
        out=ref,
    )

    out = torch.zeros((s_q, 1, h_q, d_v), dtype=dtype, device=device)
    xattention.flash_mla_with_kvcache(
        q=q.unsqueeze(1),
        k_cache=cache.unsqueeze(-2),
        block_table=None,
        cache_seqlens=None,
        head_dim_v=d_v,
        softmax_scale=sm_scale,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices.view(s_q, 1, topk),
        topk_length=topk_length,
        out=out,
    )

    assert out.isfinite().all()
    assert torch.allclose(out.squeeze(1), ref, atol=1e-2, rtol=1e-2)


def test_backend_declares_cudagraph_support():
    """``NEVER`` makes ``resolve_cudagraph_mode_and_sizes`` hard-raise.

    Every other sparse-MLA backend reports ``UNIFORM_BATCH``, which lets the
    resolver downgrade ``cudagraph_mode=FULL`` instead of failing startup. The
    capture test below is the evidence that this claim holds.
    """
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
        XPUMLASparseMetadataBuilder,
    )

    assert (
        XPUMLASparseMetadataBuilder._cudagraph_support
        is AttentionCGSupport.UNIFORM_BATCH
    )


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required")
def test_sparse_mla_kernels_are_xpu_graph_capturable():
    """The kernels the impl launches must survive capture/replay.

    Guards the ``UNIFORM_BATCH`` claim above: a replay after mutating the
    input buffers must recompute rather than return the captured result.
    """
    xattention = pytest.importorskip("xattention")
    if not hasattr(torch.xpu, "XPUGraph"):
        pytest.skip("torch.xpu.XPUGraph is required")

    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        triton_convert_req_index_to_global_index,
    )

    device, dtype = torch.device("xpu"), torch.bfloat16
    num_blocks, block_size, h_q, d_v, topk, num_tokens = 64, 64, 64, 512, 128, 8
    s_kv = num_blocks * block_size

    kv = torch.randn(s_kv, 1, d_v + 64, device=device, dtype=dtype)
    q = torch.randn(num_tokens, h_q, d_v + 64, device=device, dtype=dtype)
    req_id_per_token = torch.arange(num_tokens, dtype=torch.int32, device=device)
    block_table = (
        torch.arange(num_tokens * 8, dtype=torch.int32, device=device).view(
            num_tokens, 8
        )
        % num_blocks
    )
    topk_indices = torch.stack(
        [torch.randperm(s_kv, device=device)[:topk] for _ in range(num_tokens)]
    ).to(torch.int32)

    def run():
        global_indices, topk_length = triton_convert_req_index_to_global_index(
            req_id_per_token,
            block_table,
            topk_indices,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=topk,
            return_valid_counts=True,
        )
        out = torch.zeros((num_tokens, h_q, d_v), dtype=dtype, device=device)
        xattention.flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=global_indices.view(num_tokens, 1, -1),
            sm_scale=(d_v + 64) ** -0.5,
            d_v=d_v,
            topk_length=topk_length,
            out=out,
        )
        return out

    side = torch.xpu.Stream()
    side.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(side):
        for _ in range(3):
            run()
    torch.xpu.current_stream().wait_stream(side)
    torch.xpu.synchronize()

    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        captured_out = run()

    q.normal_()
    expected = run().clone()
    graph.replay()
    torch.xpu.synchronize()
    torch.testing.assert_close(captured_out, expected, atol=1e-2, rtol=1e-2)
