#!/usr/bin/env python3
"""
gen_attn_comparison.py

Extract real NousResearch/Llama-3.2-1B Layer 1 post-RMSNorm1 + post-RoPE Q, K, V
tensors for arbitrary M_TOKENS and N_KV (up to 1024) and emit RTL-aligned 72-bit
Log8-FP4 stimulus files plus exact FP32 Causal Attention golden reference:
  - q_words.hex     : [H_Q][M][K_PACKETS_QK]         (72-bit words, RoPE-pair permuted)
  - k_words.hex     : [H_KV][KV_TILES][K_PACKETS_QK][16] (72-bit words, Sep-KV)
  - v_words.hex     : [H_KV][DV_TILES][KV_TILES][16] (72-bit words, Sep-KV transposed)
  - g_attn_fp32.hex : [H_Q][M][D_V]                  (32-bit IEEE FP32 golden)
"""
import argparse
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gen_llama3_1b_configurable import (  # noqa: E402
    D_MODEL,
    D_K,
    D_V,
    DV_TILES,
    GQA_GRP,
    H_KV,
    H_Q,
    K_PACKETS_QK,
    REAL_PROMPT,
    fq_rtl_vec,
    metrics,
    quantize_rtl_vec,
    write_72b_words_hex,
    write_9b_pkts_hex,
    write_fp32_hex,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-tokens", type=int, default=64)
    parser.add_argument("--n-kv", type=int, default=None)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--model-id", type=str, default="NousResearch/Llama-3.2-1B")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    m_tokens = args.m_tokens
    n_kv = m_tokens if args.n_kv is None else args.n_kv
    assert 16 <= m_tokens <= 1024 and m_tokens % 16 == 0, "--m-tokens must be multiple of 16 in [16..1024]"
    assert m_tokens <= n_kv <= 1024 and n_kv % 16 == 0, "--n-kv must be >= m-tokens, <= 1024, multiple of 16"

    n_past = n_kv - m_tokens
    kv_tiles = n_kv // 16

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(HERE, "..", "sim", f"sim_attn_cmp_m{m_tokens}_kv{n_kv}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print(f"=== Generating Attention Comparison Data ({args.model_id} L{args.layer}) ===")
    print(f"    M_TOKENS={m_tokens}, N_KV={n_kv} (n_past={n_past}), OUT_DIR={out_dir}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)

    model.eval()
    ids = tok(REAL_PROMPT, return_tensors="pt")["input_ids"]
    while ids.shape[1] < n_kv:
        ids = torch.cat([ids, ids], dim=1)
    ids = ids[:, :n_kv]

    captured = {}

    def hook_fn(_mod, inp, _out):
        captured["x"] = inp[0].detach().cpu().numpy()[0]

    h = model.model.layers[args.layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(ids)
    h.remove()

    qk_scale = np.float32(D_K ** -0.25)
    lyr = model.model.layers[args.layer]
    x_all = captured["x"][:n_kv, :D_MODEL].astype(np.float16).astype(np.float32)
    g1_fp16 = lyr.input_layernorm.weight.detach().cpu().numpy().astype(np.float16).astype(np.float32)
    w_q = lyr.self_attn.q_proj.weight.detach().cpu().numpy().astype(np.float32) * qk_scale
    w_k = lyr.self_attn.k_proj.weight.detach().cpu().numpy().astype(np.float32) * qk_scale
    w_v = lyr.self_attn.v_proj.weight.detach().cpu().numpy().astype(np.float32)

    eps = 1e-5
    rms1 = 1.0 / np.sqrt(np.mean(x_all ** 2, axis=1, keepdims=True) + eps)
    xn_all = x_all * rms1 * g1_fp16
    xn_q = xn_all[n_past:, :]

    pos_all = np.arange(n_kv, dtype=np.float32)[:, None]
    inv_freq = 1.0 / (500000.0 ** (np.arange(0, D_K, 2, dtype=np.float32) / D_K))[None, :]
    freqs_all = pos_all * inv_freq
    cos_all = np.cos(freqs_all).astype(np.float16).astype(np.float32)
    sin_all = np.sin(freqs_all).astype(np.float16).astype(np.float32)

    def apply_rope(t_mat, n_heads, cos_t, sin_t):
        ntok = t_mat.shape[0]
        t_h = t_mat.reshape(ntok, n_heads, D_K)
        x1 = t_h[:, :, :D_K // 2]
        x2 = t_h[:, :, D_K // 2:]
        c = cos_t[:, None, :]
        s = sin_t[:, None, :]
        return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)

    q_fp32 = apply_rope(xn_q @ w_q.T, H_Q, cos_all[n_past:], sin_all[n_past:])      # (M, H_Q, 64)
    k_fp32 = apply_rope(xn_all @ w_k.T, H_KV, cos_all, sin_all)                     # (N_KV, H_KV, 64)
    v_fp32 = (xn_all @ w_v.T).reshape(n_kv, H_KV, D_V)                              # (N_KV, H_KV, 64)

    # Exact FP32 Causal Attention golden: shape (H_Q, M, D_V)
    q_abs = np.arange(m_tokens, dtype=np.int32)[:, None] + n_past
    k_pos = np.arange(n_kv, dtype=np.int32)[None, :]
    mask = np.where(k_pos <= q_abs, 0.0, -1e9).astype(np.float32)

    k_rep = np.repeat(k_fp32, GQA_GRP, axis=1)
    v_rep = np.repeat(v_fp32, GQA_GRP, axis=1)
    attn_fp32 = np.zeros((H_Q, m_tokens, D_V), dtype=np.float32)
    for h_i in range(H_Q):
        sc = q_fp32[:, h_i, :] @ k_rep[:, h_i, :].T + mask
        sc -= sc.max(axis=1, keepdims=True)
        pr = np.exp(sc)
        pr /= pr.sum(axis=1, keepdims=True)
        attn_fp32[h_i] = pr @ v_rep[:, h_i, :]

    # Permute Q & K pairs identically to hardware RoPE output before Log8-FP4 SLQ packing
    def rope_perm(t_h):
        p = np.empty_like(t_h)
        p[:, :, 0::2] = t_h[:, :, :D_K // 2]
        p[:, :, 1::2] = t_h[:, :, D_K // 2:]
        return p

    q_perm = rope_perm(q_fp32)
    k_perm = rope_perm(k_fp32)

    # Pack Q: [H_Q][M][K_PACKETS_QK]
    q_pkts = quantize_rtl_vec(q_perm.reshape(m_tokens * H_Q, D_K)) \
        .reshape(m_tokens, H_Q, K_PACKETS_QK, 9) \
        .transpose(1, 0, 2, 3)

    # Pack K: [H_KV][KV_TILES][K_PACKETS_QK][16]
    k_pkts = quantize_rtl_vec(k_perm.reshape(n_kv * H_KV, D_K)) \
        .reshape(kv_tiles, 16, H_KV, K_PACKETS_QK, 9) \
        .transpose(2, 0, 3, 1, 4)

    # Pack V: [H_KV][DV_TILES][KV_TILES][16] (transposed per 16-token KV tile)
    v_t_mat = np.transpose(v_fp32, (1, 2, 0)).reshape(H_KV * D_V, n_kv)
    v_pkts = quantize_rtl_vec(v_t_mat) \
        .reshape(H_KV, DV_TILES, 16, kv_tiles, 9) \
        .transpose(0, 1, 3, 2, 4)

    # Also evaluate Log8-FP4 QKV + FP32 Softmax baseline
    q_fq = fq_rtl_vec(q_perm.reshape(m_tokens * H_Q, D_K)).reshape(m_tokens, H_Q, D_K)
    k_fq = fq_rtl_vec(k_perm.reshape(n_kv * H_KV, D_K)).reshape(n_kv, H_KV, D_K)
    v_fq = np.transpose(fq_rtl_vec(v_t_mat).reshape(H_KV, D_V, n_kv), (2, 0, 1))
    k_fq_rep = np.repeat(k_fq, GQA_GRP, axis=1)
    v_fq_rep = np.repeat(v_fq, GQA_GRP, axis=1)
    attn_fq = np.zeros((H_Q, m_tokens, D_V), dtype=np.float32)
    for h_i in range(H_Q):
        sc = q_fq[:, h_i, :] @ k_fq_rep[:, h_i, :].T + mask
        sc -= sc.max(axis=1, keepdims=True)
        pr = np.exp(sc)
        pr /= pr.sum(axis=1, keepdims=True)
        attn_fq[h_i] = fq_rtl_vec(pr) @ v_fq_rep[:, h_i, :]
    sq, cs, mx = metrics(attn_fq, attn_fp32)
    print(f"    Quantized QKV+P + FP32-Softmax vs FP32 Golden: SQNR={sq:.2f} dB, Cos={cs:.6f}, MaxAbs={mx:.4f}")

    write_9b_pkts_hex(os.path.join(out_dir, "q_words.hex"), q_pkts)
    write_9b_pkts_hex(os.path.join(out_dir, "k_words.hex"), k_pkts)
    write_9b_pkts_hex(os.path.join(out_dir, "v_words.hex"), v_pkts)
    write_fp32_hex(os.path.join(out_dir, "g_attn_fp32.hex"), attn_fp32)
    write_fp32_hex(os.path.join(out_dir, "g_attn_fq.hex"), attn_fq)
    print(f"    Emitted hex files in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
