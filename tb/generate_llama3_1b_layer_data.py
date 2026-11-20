#!/usr/bin/env python3
"""
generate_llama3_1b_layer_data.py

Generates stimulus & FP32/FP16 golden reference for 1 FULL layer of
NousResearch/Llama-3.2-1B in prefill mode on snax_llm_top_clean:
  - M_TOKENS      = 16 (prefill)
  - D_MODEL       = 2048
  - D_FF          = 8192
  - H_Q           = 32, H_KV = 8 (GQA 4:1)
  - D_K = D_V     = 64
  - Total weights = 60,817,408 params (~34.2 MB FP4+Log8) stored in 512-bit DRAM
"""
import os
import sys
import math
import time
import struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "sim", "sim_llama3_1b")

# Architecture parameters (Llama-3.2-1B)
M_TOKENS = 16
D_MODEL  = 2048
D_FF     = 8192
H_Q      = 32
H_KV     = 8
D_K      = 64
D_V      = 64

K_GEMM   = D_MODEL // 16   # 128
K_DOWN   = D_FF // 16      # 512
N_TILES_MODEL = D_MODEL // 16  # 128
N_TILES_KV    = (H_KV * D_K) // 16  # 32
N_TILES_FF    = D_FF // 16     # 512
ROW_WORDS     = D_MODEL // 4   # 512

# TCDM Bank Map (BANK_DEPTH = 8192)
X_BANK     = 24
GAMMA_BANK = 28
ROPE_BANK  = 29
GAMMA1_WORD = 0
GAMMA2_WORD = 512

# DRAM byte offsets (all 64B-aligned)
DRAM_WQ = 0x0000_0000  # 128 * 128 * 144 = 2,359,296 B
DRAM_WK = 0x0024_0000  #  32 * 128 * 144 =   589,824 B
DRAM_WV = 0x002D_0000  #  32 * 128 * 144 =   589,824 B
DRAM_WO = 0x0036_0000  # 128 * 128 * 144 = 2,359,296 B
DRAM_WG = 0x005A_0000  # 512 * 128 * 144 = 9,437,184 B
DRAM_WU = 0x00EA_0000  # 512 * 128 * 144 = 9,437,184 B
DRAM_WD = 0x017A_0000  # 128 * 512 * 144 = 9,437,184 B
DRAM_TOTAL_BYTES = DRAM_WD + 128 * 512 * 144  # 34,209,792 B


def tcdm_addr(word, bank):
    return ((word & 0x1FFF) << 9) | ((bank & 0x1F) << 4)


def quantize_rtl_vec(mat_f32):
    """
    Vectorized bit-exact model of slq_encoder.sv / quantize_block_rtl.
    Input: mat_f32 shape (n_ch, kdim) where kdim % 16 == 0.
    Returns:
      pkts_9b: uint8 array shape (n_ch, kdim//16, 9) little-endian 72-bit packet
      deq_f32: float32 dequantized matrix (approximate FP4+Log8 values for FP32 ref comparison)
    """
    n_ch, kdim = mat_f32.shape
    kb = kdim // 16
    fp16 = mat_f32.astype(np.float16).view(np.uint16).reshape(n_ch, kb, 16)

    sign = (fp16 >> 15) & 1
    e    = (fp16 >> 10) & 0x1F
    m    = fp16 & 0x3FF

    mu = np.where(e == 0, 0, np.where(e == 31, 247, (e << 3) | (m >> 7))).astype(np.int32)
    mx = mu.max(axis=2, keepdims=True)

    sc = np.where(mx[:, :, 0] == 0, 0, np.clip(mx[:, :, 0] - 140, -128, 127)).astype(np.int32)
    diff = mx - mu

    # fp4_mag_of_diff thresholds: <=2:7, <=6:6, <=10:5, <=14:4, <=18:3, <=24:2, <=36:1, else 0
    mag = np.zeros_like(diff, dtype=np.uint8)
    mag = np.where(diff <= 36, 1, mag)
    mag = np.where(diff <= 24, 2, mag)
    mag = np.where(diff <= 18, 3, mag)
    mag = np.where(diff <= 14, 4, mag)
    mag = np.where(diff <= 10, 5, mag)
    mag = np.where(diff <= 6,  6, mag)
    mag = np.where(diff <= 2,  7, mag)
    mag = np.where((mu == 0) | (mx == 0), 0, mag)

    codes = np.where(mag == 0, 0, (sign.astype(np.uint8) << 3) | mag)  # (n_ch, kb, 16)

    # Pack 16 4-bit nibbles + 8-bit signed scale into 9 bytes (72 bits)
    pkts = np.zeros((n_ch, kb, 9), dtype=np.uint8)
    for b in range(8):
        pkts[:, :, b] = codes[:, :, 2 * b] | (codes[:, :, 2 * b + 1] << 4)
    pkts[:, :, 8] = sc.astype(np.uint8)
    return pkts


def fq_rtl_vec(mat_f32):
    """Dequantized FP4+Log8 Fake-Quant representation matching quantize_rtl_vec."""
    pkts = quantize_rtl_vec(mat_f32)
    n_ch, kb, _ = pkts.shape
    sc = pkts[:, :, 8].astype(np.int8).astype(np.float32)
    codes = np.zeros((n_ch, kb, 16), dtype=np.int32)
    for b in range(8):
        codes[:, :, 2 * b]     = pkts[:, :, b] & 0xF
        codes[:, :, 2 * b + 1] = (pkts[:, :, b] >> 4) & 0xF
    mag_lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    lut = np.concatenate([mag_lut, -mag_lut])
    return (lut[codes] * (2.0 ** (sc[:, :, None] / 8.0))).reshape(n_ch, kb * 16)


def permute_qk_weights_for_rope(mat, n_heads, d_k):
    """
    Permute W_Q / W_K output channels per head so adjacent even/odd channels
    correspond to RoPE pairs (j, j + d_k//2), matching run_fused_qk_rope.
    """
    out = np.empty_like(mat)
    half = d_k // 2
    for h in range(n_heads):
        base = h * d_k
        out[base + 0:base + d_k:2] = mat[base:base + half]
        out[base + 1:base + d_k:2] = mat[base + half:base + d_k]
    return out


def pack_proj_to_dram(dram_buf, byte_off, pkts_9b):
    """
    Pack pkts_9b (n_ch, kb, 9) into dram_buf in N-tile major (16 ch/tile),
    K-block outer, 16-lane inner order:
      for nt in 0..N_TILES-1:
        for k in 0..kb-1:
          for lane in 0..15: write 9 bytes
    """
    n_ch, kb, _ = pkts_9b.shape
    n_tiles = n_ch // 16
    reshaped = pkts_9b.reshape(n_tiles, 16, kb, 9).transpose(0, 2, 1, 3).reshape(-1)
    dram_buf[byte_off:byte_off + reshaped.size] = reshaped


def write_fp16_hex(path, arr_f32):
    u16 = arr_f32.astype(np.float16).view(np.uint16).reshape(-1)
    with open(path, "w") as f:
        f.write("\n".join(f"{int(v):04x}" for v in u16) + "\n")


def write_fp32_hex(path, arr_f32):
    u32 = arr_f32.astype(np.float32).view(np.uint32).reshape(-1)
    with open(path, "w") as f:
        f.write("\n".join(f"{int(v):08x}" for v in u32) + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="Load real weights & hidden states from NousResearch/Llama-3.2-1B")
    parser.add_argument("--layer", type=int, default=1,
                        help="Transformer layer index to extract (default: 1)")
    parser.add_argument("--m-tokens", type=int, default=16,
                        help="Number of prefill tokens (multiple of 16, default: 16)")
    parser.add_argument("--n-kv", type=int, default=16,
                        help="Total KV context length (must be >= m-tokens and multiple of 16)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: sim/sim_llama3_1b)")
    parser.add_argument("--skip-weights", action="store_true",
                        help="Skip writing 69MB dram_weights.hex (reuse sim_llama3_1b/dram_weights.hex)")
    parser.add_argument("--decode", action="store_true",
                        help="Generate single-token Decode data (token 0 = pos N_KV-1, attending to N_KV tokens)")
    args = parser.parse_args()
    global M_TOKENS, OUT_DIR
    M_TOKENS = args.m_tokens
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
    assert M_TOKENS >= 16 and M_TOKENS % 16 == 0, "--m-tokens must be >= 16 and divisible by 16"
    assert args.n_kv >= M_TOKENS and args.n_kv % 16 == 0, "--n-kv must be >= m-tokens and divisible by 16"
    n_kv = args.n_kv
    n_past = n_kv - M_TOKENS
    kv_tiles = n_kv // 16
    kv_tiles_past = n_past // 16

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(2026)

    print("=== Generating Llama-3.2-1B Prefill Layer Data ===")
    print(f"M_TOKENS={M_TOKENS}, N_KV={n_kv}, D_MODEL={D_MODEL}, D_FF={D_FF}, H_Q={H_Q}, H_KV={H_KV}, D_K={D_K}")

    qk_scale = D_K ** -0.25
    if args.real:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = "NousResearch/Llama-3.2-1B"
        print(f"Loading REAL model weights & input from {model_id} (layer {args.layer}) ...")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        model.eval()
        prompt = (
            "In computer architecture, systolic arrays and specialized neural processing units "
            "accelerate transformer inference by minimizing off-chip memory traffic."
        )
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        while ids.shape[1] < n_kv:
            ids = torch.cat([ids, ids], dim=1)
        ids = ids[:, :n_kv]
        captured = {}
        def hook_fn(mod, inp, out):
            captured["x"] = inp[0].detach().cpu().numpy()[0]
        h = model.model.layers[args.layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(ids)
        h.remove()

        lyr = model.model.layers[args.layer]
        x_all  = captured["x"][:n_kv, :D_MODEL].astype(np.float32)
        if args.decode:
            x_f32 = np.zeros((M_TOKENS, D_MODEL), dtype=np.float32)
            x_f32[0] = x_all[n_kv - 1]
        else:
            x_f32  = x_all[n_past:, :]
        g1_f32 = lyr.input_layernorm.weight.detach().cpu().numpy().astype(np.float32)
        g2_f32 = lyr.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float32)
        w_q = lyr.self_attn.q_proj.weight.detach().cpu().numpy().astype(np.float32) * qk_scale
        w_k = lyr.self_attn.k_proj.weight.detach().cpu().numpy().astype(np.float32) * qk_scale
        w_v = lyr.self_attn.v_proj.weight.detach().cpu().numpy().astype(np.float32)
        w_o = lyr.self_attn.o_proj.weight.detach().cpu().numpy().astype(np.float32)
        w_g = lyr.mlp.gate_proj.weight.detach().cpu().numpy().astype(np.float32)
        w_u = lyr.mlp.up_proj.weight.detach().cpu().numpy().astype(np.float32)
        w_d = lyr.mlp.down_proj.weight.detach().cpu().numpy().astype(np.float32)
        print(f"Real x_f32: std={x_f32.std():.4f}, max_abs={np.max(np.abs(x_f32)):.4f}")
    else:
        # 1. Generate realistic Llama-3.2-1B activations & weights
        x_all  = rng.normal(0.0, 0.04, (n_kv, D_MODEL)).astype(np.float32)
        if args.decode:
            x_f32 = np.zeros((M_TOKENS, D_MODEL), dtype=np.float32)
            x_f32[0] = x_all[n_kv - 1]
        else:
            x_f32  = x_all[n_past:, :]
        g1_f32 = rng.normal(1.0, 0.08, (D_MODEL,)).astype(np.float32)
        g2_f32 = rng.normal(1.0, 0.08, (D_MODEL,)).astype(np.float32)

        w_q = rng.normal(0.0, 0.025, (D_MODEL, D_MODEL)).astype(np.float32) * qk_scale
        w_k = rng.normal(0.0, 0.025, (H_KV * D_K, D_MODEL)).astype(np.float32) * qk_scale
        w_v = rng.normal(0.0, 0.025, (H_KV * D_V, D_MODEL)).astype(np.float32)
        w_o = rng.normal(0.0, 0.025, (D_MODEL, D_MODEL)).astype(np.float32)
        w_g = rng.normal(0.0, 0.025, (D_FF, D_MODEL)).astype(np.float32)
        w_u = rng.normal(0.0, 0.025, (D_FF, D_MODEL)).astype(np.float32)
        w_d = rng.normal(0.0, 0.025, (D_MODEL, D_FF)).astype(np.float32)

    # RoPE cos/sin (theta = 500000.0 for Llama-3.2)
    pos_all = np.arange(n_kv, dtype=np.float32)[:, None]
    inv_freq = 1.0 / (500000.0 ** (np.arange(0, D_K, 2, dtype=np.float32) / D_K))[None, :]
    freqs_all = pos_all * inv_freq  # (n_kv, D_K//2)
    cos_all = np.cos(freqs_all).astype(np.float16).astype(np.float32)
    sin_all = np.sin(freqs_all).astype(np.float16).astype(np.float32)
    if args.decode:
        cos_f32 = np.zeros((M_TOKENS, D_K // 2), dtype=np.float32)
        sin_f32 = np.zeros((M_TOKENS, D_K // 2), dtype=np.float32)
        cos_f32[0] = cos_all[n_kv - 1]
        sin_f32[0] = sin_all[n_kv - 1]
    else:
        cos_f32 = cos_all[n_past:]
        sin_f32 = sin_all[n_past:]

    # 2. FP32 Golden Forward Pass (exact Llama-3.2-1B math)
    eps = 1e-5
    x_fp16_in = x_f32.astype(np.float16).astype(np.float32)
    g1_fp16   = g1_f32.astype(np.float16).astype(np.float32)
    g2_fp16   = g2_f32.astype(np.float16).astype(np.float32)

    rms1 = 1.0 / np.sqrt(np.mean(x_fp16_in ** 2, axis=1, keepdims=True) + eps)
    xn_f32 = x_fp16_in * rms1 * g1_fp16

    q_lin = xn_f32 @ w_q.T  # (M, 2048)
    k_lin = xn_f32 @ w_k.T  # (M, 512)
    v_lin = xn_f32 @ w_v.T  # (M, 512)

    def apply_rope(t_mat, n_heads, cos_t=cos_f32, sin_t=sin_f32):
        ntok = t_mat.shape[0]
        t_h = t_mat.reshape(ntok, n_heads, D_K)
        x1 = t_h[:, :, :D_K // 2]
        x2 = t_h[:, :, D_K // 2:]
        c = cos_t[:, None, :]
        s = sin_t[:, None, :]
        y1 = x1 * c - x2 * s
        y2 = x1 * s + x2 * c
        return np.concatenate([y1, y2], axis=-1)

    q_rope = apply_rope(q_lin, H_Q)    # (M, 32, 64)
    k_rope = apply_rope(k_lin, H_KV)   # (M, 8, 64)
    v_h    = v_lin.reshape(M_TOKENS, H_KV, D_V)

    if args.decode:
        n_ctx_past = n_kv - 1
        x_p_fp16 = x_all[:n_ctx_past].astype(np.float16).astype(np.float32)
        rms1_p = 1.0 / np.sqrt(np.mean(x_p_fp16 ** 2, axis=1, keepdims=True) + eps)
        xn_p = x_p_fp16 * rms1_p * g1_fp16
        k_p_rope = apply_rope(xn_p @ w_k.T, H_KV, cos_all[:n_ctx_past], sin_all[:n_ctx_past])
        v_p_h    = (xn_p @ w_v.T).reshape(n_ctx_past, H_KV, D_V)
        k_all = np.concatenate([k_p_rope, k_rope[0:1]], axis=0)
        v_all = np.concatenate([v_p_h,    v_h[0:1]],    axis=0)
        mask = np.zeros((M_TOKENS, n_kv), dtype=np.float32)
    elif n_past > 0:
        x_p_fp16 = x_all[:n_past].astype(np.float16).astype(np.float32)
        rms1_p = 1.0 / np.sqrt(np.mean(x_p_fp16 ** 2, axis=1, keepdims=True) + eps)
        xn_p = x_p_fp16 * rms1_p * g1_fp16
        k_p_rope = apply_rope(xn_p @ w_k.T, H_KV, cos_all[:n_past], sin_all[:n_past])
        v_p_h    = (xn_p @ w_v.T).reshape(n_past, H_KV, D_V)
        k_all = np.concatenate([k_p_rope, k_rope], axis=0)
        v_all = np.concatenate([v_p_h,    v_h],    axis=0)
        mask = np.zeros((M_TOKENS, n_kv), dtype=np.float32)
    else:
        k_all = k_rope
        v_all = v_h
        mask = np.triu(np.full((M_TOKENS, M_TOKENS), -1e9, dtype=np.float32), k=1)

    # GQA 4:1 Attention (note: 1/sqrt(D_K) is already fused into w_q, w_k!)
    k_rep = np.repeat(k_all, H_Q // H_KV, axis=1)  # (N_KV, 32, 64)
    v_rep = np.repeat(v_all, H_Q // H_KV, axis=1)  # (N_KV, 32, 64)
    attn_out = np.zeros((M_TOKENS, H_Q, D_V), dtype=np.float32)
    for h in range(H_Q):
        scores = q_rope[:, h, :] @ k_rep[:, h, :].T + mask
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=1, keepdims=True)
        attn_out[:, h, :] = probs @ v_rep[:, h, :]
    attn_flat = attn_out.reshape(M_TOKENS, D_MODEL)

    o_proj = attn_flat @ w_o.T
    res1_f32 = x_fp16_in + o_proj

    rms2 = 1.0 / np.sqrt(np.mean(res1_f32 ** 2, axis=1, keepdims=True) + eps)
    n2_f32 = res1_f32 * rms2 * g2_fp16

    gate = n2_f32 @ w_g.T
    up   = n2_f32 @ w_u.T
    silu_gate = gate / (1.0 + np.exp(-gate))
    h_ffn = silu_gate * up
    down  = h_ffn @ w_d.T
    final_f32 = res1_f32 + down

    # 2b. FP4+Log8 Fake-Quantized Forward Pass (hardware-aligned golden)
    xn_fq = fq_rtl_vec(xn_f32)
    q_fq = fq_rtl_vec(apply_rope(xn_fq @ fq_rtl_vec(w_q).T, H_Q).reshape(M_TOKENS, D_MODEL)).reshape(M_TOKENS, H_Q, D_K)
    k_fq = fq_rtl_vec(apply_rope(xn_fq @ fq_rtl_vec(w_k).T, H_KV).reshape(M_TOKENS, H_KV * D_K)).reshape(M_TOKENS, H_KV, D_K)
    v_fq = fq_rtl_vec(xn_fq @ fq_rtl_vec(w_v).T).reshape(M_TOKENS, H_KV, D_V)
    if args.decode:
        n_ctx_past = n_kv - 1
        xn_p_fq = fq_rtl_vec(xn_p)
        k_p_raw = apply_rope(xn_p_fq @ fq_rtl_vec(w_k).T, H_KV, cos_all[:n_ctx_past], sin_all[:n_ctx_past])
        k_p_perm = np.empty_like(k_p_raw)
        k_p_perm[:, :, 0::2] = k_p_raw[:, :, :D_K // 2]
        k_p_perm[:, :, 1::2] = k_p_raw[:, :, D_K // 2:]
        k_p_fq_perm = fq_rtl_vec(k_p_perm.reshape(n_ctx_past * H_KV, D_K)).reshape(n_ctx_past, H_KV, D_K)
        k_p_fq = np.empty_like(k_p_fq_perm)
        k_p_fq[:, :, :D_K // 2] = k_p_fq_perm[:, :, 0::2]
        k_p_fq[:, :, D_K // 2:] = k_p_fq_perm[:, :, 1::2]
        v_p_raw = (xn_p_fq @ fq_rtl_vec(w_v).T).reshape(n_ctx_past, H_KV, D_V)
        v_all_fq_in = np.concatenate([v_p_raw, v_fq[0:1]], axis=0)
        v_p_t = np.transpose(v_all_fq_in, (1, 2, 0)).reshape(H_KV * D_V, n_kv)
        v_fq_all = np.transpose(fq_rtl_vec(v_p_t).reshape(H_KV, D_V, n_kv), (2, 0, 1))
        k_fq_all = np.concatenate([k_p_fq, k_fq[0:1]], axis=0)
    elif n_past > 0:
        xn_p_fq = fq_rtl_vec(xn_p)
        k_p_raw = apply_rope(xn_p_fq @ fq_rtl_vec(w_k).T, H_KV, cos_all[:n_past], sin_all[:n_past])
        k_p_perm = np.empty_like(k_p_raw)
        k_p_perm[:, :, 0::2] = k_p_raw[:, :, :D_K // 2]
        k_p_perm[:, :, 1::2] = k_p_raw[:, :, D_K // 2:]
        k_p_fq_perm = fq_rtl_vec(k_p_perm.reshape(n_past * H_KV, D_K)).reshape(n_past, H_KV, D_K)
        k_p_fq = np.empty_like(k_p_fq_perm)
        k_p_fq[:, :, :D_K // 2] = k_p_fq_perm[:, :, 0::2]
        k_p_fq[:, :, D_K // 2:] = k_p_fq_perm[:, :, 1::2]
        v_p_raw = (xn_p_fq @ fq_rtl_vec(w_v).T).reshape(n_past, H_KV, D_V)
        # V in TCDM is transposed: 16 tokens packed per packet along token axis
        v_p_t   = np.transpose(v_p_raw, (1, 2, 0)).reshape(H_KV * D_V, n_past)
        v_p_fq  = np.transpose(fq_rtl_vec(v_p_t).reshape(H_KV, D_V, n_past), (2, 0, 1))
        k_fq_all = np.concatenate([k_p_fq, k_fq], axis=0)
        v_fq_all = np.concatenate([v_p_fq, v_fq], axis=0)
    else:
        k_fq_all = k_fq
        v_fq_all = v_fq
    k_fq_rep = np.repeat(k_fq_all, H_Q // H_KV, axis=1)
    v_fq_rep = np.repeat(v_fq_all, H_Q // H_KV, axis=1)
    attn_fq = np.zeros((M_TOKENS, H_Q, D_V), dtype=np.float32)
    for h in range(H_Q):
        sc_fq = q_fq[:, h, :] @ k_fq_rep[:, h, :].T + mask
        sc_fq -= sc_fq.max(axis=1, keepdims=True)
        pr_fq = np.exp(sc_fq)
        pr_fq /= pr_fq.sum(axis=1, keepdims=True)
        attn_fq[:, h, :] = fq_rtl_vec(pr_fq) @ v_fq_rep[:, h, :]
    res1_fq = x_fp16_in + fq_rtl_vec(attn_fq.reshape(M_TOKENS, D_MODEL)) @ fq_rtl_vec(w_o).T
    rms2_fq = 1.0 / np.sqrt(np.mean(res1_fq ** 2, axis=1, keepdims=True) + eps)
    n2_fq   = res1_fq * rms2_fq * g2_fp16
    n2_qq   = fq_rtl_vec(n2_fq)
    g_fq    = n2_qq @ fq_rtl_vec(w_g).T
    u_fq    = n2_qq @ fq_rtl_vec(w_u).T
    h_fq    = fq_rtl_vec((g_fq / (1.0 + np.exp(-g_fq))) * u_fq)
    final_fq = res1_fq + h_fq @ fq_rtl_vec(w_d).T

    # 3. Quantize & pack all 7 weight matrices into 512-bit DRAM image
    dram_beats = []
    if not args.skip_weights:
        dram_buf = np.zeros(DRAM_TOTAL_BYTES, dtype=np.uint8)
        w_q_perm = permute_qk_weights_for_rope(w_q, H_Q, D_K)
        w_k_perm = permute_qk_weights_for_rope(w_k, H_KV, D_K)

        print("Quantizing & packing 34.2 MB weights to DRAM...")
        pack_proj_to_dram(dram_buf, DRAM_WQ, quantize_rtl_vec(w_q_perm))
        pack_proj_to_dram(dram_buf, DRAM_WK, quantize_rtl_vec(w_k_perm))
        pack_proj_to_dram(dram_buf, DRAM_WV, quantize_rtl_vec(w_v))
        pack_proj_to_dram(dram_buf, DRAM_WO, quantize_rtl_vec(w_o))
        pack_proj_to_dram(dram_buf, DRAM_WG, quantize_rtl_vec(w_g))
        pack_proj_to_dram(dram_buf, DRAM_WU, quantize_rtl_vec(w_u))
        pack_proj_to_dram(dram_buf, DRAM_WD, quantize_rtl_vec(w_d))

        # Write dram_weights.hex (each line = 64 bytes = 128 hex chars, MSB-first for $readmemh)
        dram_beats = dram_buf.reshape(-1, 64)[:, ::-1]
        dram_hex_path = os.path.join(OUT_DIR, "dram_weights.hex")
        with open(dram_hex_path, "w") as f:
            f.write("\n".join(row.tobytes().hex() for row in dram_beats) + "\n")
    else:
        print("Skipping dram_weights.hex (--skip-weights)...")

    # 4. Write TCDM preload (X, gamma1, gamma2, interleaved RoPE cos/sin)
    pre_addr_list = []
    pre_data_list = []

    def add_fp16_rows(bank, word_off, mat_f32):
        u16 = mat_f32.astype(np.float16).view(np.uint16).reshape(-1, 4)
        for i, row in enumerate(u16):
            w64 = (int(row[3]) << 48) | (int(row[2]) << 32) | (int(row[1]) << 16) | int(row[0])
            pre_addr_list.append(tcdm_addr(word_off + i, bank))
            pre_data_list.append(w64)

    add_fp16_rows(X_BANK, 0, x_fp16_in[:1] if args.decode else x_fp16_in)
    add_fp16_rows(GAMMA_BANK, GAMMA1_WORD, g1_fp16)
    add_fp16_rows(GAMMA_BANK, GAMMA2_WORD, g2_fp16)

    cos_u16 = cos_f32.astype(np.float16).view(np.uint16)
    sin_u16 = sin_f32.astype(np.float16).view(np.uint16)
    for nt in range(N_TILES_MODEL):
        half_idx = (nt % 4) * 8
        if args.decode:
            for w in range(4):
                j0 = half_idx + w * 2
                j1 = half_idx + w * 2 + 1
                pk = (int(sin_u16[0, j1]) << 48) | (int(cos_u16[0, j1]) << 32) | \
                     (int(sin_u16[0, j0]) << 16) | int(cos_u16[0, j0])
                pre_addr_list.append(tcdm_addr(nt * 4 + w, ROPE_BANK))
                pre_data_list.append(pk)
        else:
            for t in range(M_TOKENS):
                vec_idx = nt * M_TOKENS + t
                for w in range(4):
                    j0 = half_idx + w * 2
                    j1 = half_idx + w * 2 + 1
                    pk = (int(sin_u16[t, j1]) << 48) | (int(cos_u16[t, j1]) << 32) | \
                         (int(sin_u16[t, j0]) << 16) | int(cos_u16[t, j0])
                    pre_addr_list.append(tcdm_addr(vec_idx * 4 + w, ROPE_BANK))
                    pre_data_list.append(pk)

    with open(os.path.join(OUT_DIR, "tcdm_preload_addr.hex"), "w") as fa, \
         open(os.path.join(OUT_DIR, "tcdm_preload_data.hex"), "w") as fd:
        fa.write("\n".join(f"{a:08x}" for a in pre_addr_list) + "\n")
        fd.write("\n".join(f"{d:018x}" for d in pre_data_list) + "\n")

    if not args.decode:
        rope_mbatch = []
        for mb in range(M_TOKENS // 16):
            for nt in range(4):
                half_idx = nt * 8
                for t in range(16):
                    tok = mb * 16 + t
                    for w in range(4):
                        j0 = half_idx + w * 2
                        j1 = half_idx + w * 2 + 1
                        pk = (int(sin_u16[tok, j1]) << 48) | (int(cos_u16[tok, j1]) << 32) | \
                             (int(sin_u16[tok, j0]) << 16) | int(cos_u16[tok, j0])
                        rope_mbatch.append(pk)
        with open(os.path.join(OUT_DIR, "rope_mbatch.hex"), "w") as fr:
            fr.write("\n".join(f"{d:018x}" for d in rope_mbatch) + "\n")

    # 5. Write golden references
    write_fp16_hex(os.path.join(OUT_DIR, "g_xnorm.hex"), xn_f32)
    write_fp16_hex(os.path.join(OUT_DIR, "g_res1.hex"), res1_f32)
    write_fp16_hex(os.path.join(OUT_DIR, "g_n2.hex"), n2_f32)
    write_fp16_hex(os.path.join(OUT_DIR, "g_final_fp16.hex"), final_f32)
    write_fp32_hex(os.path.join(OUT_DIR, "g_final_fp32.hex"), final_f32)
    write_fp16_hex(os.path.join(OUT_DIR, "g_res1_fq.hex"), res1_fq)
    write_fp16_hex(os.path.join(OUT_DIR, "g_n2_fq.hex"), n2_fq)
    write_fp16_hex(os.path.join(OUT_DIR, "g_final_fq.hex"), final_fq)

    if args.decode:
        k_all_perm = np.zeros((n_kv, H_KV, D_K), dtype=np.float32)
        k_all_perm[:n_kv - 1] = k_p_perm
        k_p_pkts = quantize_rtl_vec(k_all_perm.reshape(n_kv * H_KV, D_K)) \
            .reshape(kv_tiles, 16, H_KV, D_K // 16, 9) \
            .transpose(2, 0, 3, 1, 4)  # (H_KV, kv_tiles, 4, 16, 9)
        v_p_pkts = quantize_rtl_vec(v_p_t) \
            .reshape(H_KV, D_V // 16, 16, kv_tiles, 9) \
            .transpose(0, 1, 3, 2, 4)  # (H_KV, DV_TILES, kv_tiles, 16, 9)
        # Pack DRAM KV-cache in Ring-Buffer tile order: (hkv, dvi, kj) -> 768B/tile (5*144=720B + 48B pad)
        dv_tiles = D_V // 16
        kv_tiles_bytes = np.zeros((H_KV, dv_tiles, kv_tiles, 768), dtype=np.uint8)
        k_tile_bytes = np.broadcast_to(
            k_p_pkts[:, None, :, :, :, :].reshape(H_KV, 1, kv_tiles, 576),
            (H_KV, dv_tiles, kv_tiles, 576)
        )
        v_tile_bytes = v_p_pkts.reshape(H_KV, dv_tiles, kv_tiles, 144)
        kv_tiles_bytes[:, :, :, :576] = k_tile_bytes
        kv_tiles_bytes[:, :, :, 576:720] = v_tile_bytes
        kv_dram_beats = kv_tiles_bytes.reshape(-1, 64)[:, ::-1]
        with open(os.path.join(OUT_DIR, "kv_dram.hex"), "w") as fkv:
            fkv.write("\n".join(row.tobytes().hex() for row in kv_dram_beats) + "\n")
        print(f"Wrote DECODE kv_dram.hex ({len(kv_dram_beats)} DRAM beats = {kv_tiles_bytes.size} bytes)")
    elif n_past > 0:
        k_p_pkts = quantize_rtl_vec(k_p_perm.reshape(n_past * H_KV, D_K)) \
            .reshape(kv_tiles_past, 16, H_KV, D_K // 16, 9) \
            .transpose(2, 0, 3, 1, 4).reshape(-1, 9)
        v_p_pkts = quantize_rtl_vec(v_p_t) \
            .reshape(H_KV, D_V // 16, 16, kv_tiles_past, 9) \
            .transpose(0, 1, 3, 2, 4).reshape(-1, 9)
        with open(os.path.join(OUT_DIR, "k_past.hex"), "w") as fk:
            fk.write("\n".join(f"{int.from_bytes(r.tobytes(), 'little'):018x}" for r in k_p_pkts) + "\n")
        with open(os.path.join(OUT_DIR, "v_past.hex"), "w") as fv:
            fv.write("\n".join(f"{int.from_bytes(r.tobytes(), 'little'):018x}" for r in v_p_pkts) + "\n")
        print(f"Wrote k_past.hex ({len(k_p_pkts)} pkts) and v_past.hex ({len(v_p_pkts)} pkts)")

    print(f"Done in {time.time() - t0:.2f}s! Preload words={len(pre_addr_list)}, DRAM beats={len(dram_beats)}")


if __name__ == "__main__":
    main()
