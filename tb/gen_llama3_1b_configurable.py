#!/usr/bin/env python3
"""
gen_llama3_1b_configurable.py

Configurable real-model stimulus & golden reference generator for 1 full layer of
NousResearch/Llama-3.2-1B running on snax_llm_top_clean + 8-port snax_simple_dma:
  - Supports arbitrary M_TOKENS (Prefill: multiple of 16, Decode: 1)
  - Supports arbitrary N_KV     (multiple of 16, N_KV >= M_TOKENS)
  - Extracts real layer weights, RMSNorm gammas, and hidden-state activations via PyTorch hook
  - Reuses layer weights (34.2 MB FP4+Log8) across (M, N_KV) sweeps
  - Generates separate-KV past cache files + causal FP32 & FP4 Fake-Quant golden references
"""
import argparse
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Llama-3.2-1B fixed architecture geometry
D_MODEL       = 2048
D_FF          = 8192
H_Q           = 32
H_KV          = 8
D_K           = 64
D_V           = 64
GQA_GRP       = H_Q // H_KV        # 4
ROW_WORDS     = D_MODEL // 4       # 512
N_TILES_MODEL = D_MODEL // 16      # 128
N_TILES_KV    = (H_KV * D_K) // 16 # 32
DV_TILES      = D_V // 16          # 4
K_PACKETS_QK  = D_K // 16          # 4

# DRAM byte offsets (64B-aligned)
DRAM_WQ = 0x0000_0000  # 128 * 128 * 144 = 2,359,296 B
DRAM_WK = 0x0024_0000  #  32 * 128 * 144 =   589,824 B
DRAM_WV = 0x002D_0000  #  32 * 128 * 144 =   589,824 B
DRAM_WO = 0x0036_0000  # 128 * 128 * 144 = 2,359,296 B
DRAM_WG = 0x005A_0000  # 512 * 128 * 144 = 9,437,184 B
DRAM_WU = 0x00EA_0000  # 512 * 128 * 144 = 9,437,184 B
DRAM_WD = 0x017A_0000  # 128 * 512 * 144 = 9,437,184 B
DRAM_TOTAL_BYTES = DRAM_WD + 128 * 512 * 144  # 34,209,792 B

REAL_PROMPT = (
    "Modern large language model inference on edge devices requires balancing compute density, "
    "on-chip scratchpad capacity, and external DRAM bandwidth. When an accelerator operates "
    "at low precision using logarithmic block scaling combined with four-bit floating-point "
    "formats, each memory transaction transfers nearly four times more parameters per byte "
    "compared to sixteen-bit floating point. During prefill, systolic and output-stationary "
    "matrix engines amortize weight fetches across batches of input tokens, transforming "
    "linear projections into compute-bound workloads. However, as prompt length and key-value "
    "context windows scale from dozens to hundreds of tokens, intermediate activation tensors, "
    "attention query matrices, and KV caches exceed the size of compact multi-bank SRAM "
    "scratchpads. Without autonomous macro-tiling and hardware double buffering, host processors "
    "must interrupt execution to swap tiles manually, eroding utilization. By combining "
    "ping-pong double buffering for activations and outputs, ring-buffer credit management "
    "for streamed weights, grouped-query KV pinning across shared heads, and parallel DMA "
    "ports connected cleanly through banked crossbar interconnects, an embedded NPU sustains "
    "high throughput and numerical fidelity across arbitrary sequence lengths and decode steps."
)


def quantize_rtl_vec(mat_f32):
    """Bit-exact vectorized model of slq_encoder.sv (72-bit packet = 16x FP4 + 8b Log8 scale)."""
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

    mag = np.zeros_like(diff, dtype=np.uint8)
    mag = np.where(diff <= 36, 1, mag)
    mag = np.where(diff <= 24, 2, mag)
    mag = np.where(diff <= 18, 3, mag)
    mag = np.where(diff <= 14, 4, mag)
    mag = np.where(diff <= 10, 5, mag)
    mag = np.where(diff <= 6,  6, mag)
    mag = np.where(diff <= 2,  7, mag)
    mag = np.where((mu == 0) | (mx == 0), 0, mag)

    codes = np.where(mag == 0, 0, (sign.astype(np.uint8) << 3) | mag)
    pkts = np.zeros((n_ch, kb, 9), dtype=np.uint8)
    for b in range(8):
        pkts[:, :, b] = codes[:, :, 2 * b] | (codes[:, :, 2 * b + 1] << 4)
    pkts[:, :, 8] = sc.astype(np.uint8)
    return pkts


def fq_rtl_vec(mat_f32):
    """Dequantized FP4+Log8 Fake-Quant tensor matching quantize_rtl_vec."""
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
    """Permute Q/K channels so adjacent even/odd lanes match RoPE pairs (j, j + d_k//2)."""
    out = np.empty_like(mat)
    half = d_k // 2
    for h in range(n_heads):
        base = h * d_k
        out[base + 0:base + d_k:2] = mat[base:base + half]
        out[base + 1:base + d_k:2] = mat[base + half:base + d_k]
    return out


def pack_proj_to_dram(dram_buf, byte_off, pkts_9b):
    """Pack (n_ch, kb, 9) into DRAM in N-tile major, K-block middle, 16-lane inner order."""
    n_ch, _, _ = pkts_9b.shape
    n_tiles = n_ch // 16
    reshaped = pkts_9b.reshape(n_tiles, 16, -1, 9).transpose(0, 2, 1, 3).reshape(-1)
    dram_buf[byte_off:byte_off + reshaped.size] = reshaped


def pack_fp16_72b_words(mat_f32):
    """Pack row-major float values (multiple of 4 cols) into list of 72b integers (4xFP16/word)."""
    u16 = mat_f32.astype(np.float16).view(np.uint16).reshape(-1, 4)
    # Vectorized uint64 combine
    w = (u16[:, 3].astype(np.uint64) << 48) | \
        (u16[:, 2].astype(np.uint64) << 32) | \
        (u16[:, 1].astype(np.uint64) << 16) | \
         u16[:, 0].astype(np.uint64)
    return w


def write_72b_words_hex(path, words_u64):
    with open(path, "w") as f:
        f.write("\n".join(f"{int(v):018x}" for v in words_u64) + "\n")


def write_9b_pkts_hex(path, pkts_9b):
    flat = pkts_9b.reshape(-1, 9)
    with open(path, "w") as f:
        f.write("\n".join(f"{int.from_bytes(r.tobytes(), 'little'):018x}" for r in flat) + "\n")


def write_fp16_hex(path, arr_f32):
    u16 = arr_f32.astype(np.float16).view(np.uint16).reshape(-1)
    with open(path, "w") as f:
        f.write("\n".join(f"{int(v):04x}" for v in u16) + "\n")


def write_fp32_hex(path, arr_f32):
    u32 = arr_f32.astype(np.float32).view(np.uint32).reshape(-1)
    with open(path, "w") as f:
        f.write("\n".join(f"{int(v):08x}" for v in u32) + "\n")


def metrics(act, ref):
    a = act.astype(np.float64).ravel()
    r = ref.astype(np.float64).ravel()
    err = a - r
    ref2 = float(np.dot(r, r))
    err2 = float(np.dot(err, err))
    act2 = float(np.dot(a, a))
    sqnr = 10.0 * np.log10(ref2 / err2) if err2 > 0.0 else 999.0
    cos = float(np.dot(a, r)) / (np.sqrt(act2) * np.sqrt(ref2)) if (act2 > 0 and ref2 > 0) else 0.0
    max_abs = float(np.max(np.abs(err)))
    return sqnr, cos, max_abs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--m-tokens", type=int, default=None,
                        help="Prefill input tokens (multiple of 16); Decode automatically uses 1")
    parser.add_argument("--n-kv", type=int, default=None,
                        help="KV context length (multiple of 16, >= m-tokens)")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--model-id", type=str, default="NousResearch/Llama-3.2-1B")
    parser.add_argument("--weights-dir", type=str,
                        default=os.path.join(HERE, "..", "sim", "sim_llama3_1b_weights"))
    parser.add_argument("--force-weights", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    is_decode = (args.mode == "decode")
    if is_decode:
        m_tokens = 1
        n_kv = 64 if args.n_kv is None else args.n_kv
        if args.m_tokens is not None and args.m_tokens != 1:
            print(f"[WARN] Decode mode ignores --m-tokens={args.m_tokens}, using M=1")
    else:
        m_tokens = 16 if args.m_tokens is None else args.m_tokens
        n_kv = m_tokens if args.n_kv is None else args.n_kv
        assert m_tokens >= 16 and m_tokens % 16 == 0, "--m-tokens must be multiple of 16 in prefill"

    assert n_kv >= m_tokens and n_kv % 16 == 0, "--n-kv must be >= m-tokens and multiple of 16"
    n_past = n_kv - m_tokens
    kv_tiles = n_kv // 16
    kv_tiles_past = n_past // 16  # prefill exact past tiles; decode tail slot is kv_tiles-1

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(HERE, "..", "sim", f"sim_llama3_1b_{args.mode}_m{m_tokens}_kv{n_kv}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)

    t0 = time.time()
    print(f"=== Generating Real {args.model_id} Layer {args.layer} Data ({args.mode.upper()}) ===")
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
    orig_len = ids.shape[1]
    while ids.shape[1] < n_kv:
        ids = torch.cat([ids, ids], dim=1)
    ids = ids[:, :n_kv]
    print(f"    Prompt tokens available={orig_len}, captured sequence len={ids.shape[1]}")

    captured = {}
    def hook_fn(_mod, inp, _out):
        captured["x"] = inp[0].detach().cpu().numpy()[0]

    h = model.model.layers[args.layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(ids)
    h.remove()

    qk_scale = np.float32(D_K ** -0.25)
    lyr = model.model.layers[args.layer]
    x_all  = captured["x"][:n_kv, :D_MODEL].astype(np.float32)
    x_f32  = x_all[n_past:, :]  # (m_tokens, 2048)
    g1_f32 = lyr.input_layernorm.weight.detach().cpu().numpy().astype(np.float32)
    g2_f32 = lyr.post_attention_layernorm.weight.detach().cpu().numpy().astype(np.float32)
    w_q = lyr.self_attn.q_proj.weight.detach().cpu().numpy().astype(np.float32) * qk_scale
    w_k = lyr.self_attn.k_proj.weight.detach().cpu().numpy().astype(np.float32) * qk_scale
    w_v = lyr.self_attn.v_proj.weight.detach().cpu().numpy().astype(np.float32)
    w_o = lyr.self_attn.o_proj.weight.detach().cpu().numpy().astype(np.float32)
    w_g = lyr.mlp.gate_proj.weight.detach().cpu().numpy().astype(np.float32)
    w_u = lyr.mlp.up_proj.weight.detach().cpu().numpy().astype(np.float32)
    w_d = lyr.mlp.down_proj.weight.detach().cpu().numpy().astype(np.float32)
    print(f"    Real x_f32: shape={x_f32.shape}, std={x_f32.std():.4f}, max_abs={np.max(np.abs(x_f32)):.4f}")

    # Emit/reuse DRAM weights (34.2 MB)
    weights_hex = os.path.join(args.weights_dir, f"dram_weights_l{args.layer}.hex")
    if args.force_weights or not os.path.exists(weights_hex):
        print(f"    Quantizing & packing 34.2 MB weights -> {weights_hex} ...")
        dram_buf = np.zeros(DRAM_TOTAL_BYTES, dtype=np.uint8)
        w_q_perm = permute_qk_weights_for_rope(w_q, H_Q, D_K)
        w_k_perm = permute_qk_weights_for_rope(w_k, H_KV, D_K)
        pack_proj_to_dram(dram_buf, DRAM_WQ, quantize_rtl_vec(w_q_perm))
        pack_proj_to_dram(dram_buf, DRAM_WK, quantize_rtl_vec(w_k_perm))
        pack_proj_to_dram(dram_buf, DRAM_WV, quantize_rtl_vec(w_v))
        pack_proj_to_dram(dram_buf, DRAM_WO, quantize_rtl_vec(w_o))
        pack_proj_to_dram(dram_buf, DRAM_WG, quantize_rtl_vec(w_g))
        pack_proj_to_dram(dram_buf, DRAM_WU, quantize_rtl_vec(w_u))
        pack_proj_to_dram(dram_buf, DRAM_WD, quantize_rtl_vec(w_d))
        dram_beats = dram_buf.reshape(-1, 64)[:, ::-1]
        with open(weights_hex, "w") as fw:
            fw.write("\n".join(row.tobytes().hex() for row in dram_beats) + "\n")
    else:
        print(f"    Reusing cached weights: {weights_hex}")

    # RoPE cos/sin (theta = 500000.0)
    pos_all = np.arange(n_kv, dtype=np.float32)[:, None]
    inv_freq = 1.0 / (500000.0 ** (np.arange(0, D_K, 2, dtype=np.float32) / D_K))[None, :]
    freqs_all = pos_all * inv_freq
    cos_all = np.cos(freqs_all).astype(np.float16).astype(np.float32)
    sin_all = np.sin(freqs_all).astype(np.float16).astype(np.float32)
    cos_f32 = cos_all[n_past:]
    sin_f32 = sin_all[n_past:]

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

    # 1. FP32 Golden Forward Pass
    eps = 1e-5
    x_fp16_in = x_f32.astype(np.float16).astype(np.float32)
    g1_fp16   = g1_f32.astype(np.float16).astype(np.float32)
    g2_fp16   = g2_f32.astype(np.float16).astype(np.float32)

    rms1 = 1.0 / np.sqrt(np.mean(x_fp16_in ** 2, axis=1, keepdims=True) + eps)
    xn_f32 = x_fp16_in * rms1 * g1_fp16

    q_rope = apply_rope(xn_f32 @ w_q.T, H_Q)
    k_rope = apply_rope(xn_f32 @ w_k.T, H_KV)
    v_h    = (xn_f32 @ w_v.T).reshape(m_tokens, H_KV, D_V)

    if n_past > 0:
        x_p_fp16 = x_all[:n_past].astype(np.float16).astype(np.float32)
        rms1_p = 1.0 / np.sqrt(np.mean(x_p_fp16 ** 2, axis=1, keepdims=True) + eps)
        xn_p = x_p_fp16 * rms1_p * g1_fp16
        k_p_rope = apply_rope(xn_p @ w_k.T, H_KV, cos_all[:n_past], sin_all[:n_past])
        v_p_h    = (xn_p @ w_v.T).reshape(n_past, H_KV, D_V)
        k_all = np.concatenate([k_p_rope, k_rope], axis=0)
        v_all = np.concatenate([v_p_h,    v_h],    axis=0)
    else:
        k_all = k_rope
        v_all = v_h

    if is_decode:
        mask = np.zeros((1, n_kv), dtype=np.float32)
    else:
        q_abs = np.arange(m_tokens, dtype=np.int32)[:, None] + n_past
        k_pos = np.arange(n_kv, dtype=np.int32)[None, :]
        mask = np.where(k_pos <= q_abs, 0.0, -1e9).astype(np.float32)

    k_rep = np.repeat(k_all, GQA_GRP, axis=1)
    v_rep = np.repeat(v_all, GQA_GRP, axis=1)
    attn_out = np.zeros((m_tokens, H_Q, D_V), dtype=np.float32)
    for h_i in range(H_Q):
        scores = q_rope[:, h_i, :] @ k_rep[:, h_i, :].T + mask
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=1, keepdims=True)
        attn_out[:, h_i, :] = probs @ v_rep[:, h_i, :]
    res1_f32 = x_fp16_in + attn_out.reshape(m_tokens, D_MODEL) @ w_o.T
    rms2 = 1.0 / np.sqrt(np.mean(res1_f32 ** 2, axis=1, keepdims=True) + eps)
    n2_f32 = res1_f32 * rms2 * g2_fp16
    gate = n2_f32 @ w_g.T
    up   = n2_f32 @ w_u.T
    final_f32 = res1_f32 + ((gate / (1.0 + np.exp(-gate))) * up) @ w_d.T

    # 2. FP4+Log8 Hardware-Aligned Fake-Quant Golden Pass
    wq_fq = fq_rtl_vec(w_q)
    wk_fq = fq_rtl_vec(w_k)
    wv_fq = fq_rtl_vec(w_v)
    wo_fq = fq_rtl_vec(w_o)
    wg_fq = fq_rtl_vec(w_g)
    wu_fq = fq_rtl_vec(w_u)
    wd_fq = fq_rtl_vec(w_d)

    def quant_rope_qk(raw_rope, n_h):
        # Match Fused QK RoPE permute -> SLQ quantize -> unpermute
        perm = np.empty_like(raw_rope)
        perm[:, :, 0::2] = raw_rope[:, :, :D_K // 2]
        perm[:, :, 1::2] = raw_rope[:, :, D_K // 2:]
        fq_perm = fq_rtl_vec(perm.reshape(-1, D_K)).reshape(raw_rope.shape[0], n_h, D_K)
        out = np.empty_like(fq_perm)
        out[:, :, :D_K // 2] = fq_perm[:, :, 0::2]
        out[:, :, D_K // 2:] = fq_perm[:, :, 1::2]
        return out

    xn_fq = fq_rtl_vec(xn_f32)
    q_fq = quant_rope_qk(apply_rope(xn_fq @ wq_fq.T, H_Q), H_Q)
    k_fq = quant_rope_qk(apply_rope(xn_fq @ wk_fq.T, H_KV), H_KV)
    v_raw_fq = (xn_fq @ wv_fq.T).reshape(m_tokens, H_KV, D_V)

    if is_decode:
        xn_p_fq = fq_rtl_vec(xn_p)
        k_p_raw = apply_rope(xn_p_fq @ wk_fq.T, H_KV, cos_all[:n_past], sin_all[:n_past])
        k_p_fq  = quant_rope_qk(k_p_raw, H_KV)
        # Keep k_p_perm for packing past K into DRAM_K
        k_p_perm = np.empty_like(k_p_raw)
        k_p_perm[:, :, 0::2] = k_p_raw[:, :, :D_K // 2]
        k_p_perm[:, :, 1::2] = k_p_raw[:, :, D_K // 2:]
        v_p_raw = (xn_p_fq @ wv_fq.T).reshape(n_past, H_KV, D_V)
        v_all_in = np.concatenate([v_p_raw, v_raw_fq], axis=0)  # (n_kv, H_KV, D_V)
        v_t_mat  = np.transpose(v_all_in, (1, 2, 0)).reshape(H_KV * D_V, n_kv)
        v_fq_all = np.transpose(fq_rtl_vec(v_t_mat).reshape(H_KV, D_V, n_kv), (2, 0, 1))
        k_fq_all = np.concatenate([k_p_fq, k_fq], axis=0)
    else:
        # Prefill: new M_TOKENS V are transposed in 16-token blocks
        v_new_t  = np.transpose(v_raw_fq.reshape(m_tokens // 16, 16, H_KV, D_V), (0, 2, 3, 1)).reshape(-1, 16)
        v_new_fq = np.transpose(fq_rtl_vec(v_new_t).reshape(m_tokens // 16, H_KV, D_V, 16), (0, 3, 1, 2)).reshape(m_tokens, H_KV, D_V)
        if n_past > 0:
            xn_p_fq = fq_rtl_vec(xn_p)
            k_p_raw = apply_rope(xn_p_fq @ wk_fq.T, H_KV, cos_all[:n_past], sin_all[:n_past])
            k_p_fq  = quant_rope_qk(k_p_raw, H_KV)
            k_p_perm = np.empty_like(k_p_raw)
            k_p_perm[:, :, 0::2] = k_p_raw[:, :, :D_K // 2]
            k_p_perm[:, :, 1::2] = k_p_raw[:, :, D_K // 2:]
            v_p_raw = (xn_p_fq @ wv_fq.T).reshape(n_past, H_KV, D_V)
            v_p_t   = np.transpose(v_p_raw, (1, 2, 0)).reshape(H_KV * D_V, n_past)
            v_p_fq  = np.transpose(fq_rtl_vec(v_p_t).reshape(H_KV, D_V, n_past), (2, 0, 1))
            k_fq_all = np.concatenate([k_p_fq, k_fq], axis=0)
            v_fq_all = np.concatenate([v_p_fq, v_new_fq], axis=0)
        else:
            k_fq_all = k_fq
            v_fq_all = v_new_fq

    k_fq_rep = np.repeat(k_fq_all, GQA_GRP, axis=1)
    v_fq_rep = np.repeat(v_fq_all, GQA_GRP, axis=1)
    attn_fq = np.zeros((m_tokens, H_Q, D_V), dtype=np.float32)
    for h_i in range(H_Q):
        sc_fq = q_fq[:, h_i, :] @ k_fq_rep[:, h_i, :].T + mask
        sc_fq -= sc_fq.max(axis=1, keepdims=True)
        pr_fq = np.exp(sc_fq)
        pr_fq /= pr_fq.sum(axis=1, keepdims=True)
        attn_fq[:, h_i, :] = fq_rtl_vec(pr_fq) @ v_fq_rep[:, h_i, :]
    res1_fq = x_fp16_in + fq_rtl_vec(attn_fq.reshape(m_tokens, D_MODEL)) @ wo_fq.T
    rms2_fq = 1.0 / np.sqrt(np.mean(res1_fq ** 2, axis=1, keepdims=True) + eps)
    n2_fq   = res1_fq * rms2_fq * g2_fp16
    n2_qq   = fq_rtl_vec(n2_fq)
    g_fq    = n2_qq @ wg_fq.T
    u_fq    = n2_qq @ wu_fq.T
    h_fq    = fq_rtl_vec((g_fq / (1.0 + np.exp(-g_fq))) * u_fq)
    final_fq = res1_fq + h_fq @ wd_fq.T

    sq_fq, cs_fq, mx_fq = metrics(final_fq, final_f32)
    print(f"    PyFQ vs FP32 baseline: SQNR={sq_fq:.2f} dB, CosineSim={cs_fq:.6f}, MaxAbs={mx_fq:.4f}")

    # 3. Emit hex stimulus files
    write_72b_words_hex(os.path.join(out_dir, "x_fp16.hex"), pack_fp16_72b_words(x_fp16_in))
    write_72b_words_hex(os.path.join(out_dir, "gamma1.hex"), pack_fp16_72b_words(g1_fp16))
    write_72b_words_hex(os.path.join(out_dir, "gamma2.hex"), pack_fp16_72b_words(g2_fp16))

    cos_u16 = cos_f32.astype(np.float16).view(np.uint16)
    sin_u16 = sin_f32.astype(np.float16).view(np.uint16)
    if is_decode:
        rope_words = []
        for nt in range(N_TILES_MODEL):
            half_idx = (nt % 4) * 8
            for w in range(4):
                j0, j1 = half_idx + w * 2, half_idx + w * 2 + 1
                pk = (int(sin_u16[0, j1]) << 48) | (int(cos_u16[0, j1]) << 32) | \
                     (int(sin_u16[0, j0]) << 16) | int(cos_u16[0, j0])
                rope_words.append(pk)
        write_72b_words_hex(os.path.join(out_dir, "rope_mbatch.hex"), rope_words)
    else:
        rope_mbatch = []
        for mb in range(m_tokens // 16):
            for nt in range(4):
                half_idx = nt * 8
                for t in range(16):
                    tok_i = mb * 16 + t
                    for w in range(4):
                        j0, j1 = half_idx + w * 2, half_idx + w * 2 + 1
                        pk = (int(sin_u16[tok_i, j1]) << 48) | (int(cos_u16[tok_i, j1]) << 32) | \
                             (int(sin_u16[tok_i, j0]) << 16) | int(cos_u16[tok_i, j0])
                        rope_mbatch.append(pk)
        write_72b_words_hex(os.path.join(out_dir, "rope_mbatch.hex"), rope_mbatch)

    # Golden files
    write_fp16_hex(os.path.join(out_dir, "g_xnorm.hex"), xn_f32)
    write_fp16_hex(os.path.join(out_dir, "g_res1.hex"), res1_f32)
    write_fp16_hex(os.path.join(out_dir, "g_n2.hex"), n2_f32)
    write_fp16_hex(os.path.join(out_dir, "g_final_fp16.hex"), final_f32)
    write_fp32_hex(os.path.join(out_dir, "g_final_fp32.hex"), final_f32)
    write_fp16_hex(os.path.join(out_dir, "g_res1_fq.hex"), res1_fq)
    write_fp16_hex(os.path.join(out_dir, "g_n2_fq.hex"), n2_fq)
    write_fp16_hex(os.path.join(out_dir, "g_final_fq.hex"), final_fq)

    # Past KV files in Separate-KV format (matches cfg_attn_sep_kv_i = 1)
    # K order: [H_KV][kj][kp][l]  (4*16 words per kj)
    # V order: [H_KV][DV_TILES][kj][l] (16 words per (dvi, kj))
    if is_decode:
        k_all_perm = np.zeros((n_kv, H_KV, D_K), dtype=np.float32)
        k_all_perm[:n_kv - 1] = k_p_perm
        k_p_pkts = quantize_rtl_vec(k_all_perm.reshape(n_kv * H_KV, D_K)) \
            .reshape(kv_tiles, 16, H_KV, K_PACKETS_QK, 9) \
            .transpose(2, 0, 3, 1, 4)  # (H_KV, kv_tiles, 4, 16, 9)
        v_p_pkts = quantize_rtl_vec(v_t_mat) \
            .reshape(H_KV, DV_TILES, 16, kv_tiles, 9) \
            .transpose(0, 1, 3, 2, 4)  # (H_KV, DV_TILES, kv_tiles, 16, 9)
        write_9b_pkts_hex(os.path.join(out_dir, "k_past_words.hex"), k_p_pkts)
        write_9b_pkts_hex(os.path.join(out_dir, "v_past_words.hex"), v_p_pkts)
        v_tail_words = pack_fp16_72b_words(v_p_raw[n_kv - 16:n_kv - 1].reshape(15, H_KV * D_V))
        write_72b_words_hex(os.path.join(out_dir, "v_tail_past_fp16.hex"), v_tail_words)
    elif kv_tiles_past > 0:
        k_p_pkts = quantize_rtl_vec(k_p_perm.reshape(n_past * H_KV, D_K)) \
            .reshape(kv_tiles_past, 16, H_KV, K_PACKETS_QK, 9) \
            .transpose(2, 0, 3, 1, 4)  # (H_KV, kv_tiles_past, 4, 16, 9)
        v_p_pkts = quantize_rtl_vec(v_p_t) \
            .reshape(H_KV, DV_TILES, 16, kv_tiles_past, 9) \
            .transpose(0, 1, 3, 2, 4)  # (H_KV, DV_TILES, kv_tiles_past, 16, 9)
        write_9b_pkts_hex(os.path.join(out_dir, "k_past_words.hex"), k_p_pkts)
        write_9b_pkts_hex(os.path.join(out_dir, "v_past_words.hex"), v_p_pkts)

    print(f"    Generated in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
