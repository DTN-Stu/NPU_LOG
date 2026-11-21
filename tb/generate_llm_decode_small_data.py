#!/usr/bin/env python3
"""
Generate small LLM Decode layer data (matching tb_snax_llm_layer_fused dimensions)
for tb_llm_decode_small.sv:
  M_TOKENS = 16 (HW GEMM tile; token 0 = active decode token at pos N_KV-1)
  N_KV     = 64 (default, configurable via --n-kv)
  D_MODEL  = 128, D_FF = 256, H_Q = 4, H_KV = 4, D_K = 32, D_V = 32

Outputs to sim/sim_llm_decode_small/.
"""
import argparse
import os
import time
import numpy as np

M_TOKENS = 16
D_MODEL  = 128
D_FF     = 256
H_Q      = 4
H_KV     = 4
D_K      = 32
D_V      = 32
N_TILES_MODEL = D_MODEL // 16        # 8
N_TILES_KV    = (H_KV * D_K) // 16   # 8
K_PACKETS_QKV = D_MODEL // 16        # 8
K_PACKETS_DOWN = D_FF // 16          # 16

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "sim", "sim_llm_decode_small")

X_BANK     = 24
GAMMA_BANK = 28
ROPE_BANK  = 29
GAMMA1_WORD = 0
GAMMA2_WORD = 32

# DRAM byte offsets (64B-aligned)
DRAM_WQ = 0x0000_0000  #   8 *  8 * 144 =   9,216 B
DRAM_WK = 0x0000_3000  #   8 *  8 * 144 =   9,216 B
DRAM_WV = 0x0000_6000  #   8 *  8 * 144 =   9,216 B
DRAM_WO = 0x0000_9000  #   8 *  8 * 144 =   9,216 B
DRAM_WG = 0x0000_C000  #  16 *  8 * 144 =  18,432 B
DRAM_WU = 0x0001_1000  #  16 *  8 * 144 =  18,432 B
DRAM_WD = 0x0001_6000  #   8 * 16 * 144 =  18,432 B
DRAM_TOTAL_BYTES = DRAM_WD + 8 * 16 * 144  # 108,544 B


def tcdm_addr(word, bank):
    return ((word & 0x1FFF) << 9) | ((bank & 0x1F) << 4)


def quantize_rtl_vec(mat_f32):
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
    out = np.empty_like(mat)
    half = d_k // 2
    for h in range(n_heads):
        base = h * d_k
        out[base + 0:base + d_k:2] = mat[base:base + half]
        out[base + 1:base + d_k:2] = mat[base + half:base + d_k]
    return out


def pack_proj_to_dram(dram_buf, byte_off, pkts_9b):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-kv", type=int, default=64,
                        help="Total KV context length (must be >= 16 and divisible by 16)")
    args = parser.parse_args()
    assert args.n_kv >= M_TOKENS and args.n_kv % 16 == 0
    n_kv = args.n_kv
    kv_tiles = n_kv // 16
    n_ctx_past = n_kv - 1

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(2026)

    print(f"=== Generating Small LLM Decode Layer Data (N_KV={n_kv}, D_MODEL={D_MODEL}, D_FF={D_FF}) ===")

    qk_scale = D_K ** -0.25
    x_all  = rng.normal(0.0, 0.25, (n_kv, D_MODEL)).astype(np.float32)
    x_f32  = np.zeros((M_TOKENS, D_MODEL), dtype=np.float32)
    x_f32[0] = x_all[n_kv - 1]
    g1_f32 = rng.normal(1.0, 0.1, (D_MODEL,)).astype(np.float32)
    g2_f32 = rng.normal(1.0, 0.1, (D_MODEL,)).astype(np.float32)

    w_q = rng.normal(0.0, 0.06, (D_MODEL, D_MODEL)).astype(np.float32) * qk_scale
    w_k = rng.normal(0.0, 0.06, (H_KV * D_K, D_MODEL)).astype(np.float32) * qk_scale
    w_v = rng.normal(0.0, 0.06, (H_KV * D_V, D_MODEL)).astype(np.float32)
    w_o = rng.normal(0.0, 0.06, (D_MODEL, D_MODEL)).astype(np.float32)
    w_g = rng.normal(0.0, 0.06, (D_FF, D_MODEL)).astype(np.float32)
    w_u = rng.normal(0.0, 0.06, (D_FF, D_MODEL)).astype(np.float32)
    w_d = rng.normal(0.0, 0.06, (D_MODEL, D_FF)).astype(np.float32)

    pos_all = np.arange(n_kv, dtype=np.float32)[:, None]
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, D_K, 2, dtype=np.float32) / D_K))[None, :]
    freqs_all = pos_all * inv_freq
    cos_all = np.cos(freqs_all).astype(np.float16).astype(np.float32)
    sin_all = np.sin(freqs_all).astype(np.float16).astype(np.float32)
    cos_f32 = np.zeros((M_TOKENS, D_K // 2), dtype=np.float32)
    sin_f32 = np.zeros((M_TOKENS, D_K // 2), dtype=np.float32)
    cos_f32[0] = cos_all[n_kv - 1]
    sin_f32[0] = sin_all[n_kv - 1]

    eps = 1e-5
    x_fp16_in = x_f32.astype(np.float16).astype(np.float32)
    g1_fp16   = g1_f32.astype(np.float16).astype(np.float32)
    g2_fp16   = g2_f32.astype(np.float16).astype(np.float32)

    rms1 = 1.0 / np.sqrt(np.mean(x_fp16_in ** 2, axis=1, keepdims=True) + eps)
    xn_f32 = x_fp16_in * rms1 * g1_fp16

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

    q_rope = apply_rope(xn_f32 @ w_q.T, H_Q)
    k_rope = apply_rope(xn_f32 @ w_k.T, H_KV)
    v_h    = (xn_f32 @ w_v.T).reshape(M_TOKENS, H_KV, D_V)

    x_p_fp16 = x_all[:n_ctx_past].astype(np.float16).astype(np.float32)
    rms1_p = 1.0 / np.sqrt(np.mean(x_p_fp16 ** 2, axis=1, keepdims=True) + eps)
    xn_p = x_p_fp16 * rms1_p * g1_fp16
    k_p_rope = apply_rope(xn_p @ w_k.T, H_KV, cos_all[:n_ctx_past], sin_all[:n_ctx_past])
    v_p_h    = (xn_p @ w_v.T).reshape(n_ctx_past, H_KV, D_V)
    k_all = np.concatenate([k_p_rope, k_rope[0:1]], axis=0)
    v_all = np.concatenate([v_p_h,    v_h[0:1]],    axis=0)

    attn_out = np.zeros((M_TOKENS, H_Q, D_V), dtype=np.float32)
    for h in range(H_Q):
        scores = q_rope[0:1, h, :] @ k_all[:, h, :].T
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=1, keepdims=True)
        attn_out[0:1, h, :] = probs @ v_all[:, h, :]

    res1_f32 = x_fp16_in + attn_out.reshape(M_TOKENS, D_MODEL) @ w_o.T
    rms2 = 1.0 / np.sqrt(np.mean(res1_f32 ** 2, axis=1, keepdims=True) + eps)
    n2_f32 = res1_f32 * rms2 * g2_fp16
    gate = n2_f32 @ w_g.T
    up   = n2_f32 @ w_u.T
    final_f32 = res1_f32 + ((gate / (1.0 + np.exp(-gate))) * up) @ w_d.T

    # Fake-Quant reference
    xn_fq = fq_rtl_vec(xn_f32)
    q_fq = fq_rtl_vec(apply_rope(xn_fq @ fq_rtl_vec(w_q).T, H_Q).reshape(M_TOKENS, D_MODEL)).reshape(M_TOKENS, H_Q, D_K)
    k_fq = fq_rtl_vec(apply_rope(xn_fq @ fq_rtl_vec(w_k).T, H_KV).reshape(M_TOKENS, H_KV * D_K)).reshape(M_TOKENS, H_KV, D_K)
    v_fq = fq_rtl_vec(xn_fq @ fq_rtl_vec(w_v).T).reshape(M_TOKENS, H_KV, D_V)

    xn_p_fq = fq_rtl_vec(xn_p)
    k_p_raw = apply_rope(xn_p_fq @ fq_rtl_vec(w_k).T, H_KV, cos_all[:n_ctx_past], sin_all[:n_ctx_past])
    k_p_perm = np.empty_like(k_p_raw)
    k_p_perm[:, :, 0::2] = k_p_raw[:, :, :D_K // 2]
    k_p_perm[:, :, 1::2] = k_p_raw[:, :, D_K // 2:]
    k_p_fq_perm = fq_rtl_vec(k_p_perm.reshape(n_ctx_past * H_KV, D_K)).reshape(n_ctx_past, H_KV, D_K)
    k_p_fq = np.empty_like(k_p_fq_perm)
    k_p_fq[:, :, :D_K // 2] = k_p_fq_perm[:, :, 0::2]
    k_p_fq[:, :, D_K // 2:] = k_p_fq_perm[:, :, 1::2]
    k_fq_all = np.concatenate([k_p_fq, k_fq[0:1]], axis=0)

    v_p_raw = (xn_p_fq @ fq_rtl_vec(w_v).T).reshape(n_ctx_past, H_KV, D_V)
    v_all_fq_in = np.concatenate([v_p_raw, v_fq[0:1]], axis=0)
    v_p_t = np.transpose(v_all_fq_in, (1, 2, 0)).reshape(H_KV * D_V, n_kv)
    v_fq_all = np.transpose(fq_rtl_vec(v_p_t).reshape(H_KV, D_V, n_kv), (2, 0, 1))

    attn_fq = np.zeros((M_TOKENS, H_Q, D_V), dtype=np.float32)
    for h in range(H_Q):
        sc_fq = q_fq[0:1, h, :] @ k_fq_all[:, h, :].T
        sc_fq -= sc_fq.max(axis=1, keepdims=True)
        pr_fq = np.exp(sc_fq)
        pr_fq /= pr_fq.sum(axis=1, keepdims=True)
        attn_fq[0:1, h, :] = fq_rtl_vec(pr_fq) @ v_fq_all[:, h, :]

    res1_fq = x_fp16_in + fq_rtl_vec(attn_fq.reshape(M_TOKENS, D_MODEL)) @ fq_rtl_vec(w_o).T
    rms2_fq = 1.0 / np.sqrt(np.mean(res1_fq ** 2, axis=1, keepdims=True) + eps)
    n2_fq   = res1_fq * rms2_fq * g2_fp16
    n2_qq   = fq_rtl_vec(n2_fq)
    g_fq    = n2_qq @ fq_rtl_vec(w_g).T
    u_fq    = n2_qq @ fq_rtl_vec(w_u).T
    h_fq    = fq_rtl_vec((g_fq / (1.0 + np.exp(-g_fq))) * u_fq)
    final_fq = res1_fq + h_fq @ fq_rtl_vec(w_d).T

    # Pack DRAM weights
    dram_buf = np.zeros(DRAM_TOTAL_BYTES, dtype=np.uint8)
    pack_proj_to_dram(dram_buf, DRAM_WQ, quantize_rtl_vec(permute_qk_weights_for_rope(w_q, H_Q, D_K)))
    pack_proj_to_dram(dram_buf, DRAM_WK, quantize_rtl_vec(permute_qk_weights_for_rope(w_k, H_KV, D_K)))
    pack_proj_to_dram(dram_buf, DRAM_WV, quantize_rtl_vec(w_v))
    pack_proj_to_dram(dram_buf, DRAM_WO, quantize_rtl_vec(w_o))
    pack_proj_to_dram(dram_buf, DRAM_WG, quantize_rtl_vec(w_g))
    pack_proj_to_dram(dram_buf, DRAM_WU, quantize_rtl_vec(w_u))
    pack_proj_to_dram(dram_buf, DRAM_WD, quantize_rtl_vec(w_d))

    dram_beats = dram_buf.reshape(-1, 64)[:, ::-1]
    with open(os.path.join(OUT_DIR, "dram_weights.hex"), "w") as f:
        f.write("\n".join(row.tobytes().hex() for row in dram_beats) + "\n")

    # TCDM Preload
    pre_addr_list, pre_data_list = [], []

    def add_fp16_rows(bank, word_off, mat_f32):
        u16 = mat_f32.astype(np.float16).view(np.uint16).reshape(-1, 4)
        for i, row in enumerate(u16):
            w64 = (int(row[3]) << 48) | (int(row[2]) << 32) | (int(row[1]) << 16) | int(row[0])
            pre_addr_list.append(tcdm_addr(word_off + i, bank))
            pre_data_list.append(w64)

    add_fp16_rows(X_BANK, 0, x_fp16_in)
    add_fp16_rows(GAMMA_BANK, GAMMA1_WORD, g1_fp16)
    add_fp16_rows(GAMMA_BANK, GAMMA2_WORD, g2_fp16)

    cos_u16 = cos_f32.astype(np.float16).view(np.uint16)
    sin_u16 = sin_f32.astype(np.float16).view(np.uint16)
    kp_per_head = D_K // 16  # 2
    for nt in range(N_TILES_MODEL):
        half_idx = (nt % kp_per_head) * 8
        for w in range(4):
            j0 = half_idx + w * 2
            j1 = half_idx + w * 2 + 1
            pk = (int(sin_u16[0, j1]) << 48) | (int(cos_u16[0, j1]) << 32) | \
                 (int(sin_u16[0, j0]) << 16) | int(cos_u16[0, j0])
            pre_addr_list.append(tcdm_addr(nt * 4 + w, ROPE_BANK))
            pre_data_list.append(pk)

    with open(os.path.join(OUT_DIR, "tcdm_preload_addr.hex"), "w") as fa, \
         open(os.path.join(OUT_DIR, "tcdm_preload_data.hex"), "w") as fd:
        fa.write("\n".join(f"{a:08x}" for a in pre_addr_list) + "\n")
        fd.write("\n".join(f"{d:018x}" for d in pre_data_list) + "\n")

    write_fp16_hex(os.path.join(OUT_DIR, "g_xnorm.hex"), xn_f32)
    write_fp16_hex(os.path.join(OUT_DIR, "g_res1.hex"), res1_f32)
    write_fp16_hex(os.path.join(OUT_DIR, "g_n2.hex"), n2_f32)
    write_fp16_hex(os.path.join(OUT_DIR, "g_final_fp16.hex"), final_f32)
    write_fp32_hex(os.path.join(OUT_DIR, "g_final_fp32.hex"), final_f32)
    write_fp16_hex(os.path.join(OUT_DIR, "g_res1_fq.hex"), res1_fq)
    write_fp16_hex(os.path.join(OUT_DIR, "g_n2_fq.hex"), n2_fq)
    write_fp16_hex(os.path.join(OUT_DIR, "g_final_fq.hex"), final_fq)

    k_all_perm = np.zeros((n_kv, H_KV, D_K), dtype=np.float32)
    k_all_perm[:n_ctx_past] = k_p_perm
    k_p_pkts = quantize_rtl_vec(k_all_perm.reshape(n_kv * H_KV, D_K)) \
        .reshape(kv_tiles, 16, H_KV, D_K // 16, 9) \
        .transpose(2, 0, 3, 1, 4).reshape(-1, 9)
    v_p_pkts = quantize_rtl_vec(v_p_t) \
        .reshape(H_KV, D_V // 16, 16, kv_tiles, 9) \
        .transpose(0, 1, 3, 2, 4).reshape(-1, 9)
    with open(os.path.join(OUT_DIR, "k_past.hex"), "w") as fk:
        fk.write("\n".join(f"{int.from_bytes(r.tobytes(), 'little'):018x}" for r in k_p_pkts) + "\n")
    with open(os.path.join(OUT_DIR, "v_past.hex"), "w") as fv:
        fv.write("\n".join(f"{int.from_bytes(r.tobytes(), 'little'):018x}" for r in v_p_pkts) + "\n")

    # Emit 15 tail past FP16 V tokens (tokens n_kv-16 .. n_kv-2) for HW Decode V-Transpose
    v_tail_u16 = v_p_raw[n_kv - 16:n_kv - 1].reshape(15, H_KV * D_V).astype(np.float16).view(np.uint16).reshape(15, (H_KV * D_V) // 4, 4)
    v_tail_words = [
        (int(row[3]) << 48) | (int(row[2]) << 32) | (int(row[1]) << 16) | int(row[0])
        for tok in v_tail_u16 for row in tok
    ]
    with open(os.path.join(OUT_DIR, "v_tail_past_fp16.hex"), "w") as fvt:
        fvt.write("\n".join(f"{w:018x}" for w in v_tail_words) + "\n")

    print(f"Done in {time.time() - t0:.2f}s! Preload={len(pre_addr_list)}, DRAM beats={len(dram_beats)}, KV pkts={len(k_p_pkts)}, V_tail_fp16={len(v_tail_words)}")


if __name__ == "__main__":
    main()
