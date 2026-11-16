#!/usr/bin/env python3
"""
generate_snax_llm_layer_data.py

Stimulus + bit-exact golden generator for a MINIMAL complete LLaMA-style
prefill layer on snax_llm_top_clean.  Every stage of the layer is exercised,
but the matrices are the smallest the hardware supports:

    M   = 16 tokens                (GEMM tile is fixed at 16 tokens)
    D   = 16 model channels        (K_BLOCKS = 1)
    H   = 1 head
    D_K = 16 -> Q_TILES = 2 (M_Q = 8), K_BLOCKS = 1
    D_V = 16 -> DV_TILES = 1
    N_KV= 16 -> KV_TILES = 1
    D_FF= 16 -> down-projection K = 1 block

Stages (hardware path):
    X (FP16) -> CPU RMSNorm -> HW readback-requant -> GEMM Q/K/V
    -> CPU RoPE -> HW requant -> HW FlashAttention (causal)
    -> HW requant -> GEMM O -> CPU residual
    -> CPU RMSNorm -> HW requant -> GEMM gate/up -> CPU SiLU*up
    -> HW requant -> GEMM down -> CPU residual -> final FP16

TCDM map (word = 10-bit, bank = 5-bit, addr = word<<9 | bank<<4):
    bank 16 : activation packets (Q_BASE = word 0)
    bank 17 : GEMM / attention FP16 outputs
    bank 18 : FP16 vector staging (4 words/vector)
    bank 19 : packet staging
    bank 24 : input X rows, gamma1/2, cos/sin RoPE tables
    bank 25 : Xn rows (RMSNorm 1)
    bank 26 : residual 1 rows
    bank 27 : final output rows
    banks 0..15 : FP4 weight packets (16 lanes = 16 output channels)
"""
import os
import csv
import math
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from generate_prefill_gemm_data import (
    encode_72b_packet,
    quantize_block,
    fp4_mult_lut_py,
    csd_scaler_py,
    shifter_py,
    sat_add32,
    drain_norm_fp16_py,
    fp4_code_fields,
    FP4_VALS,
)
from generate_fused_attn_8x8 import (
    LN2_Q16,
    sat32,
    softmax_requantizer_py,
    recip_div_py,
)

OUT_DIR = os.path.join(HERE, "..", "sim", "sim_snax_llm_layer")
HEX = os.path.join(OUT_DIR, "hex")
CSV = os.path.join(OUT_DIR, "csv")

# ------------------------------------------------------------------ shape ----
M_TOKENS = 16
D_MODEL = 16
D_FF = 16
H = 1
N_Q = 16
N_KV = 16
D_K = 16
D_V = 16
M_Q = 8
K_BLOCKS = D_K // 16
Q_TILES = N_Q // M_Q
DV_TILES = D_V // 16
KV_TILES = N_KV // 16
FF_KB = D_FF // 16
N_TILES_MODEL = D_MODEL // 16
N_TILES_GU = (2 * D_FF) // 16

# --------------------------------------------------------------- TCDM map ----
ACT_BANK = 16
OUT_BANK = 17
STG_BANK = 18
PKT_BANK = 19
X_BANK = 24
XN_BANK = 25
RES1_BANK = 26
FIN_BANK = 27

GAMMA1_WORD = 200
GAMMA2_WORD = 210
COS_WORD = 300
SIN_WORD = 364

W_WORD = {"q": 0, "k": 8, "v": 24, "o": 32, "gu": 40, "d": 48}
OUT_WORD = {"q": 0, "k": 64, "v": 128, "attn": 0, "o": 256, "gu": 320, "d": 384}

S_MIN = -(1 << 31)

# The sequencer registers the SM-drain slot for the causal mask (the score
# and its mask are sampled one cycle after cnt_dr_q advances), so causal
# attention is enabled and checked bit-exactly against the RTL.
CAUSAL = True


def tcdm_addr(word, bank):
    return ((word & 0x3FF) << 9) | ((bank & 0x1F) << 4)


def real_to_fp16_hex(v):
    """Bit-exact mirror of real_to_fp16() in the testbench (round half up)."""
    v = float(v)
    if v == 0.0:
        return 0
    sign = 1 if v < 0.0 else 0
    a = abs(v)
    expn = 0
    while a >= 2.0:
        a /= 2.0
        expn += 1
    while a < 1.0:
        a *= 2.0
        expn -= 1
    if expn + 15 <= 0:
        return sign << 15
    if expn + 15 >= 31:
        return (sign << 15) | (0x1F << 10)
    e = expn + 15
    mant = int((a - 1.0) * 1024.0 + 0.5)
    if mant >= 1024:
        mant = 0
        e += 1
    return (sign << 15) | ((e & 0x1F) << 10) | (mant & 0x3FF)


def fp16_hex_to_real(h):
    h = int(h) & 0xFFFF
    sign = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    mf = h & 0x3FF
    if e == 0 and mf == 0:
        return 0.0
    if e == 31:
        return -999999.0 if sign else 999999.0
    if e == 0:
        v = mf / 16777216.0
    else:
        v = (1.0 + mf / 1024.0) * (2.0 ** (e - 15))
    return -v if sign else v


def pack4(h0, h1, h2, h3):
    return ((int(h3) & 0xFFFF) << 48) | ((int(h2) & 0xFFFF) << 32) | \
           ((int(h1) & 0xFFFF) << 16) | (int(h0) & 0xFFFF)


def mu_of_hex(h):
    """Mirror of slq_encoder.sv: log8 code from the FP16 exponent/top mantissa."""
    h = int(h) & 0xFFFF
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    if e == 0:
        return 0
    if e == 31:
        return 247
    return (e << 3) | (m >> 7)


def fp4_mag_of_diff(d):
    """Mirror of fp4_mag_of() thresholds in slq_encoder.sv."""
    if d <= 2:
        return 7
    if d <= 6:
        return 6
    if d <= 10:
        return 5
    if d <= 14:
        return 4
    if d <= 18:
        return 3
    if d <= 24:
        return 2
    if d <= 36:
        return 1
    return 0


def quantize_block_rtl(seg_hex):
    """Bit-exact model of log_quantizer/slq_encoder.sv for one 16-value block."""
    mu = [mu_of_hex(h) for h in seg_hex]
    mx = max(mu)
    if mx == 0:
        sc = 0
    else:
        sc = mx - 140
        sc = max(-128, min(127, sc))
    codes = []
    for h, m in zip(seg_hex, mu):
        sign = (int(h) >> 15) & 1
        if m == 0 or mx == 0:
            mag = 0
        else:
            mag = fp4_mag_of_diff(mx - m)
        codes.append(0 if mag == 0 else ((sign << 3) | mag))
    return codes, sc


def real_matrix_to_hex(mat):
    return np.vectorize(real_to_fp16_hex)(mat).astype(np.uint16)


def quantize_matrix(mat_hex, n_blocks):
    rows = mat_hex.shape[0]
    codes = np.zeros((rows, n_blocks, 16), dtype=int)
    scales = np.zeros((rows, n_blocks), dtype=int)
    for r in range(rows):
        for b in range(n_blocks):
            seg = [int(h) for h in mat_hex[r, b * 16:(b + 1) * 16]]
            c, s = quantize_block_rtl(seg)
            codes[r, b, :] = c
            scales[r, b] = s
    return codes, scales


def gemm_golden(a_codes, a_scales, w_codes, w_scales, n_tiles, k_blocks):
    out = np.zeros((n_tiles, 16, 16), dtype=np.uint16)
    for gn in range(n_tiles):
        for m in range(16):
            for lane in range(16):
                n = gn * 16 + lane
                acc = 0
                for kb in range(k_blocks):
                    dot = 0
                    for el in range(16):
                        dot += fp4_mult_lut_py(int(a_codes[m, kb, el]),
                                               int(w_codes[n, kb, el]))
                    res_scale = int(a_scales[m, kb]) + int(w_scales[n, kb])
                    q_exp = res_scale // 8
                    r_exp = res_scale % 8
                    aligned = shifter_py(csd_scaler_py(dot, r_exp), q_exp, 16)
                    acc = sat_add32(acc, int(aligned))
                out[gn, m, lane] = drain_norm_fp16_py(int(acc), frac_bits=16)
    return out


def dequant_codes(codes, scale):
    """Fake-quant view: E2M1 code * 2^(scale/8), exact real arithmetic."""
    s = 2.0 ** (int(scale) / 8.0)
    vals = []
    for c in codes:
        c = int(c)
        mag = FP4_VALS[c & 0x7]
        vals.append((-mag if (c & 0x8) else mag) * s)
    return vals


def gemm_exact(a_codes, a_scales, w_codes, w_scales, n_tiles, k_blocks):
    """Exact real dot product of the same quantized operands (no CSD/shift)."""
    out = np.zeros((n_tiles, 16, 16), dtype=float)
    for gn in range(n_tiles):
        for m in range(16):
            for lane in range(16):
                n = gn * 16 + lane
                acc = 0.0
                for kb in range(k_blocks):
                    av = dequant_codes(a_codes[m, kb], a_scales[m, kb])
                    wv = dequant_codes(w_codes[n, kb], w_scales[n, kb])
                    for el in range(16):
                        acc += av[el] * wv[el]
                out[gn, m, lane] = acc
    return out


def cmp_metrics(name, hw_hex, exact):
    hw = np.array([fp16_hex_to_real(h) for h in np.asarray(hw_hex).flatten()])
    ex = np.asarray(exact, dtype=float).flatten()
    d = hw - ex
    ad = np.abs(d)
    rmse = float(np.sqrt(np.mean(d * d)))
    mask = np.abs(ex) > 1e-9
    if np.any(mask):
        rel = ad[mask] / np.abs(ex[mask])
        max_rel = float(np.max(rel)) * 100.0
        mean_rel = float(np.mean(rel)) * 100.0
    else:
        max_rel = 0.0
        mean_rel = 0.0
    max_ex = float(np.max(np.abs(ex))) if ex.size else 0.0
    rounded = np.array([real_to_fp16_hex(v) for v in ex], dtype=np.uint16)
    hw_u16 = np.asarray(hw_hex, dtype=np.uint16).flatten()
    match_pct = 100.0 * float(np.mean(rounded == hw_u16))
    return {
        "stage": name,
        "values": int(ex.size),
        "max_abs": float(np.max(ad)) if ad.size else 0.0,
        "mean_abs": float(np.mean(ad)) if ad.size else 0.0,
        "rmse": rmse,
        "max_rel_pct": max_rel,
        "mean_rel_pct": mean_rel,
        "max_abs_over_max": (float(np.max(ad)) / max_ex * 100.0) if max_ex else 0.0,
        "fp16_round_match_pct": match_pct,
    }


def write_hex(path, values, width):
    with open(path, "w") as f:
        for v in values:
            f.write(f"{int(v):0{width}X}\n")


def main():
    np.random.seed(2027)
    os.makedirs(HEX, exist_ok=True)
    os.makedirs(CSV, exist_ok=True)

    print("=== SNAX LLM MINIMAL full-layer (prefill) stimulus & golden ===")
    print(f"tokens={M_TOKENS} d_model={D_MODEL} d_ff={D_FF} heads={H} "
          f"d_k={D_K} d_v={D_V} n_kv={N_KV} n_q={N_Q}")
    metrics = []
    full_metrics = []

    # ------------------------------------------------------------ input ----
    x_real = np.random.normal(0.0, 1.0, (M_TOKENS, D_MODEL))
    x_hex = real_matrix_to_hex(x_real)
    gamma1_hex = real_matrix_to_hex(np.random.normal(0.0, 0.5, (1, D_MODEL)))[0]
    gamma2_hex = real_matrix_to_hex(np.random.normal(0.0, 0.5, (1, D_MODEL)))[0]

    # ---------------------------------------------------------- weights ----
    weights = {
        "q": np.random.normal(0.0, 0.6, (D_MODEL, D_MODEL)),
        "k": np.random.normal(0.0, 0.6, (D_MODEL, D_MODEL)),
        "v": np.random.normal(0.0, 0.6, (D_MODEL, D_MODEL)),
        "o": np.random.normal(0.0, 0.6, (D_MODEL, D_MODEL)),
        "gu": np.random.normal(0.0, 0.6, (2 * D_FF, D_MODEL)),
        "d": np.random.normal(0.0, 0.6, (D_MODEL, D_FF)),
    }
    w_codes, w_scales, w_packets = {}, {}, {}
    for name, mat in weights.items():
        n_ch = mat.shape[0]
        kb_total = mat.shape[1] // 16
        mat_hex = real_matrix_to_hex(mat)
        codes = np.zeros((n_ch, kb_total, 16), dtype=int)
        scales = np.zeros((n_ch, kb_total), dtype=int)
        packets = np.zeros((n_ch, kb_total), dtype=object)
        for n in range(n_ch):
            for b in range(kb_total):
                c, s = quantize_block_rtl([int(h) for h in mat_hex[n, b * 16:(b + 1) * 16]])
                codes[n, b, :] = c
                scales[n, b] = s
                packets[n, b] = encode_72b_packet(c, s)
        w_codes[name] = codes
        w_scales[name] = scales
        w_packets[name] = packets

    # ------------------------------------------------------ RMSNorm (1) ----
    xnorm_hex = np.zeros((M_TOKENS, D_MODEL), dtype=np.uint16)
    for m in range(M_TOKENS):
        xs = [fp16_hex_to_real(h) for h in x_hex[m]]
        gs = [fp16_hex_to_real(h) for h in gamma1_hex]
        acc = 0.0
        for v in xs:
            acc += v * v
        inv = 1.0 / math.sqrt(acc / D_MODEL)
        for d in range(D_MODEL):
            xnorm_hex[m, d] = real_to_fp16_hex((xs[d] * inv) * gs[d])

    a_codes, a_scales = quantize_matrix(xnorm_hex, K_BLOCKS)

    # ------------------------------------------------------- GEMM Q/K/V ----
    q_hex = gemm_golden(a_codes, a_scales, w_codes["q"], w_scales["q"],
                        N_TILES_MODEL, K_BLOCKS)
    k_hex = gemm_golden(a_codes, a_scales, w_codes["k"], w_scales["k"],
                        N_TILES_MODEL, K_BLOCKS)
    v_hex = gemm_golden(a_codes, a_scales, w_codes["v"], w_scales["v"],
                        N_TILES_MODEL, K_BLOCKS)

    metrics.append(cmp_metrics("GEMM Q", q_hex, gemm_exact(
        a_codes, a_scales, w_codes["q"], w_scales["q"], N_TILES_MODEL, K_BLOCKS)))
    metrics.append(cmp_metrics("GEMM K", k_hex, gemm_exact(
        a_codes, a_scales, w_codes["k"], w_scales["k"], N_TILES_MODEL, K_BLOCKS)))
    metrics.append(cmp_metrics("GEMM V", v_hex, gemm_exact(
        a_codes, a_scales, w_codes["v"], w_scales["v"], N_TILES_MODEL, K_BLOCKS)))

    # ------------------------------------------------------------- RoPE ----
    half = D_K // 2
    cos_hex = np.zeros((M_TOKENS, half), dtype=np.uint16)
    sin_hex = np.zeros((M_TOKENS, half), dtype=np.uint16)
    for pos in range(M_TOKENS):
        for j in range(half):
            theta = 10000.0 ** (-2.0 * j / D_K)
            angle = pos * theta
            cos_hex[pos, j] = real_to_fp16_hex(math.cos(angle))
            sin_hex[pos, j] = real_to_fp16_hex(math.sin(angle))

    def rope_apply(vec_hex):
        out = vec_hex.copy()
        for pos in range(M_TOKENS):
            for j in range(half):
                c = fp16_hex_to_real(cos_hex[pos, j])
                s = fp16_hex_to_real(sin_hex[pos, j])
                v0 = (j // 16, j % 16)
                v1 = ((j + half) // 16, (j + half) % 16)
                x0 = fp16_hex_to_real(vec_hex[v0[0], pos, v0[1]])
                x1 = fp16_hex_to_real(vec_hex[v1[0], pos, v1[1]])
                out[v0[0], pos, v0[1]] = real_to_fp16_hex(x0 * c - x1 * s)
                out[v1[0], pos, v1[1]] = real_to_fp16_hex(x0 * s + x1 * c)
        return out

    q_rope = rope_apply(q_hex)
    k_rope = rope_apply(k_hex)

    # ------------------------------------------------- quantize Q/K/V ------
    q_codes = np.zeros((N_Q, K_BLOCKS, 16), dtype=int)
    q_scales = np.zeros((N_Q, K_BLOCKS), dtype=int)
    q_packets = np.zeros((N_Q, K_BLOCKS), dtype=object)
    for q in range(N_Q):
        for kb in range(K_BLOCKS):
            seg = [int(h) for h in q_rope[kb, q, :]]
            c, s = quantize_block_rtl(seg)
            q_codes[q, kb, :] = c
            q_scales[q, kb] = s
            q_packets[q, kb] = encode_72b_packet(c, s)

    k_codes = np.zeros((N_KV, K_BLOCKS, 16), dtype=int)
    k_scales = np.zeros((N_KV, K_BLOCKS), dtype=int)
    k_packets = np.zeros((K_BLOCKS, 16), dtype=object)
    for key in range(N_KV):
        for kb in range(K_BLOCKS):
            seg = [int(h) for h in k_rope[kb, key, :]]
            c, s = quantize_block_rtl(seg)
            k_codes[key, kb, :] = c
            k_scales[key, kb] = s
    for kb in range(K_BLOCKS):
        for lane in range(16):
            k_packets[kb, lane] = encode_72b_packet(k_codes[lane, kb, :],
                                                    k_scales[lane, kb])

    v_codes = np.zeros((DV_TILES, 16, 16), dtype=int)
    v_scales = np.zeros((DV_TILES, 16), dtype=int)
    v_packets = np.zeros((DV_TILES, 16), dtype=object)
    for dvi in range(DV_TILES):
        for lane in range(16):
            seg = [int(v_hex[dvi, key, lane]) for key in range(16)]
            c, s = quantize_block_rtl(seg)
            v_codes[dvi, lane, :] = c
            v_scales[dvi, lane] = s
            v_packets[dvi, lane] = encode_72b_packet(c, s)

    # ---------------------------------------------------- FlashAttention ---
    attn_hex = np.zeros((N_Q, D_V), dtype=np.uint16)
    attn_l = np.zeros((N_Q, DV_TILES), dtype=np.int64)
    p_codes_all = np.zeros((N_Q, 16), dtype=int)
    p_scale_all = np.zeros(N_Q, dtype=int)

    for qi in range(Q_TILES):
        p_codes_q = []
        p_scale_q = []
        l_q = []
        for m in range(M_Q):
            q_glob = qi * M_Q + m
            score = []
            for lane in range(16):
                key = lane
                acc = 0
                for kb in range(K_BLOCKS):
                    dot = 0
                    for el in range(16):
                        dot += fp4_mult_lut_py(int(q_codes[q_glob, kb, el]),
                                               int(k_codes[key, kb, el]))
                    res_scale = int(q_scales[q_glob, kb]) + int(k_scales[key, kb])
                    q_exp = res_scale // 8
                    r_exp = res_scale % 8
                    aligned = shifter_py(csd_scaler_py(dot, r_exp), q_exp, 16)
                    acc = sat_add32(acc, int(aligned))
                nat = sat32((int(acc) * LN2_Q16) >> 16)
                visible = (key <= q_glob) if CAUSAL else True
                score.append(nat if visible else S_MIN)
            codes, p_blk, l_contrib, _, _, _, _ = \
                softmax_requantizer_py(score, S_MIN, True)
            p_codes_q.append(codes)
            p_scale_q.append(p_blk)
            l_q.append(l_contrib)
            p_codes_all[q_glob, :] = codes
            p_scale_all[q_glob] = p_blk

        for dvi in range(DV_TILES):
            acc_o = np.zeros((M_Q, 16), dtype=np.int64)
            for m in range(M_Q):
                p_codes = p_codes_q[m]
                p_scale = p_scale_q[m]
                for lane in range(16):
                    dot = 0
                    for key in range(16):
                        dot += fp4_mult_lut_py(int(p_codes[key]),
                                               int(v_codes[dvi, lane, key]))
                    res_scale = int(p_scale) + int(v_scales[dvi, lane])
                    q_exp = res_scale // 8
                    r_exp = res_scale % 8
                    aligned = shifter_py(csd_scaler_py(dot, r_exp), q_exp, 16)
                    acc_o[m, lane] = sat_add32(int(acc_o[m, lane]), int(aligned))
            for m in range(M_Q):
                q_glob = qi * M_Q + m
                lt = int(l_q[m])
                attn_l[q_glob, dvi] = lt
                for lane in range(16):
                    norm = recip_div_py(int(acc_o[m, lane]), lt)
                    attn_hex[q_glob, dvi * 16 + lane] = \
                        drain_norm_fp16_py(norm, frac_bits=16)

    # Fake-quant attention: exact P*V / exact sum(P) from the same P/V codes.
    attn_exact = np.zeros((N_Q, D_V), dtype=float)
    for q in range(N_Q):
        pv = dequant_codes(p_codes_all[q], p_scale_all[q])
        l_ex = sum(pv)
        for dvi in range(DV_TILES):
            for lane in range(16):
                vv = dequant_codes(v_codes[dvi, lane], v_scales[dvi, lane])
                acc = sum(pv[key] * vv[key] for key in range(16))
                attn_exact[q, dvi * 16 + lane] = acc / l_ex if l_ex != 0.0 else 0.0
    metrics.append(cmp_metrics("FlashAttention (exact P*V/l)", attn_hex, attn_exact))

    # --------------------------------------- quantization of attention out --
    a_o_codes, a_o_scales = quantize_matrix(attn_hex, DV_TILES)
    attn_packets = [encode_72b_packet(a_o_codes[q, 0, :], a_o_scales[q, 0])
                    for q in range(N_Q)]
    o_hex = gemm_golden(a_o_codes, a_o_scales, w_codes["o"], w_scales["o"],
                        N_TILES_MODEL, DV_TILES)
    metrics.append(cmp_metrics("GEMM O", o_hex, gemm_exact(
        a_o_codes, a_o_scales, w_codes["o"], w_scales["o"],
        N_TILES_MODEL, DV_TILES)))

    # ------------------------------------------------------ residual (1) ---
    res1_hex = np.zeros((M_TOKENS, D_MODEL), dtype=np.uint16)
    for q in range(M_TOKENS):
        for d in range(D_MODEL):
            ov = fp16_hex_to_real(o_hex[d // 16, q, d % 16])
            xv = fp16_hex_to_real(x_hex[q, d])
            res1_hex[q, d] = real_to_fp16_hex(ov + xv)

    # ------------------------------------------------------ RMSNorm (2) ----
    n2_hex = np.zeros((M_TOKENS, D_MODEL), dtype=np.uint16)
    for m in range(M_TOKENS):
        xs = [fp16_hex_to_real(h) for h in res1_hex[m]]
        gs = [fp16_hex_to_real(h) for h in gamma2_hex]
        acc = 0.0
        for v in xs:
            acc += v * v
        inv = 1.0 / math.sqrt(acc / D_MODEL)
        for d in range(D_MODEL):
            n2_hex[m, d] = real_to_fp16_hex((xs[d] * inv) * gs[d])

    a_ff_codes, a_ff_scales = quantize_matrix(n2_hex, K_BLOCKS)
    gu_hex = gemm_golden(a_ff_codes, a_ff_scales, w_codes["gu"], w_scales["gu"],
                         N_TILES_GU, K_BLOCKS)
    metrics.append(cmp_metrics("GEMM gate/up", gu_hex, gemm_exact(
        a_ff_codes, a_ff_scales, w_codes["gu"], w_scales["gu"],
        N_TILES_GU, K_BLOCKS)))

    # --------------------------------------------------- SiLU(gate) * up ---
    h_hex = np.zeros((M_TOKENS, D_FF), dtype=np.uint16)
    for q in range(M_TOKENS):
        for c in range(D_FF):
            g = fp16_hex_to_real(gu_hex[0, q, c])
            u = fp16_hex_to_real(gu_hex[1, q, c])
            silu = g / (1.0 + math.exp(-g))
            h_hex[q, c] = real_to_fp16_hex(silu * u)

    a_d_codes, a_d_scales = quantize_matrix(h_hex, FF_KB)
    down_hex = gemm_golden(a_d_codes, a_d_scales, w_codes["d"], w_scales["d"],
                           N_TILES_MODEL, FF_KB)
    metrics.append(cmp_metrics("GEMM down", down_hex, gemm_exact(
        a_d_codes, a_d_scales, w_codes["d"], w_scales["d"],
        N_TILES_MODEL, FF_KB)))

    # ------------------------------------------------------ residual (2) ---
    final_hex = np.zeros((M_TOKENS, D_MODEL), dtype=np.uint16)
    for q in range(M_TOKENS):
        for d in range(D_MODEL):
            dv = fp16_hex_to_real(down_hex[d // 16, q, d % 16])
            rv = fp16_hex_to_real(res1_hex[q, d])
            final_hex[q, d] = real_to_fp16_hex(dv + rv)

    # =========================================================================
    # FAKE-QUANT LAYER: same FP4 quantizers, but every GEMM and the attention
    # accumulation use exact real arithmetic (no CSD / shift / saturation).
    # =========================================================================
    def rope_real(vec):
        out = vec.copy()
        for pos in range(M_TOKENS):
            for j in range(half):
                c = fp16_hex_to_real(cos_hex[pos, j])
                s = fp16_hex_to_real(sin_hex[pos, j])
                v0 = (j // 16, j % 16)
                v1 = ((j + half) // 16, (j + half) % 16)
                x0 = vec[v0[0], pos, v0[1]]
                x1 = vec[v1[0], pos, v1[1]]
                out[v0[0], pos, v0[1]] = x0 * c - x1 * s
                out[v1[0], pos, v1[1]] = x0 * s + x1 * c
        return out

    q_ex = gemm_exact(a_codes, a_scales, w_codes["q"], w_scales["q"],
                      N_TILES_MODEL, K_BLOCKS)
    k_ex = gemm_exact(a_codes, a_scales, w_codes["k"], w_scales["k"],
                      N_TILES_MODEL, K_BLOCKS)
    v_ex = gemm_exact(a_codes, a_scales, w_codes["v"], w_scales["v"],
                      N_TILES_MODEL, K_BLOCKS)
    q_hex_f = real_matrix_to_hex(rope_real(q_ex))
    k_hex_f = real_matrix_to_hex(rope_real(k_ex))
    v_hex_f = real_matrix_to_hex(v_ex)

    qc_f = np.zeros((N_Q, K_BLOCKS, 16), dtype=int)
    qs_f = np.zeros((N_Q, K_BLOCKS), dtype=int)
    for q in range(N_Q):
        for kb in range(K_BLOCKS):
            c, s = quantize_block_rtl([int(h) for h in q_hex_f[kb, q, :]])
            qc_f[q, kb, :] = c
            qs_f[q, kb] = s
    kc_f = np.zeros((N_KV, K_BLOCKS, 16), dtype=int)
    ks_f = np.zeros((N_KV, K_BLOCKS), dtype=int)
    for key in range(N_KV):
        for kb in range(K_BLOCKS):
            c, s = quantize_block_rtl([int(h) for h in k_hex_f[kb, key, :]])
            kc_f[key, kb, :] = c
            ks_f[key, kb] = s
    vc_f = np.zeros((DV_TILES, 16, 16), dtype=int)
    vs_f = np.zeros((DV_TILES, 16), dtype=int)
    for dvi in range(DV_TILES):
        for lane in range(16):
            seg = [int(v_hex_f[dvi, key, lane]) for key in range(16)]
            c, s = quantize_block_rtl(seg)
            vc_f[dvi, lane, :] = c
            vs_f[dvi, lane] = s

    attn_f = np.zeros((N_Q, D_V), dtype=float)
    for qi in range(Q_TILES):
        for m in range(M_Q):
            q = qi * M_Q + m
            nat_scores = []
            for key in range(N_KV):
                acc = 0.0
                for kb in range(K_BLOCKS):
                    av = dequant_codes(qc_f[q, kb], qs_f[q, kb])
                    kv = dequant_codes(kc_f[key, kb], ks_f[key, kb])
                    acc += sum(av[e] * kv[e] for e in range(16))
                s16 = sat32(int(round(acc * 65536.0)))
                visible = (key <= q) if CAUSAL else True
                nat_scores.append(sat32((s16 * LN2_Q16) >> 16) if visible else S_MIN)
            pc, ps, _, _, _, _, _ = softmax_requantizer_py(nat_scores, S_MIN, True)
            pdeq = dequant_codes(pc, ps)
            l_ex = sum(pdeq)
            for dvi in range(DV_TILES):
                for lane in range(16):
                    vv = dequant_codes(vc_f[dvi, lane], vs_f[dvi, lane])
                    o = sum(pdeq[key] * vv[key] for key in range(16))
                    attn_f[q, dvi * 16 + lane] = o / l_ex if l_ex != 0.0 else 0.0

    attn_hex_f = real_matrix_to_hex(attn_f)
    a_o_codes_f, a_o_scales_f = quantize_matrix(attn_hex_f, DV_TILES)
    o_ex = gemm_exact(a_o_codes_f, a_o_scales_f, w_codes["o"], w_scales["o"],
                      N_TILES_MODEL, DV_TILES)

    res1_f = np.zeros((M_TOKENS, D_MODEL), dtype=np.uint16)
    for q in range(M_TOKENS):
        for d in range(D_MODEL):
            ov = o_ex[d // 16, q, d % 16]
            xv = fp16_hex_to_real(x_hex[q, d])
            res1_f[q, d] = real_to_fp16_hex(ov + xv)

    n2_f = np.zeros((M_TOKENS, D_MODEL), dtype=np.uint16)
    for m in range(M_TOKENS):
        xs = [fp16_hex_to_real(h) for h in res1_f[m]]
        gs = [fp16_hex_to_real(h) for h in gamma2_hex]
        acc = 0.0
        for v in xs:
            acc += v * v
        inv = 1.0 / math.sqrt(acc / D_MODEL)
        for d in range(D_MODEL):
            n2_f[m, d] = real_to_fp16_hex((xs[d] * inv) * gs[d])

    a_ff_f, a_ff_s_f = quantize_matrix(n2_f, K_BLOCKS)
    gu_ex = gemm_exact(a_ff_f, a_ff_s_f, w_codes["gu"], w_scales["gu"],
                       N_TILES_GU, K_BLOCKS)
    gu_hex_f = real_matrix_to_hex(gu_ex)
    h_f = np.zeros((M_TOKENS, D_FF), dtype=np.uint16)
    for q in range(M_TOKENS):
        for c in range(D_FF):
            g = fp16_hex_to_real(gu_hex_f[0, q, c])
            u = fp16_hex_to_real(gu_hex_f[1, q, c])
            h_f[q, c] = real_to_fp16_hex((g / (1.0 + math.exp(-g))) * u)

    a_d_f, a_d_s_f = quantize_matrix(h_f, FF_KB)
    down_ex = gemm_exact(a_d_f, a_d_s_f, w_codes["d"], w_scales["d"],
                         N_TILES_MODEL, FF_KB)

    final_exact = np.zeros((M_TOKENS, D_MODEL), dtype=float)
    final_f_hex = np.zeros((M_TOKENS, D_MODEL), dtype=np.uint16)
    for q in range(M_TOKENS):
        for d in range(D_MODEL):
            dv = down_ex[d // 16, q, d % 16]
            rv = fp16_hex_to_real(res1_f[q, d])
            final_exact[q, d] = dv + rv
            final_f_hex[q, d] = real_to_fp16_hex(dv + rv)

    full_metrics.append(cmp_metrics("fake Q vs HW Q", q_rope,
                                    np.vectorize(fp16_hex_to_real)(q_hex_f)))
    full_metrics.append(cmp_metrics("fake K vs HW K", k_rope,
                                    np.vectorize(fp16_hex_to_real)(k_hex_f)))
    full_metrics.append(cmp_metrics("fake V vs HW V", v_hex,
                                    np.vectorize(fp16_hex_to_real)(v_hex_f)))
    full_metrics.append(cmp_metrics("fake-quant attention vs HW", attn_hex, attn_f))
    full_metrics.append(cmp_metrics("fake-quant O vs HW", o_hex, o_ex))
    res1_f_real = np.vectorize(fp16_hex_to_real)(res1_f)
    full_metrics.append(cmp_metrics("fake-quant res1 vs HW", res1_hex, res1_f_real))
    full_metrics.append(cmp_metrics("fake-quant FINAL vs HW", final_hex, final_exact))

    # --------------------------------------------------------- HEX output --
    write_hex(os.path.join(HEX, "layer_x_fp16.hex"), x_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_xnorm_fp16.hex"), xnorm_hex.flatten(), 4)
    # FP16 stages are written in GEMM vector order: gn, token, lane.
    write_hex(os.path.join(HEX, "layer_q_fp16.hex"), q_rope.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_k_fp16.hex"), k_rope.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_v_fp16.hex"), v_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_q_packets.hex"),
              [q_packets[q, kb] for q in range(N_Q) for kb in range(K_BLOCKS)], 18)
    write_hex(os.path.join(HEX, "layer_k_packets.hex"),
              [k_packets[kb, lane] for kb in range(K_BLOCKS) for lane in range(16)], 18)
    write_hex(os.path.join(HEX, "layer_v_packets.hex"),
              [v_packets[dvi, lane] for dvi in range(DV_TILES) for lane in range(16)], 18)
    write_hex(os.path.join(HEX, "layer_attn_fp16.hex"), attn_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_attn_packets.hex"), attn_packets, 18)
    write_hex(os.path.join(HEX, "layer_o_fp16.hex"), o_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_res1_fp16.hex"), res1_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_norm2_fp16.hex"), n2_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_gu_fp16.hex"), gu_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_h_fp16.hex"), h_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_down_fp16.hex"), down_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_final_fp16.hex"), final_hex.flatten(), 4)

    # ------------------------------------------------------ TCDM preload ---
    preload = []
    row_words = D_MODEL // 4          # 4 words per token row
    gam_words = D_MODEL // 4          # 4 words per gamma vector
    rope_words = half // 4            # 2 words per cos/sin row

    for m in range(M_TOKENS):
        for w in range(row_words):
            preload.append((tcdm_addr(m * row_words + w, X_BANK),
                            pack4(*x_hex[m, 4 * w:4 * w + 4]),
                            f"X token={m} word={w}"))

    for w in range(gam_words):
        preload.append((tcdm_addr(GAMMA1_WORD + w, X_BANK),
                        pack4(*gamma1_hex[4 * w:4 * w + 4]), f"gamma1 word={w}"))
        preload.append((tcdm_addr(GAMMA2_WORD + w, X_BANK),
                        pack4(*gamma2_hex[4 * w:4 * w + 4]), f"gamma2 word={w}"))

    for pos in range(M_TOKENS):
        for w in range(rope_words):
            preload.append((tcdm_addr(COS_WORD + pos * rope_words + w, X_BANK),
                            pack4(*cos_hex[pos, 4 * w:4 * w + 4]),
                            f"cos pos={pos} word={w}"))
            preload.append((tcdm_addr(SIN_WORD + pos * rope_words + w, X_BANK),
                            pack4(*sin_hex[pos, 4 * w:4 * w + 4]),
                            f"sin pos={pos} word={w}"))

    for name, mat in weights.items():
        n_ch, kdim = mat.shape
        kb_total = kdim // 16
        base = W_WORD[name]
        for n in range(n_ch):
            gn, lane = n // 16, n % 16
            for kb in range(kb_total):
                word = base + gn * kb_total + kb
                preload.append((tcdm_addr(word, lane),
                                int(w_packets[name][n, kb]),
                                f"W{name} n={n} kb={kb}"))

    with open(os.path.join(HEX, "tcdm_preload_addr.hex"), "w") as fa, \
         open(os.path.join(HEX, "tcdm_preload_data.hex"), "w") as fd:
        for addr, data, _ in preload:
            fa.write(f"{addr:08X}\n")
            fd.write(f"{data:018X}\n")

    # ------------------------------------------------------------ CSV ------
    with open(os.path.join(CSV, "layer_stages.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token", "channel",
                    "x_hex", "x_real", "xnorm_hex", "xnorm_real",
                    "q_hex", "q_real", "k_hex", "k_real", "v_hex", "v_real",
                    "attn_hex", "attn_real",
                    "o_hex", "o_real", "res1_hex", "res1_real",
                    "norm2_hex", "norm2_real",
                    "gate_hex", "gate_real", "up_hex", "up_real",
                    "h_hex", "h_real", "down_hex", "down_real",
                    "final_hex", "final_real"])
        for q in range(M_TOKENS):
            for d in range(D_MODEL):
                gn, lane = d // 16, d % 16
                w.writerow([
                    q, d,
                    f"0x{x_hex[q, d]:04X}", f"{fp16_hex_to_real(x_hex[q, d]):.9f}",
                    f"0x{xnorm_hex[q, d]:04X}", f"{fp16_hex_to_real(xnorm_hex[q, d]):.9f}",
                    f"0x{q_rope[gn, q, lane]:04X}",
                    f"{fp16_hex_to_real(q_rope[gn, q, lane]):.9f}",
                    f"0x{k_rope[gn, q, lane]:04X}",
                    f"{fp16_hex_to_real(k_rope[gn, q, lane]):.9f}",
                    f"0x{v_hex[gn, q, lane]:04X}",
                    f"{fp16_hex_to_real(v_hex[gn, q, lane]):.9f}",
                    f"0x{attn_hex[q, d]:04X}",
                    f"{fp16_hex_to_real(attn_hex[q, d]):.9f}",
                    f"0x{o_hex[gn, q, lane]:04X}",
                    f"{fp16_hex_to_real(o_hex[gn, q, lane]):.9f}",
                    f"0x{res1_hex[q, d]:04X}",
                    f"{fp16_hex_to_real(res1_hex[q, d]):.9f}",
                    f"0x{n2_hex[q, d]:04X}",
                    f"{fp16_hex_to_real(n2_hex[q, d]):.9f}",
                    f"0x{gu_hex[0, q, lane]:04X}",
                    f"{fp16_hex_to_real(gu_hex[0, q, lane]):.9f}",
                    f"0x{gu_hex[1, q, lane]:04X}",
                    f"{fp16_hex_to_real(gu_hex[1, q, lane]):.9f}",
                    f"0x{h_hex[q, d]:04X}",
                    f"{fp16_hex_to_real(h_hex[q, d]):.9f}",
                    f"0x{down_hex[gn, q, lane]:04X}",
                    f"{fp16_hex_to_real(down_hex[gn, q, lane]):.9f}",
                    f"0x{final_hex[q, d]:04X}",
                    f"{fp16_hex_to_real(final_hex[q, d]):.9f}"])

    with open(os.path.join(CSV, "weights_packets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["matrix", "out_channel", "gn", "lane", "kb",
                  "word", "bank", "scale_hex", "scale_value", "packet_hex"]
        for i in range(16):
            header += [f"K{i}_code", f"K{i}_real"]
        w.writerow(header)
        for name, mat in weights.items():
            n_ch, kdim = mat.shape
            kb_total = kdim // 16
            for n in range(n_ch):
                gn, lane = n // 16, n % 16
                for kb in range(kb_total):
                    sc = int(w_scales[name][n, kb])
                    scale = 2.0 ** (sc / 8.0)
                    nibbles = [int(c) & 0xF for c in w_codes[name][n, kb]]
                    row = [f"W{name}", n, gn, lane, kb,
                           W_WORD[name] + gn * kb_total + kb, lane,
                           f"0x{sc & 0xFF:02X}", f"{scale:.9f}",
                           f"0x{int(w_packets[name][n, kb]):018X}"]
                    for nib in nibbles:
                        _, _, _, nominal = fp4_code_fields(nib)
                        row += [f"0x{nib:X}", f"{nominal * scale:.9f}"]
                    w.writerow(row)

    with open(os.path.join(CSV, "activation_packets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "index", "lane", "scale_hex", "scale_value",
                    "packet_hex", "codes"])
        for q in range(N_Q):
            for kb in range(K_BLOCKS):
                sc = int(q_scales[q, kb])
                w.writerow(["Q", q, "-", f"0x{sc & 0xFF:02X}",
                            f"{2.0 ** (sc / 8.0):.9f}",
                            f"0x{int(q_packets[q, kb]):018X}",
                            "".join(f"{int(c):X}" for c in q_codes[q, kb])])
        for kb in range(K_BLOCKS):
            for lane in range(16):
                sc = int(k_scales[lane, kb])
                w.writerow(["K", kb, lane, f"0x{sc & 0xFF:02X}",
                            f"{2.0 ** (sc / 8.0):.9f}",
                            f"0x{int(k_packets[kb, lane]):018X}",
                            "".join(f"{int(c):X}" for c in k_codes[lane, kb])])
        for dvi in range(DV_TILES):
            for lane in range(16):
                sc = int(v_scales[dvi, lane])
                w.writerow(["V", dvi, lane, f"0x{sc & 0xFF:02X}",
                            f"{2.0 ** (sc / 8.0):.9f}",
                            f"0x{int(v_packets[dvi, lane]):018X}",
                            "".join(f"{int(c):X}" for c in v_codes[dvi, lane])])

    with open(os.path.join(CSV, "tcdm_preload.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "kind", "addr_hex", "bank", "word", "data_hex", "detail"])
        for i, (addr, data, detail) in enumerate(preload):
            w.writerow([i, detail.split()[0], f"0x{addr:08X}",
                        (addr >> 4) & 0x1F, (addr >> 9) & 0x3FF,
                        f"0x{data:018X}", detail])

    with open(os.path.join(CSV, "fakequant_vs_hw.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "values", "max_abs", "mean_abs", "rmse",
                    "max_rel_pct", "mean_rel_pct",
                    "max_abs_over_max_pct", "fp16_round_match_pct"])
        for m in metrics:
            w.writerow([m["stage"], m["values"], f"{m['max_abs']:.6f}",
                        f"{m['mean_abs']:.6f}", f"{m['rmse']:.6f}",
                        f"{m['max_rel_pct']:.4f}", f"{m['mean_rel_pct']:.4f}",
                        f"{m['max_abs_over_max']:.4f}",
                        f"{m['fp16_round_match_pct']:.2f}"])

    with open(os.path.join(CSV, "fakequant_full_layer.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "values", "max_abs", "mean_abs", "rmse",
                    "max_rel_pct", "mean_rel_pct",
                    "max_abs_over_max_pct", "fp16_round_match_pct"])
        for m in full_metrics:
            w.writerow([m["stage"], m["values"], f"{m['max_abs']:.6f}",
                        f"{m['mean_abs']:.6f}", f"{m['rmse']:.6f}",
                        f"{m['max_rel_pct']:.4f}", f"{m['mean_rel_pct']:.4f}",
                        f"{m['max_abs_over_max']:.4f}",
                        f"{m['fp16_round_match_pct']:.2f}"])

    print("\n[FAKE-QUANT vs HW datapath]  (exact real arithmetic on the same FP4 codes)")
    print(f"{'stage':34s} {'max_abs':>9s} {'mean_abs':>9s} {'rmse':>9s} "
          f"{'max_rel%':>9s} {'fp16_rnd%':>9s}")
    for m in metrics:
        print(f"{m['stage']:34s} {m['max_abs']:9.5f} {m['mean_abs']:9.5f} "
              f"{m['rmse']:9.5f} {m['max_rel_pct']:9.3f} "
              f"{m['fp16_round_match_pct']:9.2f}")

    print("\n[FAKE-QUANT FULL LAYER vs HW]  (same exp-free softmax, exact arithmetic)")
    for m in full_metrics:
        print(f"{m['stage']:34s} {m['max_abs']:9.5f} {m['mean_abs']:9.5f} "
              f"{m['rmse']:9.5f} {m['max_rel_pct']:9.3f} "
              f"{m['fp16_round_match_pct']:9.2f}")

    # Sanity: exact scores from the HW Q/K codes should reproduce the HW P codes.
    p_match = 0
    for q in range(N_Q):
        nat = []
        for key in range(N_KV):
            acc = 0.0
            for kb in range(K_BLOCKS):
                av = dequant_codes(q_codes[q, kb], q_scales[q, kb])
                kv = dequant_codes(k_codes[key, kb], k_scales[key, kb])
                acc += sum(av[e] * kv[e] for e in range(16))
            s16 = sat32(int(round(acc * 65536.0)))
            nat.append(sat32((s16 * LN2_Q16) >> 16))
        pc, _, _, _, _, _, _ = softmax_requantizer_py(nat, S_MIN, True)
        p_match += sum(1 for a, b in zip(pc, p_codes_all[q]) if a == b)

    p_match_f = 0
    for q in range(N_Q):
        nat = []
        for key in range(N_KV):
            acc = 0.0
            for kb in range(K_BLOCKS):
                av = dequant_codes(qc_f[q, kb], qs_f[q, kb])
                kv = dequant_codes(kc_f[key, kb], ks_f[key, kb])
                acc += sum(av[e] * kv[e] for e in range(16))
            s16 = sat32(int(round(acc * 65536.0)))
            nat.append(sat32((s16 * LN2_Q16) >> 16))
        pc, _, _, _, _, _, _ = softmax_requantizer_py(nat, S_MIN, True)
        p_match_f += sum(1 for a, b in zip(pc, p_codes_all[q]) if a == b)

    print(f"  [dbg] exact scores -> P codes match HW: {p_match}/{N_Q*16} "
          f"(fake Q/K: {p_match_f}/{N_Q*16})")

    print(f"[SUCCESS] {len(preload)} preload words, {M_TOKENS * D_MODEL} final values")
    print(f"  HEX : {os.path.normpath(HEX)}")
    print(f"  CSV : {os.path.normpath(CSV)}")


if __name__ == "__main__":
    main()
