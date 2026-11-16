#!/usr/bin/env python3
"""
generate_snax_llm_layer_mid_data.py

Stimulus + bit-exact golden generator for a MID-SIZE complete LLaMA-style
prefill layer on snax_llm_top_clean.

Shape (consistent d_model = H * head_dim):
    M_TOKENS = 32
    D_MODEL  = 128
    D_FF     = 256
    H        = 4
    D_K      = D_V = 32   (H * 32 = 128)
    M_Q      = 8          Q_TILES = 4
    N_KV     = 32         KV_TILES = 2
    K_GEMM   = D_MODEL/16 = 8
    K_ATTN   = D_K/16     = 2
    DV_TILES = D_V/16     = 2
    FF_KB    = D_FF/16    = 16

TCDM map (word = 10-bit, bank = 5-bit, addr = word<<9 | bank<<4):
    banks 0..15 : FP4 weights (16 lanes = 16 output channels)
                  Q=0, K=64, V=128, O=192, gate/up=256, down=512
                  K/V attention ring words 0..31 overwrite Wq after Q is done:
                    K word = h*KV_TILES*K_ATTN + kj*K_ATTN + kb (0..15)
                    V word = 16 + h*DV_TILES*KV_TILES + dvi*KV_TILES + kj (16..31)
    bank 16     : activation packets / attention output / FFN activation
    bank 17     : GEMM FP16 output (one 16x16 tile at a time, 4 words/vector)
    bank 18     : FP16 vector staging (gate chunk copy)
    bank 19     : generic packet staging (unused in the final flow)
    bank 20     : O-projection activation packets (token-major)
    bank 21     : gate chunk FP16 staging
    bank 22/23  : SiLU*up FP16, tokens 0..15 / 16..31 (vector = token*16+kb)
    bank 24     : X rows (token-major, 32 words/token)
    bank 25     : Xn rows (RMSNorm 1 / RMSNorm 2)
    bank 26     : residual 1 rows
    bank 27     : final output rows
    bank 28     : gamma1 (word 0), gamma2 (word 32)
    bank 29     : cos (word 0), sin (word 128)
    bank 30     : Q requant packets (linear GEMM vector order)
    bank 31     : K requant packets (linear GEMM vector order)

Every arithmetic path mirrors the RTL:
    log_quantizer -> quantize_block_rtl
    matmul       -> gemm_golden (fp4 LUT + CSD + shifter + sat add + FP16 drain)
    softmax/P    -> softmax_requantizer_py + csd_rescaler_32b_py
    final O/l    -> recip_div_py + drain_norm_fp16_py

This generator intentionally uses D_K = D_V = 32.  The earlier proposal
(d_model=128, H=4, D_K=D_V=64) would require H*head_dim = 256 channels and
does not match the d_model=128 projection/O matrices.
"""
import os
import sys
import math
import csv

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from generate_snax_llm_layer_data import (
    quantize_block_rtl,
    quantize_matrix,
    gemm_golden,
    real_to_fp16_hex,
    fp16_hex_to_real,
    pack4,
    write_hex,
)
from generate_fused_attn_8x8 import (
    LN2_Q16,
    sat32,
    softmax_requantizer_py,
    recip_div_py,
    csd_rescaler_32b_py,
)
from generate_prefill_gemm_data import (
    encode_72b_packet,
    fp4_mult_lut_py,
    csd_scaler_py,
    shifter_py,
    sat_add32,
    drain_norm_fp16_py,
)

OUT_DIR = os.path.join(HERE, "..", "sim", "sim_snax_llm_layer_mid")
HEX = os.path.join(OUT_DIR, "hex")
CSV = os.path.join(OUT_DIR, "csv")

# ------------------------------------------------------------------ shape ----
M_TOKENS = 32
D_MODEL = 128
D_FF = 256
H = 4
D_K = 32
D_V = 32
M_Q = 8
N_Q = M_TOKENS
N_KV = M_TOKENS
K_GEMM = D_MODEL // 16          # projection reduction blocks
K_ATTN = D_K // 16              # QK reduction blocks
DV_TILES = D_V // 16
KV_TILES = N_KV // 16
Q_TILES = N_Q // M_Q
FF_KB = D_FF // 16
N_TILES_MODEL = D_MODEL // 16
N_TILES_O = (H * D_V) // 16
N_TILES_GU = (2 * D_FF) // 16

# --------------------------------------------------------------- TCDM map ----
ACT_BANK = 16
OUT_BANK = 17
STG_BANK = 18
PKT_BANK = 19
O_ACT_BANK = 20
G_STG_BANK = 21
FF1_BANK = 22
FF2_BANK = 23
X_BANK = 24
XN_BANK = 25
RES1_BANK = 26
FIN_BANK = 27
GAMMA_BANK = 28
ROPE_BANK = 29
QPKT_BANK = 30
KPKT_BANK = 31

GAMMA1_WORD = 0
GAMMA2_WORD = 32
COS_WORD = 0
SIN_WORD = 128

W_WORD = {
    "q": 0, "k": 64, "v": 128, "o": 192,
    "gu": 256, "d": 512,
}

V_BASE_BLOCK = max(H * KV_TILES * K_ATTN, 16)
CAUSAL = True
S_MIN = -(1 << 31)


def tcdm_addr(word, bank):
    return ((word & 0x3FF) << 9) | ((bank & 0x1F) << 4)


def gemm_golden_m(a_codes, a_scales, w_codes, w_scales, n_tiles, k_blocks):
    """gemm_golden for M_TOKENS, 16 tokens at a time."""
    out = np.zeros((n_tiles, M_TOKENS, 16), dtype=np.uint16)
    for mt in range(M_TOKENS // 16):
        sl = slice(mt * 16, (mt + 1) * 16)
        o = gemm_golden(a_codes[sl], a_scales[sl], w_codes, w_scales,
                        n_tiles, k_blocks)
        out[:, sl, :] = o
    return out


def main():
    np.random.seed(2029)
    os.makedirs(HEX, exist_ok=True)
    os.makedirs(CSV, exist_ok=True)

    print("=== SNAX LLM MID full-layer (prefill) stimulus & golden ===")
    print(f"tokens={M_TOKENS} d_model={D_MODEL} d_ff={D_FF} heads={H} "
          f"d_k={D_K} d_v={D_V} n_kv={N_KV} "
          f"Q_TILES={Q_TILES} KV_TILES={KV_TILES} DV_TILES={DV_TILES}")

    # ------------------------------------------------------------ input ----
    x_real = np.random.normal(0.0, 1.0, (M_TOKENS, D_MODEL))
    x_hex = np.vectorize(real_to_fp16_hex)(x_real).astype(np.uint16)
    gamma1_hex = np.vectorize(real_to_fp16_hex)(
        np.random.normal(0.0, 0.5, (1, D_MODEL)))[0]
    gamma2_hex = np.vectorize(real_to_fp16_hex)(
        np.random.normal(0.0, 0.5, (1, D_MODEL)))[0]

    # ---------------------------------------------------------- weights ----
    weights = {
        "q": np.random.normal(0.0, 0.6, (D_MODEL, D_MODEL)),
        "k": np.random.normal(0.0, 0.6, (D_MODEL, D_MODEL)),
        "v": np.random.normal(0.0, 0.6, (D_MODEL, D_MODEL)),
        "o": np.random.normal(0.0, 0.6, (D_MODEL, H * D_V)),
        "gu": np.random.normal(0.0, 0.6, (2 * D_FF, D_MODEL)),
        "d": np.random.normal(0.0, 0.6, (D_MODEL, D_FF)),
    }

    w_codes, w_scales, w_packets = {}, {}, {}
    for name, mat in weights.items():
        n_ch, kdim = mat.shape
        kb_total = kdim // 16
        mat_hex = np.vectorize(real_to_fp16_hex)(mat).astype(np.uint16)
        codes = np.zeros((n_ch, kb_total, 16), dtype=int)
        scales = np.zeros((n_ch, kb_total), dtype=int)
        packets = np.zeros((n_ch, kb_total), dtype=object)
        for n in range(n_ch):
            for b in range(kb_total):
                c, s = quantize_block_rtl(
                    [int(h) for h in mat_hex[n, b * 16:(b + 1) * 16]])
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
        acc = sum(v * v for v in xs)
        inv = 1.0 / math.sqrt(acc / D_MODEL)
        for d in range(D_MODEL):
            xnorm_hex[m, d] = real_to_fp16_hex((xs[d] * inv) * gs[d])

    a_x_codes, a_x_scales = quantize_matrix(xnorm_hex, K_GEMM)
    xn_packets = np.zeros((M_TOKENS, K_GEMM), dtype=object)
    for m in range(M_TOKENS):
        for kb in range(K_GEMM):
            xn_packets[m, kb] = encode_72b_packet(
                a_x_codes[m, kb, :], a_x_scales[m, kb])

    # ------------------------------------------------------- GEMM Q/K/V ----
    q_hex = gemm_golden_m(a_x_codes, a_x_scales, w_codes["q"], w_scales["q"],
                          N_TILES_MODEL, K_GEMM)
    k_hex = gemm_golden_m(a_x_codes, a_x_scales, w_codes["k"], w_scales["k"],
                          N_TILES_MODEL, K_GEMM)
    v_hex = gemm_golden_m(a_x_codes, a_x_scales, w_codes["v"], w_scales["v"],
                          N_TILES_MODEL, K_GEMM)

    # ------------------------------------------------------------- RoPE ----
    half = D_K // 2
    vecs_per_head = D_K // 16
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
            for h in range(H):
                for j in range(half):
                    c = fp16_hex_to_real(cos_hex[pos, j])
                    s = fp16_hex_to_real(sin_hex[pos, j])
                    n0 = h * vecs_per_head + (j // 16)
                    l0 = j % 16
                    n1 = h * vecs_per_head + ((j + half) // 16)
                    l1 = (j + half) % 16
                    x0 = fp16_hex_to_real(vec_hex[n0, pos, l0])
                    x1 = fp16_hex_to_real(vec_hex[n1, pos, l1])
                    out[n0, pos, l0] = real_to_fp16_hex(x0 * c - x1 * s)
                    out[n1, pos, l1] = real_to_fp16_hex(x0 * s + x1 * c)
        return out

    q_rope = rope_apply(q_hex)
    k_rope = rope_apply(k_hex)

    # ------------------------------------------------- Q/K requant packets --
    # Linear order exactly as the FP16 read streamer sees the GEMM output:
    #   vector index = n_tile * M_TOKENS + token
    q_codes_lin = np.zeros((N_TILES_MODEL * M_TOKENS, 16), dtype=int)
    q_scales_lin = np.zeros(N_TILES_MODEL * M_TOKENS, dtype=int)
    q_packets_lin = np.zeros(N_TILES_MODEL * M_TOKENS, dtype=object)
    k_codes_lin = np.zeros((N_TILES_MODEL * M_TOKENS, 16), dtype=int)
    k_scales_lin = np.zeros(N_TILES_MODEL * M_TOKENS, dtype=int)
    k_packets_lin = np.zeros(N_TILES_MODEL * M_TOKENS, dtype=object)
    for gn in range(N_TILES_MODEL):
        for token in range(M_TOKENS):
            idx = gn * M_TOKENS + token
            c, s = quantize_block_rtl([int(x) for x in q_rope[gn, token, :]])
            q_codes_lin[idx, :] = c
            q_scales_lin[idx] = s
            q_packets_lin[idx] = encode_72b_packet(c, s)
            c, s = quantize_block_rtl([int(x) for x in k_rope[gn, token, :]])
            k_codes_lin[idx, :] = c
            k_scales_lin[idx] = s
            k_packets_lin[idx] = encode_72b_packet(c, s)

    # Attention views: Q[q][h][kb], K[key][h][kb].
    q_codes_tok = np.zeros((N_Q, H, K_ATTN, 16), dtype=int)
    q_scales_tok = np.zeros((N_Q, H, K_ATTN), dtype=int)
    for q in range(N_Q):
        for h in range(H):
            for kb in range(K_ATTN):
                idx = (h * vecs_per_head + kb) * M_TOKENS + q
                q_codes_tok[q, h, kb, :] = q_codes_lin[idx, :]
                q_scales_tok[q, h, kb] = q_scales_lin[idx]

    k_codes_tok = np.zeros((N_KV, H, K_ATTN, 16), dtype=int)
    k_scales_tok = np.zeros((N_KV, H, K_ATTN), dtype=int)
    for key in range(N_KV):
        for h in range(H):
            for kb in range(K_ATTN):
                idx = (h * vecs_per_head + kb) * M_TOKENS + key
                k_codes_tok[key, h, kb, :] = k_codes_lin[idx, :]
                k_scales_tok[key, h, kb] = k_scales_lin[idx]

    # ------------------------------------------------------------- V ---------
    # V packet for (h,dvi,kj,lane) contains the 16 keys of one output channel.
    v_codes_attn = np.zeros((H, DV_TILES, KV_TILES, 16, 16), dtype=int)
    v_scales_attn = np.zeros((H, DV_TILES, KV_TILES, 16), dtype=int)
    v_packets = np.zeros((H * DV_TILES * KV_TILES * 16,), dtype=object)
    for h in range(H):
        for dvi in range(DV_TILES):
            for kj in range(KV_TILES):
                for lane in range(16):
                    seg = [int(v_hex[h * DV_TILES + dvi,
                                     kj * 16 + key, lane])
                           for key in range(16)]
                    c, s = quantize_block_rtl(seg)
                    v_codes_attn[h, dvi, kj, lane, :] = c
                    v_scales_attn[h, dvi, kj, lane] = s
                    idx = ((h * DV_TILES + dvi) * KV_TILES + kj) * 16 + lane
                    v_packets[idx] = encode_72b_packet(c, s)

    # ---------------------------------------------------- FlashAttention ---
    attn_hex = np.zeros((N_Q, D_MODEL), dtype=np.uint16)
    attn_packets = []
    p_codes_arr = np.zeros((H, Q_TILES, M_Q, KV_TILES, 16), dtype=int)
    p_scale_arr = np.zeros((H, Q_TILES, M_Q, KV_TILES), dtype=int)
    p_resc_arr = np.zeros((H, Q_TILES, M_Q, KV_TILES, 4), dtype=int)
    p_lc_arr = np.zeros((H, Q_TILES, M_Q, KV_TILES), dtype=np.int64)
    p_csum_arr = np.zeros((H, Q_TILES, M_Q, KV_TILES), dtype=int)
    p_mprev_arr = np.zeros((H, Q_TILES, M_Q, KV_TILES), dtype=np.int64)
    p_mnext_arr = np.zeros((H, Q_TILES, M_Q, KV_TILES), dtype=np.int64)
    score_nat_arr = np.zeros((H, Q_TILES, M_Q, KV_TILES, 16), dtype=np.int64)
    l_final_arr = np.zeros((H, Q_TILES, M_Q), dtype=np.int64)

    for h in range(H):
        for qi in range(Q_TILES):
            # P/l are independent of dvi; the RTL recomputes them per group.
            for m in range(M_Q):
                q = qi * M_Q + m
                m_run = S_MIN
                l_run = 0
                for kj in range(KV_TILES):
                    score_nat = []
                    visible_mask = []
                    for lane in range(16):
                        key = kj * 16 + lane
                        acc = 0
                        for kb in range(K_ATTN):
                            dot = 0
                            for el in range(16):
                                dot += fp4_mult_lut_py(
                                    int(q_codes_tok[q, h, kb, el]),
                                    int(k_codes_tok[key, h, kb, el]))
                            res_scale = (int(q_scales_tok[q, h, kb]) +
                                         int(k_scales_tok[key, h, kb]))
                            q_exp = res_scale // 8
                            r_exp = res_scale % 8
                            aligned = shifter_py(
                                csd_scaler_py(dot, r_exp), q_exp, 16)
                            acc = sat_add32(acc, aligned)
                        nat = sat32((acc * LN2_Q16) >> 16)
                        visible = (key <= q) if CAUSAL else True
                        visible_mask.append(1 if visible else 0)
                        score_nat.append(nat if visible else S_MIN)

                    codes, p_blk, l_contrib, resc_req, rq, rr, m_next = \
                        softmax_requantizer_py(
                            score_nat, m_run, kj == 0)
                    # RTL softmax applies key_mask per lane: an all-masked
                    # block produces codes=0, csum=0 and l_contrib=0.  The
                    # Python helper only sees S_MIN scores, which would give
                    # d=0 -> mag=7 for every lane; override that case.
                    if sum(visible_mask) == 0:
                        codes = [0] * 16
                        l_contrib = 0
                    p_codes_arr[h, qi, m, kj, :] = codes
                    p_scale_arr[h, qi, m, kj] = p_blk
                    p_resc_arr[h, qi, m, kj, :] = [1 if resc_req else 0,
                                                   rq, rr, 0]
                    p_lc_arr[h, qi, m, kj] = l_contrib
                    p_csum_arr[h, qi, m, kj] = sum(
                        [0, 1, 2, 3, 4, 6, 8, 12][int(c)] for c in codes)
                    p_mprev_arr[h, qi, m, kj] = m_run
                    p_mnext_arr[h, qi, m, kj] = m_next
                    score_nat_arr[h, qi, m, kj, :] = np.array(score_nat)
                    if kj == 0:
                        l_run = l_contrib
                    elif resc_req:
                        l_run = sat_add32(
                            csd_rescaler_32b_py(l_run, rq, rr), l_contrib)
                    else:
                        l_run = sat_add32(l_run, l_contrib)
                    m_run = m_next
                l_final_arr[h, qi, m] = l_run

            for dvi in range(DV_TILES):
                o_acc = np.zeros((M_Q, 16), dtype=np.int64)
                for kj in range(KV_TILES):
                    for m in range(M_Q):
                        resc_req = p_resc_arr[h, qi, m, kj, 0]
                        if resc_req:
                            rq = int(p_resc_arr[h, qi, m, kj, 1])
                            rr = int(p_resc_arr[h, qi, m, kj, 2])
                            for lane in range(16):
                                o_acc[m, lane] = csd_rescaler_32b_py(
                                    int(o_acc[m, lane]), rq, rr)

                        codes = p_codes_arr[h, qi, m, kj]
                        p_blk = int(p_scale_arr[h, qi, m, kj])
                        for lane in range(16):
                            dot = 0
                            for key in range(16):
                                dot += fp4_mult_lut_py(
                                    int(codes[key]),
                                    int(v_codes_attn[h, dvi, kj, lane, key]))
                            res_scale = p_blk + int(v_scales_attn[h, dvi, kj, lane])
                            q_exp = res_scale // 8
                            r_exp = res_scale % 8
                            aligned = shifter_py(
                                csd_scaler_py(dot, r_exp), q_exp, 16)
                            o_acc[m, lane] = sat_add32(
                                int(o_acc[m, lane]), aligned)

                for m in range(M_Q):
                    q = qi * M_Q + m
                    l_tot = int(l_final_arr[h, qi, m])
                    for lane in range(16):
                        norm = recip_div_py(int(o_acc[m, lane]), l_tot)
                        attn_hex[q, h * D_V + dvi * 16 + lane] = \
                            drain_norm_fp16_py(norm, frac_bits=16)
                    seg = [int(attn_hex[q, h * D_V + dvi * 16 + lane])
                           for lane in range(16)]
                    c, s = quantize_block_rtl(seg)
                    attn_packets.append(encode_72b_packet(c, s))

    # Attention output packet order must match the sequencer drain order:
    # h -> qi -> dvi -> m.
    attn_packets_reordered = []
    for h in range(H):
        for qi in range(Q_TILES):
            for dvi in range(DV_TILES):
                for m in range(M_Q):
                    q = qi * M_Q + m
                    seg = [int(attn_hex[q, h * D_V + dvi * 16 + lane])
                           for lane in range(16)]
                    c, s = quantize_block_rtl(seg)
                    attn_packets_reordered.append(encode_72b_packet(c, s))
    attn_packets = attn_packets_reordered

    # O activation: token-major, K block = h*DV_TILES+dvi.
    a_o_codes = np.zeros((M_TOKENS, N_TILES_O, 16), dtype=int)
    a_o_scales = np.zeros((M_TOKENS, N_TILES_O), dtype=int)
    for q in range(N_Q):
        for kb in range(N_TILES_O):
            c, s = quantize_block_rtl(
                [int(x) for x in attn_hex[q, kb * 16:(kb + 1) * 16]])
            a_o_codes[q, kb, :] = c
            a_o_scales[q, kb] = s

    o_hex = gemm_golden_m(a_o_codes, a_o_scales, w_codes["o"], w_scales["o"],
                          N_TILES_O, N_TILES_O)

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
        acc = sum(v * v for v in xs)
        inv = 1.0 / math.sqrt(acc / D_MODEL)
        for d in range(D_MODEL):
            n2_hex[m, d] = real_to_fp16_hex((xs[d] * inv) * gs[d])

    a_ff_codes, a_ff_scales = quantize_matrix(n2_hex, K_GEMM)
    n2_packets = np.zeros((M_TOKENS, K_GEMM), dtype=object)
    for m in range(M_TOKENS):
        for kb in range(K_GEMM):
            n2_packets[m, kb] = encode_72b_packet(
                a_ff_codes[m, kb, :], a_ff_scales[m, kb])

    gu_hex = gemm_golden_m(a_ff_codes, a_ff_scales,
                           w_codes["gu"], w_scales["gu"],
                           N_TILES_GU, K_GEMM)

    # --------------------------------------------------- SiLU(gate) * up ---
    h_hex = np.zeros((M_TOKENS, D_FF), dtype=np.uint16)
    for q in range(M_TOKENS):
        for c in range(D_FF):
            gn_gate = c // 16
            gn_up = 16 + c // 16
            g = fp16_hex_to_real(gu_hex[gn_gate, q, c % 16])
            u = fp16_hex_to_real(gu_hex[gn_up, q, c % 16])
            h_hex[q, c] = real_to_fp16_hex((g / (1.0 + math.exp(-g))) * u)

    a_d_codes, a_d_scales = quantize_matrix(h_hex, FF_KB)
    ffn_packets = np.zeros((M_TOKENS, FF_KB), dtype=object)
    for m in range(M_TOKENS):
        for kb in range(FF_KB):
            ffn_packets[m, kb] = encode_72b_packet(
                a_d_codes[m, kb, :], a_d_scales[m, kb])

    down_hex = gemm_golden_m(a_d_codes, a_d_scales,
                             w_codes["d"], w_scales["d"],
                             N_TILES_MODEL, FF_KB)

    # ------------------------------------------------------ residual (2) ---
    final_hex = np.zeros((M_TOKENS, D_MODEL), dtype=np.uint16)
    for q in range(M_TOKENS):
        for d in range(D_MODEL):
            dv = fp16_hex_to_real(down_hex[d // 16, q, d % 16])
            rv = fp16_hex_to_real(res1_hex[q, d])
            final_hex[q, d] = real_to_fp16_hex(dv + rv)

    # --------------------------------------------------------- HEX output --
    write_hex(os.path.join(HEX, "layer_x_fp16.hex"), x_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_xnorm_fp16.hex"), xnorm_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_q_fp16.hex"), q_rope.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_k_fp16.hex"), k_rope.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_v_fp16.hex"), v_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_q_packets.hex"),
              [q_packets_lin[i] for i in range(N_TILES_MODEL * M_TOKENS)], 18)
    write_hex(os.path.join(HEX, "layer_k_packets.hex"),
              [k_packets_lin[i] for i in range(N_TILES_MODEL * M_TOKENS)], 18)
    write_hex(os.path.join(HEX, "layer_v_packets.hex"),
              [v_packets[i] for i in range(v_packets.size)], 18)
    write_hex(os.path.join(HEX, "layer_attn_fp16.hex"), attn_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_attn_packets.hex"), attn_packets, 18)
    write_hex(os.path.join(HEX, "layer_o_fp16.hex"), o_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_res1_fp16.hex"), res1_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_norm2_fp16.hex"), n2_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_gu_fp16.hex"), gu_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_h_fp16.hex"), h_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_down_fp16.hex"), down_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_final_fp16.hex"), final_hex.flatten(), 4)
    write_hex(os.path.join(HEX, "layer_xn_packets.hex"),
              [xn_packets[m, kb] for m in range(M_TOKENS)
               for kb in range(K_GEMM)], 18)
    write_hex(os.path.join(HEX, "layer_norm2_packets.hex"),
              [n2_packets[m, kb] for m in range(M_TOKENS)
               for kb in range(K_GEMM)], 18)
    write_hex(os.path.join(HEX, "layer_ffn_packets.hex"),
              [ffn_packets[m, kb] for m in range(M_TOKENS)
               for kb in range(FF_KB)], 18)
    write_hex(os.path.join(HEX, "layer_l_q16.hex"),
              [int(l_final_arr[hh, qqi, mm]) & 0xFFFFFFFF
               for hh in range(H) for qqi in range(Q_TILES)
               for mm in range(M_Q)], 8)

    # ------------------------------------------------------ TCDM preload ---
    preload = []
    row_words = D_MODEL // 4
    gam_words = D_MODEL // 4

    for m in range(M_TOKENS):
        for w in range(row_words):
            preload.append((tcdm_addr(m * row_words + w, X_BANK),
                            pack4(*x_hex[m, 4 * w:4 * w + 4]),
                            f"X token={m} word={w}"))
    for w in range(gam_words):
        preload.append((tcdm_addr(GAMMA1_WORD + w, GAMMA_BANK),
                        pack4(*gamma1_hex[4 * w:4 * w + 4]), "gamma1"))
        preload.append((tcdm_addr(GAMMA2_WORD + w, GAMMA_BANK),
                        pack4(*gamma2_hex[4 * w:4 * w + 4]), "gamma2"))

    half_words = half // 4
    for pos in range(M_TOKENS):
        for w in range(half_words):
            preload.append((tcdm_addr(COS_WORD + pos * half_words + w,
                                      ROPE_BANK),
                            pack4(*cos_hex[pos, 4 * w:4 * w + 4]),
                            f"cos pos={pos}"))
            preload.append((tcdm_addr(SIN_WORD + pos * half_words + w,
                                      ROPE_BANK),
                            pack4(*sin_hex[pos, 4 * w:4 * w + 4]),
                            f"sin pos={pos}"))

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
    with open(os.path.join(CSV, "layer_mid_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "values", "note"])
        w.writerow(["x", M_TOKENS * D_MODEL, "input FP16"])
        w.writerow(["xnorm", M_TOKENS * D_MODEL, "RMSNorm1"])
        w.writerow(["q", N_TILES_MODEL * M_TOKENS * 16, "Q GEMM output"])
        w.writerow(["k", N_TILES_MODEL * M_TOKENS * 16, "K GEMM output"])
        w.writerow(["v", N_TILES_MODEL * M_TOKENS * 16, "V GEMM output"])
        w.writerow(["attn", N_Q * D_V * H, "attention FP16"])
        w.writerow(["o", N_TILES_MODEL * M_TOKENS * 16, "O GEMM output"])
        w.writerow(["h", M_TOKENS * D_FF, "SiLU*up"])
        w.writerow(["final", M_TOKENS * D_MODEL, "final FP16"])

    with open(os.path.join(CSV, "attention_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["h", "qi", "m", "kj", "lane", "score_nat_q16"])
        for hh in range(H):
            for qqi in range(Q_TILES):
                for mm in range(M_Q):
                    for kj in range(KV_TILES):
                        for ln in range(16):
                            w.writerow([hh, qqi, mm, kj, ln,
                                        int(score_nat_arr[hh, qqi, mm, kj, ln])])

    with open(os.path.join(CSV, "attention_p_blocks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["h", "qi", "m", "kj", "m_prev", "m_next", "p_blk",
                  "csum", "l_contrib", "resc_valid", "resc_q", "resc_r"]
        header += [f"K{i}_code" for i in range(16)]
        w.writerow(header)
        for hh in range(H):
            for qqi in range(Q_TILES):
                for mm in range(M_Q):
                    for kj in range(KV_TILES):
                        w.writerow([
                            hh, qqi, mm, kj,
                            int(p_mprev_arr[hh, qqi, mm, kj]),
                            int(p_mnext_arr[hh, qqi, mm, kj]),
                            int(p_scale_arr[hh, qqi, mm, kj]),
                            int(p_csum_arr[hh, qqi, mm, kj]),
                            int(p_lc_arr[hh, qqi, mm, kj]),
                            int(p_resc_arr[hh, qqi, mm, kj, 0]),
                            int(p_resc_arr[hh, qqi, mm, kj, 1]),
                            int(p_resc_arr[hh, qqi, mm, kj, 2]),
                        ] + [int(c) for c in p_codes_arr[hh, qqi, mm, kj]])

    print(f"preload words: {len(preload)}")
    print(f"Q packets: {N_TILES_MODEL * M_TOKENS}, "
          f"K packets: {N_TILES_MODEL * M_TOKENS}, "
          f"V packets: {v_packets.size}, attention packets: {len(attn_packets)}")
    print(f"[SUCCESS] HEX: {os.path.normpath(HEX)}")
    print(f"          CSV: {os.path.normpath(CSV)}")


if __name__ == "__main__":
    main()
