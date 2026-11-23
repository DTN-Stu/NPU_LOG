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
import csv
import json
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


# Hardware LUTs & constants matching RTL:
#   - log8_fp4_arith_units.sv (CSD_Q5)
#   - softmax_requantizer.sv   (SM_TH_Q16, P_RND, P_OFFSET, CSD_Q5)
#   - l_accumulator.sv         (RECIP_LUT)
#   - snax_fused_llm_core.sv   (LOG2E_Q16 = 94548)
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1_LUT = np.concatenate([E2M1_MAG, -E2M1_MAG]).astype(np.float32)
CSD_Q5 = np.array([32.0, 35.0, 38.0, 41.0, 46.0, 49.0, 54.0, 59.0], dtype=np.float32)
CSD_Q5_INT = np.array([32, 35, 38, 41, 46, 49, 54, 59], dtype=np.int64)
MINT_TABLE = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.int64)
RECIP_LUT = np.array([
    32768, 31775, 30840, 29960, 29127, 28339, 27592, 26884,
    26214, 25579, 24976, 24403, 23859, 23341, 22847, 22376,
    21926, 21495, 21083, 20687, 20308, 19944, 19594, 19258,
    18934, 18622, 18322, 18032, 17752, 17482, 17220, 16968,
], dtype=np.int64)
SM_TH_Q16 = (17238, 50961, 82774, 116497, 148310, 196608, 300480)


def decode_pkts_codes_scales(pkts):
    """Extract (codes[... , kb, 16], scales[... , kb]) from 9B SLQ packets."""
    sc = pkts[..., 8].astype(np.int8).astype(np.int32)
    codes = np.zeros(pkts.shape[:-1] + (16,), dtype=np.int32)
    for b in range(8):
        codes[..., 2 * b]     = pkts[..., b] & 0xF
        codes[..., 2 * b + 1] = (pkts[..., b] >> 4) & 0xF
    return codes, sc


def dequant_pkts_csd(pkts, use_csd=True):
    """Dequantize 9B SLQ packets using CSD_Q5/32 (default) or ideal 2^(sc/8)."""
    codes, sc = decode_pkts_codes_scales(pkts)
    if use_csd:
        q_s = np.floor(sc.astype(np.float32) / 8.0)
        r_s = sc & 7
        scale = (2.0 ** q_s) * (CSD_Q5[r_s] / 32.0)
    else:
        scale = 2.0 ** (sc.astype(np.float32) / 8.0)
    vals = E2M1_LUT[codes] * scale[..., None]
    return vals.reshape(pkts.shape[0], -1).astype(np.float32)


def fq_rtl_vec(mat_f32, use_csd=True):
    """Dequantized FP4+Log8 Fake-Quant tensor matching quantize_rtl_vec + CSD_Q5."""
    pkts = quantize_rtl_vec(mat_f32)
    return dequant_pkts_csd(pkts, use_csd=use_csd)


def matmul_log8_csd_pkts(a_pkts, w_pkts):
    """Hardware-exact paired-scale CSD matmul matching log8_fp4_arith_units.sv.

    Uses CSD_Q5[(s_a + s_w) & 7] / 32 once per block pair (instead of twice).
    a_pkts: (M, KB, 9), w_pkts: (N, KB, 9) -> returns (M, N) float32.
    """
    m_rows, kb, _ = a_pkts.shape
    n_rows = w_pkts.shape[0]
    a_codes, a_sc = decode_pkts_codes_scales(a_pkts)
    w_codes, w_sc = decode_pkts_codes_scales(w_pkts)

    a_q = (a_sc >> 3).astype(np.float32)
    a_r = a_sc & 7
    w_q = (w_sc >> 3).astype(np.float32)
    w_r = w_sc & 7

    a_vals = E2M1_LUT[a_codes] * (2.0 ** a_q)[..., None]
    w_vals = E2M1_LUT[w_codes] * (2.0 ** w_q)[..., None]

    out = np.zeros((m_rows, n_rows), dtype=np.float32)
    for ra in range(8):
        mask_a = (a_r == ra)
        if not np.any(mask_a):
            continue
        a_m = np.where(mask_a[..., None], a_vals, 0.0).reshape(m_rows, kb * 16)
        rs = ra + w_r
        fac = np.where(rs >= 8, 2.0, 1.0).astype(np.float32) * (CSD_Q5[rs & 7] / 32.0)
        w_m = (w_vals * fac[..., None]).reshape(n_rows, kb * 16)
        out += a_m @ w_m.T
    return out


def pack_p_pkts_u8(mag_idx, p_blk):
    """Pack softmax_requantizer E2M1 magnitudes (0..7) & signed int8 p_blk into 9B packets."""
    m_rows, num_blk, _ = mag_idx.shape
    pkts = np.zeros((m_rows, num_blk, 9), dtype=np.uint8)
    codes = mag_idx.astype(np.uint8)
    for b in range(8):
        pkts[:, :, b] = codes[:, :, 2 * b] | (codes[:, :, 2 * b + 1] << 4)
    pkts[:, :, 8] = p_blk.astype(np.int8).view(np.uint8)
    return pkts


def exp_free_softmax_requantizer_np(scores_f32, valid_mask_bool):
    """Vectorized model of snax_fused_llm_core LOG2E_Q16 + softmax_requantizer.sv.

    Returns:
      mag_idx : (M, num_blk, 16) int32 codes in 0..7
      p_blk   : (M, num_blk) int32 signed scale exponent
      p_unnorm: (M, K) float64 unnormalized probabilities
    """
    m_rows, k_len = scores_f32.shape
    assert k_len % 16 == 0
    num_blk = k_len // 16
    s_min = -(1 << 31)

    # Exact Q16.16 conversion + 7-term CSD LOG2E_Q16 (= 94548) multiply
    s_q16 = np.clip(np.round(scores_f32.astype(np.float64) * 65536.0),
                    -2147483648.0, 2147483647.0).astype(np.int64)
    s_prod = ((s_q16 << 16) + (s_q16 << 15) - (s_q16 << 12) +
              (s_q16 << 8) + (s_q16 << 6) + (s_q16 << 4) + (s_q16 << 2)) >> 16
    s_scaled = np.clip(s_prod, -2147483648, 2147483647)

    s_blk = s_scaled.reshape(m_rows, num_blk, 16)
    m_blk = valid_mask_bool.reshape(m_rows, num_blk, 16)
    s_masked = np.where(m_blk, s_blk, s_min)

    blk_max = s_masked.max(axis=-1, keepdims=True)
    row_max = blk_max.max(axis=-2, keepdims=True)
    score_dist = blk_max - s_masked

    th_6p0, th_4p0, th_3p0, th_2p0, th_1p5, th_1p0, th_0p5 = SM_TH_Q16
    mag_idx = np.zeros_like(score_dist, dtype=np.int32)
    mag_idx = np.where(score_dist <= th_0p5, 1, mag_idx)
    mag_idx = np.where(score_dist <= th_1p0, 2, mag_idx)
    mag_idx = np.where(score_dist <= th_1p5, 3, mag_idx)
    mag_idx = np.where(score_dist <= th_2p0, 4, mag_idx)
    mag_idx = np.where(score_dist <= th_3p0, 5, mag_idx)
    mag_idx = np.where(score_dist <= th_4p0, 6, mag_idx)
    mag_idx = np.where(score_dist <= th_6p0, 7, mag_idx)
    mag_idx = np.where(m_blk, mag_idx, 0)

    p_wide = ((blk_max - row_max + 4096) >> 13) - 21
    p_blk = np.clip(p_wide, -128, 127).astype(np.int32)

    q_s = p_blk >> 3
    r_s = p_blk & 7
    s_blk_val = (2.0 ** q_s.astype(np.float64)) * (CSD_Q5[r_s].astype(np.float64) / 32.0)

    p_unnorm = E2M1_MAG[mag_idx].astype(np.float64) * s_blk_val
    blk_any = np.any(m_blk, axis=-1, keepdims=True)
    p_unnorm = np.where(m_blk & blk_any, p_unnorm, 0.0)
    return mag_idx, p_blk[..., 0], p_unnorm.reshape(m_rows, k_len)


def csd_rescaler_32b_vec(val_in, resc_q, resc_r):
    """Vectorized mirror of vsq_rescaler_32b.sv on int64 arrays (32-bit signed range)."""
    val = val_in.astype(np.int64)
    fac = CSD_Q5_INT[resc_r]
    prod = val * fac
    net = np.where(resc_q < -26, 32, 5 + ((-resc_q) & 0x1F))
    rounded = prod + (np.int64(1) << (net - 1))
    return (rounded >> net).astype(np.int64)


def eval_attn_online_rtl_sim(scores_all_f32, v_pkts_all, valid_mask_bool, gqa_grp):
    """Vectorized over (H_Q, M): models RTL online running max, l_accum, RECIP_LUT & final_normalizer."""
    h_q, m_rows, k_len = scores_all_f32.shape
    num_blk = k_len // 16
    s_min = -(1 << 31)

    s_q16 = np.clip(np.round(scores_all_f32.astype(np.float64) * 65536.0),
                    -2147483648.0, 2147483647.0).astype(np.int64)
    s_prod = ((s_q16 << 16) + (s_q16 << 15) - (s_q16 << 12) +
              (s_q16 << 8) + (s_q16 << 6) + (s_q16 << 4) + (s_q16 << 2)) >> 16
    s_scaled = np.clip(s_prod, -2147483648, 2147483647)

    s_blk_all = s_scaled.reshape(h_q, m_rows, num_blk, 16)
    m_blk_all = valid_mask_bool.reshape(1, m_rows, num_blk, 16)
    s_masked_all = np.where(m_blk_all, s_blk_all, s_min)

    v_codes_all, v_sc_all = decode_pkts_codes_scales(v_pkts_all)  # (H_KV, D_V, num_blk, 16)
    v_codes_rep = np.repeat(v_codes_all, gqa_grp, axis=0)         # (H_Q, D_V, num_blk, 16)
    v_sc_rep = np.repeat(v_sc_all, gqa_grp, axis=0)               # (H_Q, D_V, num_blk)

    th_6p0, th_4p0, th_3p0, th_2p0, th_1p5, th_1p0, th_0p5 = SM_TH_Q16
    m_running = np.full((h_q, m_rows), s_min, dtype=np.int64)
    l_running = np.zeros((h_q, m_rows), dtype=np.int64)
    o_accum = np.zeros((h_q, m_rows, v_codes_rep.shape[1]), dtype=np.float64)

    for j in range(num_blk):
        s_j = s_masked_all[:, :, j, :]  # (H_Q, M, 16)
        m_j = np.broadcast_to(m_blk_all[:, :, j, :], (h_q, m_rows, 16))
        any_v = np.any(m_j, axis=-1)
        blk_max = s_j.max(axis=-1)

        if j == 0:
            m_next = np.where(any_v, blk_max, m_running)
            resc_req = np.zeros_like(any_v, dtype=bool)
        else:
            inc = (blk_max > m_running) & any_v
            m_next = np.where(inc, blk_max, m_running)
            resc_req = inc

        resc_delta = np.where(any_v, m_running - m_next, 0)
        resc_oct = np.clip(resc_delta >> 16, -128, 127)
        resc_r = ((resc_delta >> 13) & 0x7).astype(np.int64)

        p_blk = np.clip(((blk_max - m_next + 4096) >> 13) - 21, -128, 127)
        d = blk_max[..., None] - s_j
        mag = np.zeros_like(d, dtype=np.int64)
        mag = np.where(d <= th_0p5, 1, mag)
        mag = np.where(d <= th_1p0, 2, mag)
        mag = np.where(d <= th_1p5, 3, mag)
        mag = np.where(d <= th_2p0, 4, mag)
        mag = np.where(d <= th_3p0, 5, mag)
        mag = np.where(d <= th_4p0, 6, mag)
        mag = np.where(d <= th_6p0, 7, mag)
        mag = np.where(m_j, mag, 0)

        csum = MINT_TABLE[mag].sum(axis=-1)
        q_s = p_blk >> 3
        r_s = p_blk & 7
        prod = csum * CSD_Q5_INT[r_s]
        sh = np.minimum(q_s + 16, 15)
        l_contrib = np.where(sh >= 6, prod << np.maximum(sh - 6, 0), prod >> np.maximum(6 - sh, 0))
        l_contrib = np.where(any_v, l_contrib, 0)

        # Update l_running in exact RTL Q16.16 arithmetic
        if j == 0:
            l_running = l_contrib
        else:
            l_rescaled = csd_rescaler_32b_vec(l_running, resc_oct, resc_r)
            l_base = np.where(resc_req, l_rescaled, l_running)
            l_running = np.where(any_v, np.clip(l_base + l_contrib, -2147483648, 2147483647), l_running)

        # Online O update (paired CSD P*V in Q16.16 equivalent units: * 65536)
        resc_scale = np.where(
            resc_req,
            (2.0 ** resc_oct.astype(np.float64)) * (CSD_Q5[resc_r].astype(np.float64) / 32.0),
            1.0,
        )
        o_accum = np.where(j == 0, 0.0, o_accum * resc_scale[..., None])

        p_val_base = E2M1_MAG[mag].astype(np.float64) * (2.0 ** q_s.astype(np.float64))[..., None]
        v_j_vals = E2M1_LUT[v_codes_rep[:, :, j, :]].astype(np.float64) * (2.0 ** (v_sc_rep[:, :, j] >> 3).astype(np.float64))[..., None]
        v_j_r = v_sc_rep[:, :, j] & 7  # (H_Q, D_V)

        pv_j = np.zeros_like(o_accum)
        for rp in range(8):
            pm = (r_s == rp)
            if not np.any(pm):
                continue
            rs = rp + v_j_r  # (H_Q, D_V) via broadcast if per-head
            # Align shapes (H_Q, M, 16) x (H_Q, 16, D_V)
            fac = np.where(rs >= 8, 2.0, 1.0) * (CSD_Q5[rs & 7].astype(np.float64) / 32.0)  # (H_Q, D_V)
            p_sub = np.where(pm[:, :, None], p_val_base, 0.0)
            v_sub = np.transpose(v_j_vals * fac[:, :, None], (0, 2, 1))
            pv_j += np.matmul(p_sub, v_sub)

        o_accum += np.where(any_v[..., None], pv_j * 65536.0, 0.0)
        m_running = m_next

    # 32-entry RECIP_LUT division matching l_accumulator.sv + final_normalizer.sv
    o_out = np.zeros_like(o_accum, dtype=np.float32)
    o_int = np.clip(np.round(o_accum), -2147483648.0, 2147483647.0).astype(np.int64)
    for h in range(h_q):
        for m in range(m_rows):
            lt = int(l_running[h, m])
            if lt <= 0:
                continue
            lzc = 0
            for b in range(30, -1, -1):
                if (lt >> b) & 1:
                    break
                lzc += 1
            l_norm = (lt << lzc) & 0x7FFFFFFF
            lut_idx = (l_norm >> 25) & 0x1F
            inv_mant = int(RECIP_LUT[lut_idx])
            raw_mult = o_int[h, m] * inv_mant
            if lzc <= 29:
                shifted = raw_mult >> (29 - lzc)
            else:
                shifted = raw_mult << (lzc - 29)
            o_out[h, m] = (np.clip(shifted, -2147483648, 2147483647) / 65536.0).astype(np.float32)
    return np.transpose(o_out, (1, 0, 2))


def sfu_sigmoid_pwl_np(x):
    """Exact 4-segment PWL Sigmoid matching snax_llm_sfu_opt_pipe.sv (|x| breakpoints 1.0, 2.5, 4.5)."""
    x32 = x.astype(np.float32)
    ax = np.abs(x32)
    b0 = np.float32(0.500)
    b1 = np.float32(0.73095703)  # 20'h3F3B2
    b2 = np.float32(0.92407227)  # 20'h3F6C9
    sig_pos = np.where(
        ax < 1.0, b0 + np.float32(0.25) * ax,
        np.where(
            ax < 2.5, b1 + np.float32(0.125) * (ax - np.float32(1.0)),
            np.where(ax < 4.5, b2 + np.float32(0.03125) * (ax - np.float32(2.5)), np.float32(1.0))
        )
    )
    return np.where(x32 < 0.0, np.float32(1.0) - sig_pos, sig_pos).astype(np.float32)


def fq_rtl_vec_ideal(mat_f32):
    return fq_rtl_vec(mat_f32, use_csd=False)


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

    # 2. FP4+Log8 Hardware-Aligned Fake-Quant Golden Pass (CSD_Q5 + Exp-Free Softmax)
    wq_pkts = quantize_rtl_vec(w_q)
    wk_pkts = quantize_rtl_vec(w_k)
    wv_pkts = quantize_rtl_vec(w_v)
    wo_pkts = quantize_rtl_vec(w_o)
    wg_pkts = quantize_rtl_vec(w_g)
    wu_pkts = quantize_rtl_vec(w_u)
    wd_pkts = quantize_rtl_vec(w_d)

    def quant_rope_qk(raw_rope, n_h):
        # Match Fused QK RoPE permute -> SLQ quantize -> unpermute
        perm = np.empty_like(raw_rope)
        perm[:, :, 0::2] = raw_rope[:, :, :D_K // 2]
        perm[:, :, 1::2] = raw_rope[:, :, D_K // 2:]
        pkts = quantize_rtl_vec(perm.reshape(-1, D_K))
        fq_perm = dequant_pkts_csd(pkts).reshape(raw_rope.shape[0], n_h, D_K)
        out = np.empty_like(fq_perm)
        out[:, :, :D_K // 2] = fq_perm[:, :, 0::2]
        out[:, :, D_K // 2:] = fq_perm[:, :, 1::2]
        pkts_perm = pkts.reshape(raw_rope.shape[0], n_h, K_PACKETS_QK, 9)
        return out, pkts_perm, perm

    xn_pkts = quantize_rtl_vec(xn_f32)
    q_proj_fq = matmul_log8_csd_pkts(xn_pkts, wq_pkts)
    k_proj_fq = matmul_log8_csd_pkts(xn_pkts, wk_pkts)
    v_raw_fq  = matmul_log8_csd_pkts(xn_pkts, wv_pkts).reshape(m_tokens, H_KV, D_V)

    q_fq, q_pkts_perm, _ = quant_rope_qk(apply_rope(q_proj_fq, H_Q), H_Q)
    k_fq, k_pkts_perm, _ = quant_rope_qk(apply_rope(k_proj_fq, H_KV), H_KV)

    if is_decode:
        xn_p_pkts = quantize_rtl_vec(xn_p)
        k_p_raw = apply_rope(matmul_log8_csd_pkts(xn_p_pkts, wk_pkts), H_KV, cos_all[:n_past], sin_all[:n_past])
        k_p_fq, k_p_pkts_perm, k_p_perm = quant_rope_qk(k_p_raw, H_KV)
        v_p_raw = matmul_log8_csd_pkts(xn_p_pkts, wv_pkts).reshape(n_past, H_KV, D_V)
        v_all_in = np.concatenate([v_p_raw, v_raw_fq], axis=0)  # (n_kv, H_KV, D_V)
        v_t_mat  = np.transpose(v_all_in, (1, 2, 0)).reshape(H_KV * D_V, n_kv)
        v_all_pkts_flat = quantize_rtl_vec(v_t_mat)              # (H_KV*D_V, kv_tiles, 9)
        v_all_pkts = v_all_pkts_flat.reshape(H_KV, D_V, kv_tiles, 9)
        v_fq_all = np.transpose(dequant_pkts_csd(v_all_pkts_flat).reshape(H_KV, D_V, n_kv), (2, 0, 1))
        v_cur_fq = v_fq_all[-1:, :, :]
        k_fq_all = np.concatenate([k_p_fq, k_fq], axis=0)
        k_all_pkts_perm = np.concatenate([k_p_pkts_perm, k_pkts_perm], axis=0)
    else:
        # Prefill: new M_TOKENS V are transposed in 16-token blocks
        v_new_t  = np.transpose(v_raw_fq.reshape(m_tokens // 16, 16, H_KV, D_V), (0, 2, 3, 1)).reshape(-1, 16)
        v_new_pkts_flat = quantize_rtl_vec(v_new_t)  # (mb_cnt*H_KV*D_V, 1, 9)
        v_new_fq = np.transpose(
            dequant_pkts_csd(v_new_pkts_flat).reshape(m_tokens // 16, H_KV, D_V, 16),
            (0, 3, 1, 2)
        ).reshape(m_tokens, H_KV, D_V)
        v_new_pkts_blk = v_new_pkts_flat.reshape(m_tokens // 16, H_KV, D_V, 9).transpose(1, 2, 0, 3)  # (H_KV, D_V, mb_cnt, 9)
        v_cur_fq = v_new_fq
        if n_past > 0:
            xn_p_pkts = quantize_rtl_vec(xn_p)
            k_p_raw = apply_rope(matmul_log8_csd_pkts(xn_p_pkts, wk_pkts), H_KV, cos_all[:n_past], sin_all[:n_past])
            k_p_fq, k_p_pkts_perm, k_p_perm = quant_rope_qk(k_p_raw, H_KV)
            v_p_raw = matmul_log8_csd_pkts(xn_p_pkts, wv_pkts).reshape(n_past, H_KV, D_V)
            v_p_t   = np.transpose(v_p_raw, (1, 2, 0)).reshape(H_KV * D_V, n_past)
            v_p_pkts_flat = quantize_rtl_vec(v_p_t)  # (H_KV*D_V, kv_tiles_past, 9)
            v_p_fq  = np.transpose(dequant_pkts_csd(v_p_pkts_flat).reshape(H_KV, D_V, n_past), (2, 0, 1))
            k_fq_all = np.concatenate([k_p_fq, k_fq], axis=0)
            k_all_pkts_perm = np.concatenate([k_p_pkts_perm, k_pkts_perm], axis=0)
            v_fq_all = np.concatenate([v_p_fq, v_new_fq], axis=0)
            v_all_pkts = np.concatenate([v_p_pkts_flat.reshape(H_KV, D_V, kv_tiles_past, 9), v_new_pkts_blk], axis=2)
        else:
            k_fq_all = k_fq
            k_all_pkts_perm = k_pkts_perm
            v_fq_all = v_new_fq
            v_all_pkts = v_new_pkts_blk

    valid_mask_attn = (mask > -1e4)  # (m_tokens, n_kv) bool
    sc_all_fq = np.zeros((H_Q, m_tokens, n_kv), dtype=np.float32)
    attn_fq_raw = np.zeros((m_tokens, H_Q, D_V), dtype=np.float32)
    for h_i in range(H_Q):
        hkv = h_i // GQA_GRP
        sc_fq = matmul_log8_csd_pkts(q_pkts_perm[:, h_i, :, :], k_all_pkts_perm[:, hkv, :, :])
        sc_all_fq[h_i] = sc_fq
        mag_idx, p_blk, p_unnorm = exp_free_softmax_requantizer_np(sc_fq, valid_mask_attn)
        p_pkts = pack_p_pkts_u8(mag_idx, p_blk)
        pv_unnorm = matmul_log8_csd_pkts(p_pkts, v_all_pkts[hkv])
        l_sum = np.maximum(p_unnorm.sum(axis=-1, keepdims=True), 1e-12).astype(np.float32)
        attn_fq_raw[:, h_i, :] = pv_unnorm / l_sum

    attn_o_pkts = quantize_rtl_vec(attn_fq_raw.reshape(m_tokens, D_MODEL))
    attn_fq = dequant_pkts_csd(attn_o_pkts).reshape(m_tokens, H_Q, D_V)

    res1_fq = x_fp16_in + matmul_log8_csd_pkts(attn_o_pkts, wo_pkts)
    rms2_fq = 1.0 / np.sqrt(np.mean(res1_fq ** 2, axis=1, keepdims=True) + eps)
    n2_fq   = res1_fq * rms2_fq * g2_fp16
    n2_pkts = quantize_rtl_vec(n2_fq)
    g_fq    = matmul_log8_csd_pkts(n2_pkts, wg_pkts)
    u_fq    = matmul_log8_csd_pkts(n2_pkts, wu_pkts)
    h_pkts  = quantize_rtl_vec((g_fq / (1.0 + np.exp(-g_fq))) * u_fq)
    final_fq = res1_fq + matmul_log8_csd_pkts(h_pkts, wd_pkts)

    # Diagnostic checks (non-intrusive):
    # (a) Per-operand CSD vs Paired CSD delta on Q_proj & Final
    q_proj_perop = dequant_pkts_csd(xn_pkts) @ dequant_pkts_csd(wq_pkts).T
    sq_csd_q, cs_csd_q, _ = metrics(q_proj_fq, q_proj_perop)
    # (b) Offline Exp-Free Softmax vs Online + 32-entry RECIP_LUT delta on Attn O
    attn_online_pkts = quantize_rtl_vec(eval_attn_online_rtl_sim(sc_all_fq, v_all_pkts, valid_mask_attn, GQA_GRP).reshape(m_tokens, D_MODEL))
    attn_online_fq = dequant_pkts_csd(attn_online_pkts).reshape(m_tokens, H_Q, D_V)
    sq_onl_a, cs_onl_a, mx_onl_a = metrics(attn_online_fq, attn_fq)
    # (c) FP32 Sigmoid vs u_sfu 4-segment PWL Sigmoid delta on Final
    h_pkts_pwl = quantize_rtl_vec((g_fq * sfu_sigmoid_pwl_np(g_fq)) * u_fq)
    final_fq_pwl = res1_fq + matmul_log8_csd_pkts(h_pkts_pwl, wd_pkts)
    sq_pwl_f, cs_pwl_f, mx_pwl_f = metrics(final_fq_pwl, final_fq)
    sq_pwl_32, cs_pwl_32, mx_pwl_32 = metrics(final_fq_pwl, final_f32)

    stage_pairs = [
        ("Stage 1: RMSNorm 1",           xn_f32.astype(np.float16).astype(np.float32), xn_f32),
        ("Stage 2: Q + RoPE + LogQ",     q_fq,       q_rope),
        ("Stage 3: K + RoPE + LogQ",     k_fq,       k_rope),
        ("Stage 4: V + Trans + LogQ",    v_cur_fq,   v_h),
        ("Stage 5: Exp-Free Attn O",     attn_fq,    attn_out),
        ("Stage 6: GEMM O + Res1",       res1_fq,    res1_f32),
        ("Stage 7: RMSNorm 2",           n2_fq,      n2_f32),
        ("Stage 8: FFN + Res2 (Final)",  final_fq,   final_f32),
    ]
    print("    --- PyFQ vs. FP32 Stage Summary ---")
    pyfq_summary = {"mode": args.mode, "m_tokens": m_tokens, "n_kv": n_kv, "n_past": n_past, "stages": {}, "diagnostics": {}}
    csv_rows = []
    for st_name, st_fq, st_fp32 in stage_pairs:
        sq, cs, mx = metrics(st_fq, st_fp32)
        entry = {"sqnr_db": round(sq, 4), "cos_sim": round(cs, 6), "max_abs": round(mx, 6)}
        extra = ""
        if (not is_decode) and (n_past == 0) and (m_tokens > 1):
            sq_nb, cs_nb, mx_nb = metrics(st_fq[1:], st_fp32[1:])
            entry["no_bos_sqnr_db"] = round(sq_nb, 4)
            entry["no_bos_cos_sim"] = round(cs_nb, 6)
            entry["no_bos_max_abs"] = round(mx_nb, 6)
            extra = f" | no-BOS: {sq_nb:6.2f} dB / {cs_nb:.6f} / {mx_nb:.4f}"
        pyfq_summary["stages"][st_name] = entry
        csv_rows.append([st_name, entry["sqnr_db"], entry["cos_sim"], entry["max_abs"],
                         entry.get("no_bos_sqnr_db", ""), entry.get("no_bos_cos_sim", ""), entry.get("no_bos_max_abs", "")])
        print(f"    [PyFQ vs FP32] {st_name:<28s} : SQNR={sq:6.2f} dB | Cos={cs:.6f} | MaxAbs={mx:.4f}{extra}")

    pyfq_summary["diagnostics"] = {
        "q_proj_paired_vs_perop_csd": {"sqnr_db": round(sq_csd_q, 4), "cos_sim": round(cs_csd_q, 6)},
        "attn_o_online_recip_vs_offline_pyfq": {"sqnr_db": round(sq_onl_a, 4), "cos_sim": round(cs_onl_a, 6), "max_abs": round(mx_onl_a, 6)},
        "final_pwl_vs_fp32_sigmoid_pyfq": {"sqnr_db": round(sq_pwl_f, 4), "cos_sim": round(cs_pwl_f, 6), "max_abs": round(mx_pwl_f, 6)},
        "final_pwl_vs_fp32_true": {"sqnr_db": round(sq_pwl_32, 4), "cos_sim": round(cs_pwl_32, 6), "max_abs": round(mx_pwl_32, 6)},
    }
    print(f"    [DIAG-PY] Q paired-CSD vs per-op CSD     : SQNR={sq_csd_q:6.2f} dB | Cos={cs_csd_q:.6f}")
    print(f"    [DIAG-PY] AttnO online+RECIP_LUT vs PyFQ : SQNR={sq_onl_a:6.2f} dB | Cos={cs_onl_a:.6f} | MaxAbs={mx_onl_a:.4f}")
    print(f"    [DIAG-PY] Final PWL-Sig vs FP32-Sig PyFQ : SQNR={sq_pwl_f:6.2f} dB | Cos={cs_pwl_f:.6f} | MaxAbs={mx_pwl_f:.4f} (vs FP32: {sq_pwl_32:.2f} dB)")

    with open(os.path.join(out_dir, "pyfq_vs_fp32.json"), "w") as fj:
        json.dump(pyfq_summary, fj, indent=2)
    with open(os.path.join(out_dir, "pyfq_vs_fp32.csv"), "w", newline="") as fc:
        w = csv.writer(fc)
        w.writerow(["stage", "sqnr_db", "cos_sim", "max_abs", "no_bos_sqnr_db", "no_bos_cos_sim", "no_bos_max_abs"])
        w.writerows(csv_rows)

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

    # Golden files (existing FP16 + new FP32 intermediate stage references)
    write_fp16_hex(os.path.join(out_dir, "g_xnorm.hex"), xn_f32)
    write_fp16_hex(os.path.join(out_dir, "g_res1.hex"), res1_f32)
    write_fp16_hex(os.path.join(out_dir, "g_n2.hex"), n2_f32)
    write_fp16_hex(os.path.join(out_dir, "g_final_fp16.hex"), final_f32)
    write_fp32_hex(os.path.join(out_dir, "g_final_fp32.hex"), final_f32)
    write_fp16_hex(os.path.join(out_dir, "g_res1_fq.hex"), res1_fq)
    write_fp16_hex(os.path.join(out_dir, "g_n2_fq.hex"), n2_fq)
    write_fp16_hex(os.path.join(out_dir, "g_final_fq.hex"), final_fq)

    write_fp32_hex(os.path.join(out_dir, "g_xnorm_fp32.hex"), xn_f32)
    write_fp32_hex(os.path.join(out_dir, "g_q_fp32.hex"), q_rope)
    write_fp32_hex(os.path.join(out_dir, "g_q_fq.hex"), q_fq)
    write_fp32_hex(os.path.join(out_dir, "g_k_fp32.hex"), k_rope)
    write_fp32_hex(os.path.join(out_dir, "g_k_fq.hex"), k_fq)
    write_fp32_hex(os.path.join(out_dir, "g_v_fp32.hex"), v_h)
    write_fp32_hex(os.path.join(out_dir, "g_v_fq.hex"), v_cur_fq)
    write_fp32_hex(os.path.join(out_dir, "g_attn_o_fp32.hex"), attn_out)
    write_fp32_hex(os.path.join(out_dir, "g_attn_o_fq.hex"), attn_fq)
    write_fp32_hex(os.path.join(out_dir, "g_res1_fp32.hex"), res1_f32)
    write_fp32_hex(os.path.join(out_dir, "g_n2_fp32.hex"), n2_f32)

    # Past KV files in Separate-KV format (matches cfg_attn_sep_kv_i = 1)
    # K order: [H_KV][kj][kp][l]  (4*16 words per kj)
    # V order: [H_KV][DV_TILES][kj][l] (16 words per (dvi, kj))
    if is_decode:
        k_all_perm = np.zeros((n_kv, H_KV, D_K), dtype=np.float32)
        k_all_perm[:n_kv - 1] = k_p_perm
        k_p_pkts = quantize_rtl_vec(k_all_perm.reshape(n_kv * H_KV, D_K)) \
            .reshape(kv_tiles, 16, H_KV, K_PACKETS_QK, 9) \
            .transpose(2, 0, 3, 1, 4)  # (H_KV, kv_tiles, 4, 16, 9)
        v_p_pkts = v_all_pkts_flat \
            .reshape(H_KV, DV_TILES, 16, kv_tiles, 9) \
            .transpose(0, 1, 3, 2, 4)  # (H_KV, DV_TILES, kv_tiles, 16, 9)
        write_9b_pkts_hex(os.path.join(out_dir, "k_past_words.hex"), k_p_pkts)
        write_9b_pkts_hex(os.path.join(out_dir, "v_past_words.hex"), v_p_pkts)
        v_tail_words = pack_fp16_72b_words(v_p_raw[n_kv - 16:n_kv - 1].reshape(15, H_KV * D_V))
        write_72b_words_hex(os.path.join(out_dir, "v_tail_past_fp16.hex"), v_tail_words)
    elif kv_tiles_past > 0:
        k_p_pkts = k_p_pkts_perm.transpose(1, 0, 2, 3) \
            .reshape(H_KV, kv_tiles_past, 16, K_PACKETS_QK, 9) \
            .transpose(0, 1, 3, 2, 4)  # (H_KV, kv_tiles_past, 4, 16, 9)
        v_p_pkts = v_p_pkts_flat \
            .reshape(H_KV, DV_TILES, 16, kv_tiles_past, 9) \
            .transpose(0, 1, 3, 2, 4)  # (H_KV, DV_TILES, kv_tiles_past, 16, 9)
        write_9b_pkts_hex(os.path.join(out_dir, "k_past_words.hex"), k_p_pkts)
        write_9b_pkts_hex(os.path.join(out_dir, "v_past_words.hex"), v_p_pkts)

    print(f"    Generated in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
