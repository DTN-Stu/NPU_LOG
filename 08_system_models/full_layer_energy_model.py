#!/usr/bin/env python3
"""
Workstream 4C: Full Transformer Layer System Energy Breakdown Model.
Strictly follows 08_system_models/SYSTEM_ENERGY_EXPERIMENT_PLAN.md (WS4C).

Mandatory verification against LLAMA3_1B_PREFILL_REPORT.md (M=16):
  - Weight bytes (4.5-bit): 34,209,792 B (EXACT 0 B error)
  - Prefill cycles (M=16):  4,479,514 cyc (< 2.0% error)

Evaluates 3 Edge LLMs (Llama-3.2-1B, SmolLM2-1.7B, Qwen3-1.7B-Base) at
M = N_KV in {256, 512, 1024} across 4 system configs:
  1. FP16 Unfused        (16-bit W/A/KV/S/P, NAIVE unfused attention)
  2. NVFP4 Unfused       (4.5-bit W/A/KV, ONLINE_2A unfused attention, C2/C5 PE power)
  3. Log8-FP4 FLASH_RING (4.5-bit W/A/KV/P4, fused FlashAttention Ring KV)
  4. Log8-FP4 FLASH_KVPIN(4.5-bit W/A/KV/P4, fused FlashAttention GQA KV-Pinned)

Outputs:
  - 08_system_models/results/full_layer_energy_summary.csv
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "08_system_models" / "energy_params.json"
OUT_CSV = ROOT / "08_system_models" / "results" / "full_layer_energy_summary.csv"

# Verified from HuggingFace config.json & PAPER_BLUEPRINT_TCAD_TVLSI.md
MODELS = {
    "Llama-3.2-1B": {
        "d_model": 2048, "d_ff": 8192, "h_q": 32, "h_kv": 8, "d_k": 64, "n_layers": 16,
    },
    "SmolLM2-1.7B": {
        "d_model": 2048, "d_ff": 8192, "h_q": 32, "h_kv": 32, "d_k": 64, "n_layers": 24,
    },
    "Qwen3-1.7B-Base": {
        "d_model": 2048, "d_ff": 6144, "h_q": 16, "h_kv": 8, "d_k": 128, "n_layers": 28,
    },
}

# TCDM partition capacity (Banks 16..31 = 144 KB for ping-pong A + O)
TCDM_AO_CAP_BYTES = 144 * 1024  # 147,456 B


def layer_weight_params(cfg):
    d = cfg["d_model"]
    dff = cfg["d_ff"]
    dq = cfg["h_q"] * cfg["d_k"]
    dkv = cfg["h_kv"] * cfg["d_k"]
    # W_Q, W_K, W_V, W_O, W_gate, W_up, W_down
    return (dq * d) + (dkv * d) + (dkv * d) + (d * dq) + 3 * (dff * d)


def gemm_weight_passes(M, d_k_in, d_n_out, bytes_per_elem):
    """4-Loop Output-Stationary GEMM: number of weight streaming passes over DRAM
    given 144 KB Banks 16..31 ping-pong A/O budget.
    Each M_sub chunk holds 2 * (M_sub * K * bpe) + 2 * (M_sub * 16 * bpe) <= 144 KB
    (actually O drains per N-tile in 16-col stripes, so A double-buffer dominates).
    """
    # Conservative 4-Loop OS: A double-buffer (2 * M_sub * K * bpe) <= 128 KB (leaving 16 KB for O drain)
    max_m_sub = max(16, int((128 * 1024) // (2 * d_k_in * bytes_per_elem)))
    # Round down to multiple of 16 tokens (M_tile = 16)
    max_m_sub = max(16, (max_m_sub // 16) * 16)
    passes = math.ceil(M / max_m_sub)
    return passes


def eval_gemm_layer(cfg, M, bytes_per_elem):
    """Returns (dram_w_rd_bytes, dram_act_bytes, tcdm_rd_words, tcdm_wr_words, gemm_cycles, sfu_cycles)."""
    d = cfg["d_model"]
    dff = cfg["d_ff"]
    dq = cfg["h_q"] * cfg["d_k"]
    dkv = cfg["h_kv"] * cfg["d_k"]

    # 7 Linear projections: (name, N_out, K_in, rtl_mac_eff)
    # Efficiencies directly from LLAMA3_1B_PREFILL_REPORT.md Section 5.2
    projs = [
        ("Q",    dq,  d,   0.9214),
        ("K",    dkv, d,   0.9346),
        ("V",    dkv, d,   0.5858),
        ("O",    d,   dq,  0.9928),
        ("Gate", dff, d,   0.8934),
        ("Up",   dff, d,   0.8934),
        ("Down", d,   dff, 0.9965),
    ]

    dram_w_rd = 0
    dram_act_rw = 0
    tcdm_rd_w = 0
    tcdm_wr_w = 0
    gemm_cyc = 0.0

    word_bytes = 9.0  # 72-bit TCDM word
    for name, n_out, k_in, eta in projs:
        w_bytes = int(round(n_out * k_in * bytes_per_elem))
        passes = gemm_weight_passes(M, k_in, n_out, bytes_per_elem)
        dram_w_rd += w_bytes * passes

        # Input activation read + output activation write (1x per projection if spilled,
        # but for fused Log8-FP4 most stay in TCDM; conservatively count X_in + O_out DRAM for large M)
        a_bytes = int(round(M * k_in * bytes_per_elem))
        o_bytes = int(round(M * n_out * bytes_per_elem))
        if M >= 64:
            # At M >= 64, layer activations exceed 144 KB so inter-projection tensors stream via DRAM
            dram_act_rw += a_bytes + o_bytes

        # On-chip TCDM reads: 16 weight words + 1 act word per 256-MAC cycle
        mac_cycles = (M * n_out * k_in) / 256.0
        gemm_cyc += mac_cycles / eta
        # TCDM words: A read (ceil(M/16)*N_tiles*K_blks*16) + W read (passes*W_words) + O write
        # Exact 256-MAC array reads 17 words per MAC cycle when running 4.5-bit (16 W + 1 A)
        scale_w = bytes_per_elem / (9.0 / 16.0)
        tcdm_rd_w += int(round(mac_cycles * 17.0 * scale_w))
        tcdm_wr_w += int(round((w_bytes * passes + a_bytes + o_bytes) / word_bytes))

    # SFU + Requant + Residual cycles calibrated from LLAMA3_1B_PREFILL_REPORT.md (M=16, D=2048 -> 334,475 cyc)
    sfu_cyc = 334475.0 * (M / 16.0) * (d / 2048.0)
    return dram_w_rd, dram_act_rw, tcdm_rd_w, tcdm_wr_w, gemm_cyc, sfu_cyc


def eval_attention(cfg, M, attn_mode, bytes_per_elem):
    """Exact 4-mode attention traffic & cycle model generalized to (H_q, H_kv, D_k)."""
    h_q = cfg["h_q"]
    h_kv = cfg["h_kv"]
    d_k = cfg["d_k"]
    n_kv = M

    q_bytes = int(round(h_q * M * d_k * bytes_per_elem))
    k_bytes_1x = int(round(h_kv * n_kv * d_k * bytes_per_elem))
    v_bytes_1x = int(round(h_kv * n_kv * d_k * bytes_per_elem))
    o_bytes = int(round(h_q * M * d_k * bytes_per_elem))

    # S is FP16 (2 B/elem), P is FP4-Log8 (9/16 B/elem) or FP16 (2 B/elem) for FP16 Unfused
    s_bytes = h_q * M * n_kv * 2
    p_bytes = int(round(h_q * M * n_kv * bytes_per_elem))

    q_tiles_per_head = math.ceil(M / 8)  # M_Q = 8
    dv_tiles = math.ceil(d_k / 16)       # 16 lanes

    # Ideal MAC cycles for QK^T + PV (256 MACs/cyc)
    ideal_mac_cyc = (2.0 * h_q * M * n_kv * d_k) / 256.0
    # Reference Llama-3.2-1B has H_q=32, D_k=64 -> scale factor relative to H_q=32, D_k=64
    scale_ref = (h_q * d_k) / (32.0 * 64.0)

    if attn_mode == "NAIVE":
        # Q read 1x, K read G*(M/8)x (or per-head), S written 1x + read 2x (2-pass softmax), P written 1x + read 1x, V read
        # Exact formula validated to 0 B on Llama-3.2-1B:
        dram_rd = q_bytes + (h_q * n_kv * d_k * bytes_per_elem) + (h_q * n_kv * d_k * bytes_per_elem) + 2 * s_bytes + p_bytes
        dram_wr = o_bytes + s_bytes + p_bytes
        # Cycles calibrated from RTL NAIVE (M=256: hw=3,446,796, sfu=1,048,576)
        cyc_hw = (51.2 * M * M + 420.0 * M) * scale_ref
        cyc_sfu = (16.0 * M * M) * (h_q / 32.0)
        tcdm_rd_w = int(round(ideal_mac_cyc * 17.0 + (2 * s_bytes + p_bytes) / 9.0))
        tcdm_wr_w = int(round((dram_rd + dram_wr) / 9.0))
    elif attn_mode == "ONLINE_2A":
        # Online softmax reads S 1x instead of 2x
        dram_rd = q_bytes + (h_q * n_kv * d_k * bytes_per_elem) + (h_q * n_kv * d_k * bytes_per_elem) + s_bytes + p_bytes
        dram_wr = o_bytes + s_bytes + p_bytes
        cyc_hw = (48.9 * M * M + 380.0 * M) * scale_ref
        cyc_sfu = (8.0 * M * M) * (h_q / 32.0)
        tcdm_rd_w = int(round(ideal_mac_cyc * 17.0 + (s_bytes + p_bytes) / 9.0))
        tcdm_wr_w = int(round((dram_rd + dram_wr) / 9.0))
    elif attn_mode == "FLASH_RING":
        # Zero S/P spill! K re-read per (H_q, Q_tile, dv_tile), V re-read per (H_q, Q_tile)
        k_ring_rd = h_q * q_tiles_per_head * dv_tiles * int(round(n_kv * d_k * bytes_per_elem))
        v_ring_rd = h_q * q_tiles_per_head * int(round(n_kv * d_k * bytes_per_elem))
        dram_rd = q_bytes + k_ring_rd + v_ring_rd
        dram_wr = o_bytes
        # FlashAttention runs 2.5x MACs (Q*K^T recomputed per dv_tile)
        cyc_hw = (58.5 * M * M + 150.0 * M) * scale_ref
        cyc_sfu = 0.0
        tcdm_rd_w = int(round(ideal_mac_cyc * 2.5 * 17.0))
        tcdm_wr_w = int(round((dram_rd + dram_wr) / 9.0))
    elif attn_mode == "FLASH_KVPIN":
        # Zero S/P spill AND K, V read EXACTLY 1x per KV-head (pinned in Banks 0..15, 144 KB)!
        # Verify 1 KV-head (K+V) fits in 144 KB:
        kv_head_bytes = 2 * n_kv * d_k * bytes_per_elem
        assert kv_head_bytes <= TCDM_AO_CAP_BYTES, f"KV head {kv_head_bytes} B > 144 KB!"
        dram_rd = q_bytes + k_bytes_1x + v_bytes_1x
        dram_wr = o_bytes
        cyc_hw = (57.9 * M * M + 59.0 * M) * scale_ref
        cyc_sfu = 0.0
        tcdm_rd_w = int(round(ideal_mac_cyc * 2.5 * 17.0))
        tcdm_wr_w = int(round((dram_rd + dram_wr) / 9.0))
    else:
        raise ValueError(attn_mode)

    return int(round(dram_rd)), int(round(dram_wr)), tcdm_rd_w, tcdm_wr_w, cyc_hw, cyc_sfu


def main():
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        params = json.load(f)

    f_clk = float(params["eval_freq_hz"])
    p_gemm_mw = float(params["logic_power_mw"]["gemm_active"])
    p_attn_mw = float(params["logic_power_mw"]["attn_fused_active"])
    p_sfu_mw = float(params["logic_power_mw"]["sfu_active"])
    pe_c2_ratio = float(params["logic_power_mw"]["pe_c2_nvfp4_ratio"])
    mac_power_frac = float(params["logic_power_mw"].get("mac_core_power_fraction", 0.35))

    e_sram_rd_pj = float(params["sram_72b"]["e_rd_pJ_per_word"])
    e_sram_wr_pj = float(params["sram_72b"]["e_wr_pJ_per_word"])
    p_sram_leak_mw = float(params["sram_72b"]["p_leak_mw_32banks"])

    e_lpddr4x = float(params["dram_offchip"]["e_dram_pJ_per_bit"])
    e_lpddr5 = float(params["dram_offchip"]["sensitivity_lpddr5_pJ_per_bit"])

    # -------------------------------------------------------------------------
    # MANDATORY ASSERTION vs LLAMA3_1B_PREFILL_REPORT.md (M = 16)
    # -------------------------------------------------------------------------
    llama_cfg = MODELS["Llama-3.2-1B"]
    w_bytes_llama = int(round(layer_weight_params(llama_cfg) * (9.0 / 16.0)))
    assert w_bytes_llama == 34209792, f"Weight bytes mismatch: {w_bytes_llama} != 34209792"

    _, _, _, _, g_cyc16, s_cyc16 = eval_gemm_layer(llama_cfg, 16, 9.0 / 16.0)
    _, _, _, _, a_cyc16, as_cyc16 = eval_attention(llama_cfg, 16, "FLASH_KVPIN", 9.0 / 16.0)
    tot_cyc16 = g_cyc16 + s_cyc16 + a_cyc16 + as_cyc16
    rtl_cyc16 = 4479514
    cyc_err_pct = abs(tot_cyc16 - rtl_cyc16) / rtl_cyc16 * 100.0
    print("=== Full-Layer RTL Validation (Llama-3.2-1B, M=16) ===")
    print(f"  Weight bytes: model={w_bytes_llama:,} B  vs  RTL=34,209,792 B  (error = 0 B)  [PASS]")
    print(f"  Total cycles: model={tot_cyc16:,.0f}  vs  RTL={rtl_cyc16:,}  (error = {cyc_err_pct:.3f}%)  [PASS < 2.0%]")
    assert cyc_err_pct < 2.0, f"Full-layer cycle error {cyc_err_pct:.2f}% >= 2.0%"

    # -------------------------------------------------------------------------
    # Evaluate 3 Edge LLMs x 3 Sequence Lengths x 4 System Configs x 2 DRAM Stds
    # -------------------------------------------------------------------------
    # Effective logic power scaling: only the MAC+Acc core portion scales by C2/C5 (1.3478x)
    # for NVFP4 (and conservatively 1.3478x lower-bound for FP16 Unfused).
    sys_scale_nvfp4 = (1.0 - mac_power_frac) + mac_power_frac * pe_c2_ratio

    CONFIGS = [
        ("FP16_Unfused",       2.0,      "NAIVE",       sys_scale_nvfp4),
        ("NVFP4_Unfused",      9.0 / 16, "ONLINE_2A",   sys_scale_nvfp4),
        ("Log8_FLASH_RING",    9.0 / 16, "FLASH_RING",  1.0),
        ("Log8_FLASH_KVPIN",   9.0 / 16, "FLASH_KVPIN", 1.0),
    ]

    rows_out = []
    for dram_std, e_dram_pj in [("LPDDR4X", e_lpddr4x), ("LPDDR5", e_lpddr5)]:
        for model_name, cfg in MODELS.items():
            for M in (256, 512, 1024):
                for cfg_name, bpe, attn_mode, logic_scale in CONFIGS:
                    w_rd, act_rw, g_sram_r, g_sram_w, g_cyc, s_cyc = eval_gemm_layer(cfg, M, bpe)
                    a_rd, a_wr, a_sram_r, a_sram_w, a_cyc, a_sfu = eval_attention(cfg, M, attn_mode, bpe)

                    dram_gemm_bytes = w_rd + act_rw
                    dram_attn_bytes = a_rd + a_wr
                    dram_total_bytes = dram_gemm_bytes + dram_attn_bytes

                    tot_cyc = g_cyc + s_cyc + a_cyc + a_sfu
                    t_exec_s = tot_cyc / f_clk

                    # Logic energy (mJ)
                    e_logic_gemm_mj = p_gemm_mw * logic_scale * (g_cyc / f_clk)
                    e_logic_sfu_mj = p_sfu_mw * ((s_cyc + a_sfu) / f_clk)
                    if attn_mode in ("FLASH_RING", "FLASH_KVPIN"):
                        e_logic_attn_mj = p_attn_mw * logic_scale * (a_cyc / f_clk)
                    else:
                        e_logic_attn_mj = p_gemm_mw * logic_scale * (a_cyc / f_clk)
                    e_logic_mj = e_logic_gemm_mj + e_logic_sfu_mj + e_logic_attn_mj

                    # SRAM energy (mJ): pJ * 1e-9 = mJ
                    e_sram_dyn_mj = ((g_sram_r + a_sram_r) * e_sram_rd_pj + (g_sram_w + a_sram_w) * e_sram_wr_pj) * 1e-9
                    e_sram_leak_mj = p_sram_leak_mw * t_exec_s
                    e_sram_mj = e_sram_dyn_mj + e_sram_leak_mj

                    # DRAM energy (mJ): bits * pJ/bit * 1e-9 = mJ
                    e_dram_gemm_mj = dram_gemm_bytes * 8 * e_dram_pj * 1e-9
                    e_dram_attn_mj = dram_attn_bytes * 8 * e_dram_pj * 1e-9
                    e_dram_mj = e_dram_gemm_mj + e_dram_attn_mj

                    e_total_mj = e_logic_mj + e_sram_mj + e_dram_mj
                    e_per_tok_uj = (e_total_mj * 1e3) / M

                    rows_out.append({
                        "dram_std": dram_std,
                        "model": model_name,
                        "M": M,
                        "config": cfg_name,
                        "dram_gemm_MB": round(dram_gemm_bytes / 1e6, 3),
                        "dram_attn_MB": round(dram_attn_bytes / 1e6, 3),
                        "dram_total_MB": round(dram_total_bytes / 1e6, 3),
                        "total_cycles": int(round(tot_cyc)),
                        "t_exec_ms": round(t_exec_s * 1e3, 4),
                        "e_logic_mJ": round(e_logic_mj, 4),
                        "e_sram_mJ": round(e_sram_mj, 4),
                        "e_dram_gemm_mJ": round(e_dram_gemm_mj, 4),
                        "e_dram_attn_mJ": round(e_dram_attn_mj, 4),
                        "e_dram_mJ": round(e_dram_mj, 4),
                        "e_total_mJ": round(e_total_mj, 4),
                        "e_per_token_uJ": round(e_per_tok_uj, 3),
                        "pct_logic": round(100.0 * e_logic_mj / e_total_mj, 2),
                        "pct_sram": round(100.0 * e_sram_mj / e_total_mj, 2),
                        "pct_dram": round(100.0 * e_dram_mj / e_total_mj, 2),
                    })

    # Normalize vs FP16_Unfused and NVFP4_Unfused
    base_fp16 = {(r["dram_std"], r["model"], r["M"]): r["e_total_mJ"] for r in rows_out if r["config"] == "FP16_Unfused"}
    base_nvfp4 = {(r["dram_std"], r["model"], r["M"]): r["e_total_mJ"] for r in rows_out if r["config"] == "NVFP4_Unfused"}
    for r in rows_out:
        k = (r["dram_std"], r["model"], r["M"])
        r["norm_vs_fp16"] = round(r["e_total_mJ"] / base_fp16[k], 4)
        r["reduction_x_vs_fp16"] = round(base_fp16[k] / r["e_total_mJ"], 2)
        r["reduction_x_vs_nvfp4"] = round(base_nvfp4[k] / r["e_total_mJ"], 2)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nWrote {len(rows_out)} rows -> {OUT_CSV}")
    print("\n=== Full-Layer Energy @ M=1024, LPDDR4X (16 pJ/bit) ===")
    print(f"{'Model':16s} {'Config':18s} {'DRAM(MB)':>9} {'Logic(mJ)':>10} {'SRAM(mJ)':>9} {'DRAM(mJ)':>10} {'Total(mJ)':>10} {'vs FP16':>8} {'vs NVFP4':>9}")
    for r in rows_out:
        if r["dram_std"] == "LPDDR4X" and r["M"] == 1024:
            print(f"{r['model']:16s} {r['config']:18s} {r['dram_total_MB']:9.2f} "
                  f"{r['e_logic_mJ']:10.2f} {r['e_sram_mJ']:9.2f} {r['e_dram_mJ']:10.2f} "
                  f"{r['e_total_mJ']:10.2f} {r['reduction_x_vs_fp16']:7.2f}x {r['reduction_x_vs_nvfp4']:8.2f}x")


if __name__ == "__main__":
    main()
