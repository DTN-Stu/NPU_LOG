#!/usr/bin/env python3
"""
Workstream 4B: Attention 4-Mode System Energy Breakdown Calculator.
Strictly follows 08_system_models/SYSTEM_ENERGY_EXPERIMENT_PLAN.md (WS4B).

Reads ONLY:
  - 08_system_models/energy_params.json
  - sim/attn_comparison_results.csv
  - 08_system_models/results/tcdm_dram_4modes.csv

Outputs:
  - 08_system_models/results/attn_4modes_energy_summary.csv
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "08_system_models" / "energy_params.json"
RTL_CSV = ROOT / "sim" / "attn_comparison_results.csv"
TILING_CSV = ROOT / "08_system_models" / "results" / "tcdm_dram_4modes.csv"
OUT_CSV = ROOT / "08_system_models" / "results" / "attn_4modes_energy_summary.csv"

LOCKED_MODES = ["NAIVE", "ONLINE_2A", "FLASH_RING", "FLASH_KVPIN"]


def fit_quadratic_cycles(pts):
    """Exact quadratic fit a*M^2 + b*M + c through 3 points (M=64, 128, 256),
    plus 2-point (M=64, 128) leave-one-out check against M=256."""
    (m1, y1), (m2, y2), (m3, y3) = pts
    # 2-point check: y = a2*M^2 + b2*M fitted on (m1, m2), tested on m3
    det2 = m1 * m1 * m2 - m2 * m2 * m1
    a2 = (y1 * m2 - y2 * m1) / det2
    b2 = (m1 * m1 * y2 - m2 * m2 * y1) / det2
    pred_m3 = a2 * m3 * m3 + b2 * m3
    err_pct = abs(pred_m3 - y3) / y3 * 100.0

    # Exact 3-point fit a*M^2 + b*M + c using Lagrange / Vandermonde
    # Solve: [m1^2 m1 1; m2^2 m2 1; m3^2 m3 1] [a b c]^T = [y1 y2 y3]^T
    d12 = (y2 - y1) / (m2 - m1)
    d23 = (y3 - y2) / (m3 - m2)
    a = (d23 - d12) / (m3 - m1)
    b = d12 - a * (m1 + m2)
    c = y1 - a * m1 * m1 - b * m1
    return (a, b, c), err_pct


def main():
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        params = json.load(f)

    f_clk = float(params["eval_freq_hz"])
    p_gemm_mw = float(params["logic_power_mw"]["gemm_active"])
    p_attn_mw = float(params["logic_power_mw"]["attn_fused_active"])
    p_sfu_mw = float(params["logic_power_mw"]["sfu_active"])

    e_sram_rd_pj = float(params["sram_72b"]["e_rd_pJ_per_word"])
    e_sram_wr_pj = float(params["sram_72b"]["e_wr_pJ_per_word"])
    p_sram_leak_mw = float(params["sram_72b"]["p_leak_mw_32banks"])

    e_lpddr4x = float(params["dram_offchip"]["e_dram_pJ_per_bit"])
    e_lpddr5 = float(params["dram_offchip"]["sensitivity_lpddr5_pJ_per_bit"])

    # Load RTL rows (strictly excluding ONLINE_2B)
    rtl_map = {}
    with open(RTL_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mode = row["mode"].strip()
            if mode not in LOCKED_MODES:
                continue
            assert int(row["checks_fail"]) == 0, f"RTL fail in {row}"
            assert int(row["cpu_mem_cyc"]) == 0, f"CPU mem leak in {row}"
            key = (mode, int(row["M"]), int(row["N_KV"]), int(row["DRAM_BW"]))
            rtl_map[key] = row

    # Verify all 16 M=256 BW sweep points exist in RTL CSV
    for mode in LOCKED_MODES:
        for bw in (64, 32, 16, 8):
            assert (mode, 256, 256, bw) in rtl_map, f"Missing RTL row: {(mode, 256, 256, bw)}"

    # Load 0-Byte-verified tiling traffic table (M in {64, 128, 256, 512, 1024})
    tiling_map = {}
    with open(TILING_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mode = row["mode"].strip()
            if mode not in LOCKED_MODES:
                continue
            key = (mode, int(row["M"]), int(row["N_KV"]))
            tiling_map[key] = {k: int(v) for k, v in row.items() if k != "mode"}

    # Fit cycle extrapolation models at BW=64 for M=512, 1024 and verify < 2.0% LOO error
    cyc_hw_models = {}
    print("=== Cycle Extrapolation Model Validation (BW=64, 2-pt fit M=64,128 -> predict M=256) ===")
    for mode in LOCKED_MODES:
        pts = [(m, int(rtl_map[(mode, m, m, 64)]["cyc_hw"])) for m in (64, 128, 256)]
        coeffs, err_pct = fit_quadratic_cycles(pts)
        cyc_hw_models[mode] = coeffs
        print(f"  {mode:12s}: LOO error @ M=256 = {err_pct:.3f}%  (a={coeffs[0]:.4f}, b={coeffs[1]:.2f}, c={coeffs[2]:.1f})")
        assert err_pct < 2.0, f"Cycle extrapolation LOO error too high for {mode}: {err_pct:.3f}%"

    # Build output rows:
    # 1) Primary scaling sweep: M in {64, 128, 256, 512, 1024} at BW=64
    # 2) Bandwidth sensitivity sweep: M=256 at BW in {32, 16, 8}
    eval_points = []
    for m in (64, 128, 256, 512, 1024):
        for mode in LOCKED_MODES:
            eval_points.append((mode, m, m, 64))
    for bw in (32, 16, 8):
        for mode in LOCKED_MODES:
            eval_points.append((mode, 256, 256, bw))

    rows_out = []
    for dram_std, e_dram_pj in [("LPDDR4X", e_lpddr4x), ("LPDDR5", e_lpddr5)]:
        for mode, m, n_kv, bw in eval_points:
            t = tiling_map[(mode, m, n_kv)]
            if (mode, m, n_kv, bw) in rtl_map:
                r = rtl_map[(mode, m, n_kv, bw)]
                cyc_hw = int(r["cyc_hw"])
                p2_sfu = int(r["p2_sfu_est"])
                cyc_tot = int(r["cyc_total_w_sfu"])
                cyc_src = "RTL_EXACT"
                # Cross-check 0-Byte error against tiling model
                assert int(r["rd_bytes"]) == t["dram_rd_bytes"]
                assert int(r["wr_bytes"]) == t["dram_wr_bytes"]
            else:
                a, b, c = cyc_hw_models[mode]
                cyc_hw = int(round(a * m * m + b * m + c))
                if mode == "NAIVE":
                    p2_sfu = 16 * m * m
                elif mode == "ONLINE_2A":
                    p2_sfu = 8 * m * m
                else:
                    p2_sfu = 0
                cyc_tot = cyc_hw + p2_sfu
                cyc_src = "CAL_MODEL"

            t_exec_s = cyc_tot / f_clk
            t_hw_s = cyc_hw / f_clk
            t_sfu_s = p2_sfu / f_clk

            # Energy in microjoules (uJ):
            # P (mW) * T (s) = mJ -> * 1000 = uJ
            # pJ -> * 1e-6 = uJ
            if mode in ("FLASH_RING", "FLASH_KVPIN"):
                e_logic_uj = p_attn_mw * t_exec_s * 1e3
            else:
                e_logic_uj = (p_gemm_mw * t_hw_s + p_sfu_mw * t_sfu_s) * 1e3

            e_sram_dyn_uj = (t["tcdm_rd_words"] * e_sram_rd_pj + t["tcdm_wr_words"] * e_sram_wr_pj) * 1e-6
            e_sram_leak_uj = p_sram_leak_mw * t_exec_s * 1e3
            e_sram_uj = e_sram_dyn_uj + e_sram_leak_uj

            e_dram_uj = (t["dram_rd_bytes"] + t["dram_wr_bytes"]) * 8 * e_dram_pj * 1e-6
            e_total_uj = e_logic_uj + e_sram_uj + e_dram_uj

            rows_out.append({
                "dram_std": dram_std,
                "e_dram_pJ_per_bit": e_dram_pj,
                "mode": mode,
                "M": m,
                "N_KV": n_kv,
                "DRAM_BW": bw,
                "cyc_source": cyc_src,
                "cyc_hw": cyc_hw,
                "p2_sfu_est": p2_sfu,
                "cyc_total_w_sfu": cyc_tot,
                "t_exec_ms": round(t_exec_s * 1e3, 6),
                "dram_rd_bytes": t["dram_rd_bytes"],
                "dram_wr_bytes": t["dram_wr_bytes"],
                "dram_total_bytes": t["dram_total_bytes"],
                "tcdm_rd_words": t["tcdm_rd_words"],
                "tcdm_wr_words": t["tcdm_wr_words"],
                "e_logic_uJ": round(e_logic_uj, 4),
                "e_sram_dyn_uJ": round(e_sram_dyn_uj, 4),
                "e_sram_leak_uJ": round(e_sram_leak_uj, 4),
                "e_sram_uJ": round(e_sram_uj, 4),
                "e_dram_uJ": round(e_dram_uj, 4),
                "e_total_uJ": round(e_total_uj, 4),
                "pct_logic": round(100.0 * e_logic_uj / e_total_uj, 2),
                "pct_sram": round(100.0 * e_sram_uj / e_total_uj, 2),
                "pct_dram": round(100.0 * e_dram_uj / e_total_uj, 2),
            })

    # Add Normalized Energy vs NAIVE at same (dram_std, M, N_KV, DRAM_BW)
    naive_lookup = {
        (r["dram_std"], r["M"], r["N_KV"], r["DRAM_BW"]): r["e_total_uJ"]
        for r in rows_out if r["mode"] == "NAIVE"
    }
    for r in rows_out:
        base = naive_lookup[(r["dram_std"], r["M"], r["N_KV"], r["DRAM_BW"])]
        r["norm_energy_vs_naive"] = round(r["e_total_uJ"] / base, 4)
        r["energy_reduction_x_vs_naive"] = round(base / r["e_total_uJ"], 2)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nWrote {len(rows_out)} rows -> {OUT_CSV}")
    print("\n=== Summary @ BW=64 (LPDDR4X 16 pJ/bit vs LPDDR5 9 pJ/bit) ===")
    for std in ("LPDDR4X", "LPDDR5"):
        print(f"\n[{std}]")
        print(f"{'M':>5} {'Mode':12s} {'Cyc':>10} {'Logic(uJ)':>10} {'SRAM(uJ)':>10} {'DRAM(uJ)':>10} {'Total(uJ)':>10} {'vs NAIVE':>9}")
        for r in rows_out:
            if r["dram_std"] == std and r["DRAM_BW"] == 64:
                print(f"{r['M']:5d} {r['mode']:12s} {r['cyc_total_w_sfu']:10d} "
                      f"{r['e_logic_uJ']:10.2f} {r['e_sram_uJ']:10.2f} {r['e_dram_uJ']:10.2f} "
                      f"{r['e_total_uJ']:10.2f} {r['energy_reduction_x_vs_naive']:8.2f}x")


if __name__ == "__main__":
    main()
