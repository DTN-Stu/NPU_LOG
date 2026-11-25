#!/usr/bin/env python3
"""
Generate the 3 IEEE 2-Column 300-DPI Publication Figures for Section V:
  1. fig_attn_bw_sensitivity.{pdf,png}       (M=256, BW in {64,32,16,8}, 4 locked modes)
  2. fig_attn_energy_breakdown.{pdf,png}     (M=64..1024, 4 locked modes, Logic/SRAM/DRAM)
  3. fig_full_layer_energy_breakdown.{pdf,png} (3 Edge LLMs + Geo. Mean @ M=1024, 4 configs)

Strictly reads from:
  - sim/attn_comparison_results.csv
  - 08_system_models/results/attn_4modes_energy_summary.csv
  - 08_system_models/results/full_layer_energy_summary.csv
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RTL_CSV = ROOT / "sim" / "attn_comparison_results.csv"
ATTN_ENERGY_CSV = ROOT / "08_system_models" / "results" / "attn_4modes_energy_summary.csv"
FULL_ENERGY_CSV = ROOT / "08_system_models" / "results" / "full_layer_energy_summary.csv"
OUT_DIR = ROOT / "08_system_models" / "results"

# IEEE publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colorblind-safe IEEE palette
C_LOGIC = "#2B5B84"   # Deep steel blue
C_SRAM  = "#E69F00"   # Amber/gold
C_DRAM  = "#C0392B"   # Muted crimson
C_DRAM_W = "#D95F02"  # Burnt orange (GEMM DRAM)
C_DRAM_A = "#9E1B32"  # Dark crimson (Attn DRAM)


def plot_fig1_bw_sensitivity():
    """Fig 1: Normalized Latency vs DRAM Bandwidth (M=N_KV=256, RTL-verified)."""
    rows = {}
    with open(RTL_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m, n, bw = int(r["M"]), int(r["N_KV"]), int(r["DRAM_BW"])
            mode = r["mode"].strip()
            if m == 256 and n == 256 and mode in ("NAIVE", "ONLINE_2A", "FLASH_RING", "FLASH_KVPIN"):
                rows[(mode, bw)] = int(r["cyc_total_w_sfu"])

    bws = [64, 32, 16, 8]
    x = np.arange(len(bws))
    base_kvpin_64 = rows[("FLASH_KVPIN", 64)]

    styles = [
        ("NAIVE",       "NAIVE (Unfused 2-Pass)",      "#7F8C8D", "s", "--"),
        ("ONLINE_2A",   "ONLINE_2A (Unfused Online)",  "#E69F00", "^", "-."),
        ("FLASH_RING",  "FLASH_RING (Fused, No Pin)",  "#C0392B", "D", ":"),
        ("FLASH_KVPIN", "FLASH_KVPIN (Ours, Pinned)",  "#1B7837", "o", "-"),
    ]

    fig, ax = plt.subplots(figsize=(3.55, 2.45), dpi=300)
    for mode, label, color, marker, ls in styles:
        ys = [rows[(mode, bw)] / base_kvpin_64 for bw in bws]
        lw = 2.0 if mode == "FLASH_KVPIN" else 1.4
        ms = 5.5 if mode == "FLASH_KVPIN" else 4.5
        ax.plot(x, ys, label=label, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=ms)
        # Annotate endpoints at BW=8
        if mode == "FLASH_KVPIN":
            ax.annotate(f"{ys[-1]:.3f}× (+0.23%)", xy=(x[-1], ys[-1]), xytext=(x[-1]-1.25, ys[-1]-0.18),
                        fontsize=7.2, fontweight="bold", color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
        elif mode == "FLASH_RING":
            ax.annotate(f"{ys[-1]:.2f}×", xy=(x[-1], ys[-1]), xytext=(x[-1]-0.38, ys[-1]+0.06),
                        fontsize=7.2, fontweight="bold", color=color)
        elif mode == "NAIVE":
            ax.annotate(f"{ys[-1]:.2f}×", xy=(x[-1], ys[-1]), xytext=(x[-1]-0.62, ys[-1]+0.02),
                        fontsize=7.0, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{bw} B/cyc\n({bw*0.5:.0f} GB/s)" for bw in bws])
    ax.set_xlabel("Off-Chip DRAM Bandwidth @ 500 MHz")
    ax.set_ylabel("Normalized Cycles (vs. KVPIN @ 64 B/cyc)")
    ax.set_title("Attention Bandwidth Sensitivity ($M = N_{KV} = 256$, RTL)", pad=6)
    ax.set_ylim(0.75, 2.20)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_attn_bw_sensitivity.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_attn_bw_sensitivity.{pdf,png}")


def plot_fig2_attn_energy_breakdown():
    """Fig 2: Attention System Energy Breakdown (M=64..1024, LPDDR4X 16 pJ/bit)."""
    data = {}
    with open(ATTN_ENERGY_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dram_std"] == "LPDDR4X" and int(r["DRAM_BW"]) == 64:
                data[(r["mode"], int(r["M"]))] = {
                    "logic": float(r["e_logic_uJ"]) / 1000.0,  # mJ
                    "sram":  float(r["e_sram_uJ"]) / 1000.0,
                    "dram":  float(r["e_dram_uJ"]) / 1000.0,
                    "total": float(r["e_total_uJ"]) / 1000.0,
                    "red_x": float(r["energy_reduction_x_vs_naive"]),
                }

    ms = [64, 128, 256, 512, 1024]
    modes = ["NAIVE", "ONLINE_2A", "FLASH_RING", "FLASH_KVPIN"]
    mode_short = ["N", "O", "R", "K*"]

    fig, axes = plt.subplots(1, 5, figsize=(7.16, 2.45), dpi=300, sharey=False)
    for idx, (ax, m) in enumerate(zip(axes, ms)):
        x = np.arange(len(modes))
        w = 0.62
        logic = np.array([data[(md, m)]["logic"] for md in modes])
        sram  = np.array([data[(md, m)]["sram"]  for md in modes])
        dram  = np.array([data[(md, m)]["dram"]  for md in modes])
        totals = logic + sram + dram

        ax.bar(x, logic, w, color=C_LOGIC, label="Compute Logic (500 MHz SAIF)")
        ax.bar(x, sram,  w, bottom=logic, color=C_SRAM, label="288-KB 72b SRAM")
        ax.bar(x, dram,  w, bottom=logic + sram, color=C_DRAM, label="Off-Chip DRAM (LPDDR4X)")

        # Annotate KVPIN energy reduction vs NAIVE
        kv_tot = totals[3]
        red_x = data[("FLASH_KVPIN", m)]["red_x"]
        ax.text(3, kv_tot + 0.04 * max(totals), f"{red_x:.1f}×\nless",
                ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#1B7837")

        ax.set_xticks(x)
        ax.set_xticklabels(mode_short, fontsize=7.5)
        ax.set_title(f"$M = N_{{KV}} = {m}$", fontsize=8.2, pad=4)
        ax.set_ylim(0, max(totals) * 1.24)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        if idx == 0:
            ax.set_ylabel("Attention Energy (mJ)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03),
               frameon=True, framealpha=0.95, edgecolor="#CCCCCC")
    fig.text(0.5, -0.02, "Modes: N = NAIVE, O = ONLINE_2A, R = FLASH_RING, K* = FLASH_KVPIN (Ours)",
             ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_attn_energy_breakdown.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_attn_energy_breakdown.{pdf,png}")


def plot_fig3_full_layer_energy_breakdown():
    """Fig 3: Full-Layer System Energy Breakdown on 3 Edge LLMs + Geo. Mean (M=1024, LPDDR4X)."""
    models = ["Llama-3.2-1B", "SmolLM2-1.7B", "Qwen3-1.7B-Base"]
    configs = ["FP16_Unfused", "NVFP4_Unfused", "Log8_FLASH_RING", "Log8_FLASH_KVPIN"]
    cfg_labels = ["FP16\nUnfused", "NVFP4\nUnfused", "Log8\nRING", "Log8\nKVPIN*"]

    data = {}
    with open(FULL_ENERGY_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dram_std"] == "LPDDR4X" and int(r["M"]) == 1024:
                data[(r["model"], r["config"])] = {
                    "logic":     float(r["e_logic_mJ"]),
                    "sram":      float(r["e_sram_mJ"]),
                    "dram_gemm": float(r["e_dram_gemm_mJ"]),
                    "dram_attn": float(r["e_dram_attn_mJ"]),
                    "total":     float(r["e_total_mJ"]),
                    "norm_fp16": float(r["norm_vs_fp16"]),
                }

    # Compute normalized components relative to FP16_Unfused = 1.00 per model + Geo. Mean
    panel_names = models + ["Geo. Mean"]
    fig, ax = plt.subplots(figsize=(7.16, 2.65), dpi=300)

    n_groups = len(panel_names)
    n_bars = len(configs)
    group_gap = 1.0
    bar_w = 0.19
    x_centers = np.arange(n_groups) * (n_bars * bar_w + 0.35)

    for b_idx, cfg in enumerate(configs):
        xs = x_centers + (b_idx - (n_bars - 1) / 2.0) * bar_w
        l_vals, s_vals, dg_vals, da_vals, tot_vals = [], [], [], [], []
        for m in models:
            base = data[(m, "FP16_Unfused")]["total"]
            d = data[(m, cfg)]
            l_vals.append(d["logic"] / base)
            s_vals.append(d["sram"] / base)
            dg_vals.append(d["dram_gemm"] / base)
            da_vals.append(d["dram_attn"] / base)
            tot_vals.append(d["total"] / base)
        # Geometric mean of normalized total, split by average component share
        geo_tot = math.exp(sum(math.log(v) for v in tot_vals) / len(tot_vals))
        avg_l = np.mean([l / t for l, t in zip(l_vals, tot_vals)]) * geo_tot
        avg_s = np.mean([s / t for s, t in zip(s_vals, tot_vals)]) * geo_tot
        avg_dg = np.mean([g / t for g, t in zip(dg_vals, tot_vals)]) * geo_tot
        avg_da = np.mean([a / t for a, t in zip(da_vals, tot_vals)]) * geo_tot

        l_arr = np.array(l_vals + [avg_l])
        s_arr = np.array(s_vals + [avg_s])
        dg_arr = np.array(dg_vals + [avg_dg])
        da_arr = np.array(da_vals + [avg_da])
        t_arr = l_arr + s_arr + dg_arr + da_arr

        lbl_l  = "Compute Logic (500 MHz SAIF)" if b_idx == 0 else None
        lbl_s  = "288-KB 72b SRAM" if b_idx == 0 else None
        lbl_dg = "DRAM: GEMM Weights/Act" if b_idx == 0 else None
        lbl_da = "DRAM: Attention (Q/K/V/S/P)" if b_idx == 0 else None

        ax.bar(xs, l_arr,  bar_w * 0.88, color=C_LOGIC,  edgecolor="black", linewidth=0.35, label=lbl_l)
        ax.bar(xs, s_arr,  bar_w * 0.88, bottom=l_arr, color=C_SRAM, edgecolor="black", linewidth=0.35, label=lbl_s)
        ax.bar(xs, dg_arr, bar_w * 0.88, bottom=l_arr + s_arr, color=C_DRAM_W, edgecolor="black", linewidth=0.35, label=lbl_dg)
        ax.bar(xs, da_arr, bar_w * 0.88, bottom=l_arr + s_arr + dg_arr, color=C_DRAM_A, edgecolor="black", linewidth=0.35, label=lbl_da)

        # Bar labels on top
        for xi, ti in zip(xs, t_arr):
            if cfg == "Log8_FLASH_KVPIN":
                ax.text(xi, ti + 0.025, f"{1.0/ti:.1f}×", ha="center", va="bottom",
                        fontsize=7.0, fontweight="bold", color="#1B7837")
            elif cfg == "FP16_Unfused":
                ax.text(xi, ti + 0.02, "1.00", ha="center", va="bottom", fontsize=6.5, color="#333333")
            else:
                ax.text(xi, ti + 0.02, f"{ti:.2f}", ha="center", va="bottom", fontsize=6.5, color="#444444")

        # Config short label under each bar
        for xi in xs:
            ax.text(xi, -0.04, ["FP16", "NV4", "RING", "KVP*"][b_idx],
                    ha="center", va="top", fontsize=6.3, rotation=0)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(panel_names, fontsize=8.5, fontweight="bold")
    ax.tick_params(axis="x", pad=14)
    ax.set_ylabel("Normalized Full-Layer Energy\n(vs. FP16 Unfused = 1.00)")
    ax.set_title("Full-Layer System Energy Breakdown Across Edge LLMs ($M = N_{KV} = 1024$, LPDDR4X 16 pJ/bit)", pad=26)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
              frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=7.0, columnspacing=1.0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_full_layer_energy_breakdown.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_full_layer_energy_breakdown.{pdf,png}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_fig1_bw_sensitivity()
    plot_fig2_attn_energy_breakdown()
    plot_fig3_full_layer_energy_breakdown()
