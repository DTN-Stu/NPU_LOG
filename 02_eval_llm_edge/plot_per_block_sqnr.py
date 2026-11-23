"""
Per-Block SQNR Analysis (Fig. 8 in IEEE TCAD/TVLSI Blueprint) for 02_eval_llm_edge.

Measures per-block (B=16) Signal-to-Quantization-Noise Ratio (SQNR in dB) across:
  (a) Block Dynamic Range Ratio (DR = max|x| / mean|x|)
  (b) Block-Max Octave Phase (phi = log2(max|x|) mod 1)

Reuses bit-exact quantizers from 02_eval_llm_edge/quantizers.py directly on real
Edge LLM tensors (e.g., NousResearch/Llama-3.2-1B, Qwen/Qwen2.5-0.5B), and also
supports fast offline preview from local .npz tensor dumps.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Quantizer Registry (Primary: sibling quantizers.py; Fallback: NumPy mirror)
# ---------------------------------------------------------------------------

E2M1_LEVELS_NP = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1_MIDPOINTS_NP = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
CSD_Q5_NP = np.array([32.0, 35.0, 38.0, 41.0, 46.0, 49.0, 54.0, 59.0], dtype=np.float32)


def _pad_to_block_np(x: np.ndarray, block_size: int) -> Tuple[np.ndarray, Tuple[int, ...], int]:
    orig_shape = x.shape
    last_dim = orig_shape[-1]
    pad = (block_size - (last_dim % block_size)) % block_size
    if pad > 0:
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, pad)]
        x = np.pad(x, pad_width, mode="constant", constant_values=0.0)
    return x.reshape(*orig_shape[:-1], -1, block_size), orig_shape, pad


def _unpad_from_block_np(blocked: np.ndarray, orig_shape: Tuple[int, ...], pad: int) -> np.ndarray:
    out = blocked.reshape(*orig_shape[:-1], -1)
    if pad > 0:
        out = out[..., : orig_shape[-1]]
    return out


def _nearest_e2m1_np(val: np.ndarray) -> np.ndarray:
    sign = np.where(val < 0.0, -1.0, 1.0).astype(np.float32)
    idx = np.searchsorted(E2M1_MIDPOINTS_NP, np.abs(val), side="left")
    q_mag = E2M1_LEVELS_NP[idx]
    return np.where(q_mag == 0.0, 0.0, q_mag * sign).astype(np.float32)


def _np_log8_fp4_csd(x: np.ndarray, block_size: int = 16) -> np.ndarray:
    """Bit-exact NumPy mirror of quantize_log8_fp4_csd() in quantizers.py."""
    blocked, orig_shape, pad = _pad_to_block_np(x.astype(np.float32), block_size)
    fp16 = blocked.astype(np.float16)
    bits = fp16.view(np.uint16).astype(np.int32)

    sign_bit = (bits >> 15) & 0x1
    exp_fp16 = (bits >> 10) & 0x1F
    man_top3 = (bits >> 7) & 0x7

    mu = (exp_fp16 << 3) | man_top3
    mu = np.where(exp_fp16 == 0, 0, mu)
    mu = np.where(exp_fp16 == 31, 247, mu)

    mu_max = np.max(mu, axis=-1, keepdims=True)
    scale_sat = np.clip(mu_max - 140, -128, 127)
    scale_sat = np.where(mu_max == 0, 0, scale_sat)

    diff = mu_max - mu
    mag_idx = np.zeros_like(diff)
    for k, th in enumerate((36, 24, 18, 14, 10, 6, 2), start=1):
        mag_idx = np.where(diff <= th, k, mag_idx)

    zero_elem = (mu == 0) | (mu_max == 0)
    mag_idx = np.where(zero_elem, 0, mag_idx)

    mag_val = E2M1_LEVELS_NP[mag_idx]
    sign_val = np.where((sign_bit == 1) & (mag_idx != 0), -1.0, 1.0).astype(np.float32)

    q_s = np.floor(scale_sat.astype(np.float32) / 8.0)
    r_s = scale_sat.astype(np.int64) & 7
    scale_val = np.power(2.0, q_s) * (CSD_Q5_NP[r_s] / 32.0)

    dequant = (sign_val * mag_val * scale_val).astype(np.float32)
    return _unpad_from_block_np(dequant, orig_shape, pad)


def _np_mxfp4(x: np.ndarray, block_size: int, scale_mode: str) -> np.ndarray:
    """NumPy mirror of quantize_mxfp4() in quantizers.py."""
    blocked, orig_shape, pad = _pad_to_block_np(x.astype(np.float32), block_size)
    max_abs = np.max(np.abs(blocked), axis=-1, keepdims=True)
    if scale_mode == "floor":
        safe_max = np.maximum(max_abs, 2.0 ** -125)
        exp_e8m0 = np.clip(np.floor(np.log2(safe_max)) - 2.0, -127.0, 127.0)
    else:
        safe_max = np.maximum(max_abs / 6.0, 2.0 ** -127)
        exp_e8m0 = np.clip(np.ceil(np.log2(safe_max)), -127.0, 127.0)
    scale = np.where(max_abs == 0.0, 1.0, np.power(2.0, exp_e8m0)).astype(np.float32)
    dequant = _nearest_e2m1_np(blocked / scale) * scale
    return _unpad_from_block_np(dequant, orig_shape, pad)


def _np_nvfp4(x: np.ndarray, block_size: int = 16) -> np.ndarray:
    """NumPy mirror of quantize_nvfp4() fallback in quantizers.py."""
    x_f = x.astype(np.float32)
    global_max = float(np.max(np.abs(x_f)))
    if global_max == 0.0:
        return x_f
    tensor_scale = max(global_max / (6.0 * 448.0), 1e-12)
    blocked, orig_shape, pad = _pad_to_block_np(x_f / tensor_scale, block_size)
    max_abs = np.max(np.abs(blocked), axis=-1, keepdims=True)
    blk_scale_fp32 = np.maximum(max_abs / 6.0, 1e-12)
    exp = np.floor(np.log2(blk_scale_fp32))
    mant = blk_scale_fp32 / np.power(2.0, exp)
    blk_scale_fp8 = np.rint(mant * 8.0) / 8.0 * np.power(2.0, exp)
    blk_scale_fp8 = np.where(max_abs == 0.0, 1.0, blk_scale_fp8).astype(np.float32)
    dequant = _nearest_e2m1_np(blocked / blk_scale_fp8) * blk_scale_fp8 * tensor_scale
    return _unpad_from_block_np(dequant, orig_shape, pad)


def build_quantizer_specs(use_torch: bool, include_g32: bool = False) -> List[Dict[str, object]]:
    """Return ordered plot/eval specifications for each format."""
    if use_torch:
        script_dir = str(Path(__file__).resolve().parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from quantizers import (
            quantize_log8_fp4_csd,
            quantize_mxfp4_g16_ceil,
            quantize_mxfp4_g16_floor,
            quantize_mxfp4_g32_floor,
            quantize_nvfp4,
        )

        fn_map = {
            "mxfp4_g32_floor": quantize_mxfp4_g32_floor,
            "mxfp4_g16_floor": quantize_mxfp4_g16_floor,
            "mxfp4_g16_ceil": quantize_mxfp4_g16_ceil,
            "nvfp4": quantize_nvfp4,
            "log8_fp4_csd": quantize_log8_fp4_csd,
        }
    else:
        fn_map = {
            "mxfp4_g32_floor": lambda a: _np_mxfp4(a, 32, "floor"),
            "mxfp4_g16_floor": lambda a: _np_mxfp4(a, 16, "floor"),
            "mxfp4_g16_ceil": lambda a: _np_mxfp4(a, 16, "ceil"),
            "nvfp4": lambda a: _np_nvfp4(a, 16),
            "log8_fp4_csd": lambda a: _np_log8_fp4_csd(a, 16),
        }

    specs: List[Dict[str, object]] = []
    if include_g32:
        specs.append({
            "key": "mxfp4_g32_floor",
            "label": "MXFP4 g32 (OCP, 4.25b)",
            "fn": fn_map["mxfp4_g32_floor"],
            "color": "#6c757d",
            "linestyle": "--",
            "marker": "x",
            "lw": 1.35,
            "zorder": 2,
        })

    specs.extend([
        {
            "key": "mxfp4_g16_floor",
            "label": "MXFP4 g16 floor (4.50b)",
            "fn": fn_map["mxfp4_g16_floor"],
            "color": "#d95f02",
            "linestyle": "-.",
            "marker": "s",
            "lw": 1.45,
            "zorder": 3,
        },
        {
            "key": "mxfp4_g16_ceil",
            "label": "MXFP4 g16 ceil (4.50b)",
            "fn": fn_map["mxfp4_g16_ceil"],
            "color": "#1b9e77",
            "linestyle": ":",
            "marker": "^",
            "lw": 1.6,
            "zorder": 3,
        },
        {
            "key": "nvfp4",
            "label": "NVFP4 (E4M3, 4.50b)",
            "fn": fn_map["nvfp4"],
            "color": "#1f77b4",
            "linestyle": "-",
            "marker": "o",
            "lw": 1.65,
            "zorder": 4,
        },
        {
            "key": "log8_fp4_csd",
            "label": "Log8-FP4 CSD (Ours, 4.50b)",
            "fn": fn_map["log8_fp4_csd"],
            "color": "#c62828",
            "linestyle": "-",
            "marker": "D",
            "lw": 2.0,
            "zorder": 5,
        },
    ])
    return specs


# ---------------------------------------------------------------------------
# Real Tensor Collection (--model HuggingFace pass OR --npz slice preview)
# ---------------------------------------------------------------------------

FALLBACK_TEXT = (
    "In transformer-based large language models deployed on edge NPUs, linear projections "
    "and scaled dot-product attention dominate both off-chip memory traffic and on-chip MAC "
    "energy. Block microscaling formats such as MXFP4 suffer severe dynamic-range clipping "
    "or under-utilization whenever block maxima cross power-of-two octave boundaries. "
) * 32


def collect_tensors_from_hf(
    model_id: str,
    device: str,
    layers: List[int],
    tensor_source: str,
    seq_len: int,
) -> List[Tuple[str, "torch.Tensor"]]:
    """Capture real activation/weight tensors from a single forward pass."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HF model {model_id} on {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(device)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        ).to(device)
    model.eval()

    text = FALLBACK_TEXT
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        joined = "\n\n".join(t for t in ds["text"][:120] if t.strip())
        if len(joined) > 200:
            text = joined
    except Exception as exc:
        print(f"[INFO] Using built-in sample text (datasets unavailable: {exc}).")

    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(device)

    captured: List[Tuple[str, torch.Tensor]] = []
    target_projs = {"q_proj", "o_proj", "gate_proj", "down_proj"}
    layer_set = set(layers)
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parts = name.split(".")
        proj_name = parts[-1]
        if proj_name not in target_projs:
            continue
        layer_idx = None
        for idx, token in enumerate(parts):
            if token == "layers" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                layer_idx = int(parts[idx + 1])
                break
        if layer_idx is None or layer_idx not in layer_set:
            continue

        if tensor_source in ("weights", "both"):
            captured.append((f"L{layer_idx}.{proj_name}.weight", module.weight.detach()))

        if tensor_source in ("activations", "both"):
            def _make_hook(tag: str):
                def _pre_hook(_mod, inputs):
                    if inputs and isinstance(inputs[0], torch.Tensor):
                        captured.append((tag, inputs[0].detach()))
                return _pre_hook

            hooks.append(module.register_forward_pre_hook(_make_hook(f"L{layer_idx}.{proj_name}.act")))

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    print(f"Captured {len(captured)} real tensors from layers {sorted(layer_set)} (seq_len={input_ids.shape[1]}).")
    return captured


def collect_tensors_from_npz(
    npz_path: Path,
    tensor_source: str,
    npz_keys: Optional[List[str]],
) -> List[Tuple[str, np.ndarray]]:
    """Load real activation/weight matrices from a local .npz dump."""
    data = np.load(npz_path)
    selected: List[Tuple[str, np.ndarray]] = []
    for k in data.files:
        arr = data[k]
        if arr.ndim < 2:
            continue
        if npz_keys is not None:
            if k not in npz_keys:
                continue
        else:
            k_low = k.lower()
            is_act = k_low.startswith("x") or "act" in k_low
            is_wt = k_low.startswith("w")
            if tensor_source == "activations" and not is_act:
                continue
            if tensor_source == "weights" and not is_wt:
                continue
        selected.append((k, arr.astype(np.float32)))

    if not selected:
        raise ValueError(f"No matching 2D tensors found in {npz_path} for tensor_source={tensor_source!r}.")
    print(f"Loaded {len(selected)} tensors from {npz_path}: {[k for k, _ in selected]}.")
    return selected


# ---------------------------------------------------------------------------
# Per-Block Metric Computation & Binning
# ---------------------------------------------------------------------------

def _to_blocks16_np(t, block_size: int = 16) -> np.ndarray:
    if HAS_TORCH and isinstance(t, torch.Tensor):
        arr = t.detach().float().cpu().numpy()
    else:
        arr = np.asarray(t, dtype=np.float32)
    blocked, _, _ = _pad_to_block_np(arr, block_size)
    return blocked.reshape(-1, block_size)


def compute_per_block_metrics(
    tensors: List[Tuple[str, object]],
    specs: List[Dict[str, object]],
    max_blocks: int,
    seed: int,
    sqnr_cap_db: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Compute DR, octave phase, and SQNR (dB) on identical 16-element blocks."""
    dr_chunks: List[np.ndarray] = []
    phase_chunks: List[np.ndarray] = []
    sqnr_chunks: Dict[str, List[np.ndarray]] = {str(s["key"]): [] for s in specs}

    for _name, orig_t in tensors:
        x_blk = _to_blocks16_np(orig_t, 16)
        sig_pow = np.sum(x_blk * x_blk, axis=1)
        abs_x = np.abs(x_blk)
        max_abs = np.max(abs_x, axis=1)
        mean_abs = np.mean(abs_x, axis=1)

        valid = (sig_pow > 1e-12) & (max_abs > 1e-7) & (mean_abs > 1e-12)
        if not np.any(valid):
            continue

        x_valid = x_blk[valid]
        sig_valid = sig_pow[valid]
        dr_valid = max_abs[valid] / (mean_abs[valid] + 1e-12)
        phase_valid = np.mod(np.log2(max_abs[valid]), 1.0)

        dr_chunks.append(dr_valid)
        phase_chunks.append(phase_valid)

        for spec in specs:
            key = str(spec["key"])
            q_fn: Callable = spec["fn"]  # type: ignore[assignment]
            q_t = q_fn(orig_t)
            q_blk = _to_blocks16_np(q_t, 16)[valid]
            err_pow = np.sum((x_valid - q_blk) ** 2, axis=1)
            sqnr = 10.0 * np.log10(sig_valid / np.maximum(err_pow, 1e-12))
            sqnr_chunks[key].append(np.minimum(sqnr, sqnr_cap_db))

    if not dr_chunks:
        raise RuntimeError("No active non-zero 16-element blocks found in collected tensors.")

    dr_all = np.concatenate(dr_chunks)
    phase_all = np.concatenate(phase_chunks)
    sqnr_all = {k: np.concatenate(v) for k, v in sqnr_chunks.items()}

    n_total = dr_all.shape[0]
    if n_total > max_blocks:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(n_total, size=max_blocks, replace=False))
        dr_all = dr_all[sel]
        phase_all = phase_all[sel]
        sqnr_all = {k: v[sel] for k, v in sqnr_all.items()}
        print(f"Subsampled {max_blocks:,} / {n_total:,} active blocks (seed={seed}).")
    else:
        print(f"Analyzed all {n_total:,} active 16-element blocks.")

    return dr_all, phase_all, sqnr_all


def bin_sqnr_statistics(
    x_vals: np.ndarray,
    sqnr_by_fmt: Dict[str, np.ndarray],
    edges: np.ndarray,
    min_bin_count: int,
) -> Dict[str, object]:
    """Bin 1-D x_vals and compute mean, p25, p75 SQNR per format."""
    n_bins = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(x_vals, edges) - 1
    bin_idx = np.where((x_vals >= edges[0]) & (x_vals <= edges[-1]), np.clip(bin_idx, 0, n_bins - 1), -1)

    counts = np.array([int(np.sum(bin_idx == b)) for b in range(n_bins)])
    valid_mask = counts >= min_bin_count

    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for key, arr in sqnr_by_fmt.items():
        means = np.full(n_bins, np.nan)
        p25s = np.full(n_bins, np.nan)
        p75s = np.full(n_bins, np.nan)
        for b in range(n_bins):
            if not valid_mask[b]:
                continue
            vals = arr[bin_idx == b]
            means[b] = float(np.mean(vals))
            p25s[b] = float(np.percentile(vals, 25))
            p75s[b] = float(np.percentile(vals, 75))
        stats[key] = {"mean": means, "p25": p25s, "p75": p75s}

    return {"centers": centers, "counts": counts, "valid": valid_mask, "stats": stats}


# ---------------------------------------------------------------------------
# CSV Export & IEEE Double-Column Plot
# ---------------------------------------------------------------------------

def save_csv_outputs(
    out_dir: Path,
    specs: List[Dict[str, object]],
    sqnr_all: Dict[str, np.ndarray],
    dr_binned: Dict[str, object],
    phase_binned: Dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Global summary CSV
    with open(out_dir / "sqnr_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["format", "mean_sqnr_db", "median_sqnr_db", "p10_sqnr_db", "p90_sqnr_db", "num_blocks"])
        for spec in specs:
            k = str(spec["key"])
            v = sqnr_all[k]
            writer.writerow([
                k,
                f"{np.mean(v):.3f}",
                f"{np.median(v):.3f}",
                f"{np.percentile(v, 10):.3f}",
                f"{np.percentile(v, 90):.3f}",
                len(v),
            ])

    # 2. Binned tables (DR and Octave Phase)
    for fname, x_col, binned in (
        ("sqnr_vs_dr.csv", "dr_center", dr_binned),
        ("sqnr_vs_octave_phase.csv", "octave_phase_center", phase_binned),
    ):
        headers = [x_col, "block_count"]
        for spec in specs:
            k = str(spec["key"])
            headers.extend([f"{k}_mean", f"{k}_p25", f"{k}_p75"])
        centers = binned["centers"]  # type: ignore[index]
        counts = binned["counts"]    # type: ignore[index]
        valid = binned["valid"]      # type: ignore[index]
        stats = binned["stats"]      # type: ignore[index]

        with open(out_dir / fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for b in range(len(centers)):
                if not valid[b]:
                    continue
                row = [f"{centers[b]:.4f}", int(counts[b])]
                for spec in specs:
                    k = str(spec["key"])
                    s = stats[k]
                    row.extend([f"{s['mean'][b]:.3f}", f"{s['p25'][b]:.3f}", f"{s['p75'][b]:.3f}"])
                writer.writerow(row)


def render_figure(
    out_dir: Path,
    specs: List[Dict[str, object]],
    dr_binned: Dict[str, object],
    phase_binned: Dict[str, object],
) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), sharey=True)
    panel_cfg = [
        (axes[0], dr_binned, r"Block Dynamic Range Ratio ($\max|x| / \mathrm{mean}|x|$)", "(a) SQNR vs. Block Dynamic Range"),
        (axes[1], phase_binned, r"Block-Max Octave Phase ($\log_2(\max|x|)\ \mathrm{mod}\ 1$)", "(b) SQNR vs. Octave Phase Boundary"),
    ]

    for ax, binned, xlabel, subtitle in panel_cfg:
        valid = binned["valid"]      # type: ignore[index]
        x = binned["centers"][valid] # type: ignore[index]
        stats = binned["stats"]      # type: ignore[index]

        for spec in specs:
            k = str(spec["key"])
            y_mean = stats[k]["mean"][valid]
            y_p25 = stats[k]["p25"][valid]
            y_p75 = stats[k]["p75"][valid]
            ax.plot(
                x, y_mean,
                label=str(spec["label"]),
                color=str(spec["color"]),
                linestyle=str(spec["linestyle"]),
                marker=str(spec["marker"]),
                markersize=3.6,
                linewidth=float(spec["lw"]),
                zorder=int(spec["zorder"]),
            )
            if k in ("log8_fp4_csd", "mxfp4_g16_ceil", "nvfp4"):
                ax.fill_between(x, y_p25, y_p75, color=str(spec["color"]), alpha=0.10, zorder=1)

        ax.set_xlabel(xlabel)
        ax.set_title(subtitle, pad=4)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

    axes[0].set_ylabel("Per-Block SQNR (dB)")
    axes[1].set_xlim(0.0, 1.0)

    # Mark exact MXFP4 1.5 * 2^k threshold at phi = log2(1.5) ~= 0.585
    phi_b = float(np.log2(1.5))
    axes[1].axvline(phi_b, color="#444444", linestyle="--", linewidth=0.95, alpha=0.75, zorder=2)

    p_valid = phase_binned["valid"]  # type: ignore[index]
    px = phase_binned["centers"][p_valid]  # type: ignore[index]
    py_floor = phase_binned["stats"]["mxfp4_g16_floor"]["mean"][p_valid]  # type: ignore[index]
    py_ceil = phase_binned["stats"]["mxfp4_g16_ceil"]["mean"][p_valid]    # type: ignore[index]

    y_lo, y_hi = axes[1].get_ylim()
    y_span = max(y_hi - y_lo, 1.0)
    axes[1].set_ylim(y_lo - 0.14 * y_span, y_hi + 0.06 * y_span)
    y_lo, y_hi = axes[1].get_ylim()
    y_span = y_hi - y_lo

    axes[1].text(
        phi_b - 0.02, y_lo + 0.05 * y_span,
        r"$\phi=\log_2(1.5)\approx 0.585$",
        ha="right", va="bottom",
        fontsize=7.2, color="#444444",
    )

    if len(px) > 0 and not np.all(np.isnan(py_floor)):
        min_i = int(np.nanargmin(py_floor))
        x_clip = float(px[min_i])
        y_clip = float(py_floor[min_i])
        tx_clip = max(0.06, x_clip - 0.36) if x_clip > 0.55 else min(0.60, x_clip + 0.10)
        axes[1].annotate(
            r"MXFP4 floor clipping",
            xy=(x_clip, y_clip),
            xytext=(tx_clip, y_clip + 0.05 * y_span),
            arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#d95f02"},
            fontsize=7.2,
            color="#d95f02",
        )

    post_mask = px > phi_b
    if np.any(post_mask) and not np.all(np.isnan(py_ceil[post_mask])):
        post_idx = np.where(post_mask)[0]
        target_post_i = int(post_idx[min(1, len(post_idx) - 1)])
        x_ceil = float(px[target_post_i])
        y_ceil = float(py_ceil[target_post_i])
        tx_ceil = max(0.04, phi_b - 0.03)
        axes[1].annotate(
            r"MXFP4 ceil scale-up",
            xy=(x_ceil, y_ceil),
            xytext=(tx_ceil, y_lo + 0.18 * y_span),
            ha="right",
            arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#1b9e77"},
            fontsize=7.2,
            color="#1b9e77",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=2 if len(specs) == 4 else 3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.5,
        handlelength=2.2,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))

    fig.savefig(out_dir / "fig8_per_block_sqnr.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig8_per_block_sqnr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-Block SQNR Analysis (Fig. 8) for SNAX Log8-FP4")
    parser.add_argument("--model", type=str, default="NousResearch/Llama-3.2-1B", help="HF model ID or local path")
    parser.add_argument("--npz", type=str, default=None, help="Optional local .npz tensor file for instant offline run")
    parser.add_argument("--npz_keys", nargs="+", default=None, help="Specific array keys inside --npz")
    parser.add_argument("--tensor_source", type=str, default="activations", choices=["activations", "weights", "both"])
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 1, 7, 15], help="Layer indices for --model hooks")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length for single forward pass")
    parser.add_argument("--device", type=str, default="cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
    parser.add_argument("--num_bins", type=int, default=16, help="Number of bins for DR and octave phase")
    parser.add_argument("--min_bin_count", type=int, default=15, help="Minimum blocks required to plot a bin")
    parser.add_argument("--max_blocks", type=int, default=100000, help="Uniform block subsample cap for speed")
    parser.add_argument("--include_g32", action="store_true", help="Include OCP MXFP4 g32 (4.25b) curve alongside B=16 (4.50b) formats")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results/fig8_sqnr")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.npz is not None:
        if not HAS_TORCH:
            print("[INFO] PyTorch not installed; using bit-exact NumPy mirror of quantizers.py for --npz preview.")
        specs = build_quantizer_specs(use_torch=HAS_TORCH, include_g32=args.include_g32)
        tensors_raw = collect_tensors_from_npz(Path(args.npz), args.tensor_source, args.npz_keys)
        tensors = [
            (k, torch.from_numpy(v) if HAS_TORCH else v)
            for k, v in tensors_raw
        ]
    else:
        if not HAS_TORCH:
            raise SystemExit(
                "Error: PyTorch/transformers are required for --model mode. "
                "Install `02_eval_llm_edge/requirements.txt` or pass `--npz ../sim/sim_snax_llm_layer_mid/llama32_1b_layer1_slice.npz`."
            )
        specs = build_quantizer_specs(use_torch=True, include_g32=args.include_g32)
        tensors = collect_tensors_from_hf(args.model, args.device, args.layers, args.tensor_source, args.seq_len)

    dr_all, phase_all, sqnr_all = compute_per_block_metrics(
        tensors=tensors,
        specs=specs,
        max_blocks=args.max_blocks,
        seed=args.seed,
    )

    dr_lo = max(1.0, float(np.percentile(dr_all, 0.5)))
    dr_hi = float(np.percentile(dr_all, 99.0))
    if dr_hi <= dr_lo + 0.25:
        dr_hi = dr_lo + 1.0
    dr_edges = np.linspace(dr_lo, dr_hi, args.num_bins + 1)
    phase_edges = np.linspace(0.0, 1.0, args.num_bins + 1)

    dr_binned = bin_sqnr_statistics(dr_all, sqnr_all, dr_edges, args.min_bin_count)
    phase_binned = bin_sqnr_statistics(phase_all, sqnr_all, phase_edges, args.min_bin_count)

    save_csv_outputs(out_dir, specs, sqnr_all, dr_binned, phase_binned)
    render_figure(out_dir, specs, dr_binned, phase_binned)

    print("\n=== Per-Block SQNR Summary ===")
    for spec in specs:
        k = str(spec["key"])
        v = sqnr_all[k]
        print(f"  {k:<16s} | mean={np.mean(v):6.2f} dB | median={np.median(v):6.2f} dB | p10={np.percentile(v, 10):6.2f} dB")
    print(f"\nSaved figure & CSVs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
