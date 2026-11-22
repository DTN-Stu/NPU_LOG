"""
Parse ASAP7 7nm DC Synthesis & SAIF Power reports from paper_syn/reports/
and print clean Markdown tables for the paper.
"""
import re
from pathlib import Path

CONFIGS = [
    ("C1: MXFP4 + FP32 + FF",           "cfg1_mxfp4_g16_fp32_ff",    "u_ff_acc"),
    ("C2: NVFP4 + FP32 + FF",           "cfg2_nvfp4_fp32_ff",        "u_ff_acc"),
    ("C3: Log8 + FP32 + FF",            "cfg6_log8_fp32_ff",         "u_ff_acc"),
    ("C4: Log8 + INT32 Lazy + FF",      "cfg4_log8_int32_lazy_ff",   "u_core/u_acc"),
    ("C5: Log8 + INT32 Lazy + D-Latch", "cfg5_log8_int32_lazy_latch","u_core/u_acc"),
]

TOPS = 0.256  # 256 MACs/cycle * 2 OPs/MAC * 0.5 GHz


def parse_area(path: Path) -> dict:
    txt = path.read_text(errors="ignore")
    total = re.search(r"Total cell area:\s+([\d.]+)", txt)
    comb  = re.search(r"Combinational area:\s+([\d.]+)", txt)
    seq   = re.search(r"Noncombinational area:\s+([\d.]+)", txt)
    return {
        "total": float(total.group(1)) if total else 0.0,
        "comb":  float(comb.group(1))  if comb  else 0.0,
        "seq":   float(seq.group(1))   if seq   else 0.0,
    }


def parse_hier_area(path: Path, inst: str) -> float:
    txt = path.read_text(errors="ignore")
    m = re.search(rf"^\s*{re.escape(inst)}\s+([\d.]+)", txt, re.MULTILINE)
    return float(m.group(1)) if m else 0.0


def parse_qor_timing(path: Path) -> tuple[float, float]:
    txt = path.read_text(errors="ignore")
    wns = re.search(r"Critical Path Slack:\s+([-+]?[\d.]+)", txt)
    tns = re.search(r"Total Negative Slack:\s+([-+]?[\d.]+)", txt)
    return (
        float(wns.group(1)) if wns else float("nan"),
        float(tns.group(1)) if tns else float("nan"),
    )


def parse_power_mw(path: Path, top_name: str) -> dict:
    if not path.exists():
        return {"total": 0.0, "int": 0.0, "sw": 0.0, "leak": 0.0}
    txt = path.read_text(errors="ignore")
    dyn_u = re.search(r"Dynamic Power Units\s*=\s*1(\S+)", txt)
    leak_u = re.search(r"Leakage Power Units\s*=\s*1(\S+)", txt)
    unit_map = {"W": 1e3, "mW": 1.0, "uW": 1e-3, "nW": 1e-6, "pW": 1e-9}
    d_scale = unit_map.get(dyn_u.group(1), 1.0) if dyn_u else 1.0
    l_scale = unit_map.get(leak_u.group(1), 1e-9) if leak_u else 1e-9

    # Top-level hierarchy line: "<top_name>  <sw>  <int>  <leak>  <total>  100.0"
    m = re.search(
        rf"^\s*{re.escape(top_name)}\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+100\.0",
        txt,
        re.MULTILINE,
    )
    if not m:
        return {"total": 0.0, "int": 0.0, "sw": 0.0, "leak": 0.0}
    return {
        "sw":    float(m.group(1)) * d_scale,
        "int":   float(m.group(2)) * d_scale,
        "leak":  float(m.group(3)) * l_scale,
        "total": float(m.group(4)) * d_scale,
    }


def main():
    rep_dir = Path(__file__).resolve().parent / "reports"
    print("### 1. Full-Chip PPA (500 MHz, ASAP7 7nm RVT TT, Real Llama-3.2-1B & Uniform SAIF)")
    print("| Config | WNS (ps) | TNS (ps) | Comb (um2) | Seq (um2) | Total Area (um2) | Acc Area (um2) | Datapath Area (um2) | Pwr Real (mW) | Pwr Unif (mW) | Area Eff (TOPS/mm2) | Pwr Eff Real (TOPS/W) | Comp. Eff |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for label, top, acc_inst in CONFIGS:
        ar = parse_area(rep_dir / f"{top}_area.rpt")
        acc_a = parse_hier_area(rep_dir / f"{top}_area.rpt", acc_inst)
        dp_a = ar["total"] - acc_a
        wns, tns = parse_qor_timing(rep_dir / f"{top}_qor.rpt")
        pr = parse_power_mw(rep_dir / f"{top}_power_saif_real.rpt", top)
        pu = parse_power_mw(rep_dir / f"{top}_power_saif_uniform.rpt", top)
        aeff = (TOPS * 1e6 / ar["total"]) if ar["total"] > 0 else 0.0
        peff = (TOPS * 1e3 / pr["total"]) if pr["total"] > 0 else 0.0
        ceff = aeff * peff
        print(
            f"| {label} | {wns:+.2f} | {tns:.2f} | {ar['comb']:.2f} | {ar['seq']:.2f} | **{ar['total']:.2f}** | "
            f"{acc_a:.2f} | {dp_a:.2f} | **{pr['total']:.3f}** | {pu['total']:.3f} | "
            f"{aeff:.2f} | {peff:.2f} | **{ceff:.1f}** |"
        )

    print("\n### 2. Power Breakdown (Real Llama-3.2-1B SAIF, mW) & Datapath-Only Efficiency")
    print("| Config | Sw (mW) | Int (mW) | Leak (mW) | Total (mW) | Acc Pwr (mW) | DP Pwr (mW) | DP Area Eff | DP Pwr Eff |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for label, top, acc_inst in CONFIGS:
        ar = parse_area(rep_dir / f"{top}_area.rpt")
        acc_a = parse_hier_area(rep_dir / f"{top}_area.rpt", acc_inst)
        dp_a = ar["total"] - acc_a
        pr = parse_power_mw(rep_dir / f"{top}_power_saif_real.rpt", top)
        acc_leaf = acc_inst.split("/")[-1]
        # parse acc power from hier power report
        txt_p = (rep_dir / f"{top}_power_saif_real.rpt").read_text(errors="ignore")
        m_acc = re.search(rf"^\s*{re.escape(acc_leaf)}\s+\(.*?\)\s*\n\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", txt_p, re.MULTILINE)
        if not m_acc:
            m_acc = re.search(rf"^\s*{re.escape(acc_leaf)}\s+\(.*?\)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", txt_p, re.MULTILINE)
        acc_p = float(m_acc.group(4)) if m_acc else 0.0
        dp_p = pr["total"] - acc_p
        dp_aeff = (TOPS * 1e6 / dp_a) if dp_a > 0 else 0.0
        dp_peff = (TOPS * 1e3 / dp_p) if dp_p > 0 else 0.0
        print(
            f"| {label} | {pr['sw']:.3f} | {pr['int']:.3f} | {pr['leak']:.4f} | {pr['total']:.3f} | "
            f"{acc_p:.3f} | {dp_p:.3f} | {dp_aeff:.2f} | {dp_peff:.2f} |"
        )


if __name__ == "__main__":
    main()
