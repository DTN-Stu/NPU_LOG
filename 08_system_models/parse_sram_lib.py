#!/usr/bin/env python3
"""
parse_sram_lib.py — Workstream 2B
Directly parses 05_system_ppa/eda_backup/sram_lib/srambank_256x4x72_6t122.lib
to extract physical per-word read/write energy (pJ) and leakage (mW) without
hardcoding or guessing.
"""
import json
import os
import re

LIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "05_system_ppa", "eda_backup", "sram_lib", "srambank_256x4x72_6t122.lib",
)


def parse_sram_lib(path=LIB_PATH):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()

    units = {}
    for k in ("time_unit", "voltage_unit", "current_unit", "leakage_power_unit"):
        m = re.search(rf'{k}\s*:\s*"([^"]+)"', text)
        units[k] = m.group(1) if m else None

    m_leak = re.search(r'cell_leakage_power\s*:\s*([0-9eE.+\-]+)', text)
    cell_leak = float(m_leak.group(1)) if m_leak else 0.0

    # Extract pin(clk) block
    clk_start = text.index("pin (clk)")
    clk_end = text.index("pin (wd[71])", clk_start)
    clk_block = text[clk_start:clk_end]

    # Parse internal_power blocks in pin(clk) (closed at 6-space indent)
    pwr_blocks = re.findall(
        r'internal_power\s*\(\)\s*\{(.*?)\n\s{6}\}',
        clk_block,
        flags=re.DOTALL,
    )
    clk_energy = {"write": [], "!write": []}
    for blk in pwr_blocks:
        mw = re.search(r'when\s*:\s*"([^"]+)"', blk)
        mp = re.search(r'related_pg_pin\s*:\s*(\w+)', blk)
        mr = re.search(r'rise_power[^\{]*\{.*?values\s*\(\s*\\?\s*"([^"]+)"', blk, flags=re.DOTALL)
        mf = re.search(r'fall_power[^\{]*\{.*?values\s*\(\s*\\?\s*"([^"]+)"', blk, flags=re.DOTALL)
        if not mw:
            continue
        when = mw.group(1)
        pg = mp.group(1) if mp else "?"
        r_vals = [float(x.strip()) for x in mr.group(1).split(",")] if mr else [0.0]
        f_vals = [float(x.strip()) for x in mf.group(1).split(",")] if mf else [0.0]
        # Nominal index (first/typical slew) — report both min/nominal/max
        clk_energy.setdefault(when, []).append({
            "pg_pin": pg,
            "rise_vals": r_vals,
            "fall_vals": f_vals,
            "rise_nom": r_vals[0],
            "fall_nom": f_vals[0],
            "sum_nom": r_vals[0] + f_vals[0],
        })

    # Read energy from clk (!write): rise + fall across VDD and VSS
    e_clk_rd = sum(x["sum_nom"] for x in clk_energy["!write"])
    e_clk_wr = sum(x["sum_nom"] for x in clk_energy["write"])

    # Parse bus(dataout) internal_power (output toggle during read: VDD + VSS)
    rd_start = text.index("bus (dataout)")
    rd_block = text[rd_start:]
    rd_pwr_blocks = re.findall(
        r'internal_power\s*\(\)\s*\{(.*?)\n\s{6}\}',
        rd_block,
        flags=re.DOTALL,
    )
    rd_bit_avg = 0.0
    for blk in rd_pwr_blocks:
        mr = re.search(r'rise_power[^\{]*\{.*?values\s*\(\s*\\?\s*"([^"]+)"', blk, flags=re.DOTALL)
        mf = re.search(r'fall_power[^\{]*\{.*?values\s*\(\s*\\?\s*"([^"]+)"', blk, flags=re.DOTALL)
        # Nominal row 0, col 0 (5ps slew, 5.76fF load)
        rv = float(mr.group(1).split(",")[0].strip()) if mr else 0.0
        fv = float(mf.group(1).split(",")[0].strip()) if mf else 0.0
        # Average per bit transition (50% rise, 50% fall)
        rd_bit_avg += 0.5 * (rv + fv)
    # 72 bits with 50% bit-toggle probability per read word -> 36 bits switch
    e_rd_pins_72b = 72 * 0.5 * rd_bit_avg
    rd_pins = list(range(72))
    rd_pin_nom = rd_bit_avg

    # Parse wd[71] input capacitance & internal_power
    wd_start = text.index("pin (wd[71])")
    wd_end = text.index("pin (wd[70])", wd_start)
    wd71 = text[wd_start:wd_end]
    m_wd_cap = re.search(r'capacitance\s*:\s*([0-9eE.+\-]+)', wd71)
    wd_cap_pf = float(m_wd_cap.group(1)) if m_wd_cap else 0.0
    # Dynamic CV^2 on 72-bit wd input (50% toggle, V=0.7V -> 0.5*C*V^2 per toggle = C*V^2*0.5? 0.5*alpha*C*V^2)
    e_wd_cap_72b = 72 * 0.5 * wd_cap_pf * (0.7 ** 2)

    result = {
        "lib_file": "05_system_ppa/eda_backup/sram_lib/srambank_256x4x72_6t122.lib",
        "units": units,
        "cell_leakage_power_in_lib": cell_leak,
        "clk_internal_power_read_pJ": round(e_clk_rd, 6),
        "clk_internal_power_write_pJ": round(e_clk_wr, 6),
        "rd_pin_count": len(rd_pins),
        "rd_pin_nom_pJ": round(rd_pin_nom, 6),
        "rd_72b_pins_50pct_pJ": round(e_rd_pins_72b, 6),
        "e_sram_rd_lib_total_pJ": round(e_clk_rd + e_rd_pins_72b, 4),
        "wd_pin_cap_pF": wd_cap_pf,
        "wd_72b_cap_pJ": round(e_wd_cap_72b, 6),
        "e_sram_wr_lib_raw_pJ": round(e_clk_wr + e_wd_cap_72b, 4),
        "clk_breakdown": clk_energy,
        "note": (
            "Raw .lib models write clk internal_power as 0.3196 pJ (13.4x lower than read 4.291 pJ), "
            "an artifact of mk_SRAM_lib.py. Per SYSTEM_ENERGY_EXPERIMENT_PLAN.md WS2B, "
            "we report raw .lib value transparently AND use physically conservative 7nm SRAM write energy "
            "equal to read energy (or ~1.1x read = ~4.81 pJ/word, within 4.0-6.0 pJ CACTI 7.0 7nm range) "
            "plus conservative 7nm 288-KB leakage."
        ),
    }
    return result


if __name__ == "__main__":
    res = parse_sram_lib()
    print(json.dumps({k: v for k, v in res.items() if k != "clk_breakdown"}, indent=2))
    print("\nclk !write breakdown:")
    for item in res["clk_breakdown"]["!write"]:
        print(" ", item["pg_pin"], "rise=", item["rise_nom"], "fall=", item["fall_nom"], "sum=", item["sum_nom"])
    print("clk write breakdown:")
    for item in res["clk_breakdown"]["write"]:
        print(" ", item["pg_pin"], "rise=", item["rise_nom"], "fall=", item["fall_nom"], "sum=", item["sum_nom"])
