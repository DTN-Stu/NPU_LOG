#!/usr/bin/env python3
# =============================================================================
# gen_ws_data.py
# Sinh du lieu Weight/Activation cho flow WEIGHT-STATIONARY (mode 00, W*A)
# cua snax_fused_llm_core. Cung phuong phap voi gen_gemm_data.py:
#   - Quantize tung block 16 phan tu, chon 1 scale log8 chung (max -> gan 6.0)
#   - Model BIT-EXACT RTL: LUT E2M1, CSD scaler, bidirectional shift, sat_add
#   - CSV co ca DIGITAL (code/hex) va REAL (float)
#   - HEX de nap vao TB ($readmemh)
#
# value = sign * E2M1[mag] * 2^(scale_code/8)
#
# Output (mac dinh ./data_ws):
#   act.hex             72-bit packet / (in_channel, k-block)  idx = m*KB + kb
#   wt.hex              72-bit packet / (k-block, lane)        idx = kb*16 + lane
#   expected_acc.hex    M dong x 512-bit = 16 lane x 32-bit Q16.16
#   expected_fp16.hex   (M*16) dong, 16-bit: idx = m*16 + lane (bit-exact drain)
#   tb_case_params.svh  CASE_M / CASE_K / CASE_KB
#   act_matrix.csv      M x K float (real)
#   wt_matrix.csv       16 x K float (real)
#   act_blocks.csv      1 dong/block: scale + 16 code + 16 value
#   wt_blocks.csv       nhu tren cho weight
#   act_packets.csv     packet hex72 + 16 gia tri decoded (digital+real)
#   wt_packets.csv
#   expected_acc.csv    in_channel,lane,acc_q16,acc_real,fp16_hex
#   README.txt
#
# Vi du:
#   python gen_ws_data.py -M 4 -K 32 --seed 7
#   python gen_ws_data.py -M 8 -K 64 --seed 3 --stress
# =============================================================================
import argparse
import csv
import math
import os
import random

# --- E2M1 ---------------------------------------------------------------
E2M1_VAL = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_MAG = [0, 1, 2, 3, 4, 6, 8, 12]          # == 2 x value (RTL encoding)
MID      = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]

# --- RTL bit-exact model -------------------------------------------------
CSD = [32, 35, 38, 41, 46, 49, 54, 59]
INT32_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)


def s8(v):
    v &= 0xFF
    return v - 256 if v >= 128 else v


def clamp32(v):
    return INT32_MAX if v > INT32_MAX else (INT32_MIN if v < INT32_MIN else v)


def lut_prod(a, b):
    p = E2M1_MAG[a & 7] * E2M1_MAG[b & 7]
    return -p if ((a >> 3) ^ (b >> 3)) & 1 else p


def pack(codes, scale_code):
    w = (scale_code & 0xFF) << 64
    for e in range(16):
        w |= (codes[e] & 0xF) << (4 * e)
    return w


def align_step(act_w, wt_w, frac=16):
    """Q16.16 contribution of one packet pair (matches bidirectional_shifter)."""
    dot = 0
    for e in range(16):
        dot += lut_prod((act_w >> (4 * e)) & 0xF, (wt_w >> (4 * e)) & 0xF)
    res = s8(act_w >> 64) + s8(wt_w >> 64)
    q, r = res >> 3, res & 7
    scaled = dot * CSD[r]
    net = q + (frac - 7)
    wide = (scaled << net) if net >= 0 else (scaled >> (-net))
    return clamp32(wide)


def sat_add(a, b):
    return clamp32(a + b)


def fp16_from_q16(norm_in):
    """Bit-exact logscale_lazy_drain_norm (FRAC_BITS=16) -> 16-bit FP16."""
    x = norm_in & 0xFFFFFFFF
    if x & 0x80000000:
        x -= (1 << 32)
    sign = 1 if x < 0 else 0
    a = -x if sign else x
    if a == 0:
        return 0
    lzc = 0
    for i in range(31, -1, -1):
        if (a >> i) & 1:
            break
        lzc += 1
    msb = 31 - lzc
    raw_exp = msb - 16 + 15
    ns = (a << lzc) & 0xFFFFFFFF
    mant = (ns >> 21) & 0x3FF
    guard = (ns >> 20) & 1
    rnd = (ns >> 19) & 1
    sticky = 1 if (ns & ((1 << 19) - 1)) else 0
    round_up = guard & (rnd | sticky | (mant & 1))
    rm = mant + round_up
    final_exp = raw_exp + (1 if rm > 0x3FF else 0)
    rm &= 0x3FF
    if final_exp <= 0:
        return 0
    if final_exp >= 31:
        return (sign << 15) | (0x1E << 10) | 0x3FF
    return (sign << 15) | ((final_exp & 0x1F) << 10) | rm


# --- Quantizer -----------------------------------------------------------
def quantize_block(xs):
    """Return (codes, quantized_values, scale_code)."""
    amax = max((abs(x) for x in xs), default=0.0)
    if amax == 0.0:
        return [0] * 16, [0.0] * 16, 0
    s = int(round(8.0 * math.log2(amax / 6.0)))
    s = max(-128, min(127, s))
    scale = 2.0 ** (s / 8.0)

    codes, vals = [], []
    for x in xs:
        y = abs(x) / scale
        idx = 0
        for i, m in enumerate(MID):
            if y >= m:
                idx = i + 1
        sign = 1 if x < 0 else 0
        codes.append((sign << 3) | idx)
        vals.append((-1.0 if sign else 1.0) * E2M1_VAL[idx] * scale)
    return codes, vals, s


def make_matrix(rows, cols, rng, lo, hi, gain_lo, gain_hi):
    """rows x cols floats; moi block 16 phan tu co gain rieng de scale da dang."""
    out = []
    for _ in range(rows):
        row = []
        for _kb in range(cols // 16):
            g = 2.0 ** rng.uniform(gain_lo, gain_hi)
            for _ in range(16):
                row.append(rng.uniform(lo, hi) * g)
        out.append(row)
    return out


# --- Output helpers ------------------------------------------------------
def write_hex(path, words, width_bits):
    digits = (width_bits + 3) // 4
    with open(path, "w", newline="\n") as f:
        for w in words:
            f.write(f"{w & ((1 << width_bits) - 1):0{digits}x}\n")


def write_matrix_csv(path, mat, row_label, cols):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([row_label] + [f"k{k}" for k in range(cols)])
        for r, row in enumerate(mat):
            wr.writerow([r] + [f"{v:.6f}" for v in row])


def write_blocks_csv(path, rows_label, blocks):
    header = [rows_label, "kblock", "scale_code", "scale_value"]
    header += [f"c{e}" for e in range(16)]
    header += [f"v{e}" for e in range(16)]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for label, kb, sc, codes, vals in blocks:
            wr.writerow([label, kb, sc, f"{2.0 ** (sc / 8.0):.6f}"]
                        + [f"0x{c:x}" for c in codes]
                        + [f"{v:.6f}" for v in vals])


def write_packets_csv(path, rows_label, packets, blocks):
    header = [rows_label, "kblock", "scale_code", "packet_hex72", "data_hex64"]
    header += [f"v{e}" for e in range(16)]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for (label, kb, sc, _codes, vals), p in zip(blocks, packets):
            wr.writerow([label, kb, sc, f"{p & ((1 << 72) - 1):018x}",
                         f"{(p >> 0) & ((1 << 64) - 1):016x}"]
                        + [f"{v:.6f}" for v in vals])


# --- Main ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate W*A weight-stationary data (FP4 log-scale)")
    ap.add_argument("-M", "--tokens", type=int, default=4, help="so in-channel (slot) = so cot activation")
    ap.add_argument("-K", "--inner", type=int, default=32, help="chieu K (boi so 16)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("-o", "--out", default="data_ws", help="thu muc output")
    ap.add_argument("--stress", action="store_true",
                    help="gain rong de cham bao hoa va shift sau")
    args = ap.parse_args()

    if args.inner % 16 != 0:
        raise SystemExit("K phai la boi so cua 16")
    M, K, KB = args.tokens, args.inner, args.inner // 16
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    if args.stress:
        act_mat = make_matrix(M, K, rng, -6.0, 6.0, -14.0, 14.0)
        wt_mat = make_matrix(16, K, rng, -6.0, 6.0, -14.0, 14.0)
    else:
        act_mat = make_matrix(M, K, rng, -4.0, 4.0, -3.0, 3.0)
        wt_mat = make_matrix(16, K, rng, -4.0, 4.0, -3.0, 3.0)

    # ---- quantize + pack ------------------------------------------------
    def quant_rows(mat):
        packets, blocks = [], []
        for r, row in enumerate(mat):
            for kb in range(KB):
                seg = row[kb * 16:(kb + 1) * 16]
                codes, vals, sc = quantize_block(seg)
                packets.append(pack(codes, sc))
                blocks.append((r, kb, sc, codes, vals))
        return packets, blocks

    act_pkts, act_blocks = quant_rows(act_mat)
    wt_pkts, wt_blocks = quant_rows(wt_mat)

    # ---- bit-exact expected accumulator (Q16.16) ----------------------
    # Dung cho flow weight-stationary: weight block kb dung chung cho moi in-channel
    exp_words, exp_rows = [], []
    for m in range(M):
        acc = [0] * 16
        for kb in range(KB):
            a = act_pkts[m * KB + kb]
            for lane in range(16):
                acc[lane] = sat_add(acc[lane], align_step(a, wt_pkts[kb * 16 + lane]))
        exp_rows.append(acc)
        w = 0
        for lane in range(16):
            w |= (acc[lane] & 0xFFFFFFFF) << (32 * lane)
        exp_words.append(w)

    # ---- write hex ------------------------------------------------------
    write_hex(os.path.join(args.out, "act.hex"), act_pkts, 72)
    write_hex(os.path.join(args.out, "wt.hex"), wt_pkts, 72)
    write_hex(os.path.join(args.out, "expected_acc.hex"), exp_words, 512)
    # expected FP16 (bit-exact drain) - order in_channel-major, lane-minor
    with open(os.path.join(args.out, "expected_fp16.hex"), "w", newline="\n") as f:
        for m in range(M):
            for lane in range(16):
                f.write(f"{fp16_from_q16(exp_rows[m][lane]):04x}\n")

    with open(os.path.join(args.out, "tb_case_params.svh"), "w", newline="\n") as f:
        f.write("// auto-generated by gen_ws_data.py\n")
        f.write(f"localparam int CASE_M  = {M};\n")
        f.write(f"localparam int CASE_K  = {K};\n")
        f.write(f"localparam int CASE_KB = {KB};\n")

    # ---- CSV ------------------------------------------------------------
    write_matrix_csv(os.path.join(args.out, "act_matrix.csv"), act_mat, "in_channel", K)
    write_matrix_csv(os.path.join(args.out, "wt_matrix.csv"), wt_mat, "lane", K)
    write_blocks_csv(os.path.join(args.out, "act_blocks.csv"), "in_channel", act_blocks)
    write_blocks_csv(os.path.join(args.out, "wt_blocks.csv"), "lane", wt_blocks)
    write_packets_csv(os.path.join(args.out, "act_packets.csv"), "in_channel", act_pkts, act_blocks)
    write_packets_csv(os.path.join(args.out, "wt_packets.csv"), "lane", wt_pkts, wt_blocks)

    with open(os.path.join(args.out, "expected_acc.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["in_channel", "lane", "acc_q16_16", "acc_real", "fp16_hex"])
        for m in range(M):
            for lane in range(16):
                v = exp_rows[m][lane]
                wr.writerow([m, lane, v, f"{v / 65536.0:.6f}",
                             f"{fp16_from_q16(v):04x}"])

    # ---- README ---------------------------------------------------------
    with open(os.path.join(args.out, "README.txt"), "w", newline="\n") as f:
        f.write(
            f"W*A weight-stationary data (M={M} in-channel, K={K}, KB={KB}, "
            f"seed={args.seed}, stress={args.stress})\n"
            "================================================================\n"
            "act.hex : 72-bit packet / (in_channel, k-block)  idx = m*KB + kb\n"
            "wt.hex  : 72-bit packet / (k-block, lane)        idx = kb*16 + lane\n"
            "packet  = {{scale_code[7:0], 16 x {{sign,mag[2:0]}}}}\n"
            "          value_i = sign_i * E2M1[mag_i] * 2^(scale_code/8)\n"
            "expected_acc.hex : M lines x 512 bit = 16 lane x 32-bit Q16.16\n"
            "                   lane L = bits [32L +: 32]\n"
            "expected_fp16.hex: M*16 lines, idx = m*16 + lane (bit-exact drain)\n"
            "tb_case_params.svh : CASE_M / CASE_K / CASE_KB\n"
            "CSV: matrix (real), blocks (codes + values), packets (hex + values)\n"
        )

    print(f"out dir : {os.path.abspath(args.out)}")
    print(f"shape   : M={M} K={K} (KB={KB}) seed={args.seed} stress={args.stress}")
    print(f"files   : act.hex({len(act_pkts)}), wt.hex({len(wt_pkts)}), "
          f"expected_acc.hex({M}), expected_fp16.hex({M*16})")
    print("\nBlock scale codes (act in_channel0):",
          [b[2] for b in act_blocks if b[0] == 0])
    print("Expected acc (in_channel 0), lane 0..3:")
    for lane in range(4):
        v = exp_rows[0][lane]
        print(f"  lane{lane}: q16={v:>12d}  real={v/65536.0:+.6f}  "
              f"fp16=0x{fp16_from_q16(v):04x}")


if __name__ == "__main__":
    main()
