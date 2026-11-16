#!/usr/bin/env python3
"""
generate_snax_gemm_k1024_data.py

Single-tile Dense GEMM stimulus/golden for snax_llm_top_clean:
    M = 16 tokens, N = 16 channels, K = 1024  (K_BLOCKS = 64)

TCDM layout
    A (activations) : bank 16, word = token*64 + kb      (16 x 64 = 1024 words)
    W (weights)     : banks 0..15, word = kb, bank = lane (64 words/bank)
    C (output)      : bank 17, word = 4*token + wd       (64 words)

This is the largest K that fits the current one-bank activation layout
(16 tokens * 64 blocks = 1024 words = bank depth).  It is intended to measure
the compute/drain balance at a more realistic K than the 16x16x16 layer test.
"""
import os
import sys
import csv
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
    fp16_hex_to_float,
)

OUT_DIR = os.path.join(HERE, "..", "sim", "sim_snax_gemm_k1024")
HEX = os.path.join(OUT_DIR, "hex")
CSV = os.path.join(OUT_DIR, "csv")

M = 16
N = 16
K = 1024
KB = K // 16          # 64

ACT_BANK = 16
OUT_BANK = 17
NUM_VEC = 16
NUM_WORD = NUM_VEC * 4   # 64


def tcdm_addr(word, bank):
    return ((word & 0x3FF) << 9) | ((bank & 0x1F) << 4)


def main():
    np.random.seed(1024)
    os.makedirs(HEX, exist_ok=True)
    os.makedirs(CSV, exist_ok=True)

    print("=== SNAX single-tile GEMM M=16 N=16 K=1024 (64 K blocks) ===")
    A = np.random.normal(0.0, 1.0, (M, K))
    W = np.random.normal(0.0, 1.0, (K, N))

    a_codes = np.zeros((M, KB, 16), dtype=int)
    a_scales = np.zeros((M, KB), dtype=int)
    a_packets = np.zeros((M, KB), dtype=object)
    for m in range(M):
        for kb in range(KB):
            c, _, s = quantize_block(A[m, kb * 16:(kb + 1) * 16])
            a_codes[m, kb, :] = c
            a_scales[m, kb] = s
            a_packets[m, kb] = encode_72b_packet(c, s)

    w_codes = np.zeros((N, KB, 16), dtype=int)
    w_scales = np.zeros((N, KB), dtype=int)
    w_packets = np.zeros((N, KB), dtype=object)
    for n in range(N):
        for kb in range(KB):
            c, _, s = quantize_block(W[kb * 16:(kb + 1) * 16, n])
            w_codes[n, kb, :] = c
            w_scales[n, kb] = s
            w_packets[n, kb] = encode_72b_packet(c, s)

    # Bit-exact hardware model: gn=0, gm=0, token m, lane n.
    golden = np.zeros((M, N), dtype=np.uint16)
    acc_dbg = np.zeros((M, N), dtype=np.int64)
    for m in range(M):
        for n in range(N):
            acc = 0
            for kb in range(KB):
                dot = 0
                for el in range(16):
                    dot += fp4_mult_lut_py(int(a_codes[m, kb, el]),
                                           int(w_codes[n, kb, el]))
                res = int(a_scales[m, kb]) + int(w_scales[n, kb])
                q_exp = res // 8
                r_exp = res % 8
                aligned = shifter_py(csd_scaler_py(dot, r_exp), q_exp, 16)
                acc = sat_add32(acc, int(aligned))
            acc_dbg[m, n] = acc
            golden[m, n] = drain_norm_fp16_py(int(acc), frac_bits=16)

    # TCDM preload
    preload = []
    for m in range(M):
        for kb in range(KB):
            preload.append((tcdm_addr(m * KB + kb, ACT_BANK),
                            int(a_packets[m, kb]),
                            f"Act token={m} kb={kb}"))
    for n in range(N):
        for kb in range(KB):
            preload.append((tcdm_addr(kb, n),
                            int(w_packets[n, kb]),
                            f"Wt lane={n} kb={kb}"))

    with open(os.path.join(HEX, "tcdm_preload_addr.hex"), "w") as fa, \
         open(os.path.join(HEX, "tcdm_preload_data.hex"), "w") as fd:
        for addr, data, _ in preload:
            fa.write(f"{addr:08X}\n")
            fd.write(f"{data:018X}\n")

    # Expected output words: vector = token, 4 words of 4 FP16 lanes.
    with open(os.path.join(HEX, "expected_out_words.hex"), "w") as f:
        for m in range(M):
            for wd in range(4):
                lanes = [int(golden[m, 4 * wd + j]) for j in range(4)]
                packed = (lanes[3] << 48) | (lanes[2] << 32) | \
                         (lanes[1] << 16) | lanes[0]
                f.write(f"{packed:018X}\n")

    with open(os.path.join(HEX, "golden_gemm_fp16.hex"), "w") as f:
        for m in range(M):
            for n in range(N):
                f.write(f"{int(golden[m, n]):04X}\n")

    # CSV debug views
    with open(os.path.join(CSV, "golden_gemm.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token", "lane", "accum_q16", "accum_real",
                    "fp16_hex", "fp16_real"])
        for m in range(M):
            for n in range(N):
                h = int(golden[m, n])
                real = fp16_hex_to_float(h)
                w.writerow([m, n, int(acc_dbg[m, n]),
                            f"{int(acc_dbg[m, n]) / 65536.0:.9f}",
                            f"0x{h:04X}", f"{float(real):.9f}"])

    with open(os.path.join(CSV, "tcdm_preload.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "kind", "addr_hex", "bank", "word",
                    "data_hex", "detail"])
        for i, (addr, data, detail) in enumerate(preload):
            w.writerow([i, detail.split()[0], f"0x{addr:08X}",
                        (addr >> 4) & 0x1F, (addr >> 9) & 0x3FF,
                        f"0x{data:018X}", detail])

    with open(os.path.join(CSV, "quantized_blocks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "row", "kb", "scale_hex", "scale_value",
                    "packet_hex", "codes"])
        for m in range(M):
            for kb in range(KB):
                sc = int(a_scales[m, kb])
                w.writerow(["A", m, kb, f"0x{sc & 0xFF:02X}",
                            f"{2.0 ** (sc / 8.0):.9f}",
                            f"0x{int(a_packets[m, kb]):018X}",
                            "".join(f"{int(c):X}" for c in a_codes[m, kb])])
        for n in range(N):
            for kb in range(KB):
                sc = int(w_scales[n, kb])
                w.writerow(["W", n, kb, f"0x{sc & 0xFF:02X}",
                            f"{2.0 ** (sc / 8.0):.9f}",
                            f"0x{int(w_packets[n, kb]):018X}",
                            "".join(f"{int(c):X}" for c in w_codes[n, kb])])

    print(f"[SUCCESS] preload={len(preload)} words, expected={NUM_WORD} words")
    print(f"  HEX: {os.path.normpath(HEX)}")
    print(f"  CSV: {os.path.normpath(CSV)}")


if __name__ == "__main__":
    main()
