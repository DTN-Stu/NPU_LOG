#!/usr/bin/env python3
"""
attn_4modes_tiling_model.py — Workstream 3
Exact architectural tiling & memory-traffic model for all 4 Attention modes:
  1. NAIVE       (Unfused 3-Phase: QK^T -> DRAM S -> 2-pass Softmax -> DRAM P -> PV)
  2. ONLINE_2A   (Unfused Online SM: QK^T -> DRAM S -> 1-pass Softmax -> DRAM P -> PV)
  3. FLASH_RING  (Fused FlashAttn, m_q=8, kv_pin=0: reloads K+V via Ring Buffer)
  4. FLASH_KVPIN (Fused FlashAttn + GQA 4:1 KV-Pinning in Banks 0..15, m_q=8, kv_pin=1)

Verifies 0-Byte error against RTL ground truth in sim/attn_comparison_results.csv
for M = N_KV in {64, 128, 256}, and exports 08_system_models/results/tcdm_dram_4modes.csv
for M = N_KV in {64, 128, 256, 512, 1024}.
"""
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTL_CSV = os.path.join(ROOT, "sim", "attn_comparison_results.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_CSV = os.path.join(OUT_DIR, "tcdm_dram_4modes.csv")

# Llama-3.2-1B Layer 1 Attention Geometry (matches tb/tb_attn_comparison.sv)
H_Q = 32
H_KV = 8
D_K = 64
D_V = 64
K_PACKETS_QK = D_K // 16   # 4
DV_TILES = D_V // 16       # 4
M_BATCH = 16               # 16 tokens per GEMM M-tile
WORD_BYTES = 9             # 72-bit Log8-FP4 packet
DRAM_BEAT_BYTES = 64       # 512-bit DRAM bus


def dma_bytes(addr: int, n_bytes: int) -> int:
    """Exact 512-bit (64B) DRAM beat count for an arbitrary byte address & length."""
    if n_bytes <= 0:
        return 0
    first_beat = addr // DRAM_BEAT_BYTES
    last_beat = (addr + n_bytes - 1) // DRAM_BEAT_BYTES
    return (last_beat - first_beat + 1) * DRAM_BEAT_BYTES


def align64(n_bytes: int) -> int:
    return dma_bytes(0, n_bytes)


def simulate_4modes(mode: str, m_tokens: int, n_kv: int) -> dict:
    """
    Explicit tile-by-tile DRAM <-> TCDM <-> Core traffic profiler matching
    tb/tb_attn_comparison.sv and snax_llm_sequencer.sv.
    """
    assert m_tokens % 16 == 0 and n_kv % 16 == 0
    kv_tiles = n_kv // 16
    m_chunks = m_tokens // M_BATCH
    q_tiles = m_tokens // 8  # m_q = 8 for FLASH_*

    rd_bytes = 0
    wr_bytes = 0
    sp_rd_bytes = 0
    sp_wr_bytes = 0
    tcdm_rd_words = 0
    tcdm_wr_words = 0

    if mode in ("NAIVE", "ONLINE_2A"):
        s_strip_bytes = kv_tiles * 64 * WORD_BYTES   # 4 words * 16 banks * kv_tiles * 9B
        p_strip_bytes = kv_tiles * 16 * WORD_BYTES   # 1 word  * 16 banks * kv_tiles * 9B
        s_strip_words = kv_tiles * 64
        p_strip_words = kv_tiles * 16

        for h in range(H_Q):
            for mb in range(m_chunks):
                # Phase 1: DMA Load Q_h[mb] (16 tokens x 36B at exact byte offsets -> 1 or 2 beats)
                for t in range(M_BATCH):
                    q_addr = ((h * m_tokens + mb * M_BATCH + t) * K_PACKETS_QK) * WORD_BYTES
                    rd_bytes += dma_bytes(q_addr, K_PACKETS_QK * WORD_BYTES)
                    tcdm_wr_words += K_PACKETS_QK

                # Phase 1: GEMM S = Q * K^T (DMA loads kv_tiles of K, each 4*16*9 = 576B)
                for _kv in range(kv_tiles):
                    rd_bytes += align64(K_PACKETS_QK * 16 * WORD_BYTES)
                    tcdm_wr_words += K_PACKETS_QK * 16
                    # Core reads Q (16 tokens x 4 pkts) + K (16 lanes x 4 pkts)
                    tcdm_rd_words += M_BATCH * K_PACKETS_QK + 16 * K_PACKETS_QK
                    # Write-out FP16 S tile (16 tokens x 4 words = 64 words)
                    tcdm_wr_words += 64

                # Phase 1: DMA WB S -> DRAM (spill write)
                b_s = align64(s_strip_bytes)
                wr_bytes += b_s
                sp_wr_bytes += b_s
                tcdm_rd_words += s_strip_words

                # Phase 2: DMA Read S from DRAM (2 passes for NAIVE, 1 pass for ONLINE_2A)
                passes = 2 if mode == "NAIVE" else 1
                for _ in range(passes):
                    rd_bytes += b_s
                    sp_rd_bytes += b_s
                    tcdm_wr_words += s_strip_words

                # Phase 2: RTL Requant S -> P (reads S words, writes P words)
                tcdm_rd_words += s_strip_words
                tcdm_wr_words += p_strip_words

                # Phase 2: DMA WB P -> DRAM (spill write)
                b_p = align64(p_strip_bytes)
                wr_bytes += b_p
                sp_wr_bytes += b_p
                tcdm_rd_words += p_strip_words

                # Phase 3: DMA Load P from DRAM (spill read)
                rd_bytes += b_p
                sp_rd_bytes += b_p
                tcdm_wr_words += p_strip_words

                # Phase 3: GEMM O = P * V (DMA loads dv_tiles of V, each kv_tiles*16*9 B)
                for _dv in range(DV_TILES):
                    rd_bytes += align64(kv_tiles * 16 * WORD_BYTES)
                    tcdm_wr_words += kv_tiles * 16
                    # Core reads P (16 tokens x kv_tiles pkts) + V (16 lanes x kv_tiles pkts)
                    tcdm_rd_words += M_BATCH * kv_tiles + 16 * kv_tiles
                    # Write-out FP4 O tile (16 tokens x 1 word = 16 words)
                    tcdm_wr_words += 16

                # Phase 3: DMA WB O -> DRAM (16 tokens x 36B at exact byte offsets -> 1 or 2 beats)
                for t in range(M_BATCH):
                    o_addr = ((h * m_tokens + mb * M_BATCH + t) * DV_TILES) * WORD_BYTES
                    wr_bytes += dma_bytes(o_addr, DV_TILES * WORD_BYTES)
                    tcdm_rd_words += DV_TILES

        # MAC ops: QK^T (H_Q * M * N_KV * D_K) + PV (H_Q * M * N_KV * D_V)
        mac_ops = 2 * H_Q * m_tokens * n_kv * D_K

    elif mode == "FLASH_KVPIN":
        # Q macro-DMA: q_sub = min(q_tiles, 8), chunks of (q_sub * 8 * 4 * 9 B)
        q_sub = min(q_tiles, 8)
        n_q_chunks = (q_tiles + q_sub - 1) // q_sub
        for _h in range(H_Q):
            for _c in range(n_q_chunks):
                chunk_q_bytes = q_sub * 8 * K_PACKETS_QK * WORD_BYTES
                rd_bytes += align64(chunk_q_bytes)
                tcdm_wr_words += q_sub * 8 * K_PACKETS_QK
                # Write-out O macro-DMA chunk
                chunk_o_bytes = q_sub * 8 * DV_TILES * WORD_BYTES
                wr_bytes += align64(chunk_o_bytes)
                tcdm_rd_words += q_sub * 8 * DV_TILES

        # K + V pinned ONCE per KV-head (GQA 4:1 -> H_KV = 8)
        for _hkv in range(H_KV):
            for _kv in range(kv_tiles):
                # K tile: 16 * 4 * 9 = 576 B (9 beats)
                rd_bytes += align64(16 * K_PACKETS_QK * WORD_BYTES)
                tcdm_wr_words += 16 * K_PACKETS_QK
                # V tiles (Sep-KV): 4 dv_tiles x (16 * 9 = 144 B -> 192 B aligned)
                for _dv in range(DV_TILES):
                    rd_bytes += align64(16 * WORD_BYTES)
                    tcdm_wr_words += 16

        # On-chip TCDM reads during fused FlashAttention compute:
        # For each head, each Q-tile (8 tokens), each KV-tile, QK^T + 4x DV PV
        for _h in range(H_Q):
            for _qt in range(q_tiles):
                for _kv in range(kv_tiles):
                    # QK^T reads Q (8*4) + K (16*4) from TCDM
                    tcdm_rd_words += 8 * K_PACKETS_QK + 16 * K_PACKETS_QK
                    # PV reads V (4 * 16) from TCDM (P comes from on-chip P-FIFO!)
                    tcdm_rd_words += DV_TILES * 16
                # Write 8 tokens x 4 DV FP4 words to TCDM output bank
                tcdm_wr_words += 8 * DV_TILES

        # Effective MAC ops (2.5x due to m_q=8 half-array + 4-way DV streaming in RTL)
        mac_ops = 2 * H_Q * m_tokens * n_kv * D_K

    elif mode == "FLASH_RING":
        q_sub = min(q_tiles, 8)
        n_q_chunks = (q_tiles + q_sub - 1) // q_sub
        for _h in range(H_Q):
            for _c in range(n_q_chunks):
                chunk_q_bytes = q_sub * 8 * K_PACKETS_QK * WORD_BYTES
                rd_bytes += align64(chunk_q_bytes)
                tcdm_wr_words += q_sub * 8 * K_PACKETS_QK
                chunk_o_bytes = q_sub * 8 * DV_TILES * WORD_BYTES
                wr_bytes += align64(chunk_o_bytes)
                tcdm_rd_words += q_sub * 8 * DV_TILES

            # Ring Buffer (kv_pin=0): streams K (576 B) + V (192 B) per (qt, kv, dv)
            for _qt in range(q_tiles):
                for _kv in range(kv_tiles):
                    for _dv in range(DV_TILES):
                        rd_bytes += align64(16 * K_PACKETS_QK * WORD_BYTES)  # K = 576 B
                        rd_bytes += align64(16 * WORD_BYTES)                 # V = 192 B
                        tcdm_wr_words += 16 * K_PACKETS_QK + 16
                    # Core TCDM reads
                    tcdm_rd_words += 8 * K_PACKETS_QK + 16 * K_PACKETS_QK + DV_TILES * 16
                tcdm_wr_words += 8 * DV_TILES

        mac_ops = 2 * H_Q * m_tokens * n_kv * D_K
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "mode": mode,
        "M": m_tokens,
        "N_KV": n_kv,
        "dram_rd_bytes": rd_bytes,
        "dram_wr_bytes": wr_bytes,
        "dram_total_bytes": rd_bytes + wr_bytes,
        "sp_rd_bytes": sp_rd_bytes,
        "sp_wr_bytes": sp_wr_bytes,
        "sp_total_bytes": sp_rd_bytes + sp_wr_bytes,
        "tcdm_rd_words": tcdm_rd_words,
        "tcdm_wr_words": tcdm_wr_words,
        "mac_ops": mac_ops,
    }


def verify_against_rtl():
    """Strict 0-Byte assertion against sim/attn_comparison_results.csv."""
    rtl_rows = {}
    with open(RTL_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["mode"], int(row["M"]), int(row["N_KV"]))
            if key not in rtl_rows and int(row["checks_fail"]) == 0:
                rtl_rows[key] = row

    checked = 0
    for m in (64, 128, 256):
        for mode in ("NAIVE", "ONLINE_2A", "FLASH_RING", "FLASH_KVPIN"):
            key = (mode, m, m)
            assert key in rtl_rows, f"Missing RTL ground truth for {key}"
            rtl = rtl_rows[key]
            sim = simulate_4modes(mode, m, m)
            for col_sim, col_rtl in [
                ("dram_rd_bytes", "rd_bytes"),
                ("dram_wr_bytes", "wr_bytes"),
                ("sp_rd_bytes", "sp_rd_bytes"),
                ("sp_wr_bytes", "sp_wr_bytes"),
            ]:
                got = sim[col_sim]
                exp = int(rtl[col_rtl])
                assert got == exp, f"BYTE MISMATCH {key} {col_sim}: got={got}, rtl={exp} (diff={got-exp})"
            checked += 1
            print(f"[PASS 0-BYTE] {mode:11s} M={m:3d} N_KV={m:3d} | "
                  f"rd={sim['dram_rd_bytes']:10,d} B | wr={sim['dram_wr_bytes']:9,d} B | "
                  f"sp={sim['sp_total_bytes']:10,d} B | tcdm_rd={sim['tcdm_rd_words']:9,d}w")
    print(f"\n>>> ALL {checked}/12 RTL ASSERTIONS PASSED WITH EXACT 0-BYTE ERROR! <<<")


def export_csv():
    os.makedirs(OUT_DIR, exist_ok=True)
    fieldnames = [
        "mode", "M", "N_KV", "dram_rd_bytes", "dram_wr_bytes", "dram_total_bytes",
        "sp_rd_bytes", "sp_wr_bytes", "sp_total_bytes",
        "tcdm_rd_words", "tcdm_wr_words", "mac_ops",
    ]
    rows = []
    for m in (64, 128, 256, 512, 1024):
        for mode in ("NAIVE", "ONLINE_2A", "FLASH_RING", "FLASH_KVPIN"):
            rows.append(simulate_4modes(mode, m, m))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Exported {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    verify_against_rtl()
    export_csv()
