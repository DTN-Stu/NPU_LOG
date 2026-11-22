"""
Generate bit-exact 72-bit packet traces (MXFP4, NVFP4, Log8) from real
Llama-3.2-1B Layer 1 tensors (sim/sim_snax_llm_layer_mid/real_llama32_1b_layer1_slice.npz).

Each format quantizes the exact same FP16/FP32 tensors so SAIF power comparison
reflects real LLM activity & scale distributions per format.
"""
from pathlib import Path
import numpy as np

E2M1_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1_MIDS   = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)

TRACE_LEN = 4096
NUM_LANES = 16


def encode_e2m1_nibbles(vals: np.ndarray) -> np.ndarray:
    """Quantize float values to 4-bit E2M1 codes [0..15] (signed)."""
    signs = (vals < 0).astype(np.uint8)
    abs_v = np.abs(vals)
    mag = np.searchsorted(E2M1_MIDS, abs_v, side="left").astype(np.uint8)
    codes = np.where(mag == 0, 0, (signs << 3) | mag)
    return codes


def pack_72b_hex(scales_u8: np.ndarray, nibbles_16: np.ndarray) -> list[str]:
    """Pack [N] uint8 scales and [N, 16] uint8 nibbles into 18-hex-char 72b strings."""
    lines = []
    for s, row in zip(scales_u8, nibbles_16):
        lo64 = 0
        for e in range(16):
            lo64 |= (int(row[e]) & 0xF) << (e * 4)
        pkt72 = ((int(s) & 0xFF) << 64) | lo64
        lines.append(f"{pkt72:018x}")
    return lines


def quantize_blocks_log8(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact slq_encoder.sv (blocks: [N, 16])."""
    fp16 = blocks.astype(np.float16)
    bits = fp16.view(np.uint16).astype(np.int32)
    sign_bit = (bits >> 15) & 0x1
    exp_fp16 = (bits >> 10) & 0x1F
    man_top3 = (bits >> 7) & 0x7

    mu = (exp_fp16 << 3) | man_top3
    mu = np.where(exp_fp16 == 0, 0, mu)
    mu = np.where(exp_fp16 == 31, 247, mu)

    mu_max = mu.max(axis=1, keepdims=True)
    scale_sat = np.clip(mu_max - 140, -128, 127)
    scale_sat = np.where(mu_max == 0, 0, scale_sat)

    diff = mu_max - mu
    mag = np.zeros_like(diff)
    for idx, th in [(1, 36), (2, 24), (3, 18), (4, 14), (5, 10), (6, 6), (7, 2)]:
        mag = np.where(diff <= th, idx, mag)
    mag = np.where((mu == 0) | (mu_max == 0), 0, mag)
    nibbles = np.where(mag == 0, 0, (sign_bit << 3) | mag).astype(np.uint8)
    scales_u8 = (scale_sat[:, 0].astype(np.int8)).view(np.uint8)
    return scales_u8, nibbles


def quantize_blocks_mxfp4(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OCP MXFP4 g16: E8M0 (bias 127) + E2M1."""
    max_abs = np.abs(blocks).max(axis=1, keepdims=True)
    safe_max = np.maximum(max_abs / 6.0, 2.0 ** -127)
    exp_e8m0 = np.clip(np.ceil(np.log2(safe_max)), -127, 127).astype(np.int32)
    exp_e8m0 = np.where(max_abs == 0.0, 0, exp_e8m0)
    scale = np.where(max_abs == 0.0, 1.0, np.power(2.0, exp_e8m0))
    nibbles = encode_e2m1_nibbles(blocks / scale)
    scales_u8 = np.clip(exp_e8m0[:, 0] + 127, 0, 254).astype(np.uint8)
    return scales_u8, nibbles


def encode_e4m3_u8(val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize positive scale values to unsigned E4M3 (bias=7), return (u8, float)."""
    v = np.clip(val, 2.0 ** -9, 448.0)
    exp = np.floor(np.log2(v)).astype(np.int32)
    # Subnormal handling for exp < -6
    is_sub = exp < -6
    mant_norm = np.round((v / np.power(2.0, np.maximum(exp, -6)) - 1.0) * 8.0).astype(np.int32)
    carry = mant_norm >= 8
    exp_n = np.where(carry, exp + 1, exp)
    mant_n = np.where(carry, 0, mant_norm)
    e_field = np.clip(exp_n + 7, 1, 15)
    # Clamp max E4M3 normal (448 = 0_1111_110)
    mant_n = np.where((e_field == 15) & (mant_n > 6), 6, mant_n)

    mant_sub = np.clip(np.round(v / (2.0 ** -9)).astype(np.int32), 1, 7)
    u8 = np.where(is_sub, mant_sub, (e_field << 3) | mant_n).astype(np.uint8)
    flt = np.where(
        is_sub,
        mant_sub * (2.0 ** -9),
        (1.0 + mant_n / 8.0) * np.power(2.0, e_field - 7),
    ).astype(np.float32)
    return u8, flt


def quantize_blocks_nvfp4(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NVIDIA NVFP4: tensor_scale + E4M3 block scale + E2M1."""
    gmax = float(np.abs(blocks).max())
    t_scale = max(gmax / (6.0 * 448.0), 1e-12)
    scaled = blocks / t_scale
    max_abs = np.abs(scaled).max(axis=1, keepdims=True)
    blk_s = np.maximum(max_abs / 6.0, 2.0 ** -9)
    s_u8, s_flt = encode_e4m3_u8(blk_s[:, 0])
    s_u8 = np.where(max_abs[:, 0] == 0.0, 0x38, s_u8).astype(np.uint8)  # 1.0 in E4M3 = 0x38
    s_flt = np.where(max_abs[:, 0] == 0.0, 1.0, s_flt)[:, None]
    nibbles = encode_e2m1_nibbles(scaled / s_flt)
    return s_u8, nibbles


def main():
    root = Path(__file__).resolve().parent
    npz_path = root.parent / "sim" / "sim_snax_llm_layer_mid" / "real_llama32_1b_layer1_slice.npz"
    out_dir = root / "paper_syn" / "stim"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(npz_path)
    x_blocks = d["x_real"].astype(np.float32).reshape(-1, 16)
    w_all = np.concatenate([d[k].astype(np.float32).reshape(-1, 16) for k in ("wq", "wk", "wv", "wo", "wgu", "wd")], axis=0)

    act_idx = np.arange(TRACE_LEN) % len(x_blocks)
    wt_idx  = np.arange(TRACE_LEN * NUM_LANES) % len(w_all)
    act_blk = x_blocks[act_idx]
    wt_blk  = w_all[wt_idx]

    for fmt, qfn in [("mxfp4", quantize_blocks_mxfp4), ("nvfp4", quantize_blocks_nvfp4), ("log8", quantize_blocks_log8)]:
        as_u8, anib = qfn(act_blk)
        ws_u8, wnib = qfn(wt_blk)
        (out_dir / f"stim_{fmt}_act.hex").write_text("\n".join(pack_72b_hex(as_u8, anib)) + "\n")
        (out_dir / f"stim_{fmt}_wt.hex").write_text("\n".join(pack_72b_hex(ws_u8, wnib)) + "\n")
        print(f"Wrote {fmt}: act={len(as_u8)} pkts, wt={len(ws_u8)} pkts, "
              f"act_scale_min/max=({int(as_u8.min())},{int(as_u8.max())})")


if __name__ == "__main__":
    main()
