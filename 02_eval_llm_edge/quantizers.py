"""
Bit-exact PyTorch/CUDA models for SNAX FP4 LLM Subsystem (Tier A Evaluation).
Matches RTL modules in 01_snax_llm_fp4_subsystem/:
  - slq_encoder.sv          -> quantize_log8_fp4()
  - softmax_requantizer.sv  -> exp_free_softmax_requantizer()
  - Baselines               -> quantize_mxfp4(), quantize_nvfp4()
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# E2M1 magnitudes: 3'b000..3'b111 -> [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
E2M1_MINT   = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0], dtype=torch.float32)

# Midpoints for nearest E2M1 rounding (used by MXFP4 / NVFP4)
E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)

# CSD_Q5 table from softmax_requantizer.sv: 32 * 2^(r/8) for r in 0..7
CSD_Q5 = torch.tensor([32.0, 35.0, 38.0, 41.0, 46.0, 49.0, 54.0, 59.0], dtype=torch.float32)


def _pad_to_block(x: torch.Tensor, block_size: int):
    """Pad last dimension of x to a multiple of block_size."""
    orig_shape = x.shape
    last_dim = orig_shape[-1]
    pad = (block_size - (last_dim % block_size)) % block_size
    if pad > 0:
        x = F.pad(x, (0, pad), value=0.0)
    blocked = x.reshape(*orig_shape[:-1], -1, block_size)
    return blocked, orig_shape, pad


def _unpad_from_block(blocked: torch.Tensor, orig_shape, pad: int):
    out = blocked.reshape(*orig_shape[:-1], -1)
    if pad > 0:
        out = out[..., : orig_shape[-1]]
    return out


def _nearest_e2m1(val: torch.Tensor) -> torch.Tensor:
    """Round arbitrary tensor to nearest signed E2M1 value in [-6, 6]."""
    sign = torch.where(val < 0, -1.0, 1.0)
    abs_v = val.abs()
    mids = E2M1_MIDPOINTS.to(val.device)
    # bucketize returns index 0..7 corresponding to E2M1_LEVELS
    idx = torch.bucketize(abs_v, mids, right=False)
    levels = E2M1_LEVELS.to(val.device)
    q_mag = levels[idx]
    # Ensure zero stays +0
    q_mag = torch.where(q_mag == 0.0, 0.0, q_mag * sign)
    return q_mag


@torch.no_grad()
def quantize_log8_fp4(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """
    Bit-exact simulation of slq_encoder.sv (log_quantizer):
      - Converts input to IEEE-754 FP16
      - Extracts mu = {E_fp16[4:0], M_fp16[9:7]} (0 if E==0, 247 if E==31)
      - Computes mu_max per block of 16
      - scale_sat = clamp(mu_max - 140, -128, 127) (0 if mu_max == 0)
      - diff = mu_max - mu
      - Thresholds TH = [2, 6, 10, 14, 18, 24, 36] -> E2M1 magnitude
      - Dequantized value: sign * E2M1[mag] * 2^(scale_sat / 8)
    """
    orig_dtype = x.dtype
    blocked, orig_shape, pad = _pad_to_block(x.float(), block_size)

    # Convert to IEEE FP16 bit representation (uint16 stored in int32)
    fp16 = blocked.to(torch.float16)
    bits = fp16.view(torch.int16).to(torch.int32) & 0xFFFF

    sign_bit = (bits >> 15) & 0x1
    exp_fp16 = (bits >> 10) & 0x1F
    man_top3 = (bits >> 7) & 0x7

    mu = (exp_fp16 << 3) | man_top3
    mu = torch.where(exp_fp16 == 0, torch.zeros_like(mu), mu)
    mu = torch.where(exp_fp16 == 31, torch.full_like(mu, 247), mu)

    mu_max = mu.max(dim=-1, keepdim=True).values
    scale_raw = mu_max - 140
    scale_sat = torch.clamp(scale_raw, -128, 127)
    scale_sat = torch.where(mu_max == 0, torch.zeros_like(scale_sat), scale_sat)

    diff = mu_max - mu

    # Thresholds from slq_encoder.sv:
    # TH_6P0=2, TH_4P0=6, TH_3P0=10, TH_2P0=14, TH_1P5=18, TH_1P0=24, TH_0P5=36
    mag_idx = torch.zeros_like(diff)
    mag_idx = torch.where(diff <= 36, torch.full_like(mag_idx, 1), mag_idx)
    mag_idx = torch.where(diff <= 24, torch.full_like(mag_idx, 2), mag_idx)
    mag_idx = torch.where(diff <= 18, torch.full_like(mag_idx, 3), mag_idx)
    mag_idx = torch.where(diff <= 14, torch.full_like(mag_idx, 4), mag_idx)
    mag_idx = torch.where(diff <= 10, torch.full_like(mag_idx, 5), mag_idx)
    mag_idx = torch.where(diff <= 6,  torch.full_like(mag_idx, 6), mag_idx)
    mag_idx = torch.where(diff <= 2,  torch.full_like(mag_idx, 7), mag_idx)

    zero_elem = (mu == 0) | (mu_max == 0)
    mag_idx = torch.where(zero_elem, torch.zeros_like(mag_idx), mag_idx)

    levels = E2M1_LEVELS.to(x.device)
    mag_val = levels[mag_idx]
    sign_val = torch.where(sign_bit == 1, -1.0, 1.0)
    sign_val = torch.where(mag_idx == 0, 1.0, sign_val)

    scale_val = torch.pow(2.0, scale_sat.float() / 8.0)
    dequant = sign_val * mag_val * scale_val

    out = _unpad_from_block(dequant, orig_shape, pad)
    return out.to(orig_dtype)


@torch.no_grad()
def quantize_mxfp4(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """
    OCP MXFP4 Baseline:
      - Block size = 32
      - Scale = E8M0 (pure power-of-2: 2^floor(log2(max_abs / 6.0)))
      - Data  = E2M1
    """
    orig_dtype = x.dtype
    blocked, orig_shape, pad = _pad_to_block(x.float(), block_size)

    max_abs = blocked.abs().max(dim=-1, keepdim=True).values
    # E8M0: largest power of 2 <= max_abs / 6.0 (or ceil so no clipping; OCP uses floor(log2(max)) - 2)
    safe_max = torch.clamp(max_abs / 6.0, min=2.0 ** -127)
    exp_e8m0 = torch.ceil(torch.log2(safe_max)).clamp(-127, 127)
    scale = torch.pow(2.0, exp_e8m0)
    scale = torch.where(max_abs == 0.0, torch.ones_like(scale), scale)

    q_e2m1 = _nearest_e2m1(blocked / scale)
    dequant = q_e2m1 * scale
    return _unpad_from_block(dequant, orig_shape, pad).to(orig_dtype)


@torch.no_grad()
def quantize_nvfp4(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """
    NVIDIA NVFP4 Baseline:
      - Block size = 16
      - Scale = E4M3 FP8 (simulated via torch.float8_e4m3fn if available, else E4M3 rounding)
      - Tensor-level FP32 scale to keep E4M3 in dynamic range
      - Data = E2M1
    """
    orig_dtype = x.dtype
    x_f = x.float()
    global_max = x_f.abs().max()
    if global_max == 0:
        return x
    # E4M3 max normal is 448.0; scale tensor so max block scale fits nicely in E4M3
    tensor_scale = global_max / (6.0 * 448.0)
    tensor_scale = torch.clamp(tensor_scale, min=1e-12)

    blocked, orig_shape, pad = _pad_to_block(x_f / tensor_scale, block_size)
    max_abs = blocked.abs().max(dim=-1, keepdim=True).values
    blk_scale_fp32 = torch.clamp(max_abs / 6.0, min=1e-12)

    if hasattr(torch, "float8_e4m3fn"):
        blk_scale_fp8 = blk_scale_fp32.to(torch.float8_e4m3fn).float()
    else:
        # Fallback: 3-bit mantissa rounding
        exp = torch.floor(torch.log2(blk_scale_fp32))
        mant = blk_scale_fp32 / torch.pow(2.0, exp)
        blk_scale_fp8 = torch.round(mant * 8.0) / 8.0 * torch.pow(2.0, exp)

    blk_scale_fp8 = torch.where(max_abs == 0.0, torch.ones_like(blk_scale_fp8), blk_scale_fp8)
    q_e2m1 = _nearest_e2m1(blocked / blk_scale_fp8)
    dequant = q_e2m1 * blk_scale_fp8 * tensor_scale
    return _unpad_from_block(dequant, orig_shape, pad).to(orig_dtype)


@torch.no_grad()
def exp_free_softmax_requantizer(
    scores: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    base_scale_mode: str = "log2e",
) -> torch.Tensor:
    """
    Bit-exact simulation of softmax_requantizer.sv + snax_fused_llm_core.sv:
      - scores: [..., Q_len, K_len] (already scaled by 1/sqrt(d_k))
      - attn_mask: additive mask (0 for valid, large negative for masked) or boolean mask
      - base_scale_mode:
          'rtl_ln2' : exact constant from snax_fused_llm_core.sv (LN2_Q16 = 45426/65536)
          'log2e'   : mathematically exact base-2 conversion 1/ln(2) = log2(e)
    Returns effective normalized probability matrix P_hat [..., Q_len, K_len]
    equivalent to quantized P * V / l in the hardware datapath.
    """
    orig_dtype = scores.dtype
    s = scores.float()

    # Determine valid key mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            valid_mask = attn_mask
        else:
            valid_mask = attn_mask > -1e4
    else:
        valid_mask = torch.ones_like(s, dtype=torch.bool)

    # Convert to Q16.16 fixed-point as in vsq_matmul_engine drain -> softmax_requantizer
    # Apply base conversion
    if base_scale_mode == "rtl_ln2":
        scale_factor = 45426.0 / 65536.0
    else:
        scale_factor = 1.0 / math.log(2.0)

    s_q16 = torch.round(s * scale_factor * 65536.0).clamp(-2147483648.0, 2147483647.0)

    # Pad K_len to blocks of 16
    block_size = 16
    k_len = s_q16.shape[-1]
    pad = (block_size - (k_len % block_size)) % block_size
    if pad > 0:
        s_q16 = F.pad(s_q16, (0, pad), value=-2147483648.0)
        valid_mask = F.pad(valid_mask, (0, pad), value=False)

    s_blk = s_q16.reshape(*s_q16.shape[:-1], -1, block_size)       # [..., num_blk, 16]
    m_blk = valid_mask.reshape(*valid_mask.shape[:-1], -1, block_size)

    S_MIN = -2147483648.0
    s_masked = torch.where(m_blk, s_blk, torch.full_like(s_blk, S_MIN))

    # Block max & global row max m
    blk_max = s_masked.max(dim=-1, keepdim=True).values            # [..., num_blk, 1]
    row_max = blk_max.max(dim=-2, keepdim=True).values             # [..., 1, 1]

    # Per-key distance in Q16: d = blk_max - s_score
    score_dist = blk_max - s_masked

    # Thresholds from softmax_requantizer.sv (Q16)
    # [17238, 50961, 82774, 116497, 148310, 196608, 300480]
    mag_idx = torch.zeros_like(score_dist, dtype=torch.long)
    mag_idx = torch.where(score_dist <= 300480.0, torch.full_like(mag_idx, 1), mag_idx)
    mag_idx = torch.where(score_dist <= 196608.0, torch.full_like(mag_idx, 2), mag_idx)
    mag_idx = torch.where(score_dist <= 148310.0, torch.full_like(mag_idx, 3), mag_idx)
    mag_idx = torch.where(score_dist <= 116497.0, torch.full_like(mag_idx, 4), mag_idx)
    mag_idx = torch.where(score_dist <= 82774.0,  torch.full_like(mag_idx, 5), mag_idx)
    mag_idx = torch.where(score_dist <= 50961.0,  torch.full_like(mag_idx, 6), mag_idx)
    mag_idx = torch.where(score_dist <= 17238.0,  torch.full_like(mag_idx, 7), mag_idx)
    mag_idx = torch.where(m_blk, mag_idx, torch.zeros_like(mag_idx))

    levels = E2M1_LEVELS.to(s.device)
    code_val = levels[mag_idx]  # [..., num_blk, 16]

    # Block scale: p_blk = clamp(((blk_max - row_max + P_RND) >>> 13) - 21, -128, 127)
    # P_RND = 2^(16-4) = 4096
    p_wide = torch.floor((blk_max - row_max + 4096.0) / 8192.0) - 21.0
    p_blk = torch.clamp(p_wide, -128.0, 127.0)

    # S_blk = 2^q * CSD_Q5[r] / 32, where q = floor(p_blk / 8), r = p_blk mod 8
    q_s = torch.floor(p_blk / 8.0)
    r_s = (p_blk.long() & 7)
    csd = CSD_Q5.to(s.device)[r_s]
    s_blk_val = torch.pow(2.0, q_s) * (csd / 32.0)

    # Unnormalized P = code_val * S_blk; l = sum(P) across row
    p_unnorm = code_val * s_blk_val
    p_unnorm = torch.where(m_blk, p_unnorm, torch.zeros_like(p_unnorm))
    p_unnorm = p_unnorm.reshape(*s.shape[:-1], -1)[..., :k_len]

    l_sum = p_unnorm.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    p_hat = p_unnorm / l_sum
    return p_hat.to(orig_dtype)
