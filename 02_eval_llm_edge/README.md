# Tier-A Algorithmic Evaluation for SNAX FP4 Edge LLM Subsystem

Bộ script Python mô phỏng **bit-exact** các khối lượng tử hóa trong `01_snax_llm_fp4_subsystem/`:
- `slq_encoder.sv` (`quantize_log8_fp4`)
- `softmax_requantizer.sv` (`exp_free_softmax_requantizer`)
- Các baseline chuẩn: `FP16`, `MXFP4` (OCP, block 32, E8M0), `NVFP4` (block 16, E4M3).

## Cài đặt
```bash
pip install -r requirements.txt
```

## Chạy thực nghiệm nhanh (PPL + Attention O Error)
```bash
python eval_edge_llm.py --model Qwen/Qwen2.5-0.5B --eval_ppl --eval_attn_err
```

## Chạy đầy đủ cho Paper (PPL + Attention Error + Zero-shot Benchmarks)
```bash
python eval_edge_llm.py \
  --model meta-llama/Llama-3.2-1B \
  --eval_ppl --eval_attn_err --eval_zeroshot \
  --seq_len 1024 --max_chunks 64
```

## Chế độ scale trong Softmax
- `--base_scale_mode log2e`: dùng hệ số toán học chính xác $1/\ln(2) = \log_2(e)$.
- `--base_scale_mode rtl_ln2`: dùng hằng số `LN2_Q16 = 45426/65536` khớp chính xác 100% với `snax_fused_llm_core.sv`.
