# Tier-A Algorithmic Evaluation for SNAX FP4 Edge LLM Subsystem

Bộ script Python mô phỏng **bit-exact** các khối lượng tử hóa trong `01_snax_llm_fp4_subsystem/`:
- `slq_encoder.sv` (`quantize_log8_fp4`)
- `logscale_frac_scaler` trong `log8_fp4_arith_units.sv` (`quantize_log8_fp4_csd`, bảng `CSD_Q5/32`)
- `softmax_requantizer.sv` (`exp_free_softmax_requantizer`)
- `snax_transpose_buffer.sv` / `snax_v_quant_path.sv` (`--v_quant_axis token`, lượng tử hóa `V` theo chiều token `N_KV`)
- Các baseline chuẩn & công bằng iso-bitwidth (`4.50 b/elem`, `B=16`):
  - `fp16`: FP16 gốc
  - `mxfp4`: OCP MXFP4 `B=32` (`4.25b`, `ceil` không clip)
  - `mxfp4_g32_floor`: OCP v1.0 MXFP4 chuẩn `B=32` (`4.25b`, `floor(log2(max)) - 2`)
  - `mxfp4_g16_floor`: OCP v1.0 MXFP4 `B=16` (`4.50b`, `floor` — đối chứng MiX Table IV)
  - `mxfp4_g16_ceil`: MXFP4 `B=16` (`4.50b`, `ceil` — sửa lỗi clip, cùng vạch bitwidth với NVFP4/Log8-FP4)
  - `nvfp4`: NVIDIA NVFP4 (`B=16`, `E4M3` + global FP32 scale)
  - `log8_fp4`: Log8-FP4 toán học (`2^(scale/8)`) + Softmax chuẩn FP16
  - `log8_fp4_csd`: Log8-FP4 với 3-term `CSD_Q5` + Softmax chuẩn FP16
  - `log8_fp4_expfree`: Full RTL datapath (`slq_encoder.sv` cho cả W và A/KV + `CSD_Q5` + `exp_free_softmax_requantizer` + V-transpose)
  - `log8_fp4_exact`: Exact-Log8 toán học (`round(8*log2(max/6))`, `_nearest_e2m1`) cho cả W và A/KV + Softmax chuẩn FP16 (ablation giới hạn trên của Log8)
  - `log8_fp4_exactw_csd`: **Offline CPU Exact-Log8 cho Weights DRAM (`CSD_Q5`)** + **Online `slq_encoder.sv` (`CSD_Q5`) cho A/KV** + Softmax chuẩn FP16 (**0 thay đổi RTL**)
  - `log8_fp4_exactw_expfree`: **Offline CPU Exact-Log8 cho Weights DRAM (`CSD_Q5`)** + **Online `slq_encoder.sv` (`CSD_Q5`) cho A/KV** + **Exp-Free `P4` Softmax** (**0 thay đổi RTL**, khớp luồng nạp `dram_weights.hex` thực tế)

## Cài đặt
```bash
pip install -r requirements.txt
```

## Chạy thực nghiệm nhanh (PPL + Attention O Error)
```bash
python eval_edge_llm.py --model Qwen/Qwen2.5-0.5B --eval_ppl --eval_attn_err
```

## Chạy đối chứng Offline Exact-Log8 Weights + Online SLQ A/KV (trên Colab, 0 thay đổi RTL)
```bash
python eval_edge_llm.py \
  --model Qwen/Qwen3-1.7B-Base \
  --configs fp16 nvfp4 log8_fp4_expfree log8_fp4_exactw_csd log8_fp4_exactw_expfree \
  --eval_ppl --seq_len 1024 --max_chunks 16
```

## Chạy đầy đủ cho Paper (PPL + Attention Error + Zero-shot Benchmarks)
```bash
python eval_edge_llm.py \
  --model meta-llama/Llama-3.2-1B \
  --eval_ppl --eval_attn_err --eval_zeroshot \
  --seq_len 1024 --max_chunks 64
```

## Tùy chọn cho Softmax & V-Transpose
- `--base_scale_mode log2e`: dùng hệ số toán học chính xác $1/\ln(2) = \log_2(e)$.
- `--base_scale_mode rtl_ln2`: dùng hằng số `LN2_Q16 = 45426/65536` khớp chính xác 100% với `snax_fused_llm_core.sv`.
- `--v_quant_axis {token,channel}`: mặc định `token` (lượng tử hóa `V` dọc chiều `N_KV` khớp 100% `snax_transpose_buffer.sv`; chọn `channel` nếu muốn ablation dọc `D_V`).

## Chạy trên Google Colab (GPU, `datasets >= 4.0`, `huggingface_hub >= 1.x`, `HF_TOKEN`)
- Script đã dùng dataset namespace chuẩn `Salesforce/wikitext` (tránh lỗi `Invalid HF URI` trên `huggingface_hub >= 1.x`) và hỗ trợ streaming C4 qua Parquet/JSON tương thích `datasets >= 4.0`.
- Truyền `HF_TOKEN` theo một trong 3 cách (ưu tiên từ trên xuống):
  1. Cờ CLI: `--hf_token <YOUR_TOKEN>`
  2. Biến môi trường Colab/Shell: `export HF_TOKEN=<YOUR_TOKEN>` (hoặc `%env HF_TOKEN=...` trong cell)
  3. **Google Colab Secrets**: tạo secret tên `HF_TOKEN`, script sẽ tự động đọc qua `google.colab.userdata`.
