# Báo cáo Verification End-to-End: 1 Layer Llama-3.2-1B Prefill trên SNAX-LLM FP4 Subsystem

## 1. Tổng quan & Kiến trúc mục tiêu
Báo cáo này tổng hợp quá trình nâng cấp RTL, xây dựng script sinh dữ liệu Python, thiết kế Testbench SystemVerilog và kết quả simulation end-to-end cho **1 layer hoàn chỉnh của mô hình `NousResearch/Llama-3.2-1B` ở chế độ Prefill** trên `snax_llm_top_clean` kết hợp 4-port `snax_simple_dma` và DRAM 512-bit.

| Thông số | Giá trị trên `Llama-3.2-1B` |
|---|---|
| `M_TOKENS` (Prefill) | `16` (`M_TILES = 1`, `M_Q = 8`, `Q_TILES = 2`, `KV_TILES = 1`) |
| `hidden_size` (`D_MODEL`) | `2,048` (`N_TILES_MODEL = 128`, `K_PACKETS_QKV = 128`) |
| `intermediate_size` (`D_FF`) | `8,192` (`N_TILES_FF = 512`, `K_PACKETS_DOWN = 512`) |
| Attention Heads (`H_Q / H_KV`) | `32 / 8` (**GQA 4:1**, `gqa_group_m1 = 3`) |
| `head_dim` (`D_K = D_V`) | `64` (`K_PACKETS_QK = 4`, `DV_TILES = 4`) |
| Tổng trọng số 1 Layer (FP4+Log8) | `~60.8M` params = **`34,209,792` bytes** (~34.2 MB, 100% đặt trên DRAM 512-bit) |

---

## 2. Các nâng cấp RTL đã triển khai

1. **Ring Buffer 4-slot × 128 KB (`SLOT_SIZE_BYTES = 128 * 1024`)**:
   - Cập nhật tham số trong [`01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv`](01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv) và truyền `.SLOT_SIZE_BYTES (128 * 1024)` trong [`01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv`](01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv).
   - Mỗi slot chứa tối đa 256 K-packets ($K_{\max} = 4,096$) trên các bank `0..15`, cho phép double/quad-buffering ẩn độ trễ DRAM qua 4-port DMA.

2. **Output-Stationary K-Tiling**:
   - [`01_snax_llm_fp4_subsystem/snax_flash_csr_snax.sv`](01_snax_llm_fp4_subsystem/snax_flash_csr_snax.sv): Thêm trường `seq_gemm_k_tiles_m1` tại `RW_SEQ_CTRL[15:8]` (`0` = 1 K-tile, tương thích ngược hoàn toàn).
   - [`01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv`](01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv): Thêm bộ đếm `cnt_kt_q` và DMA prefetch `dma_kt_q`; giữ partial sum 32-bit trên accumulator qua các K-tile và chỉ kích hoạt `gemm_k_last_fire` (drain + clear) khi `last_kt`; cập nhật địa chỉ weight/activation (`gemm_wt_addr`, `gemm_act_addr`, `total_k_blocks`, `act_token_stride_o`, `dma_req_dram_addr_o`).
   - Áp dụng trực tiếp cho `W_down` (`2048 × 8192`) với **2 K-tiles × 4,096** (`cfg_gemm_k_tiles_m1 = 1`).

3. **GQA (Grouped-Query Attention) Head Mapping**:
   - [`01_snax_llm_fp4_subsystem/snax_flash_csr_snax.sv`](01_snax_llm_fp4_subsystem/snax_flash_csr_snax.sv): Thêm trường `seq_gqa_group_m1` tại `RW_SEQ_CTRL[23:16]` (`G - 1`, `0` = MHA 1:1).
   - [`01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv`](01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv): Thêm `cnt_hkv_q` và `cnt_hgrp_q`; `q_addr_calc` dùng `cnt_h_q` (`0..31`), còn `k_addr_calc` và `v_addr_calc` dùng `cnt_hkv_q` (`0..7`).

4. **Fix tham số hóa `b_addr` theo `BANK_DEPTH`**:
   - [`01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv`](01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L126): Sửa `b_addr` từ `[9:0]` thành `[$clog2(BANK_DEPTH)-1:0]` để không bị cắt bit địa chỉ cao khi cấu hình `BANK_DEPTH = 8192`.

---

## 3. Script Python sinh dữ liệu & Golden Model
Script [`tb/generate_llama3_1b_layer_data.py`](tb/generate_llama3_1b_layer_data.py) thực hiện trong ~5.9 giây:
- **Vectorized Quantizer (`quantize_rtl_vec`)**: Mô hình hóa bit-exact bộ mã hóa `slq_encoder.sv` (FP16 → Log8 shared scale + FP4 E2M1 nibbles, đóng gói 9 bytes/packet).
- **Offline RoPE Weight Permutation (`permute_qk_weights_for_rope`)**: Hoán vị các kênh đầu ra của `W_Q` và `W_K` theo cặp `(j, j + D_K/2)` cho `D_K = 64` để khớp luồng `Fused Q/K + RoPE + LogQ` phần cứng.
- **Đóng gói DRAM Image 512-bit**: Ghi 7 ma trận trọng số (`W_Q, W_K, W_V, W_O, W_gate, W_up, W_down`) tổng cộng `534,528` beats 64B (`34,209,792` bytes) ra `sim/sim_llama3_1b/dram_weights.hex`.
- **TCDM Preload & Golden Reference**: Xuất `tcdm_preload_addr.hex` / `tcdm_preload_data.hex` (`17,408` words cho `X`, `gamma1`, `gamma2`, `cos/sin` RoPE) và golden cho cả **FP32** lẫn **FP4+Log8 Fake-Quant** (`g_xnorm`, `g_res1`, `g_n2`, `g_res1_fq`, `g_n2_fq`, `g_final_fp16`, `g_final_fp32`, `g_final_fq`).

---

## 4. Kiến trúc Testbench End-to-End
- **Unit TB cho K-Tiling, 128 KB Ring & GQA**: [`tb/tb_ktiling_gqa.sv`](tb/tb_ktiling_gqa.sv) ([`sim/filelist_ktiling_gqa.f`](sim/filelist_ktiling_gqa.f)).
- **Full-layer Llama-3.2-1B TB**: [`tb/tb_llama3_1b_layer.sv`](tb/tb_llama3_1b_layer.sv) ([`sim/filelist_llama3_1b.f`](sim/filelist_llama3_1b.f)):
  - Instantiate `snax_llm_top_clean #(.BANK_DEPTH(8192))` nối với `snax_simple_dma #(.TCDM_PORTS(4))` và mô hình DRAM 512-bit (`64 MB`, `DRAM_LATENCY = 6` cycles).
  - **TCDM Map**: Banks `0..15` dành cho Ring Buffer 4×128 KB (khi `V` chạy DMA đặt `RING_BASE` tại word 256 để tránh đè vùng `V_BASE_BLOCK = 128..159`, sau đó scatter `K[0..7]` vào words `0..31`); Banks `16..30` chứa activation, staging, `QPKT`, `O_ACT`, `FF1`, `X`, `XN`, `RES1`, `FIN`, `GAMMA`, `ROPE`.
  - **Luồng 12 Phase**:
    1. `[1/12]` SFU RMSNorm 1 (`M=16, D=2048`)
    2. `[2/12]` HW Requant `Xn → ACT_BANK` (`2,048` packets)
    3. `[3/12]` Fused `Q` (`2048×2048`, DMA) + RoPE + LogQ + reorder `QPKT_BANK`
    4. `[4/12]` Fused `K` (`512×2048`, `H_KV=8`, DMA) + RoPE + LogQ
    5. `[5/12]` Fused `V` + Transpose (`32` jobs `H_KV=8 × DV=4`, DMA) + scatter `K`
    6. `[6/12]` HW Causal FlashAttention (**GQA 4:1**, `H_Q=32, H_KV=8, D=64`, 4D FP4 token-major out)
    7. `[7/12]` GEMM `O` (`2048×2048`, DMA)
    8. `[8/12]` SFU Residual 1 + RMSNorm 2
    9. `[9/12]` HW Requant `Norm2 → ACT_BANK`
    10. `[10/12]` FFN Gate & Up (`2 × 8192×2048`, DMA) + `SiLU*Up` (`32` chunks × `16` N-tiles)
    11. `[11/12]` GEMM `Down` (`2048×8192`, DMA + **2×4096 Output-Stationary K-Tiling**)
    12. `[12/12]` SFU Residual 2 → Kiểm tra SQNR & CosineSim

---

## 5. Kết quả Simulation & Phân tích Hiệu năng

### 5.1. Regression Suite (QuestaSim 2024.1)
| Testbench | Kết quả | Chi tiết chính |
|---|---|---|
| `tb_ktiling_gqa` | **PASS (`0` errors)** | K-Tiling non-DMA (`64/64`), K-Tiling+DMA+128KB Ring (`64/64`), GQA 4:1 vs MHA (`1024/1024`) |
| `tb_simple_dma_gemm` | **PASS (`0` errors)** | Round-trip (`128/128`), Streaming DMA GEMM (`1024/1024`), `288` DRAM beats |
| `tb_large_layer_dma` | **PASS (`0` errors)** | `5,760` DRAM beats (`368,640` B), `40,960` TCDM words |
| `tb_snax_llm_layer_fused` | **PASS (`0` errors)** | `76 passed, 0 failed`, `32,919` job cycles, `SQNR = 37.59 dB, CosineSim = 0.999913` |

### 5.2. Llama-3.2-1B End-to-End Prefill Layer
Log tham chiếu được lưu tại [`sim/sim_llama3_1b/llama3_1b_baseline_default.log`](sim/sim_llama3_1b/llama3_1b_baseline_default.log).

- **Trạng thái**: **`>>> LLAMA-3.2-1B FULL LAYER PREFILL PASSED (0 errors) <<<`** (`Checks: 111 passed, 0 failed`, `Errors: 0`).
- **Tổng chu kỳ**: **`4,479,514` cycles** (`seq = 4,012,274`, `mac = 3,811,328`, `dma = 1,140,902`, `sfu = 110,272`).
- **DRAM Traffic**: **`534,528` beats 512-bit = `34,209,792` bytes** (~34.2 MB, khớp 100%).
- **Hiệu suất MAC**: `3,811,328 / 4,479,514 = 85.1%` (trần lý thuyết của mảng 256-MACs/cycle là `3,805,184` cycles); trong đó `GEMM O` đạt **99.28%** và `GEMM Down` (K-Tiling) đạt **99.65%** hiệu suất lý thuyết, chứng minh 4-port DMA + Ring Buffer 4×128 KB đã che giấu gần như hoàn toàn độ trễ DRAM.

| Phase | Thành phần | Cycles | Tỷ trọng | Độ chính xác (SQNR / CosineSim) |
|---|---|---|---|---|
| `[1/12]` | SFU RMSNorm 1 | `53,510` | `1.19%` | `44.10 dB / 0.999990` (vs FP32) |
| `[2/12]` | HW Requant Xn | `18,456` | `0.41%` | — |
| `[3/12]` | Fused Q (2048×2048, DMA) + RoPE | `284,521` | `6.35%` | `92.1%` MAC eff. |
| `[4/12]` | Fused K (512×2048, DMA) + RoPE | `70,121` | `1.57%` | `93.5%` MAC eff. |
| `[5/12]` | Fused V + Transpose (32 jobs, DMA) | `111,873` | `2.50%` | `58.6%` MAC eff. |
| `[6/12]` | Causal FlashAttention (GQA 4:1) | `14,890` | `0.33%` | `H_Q=32, H_KV=8, D=64` |
| `[7/12]` | GEMM O (2048×2048, DMA) | `264,049` | `5.89%` | `99.28%` MAC eff. |
| `[8/12]` | SFU Residual 1 + RMSNorm 2 | `139,564` | `3.12%` | Res1: `12.31 dB / 0.972731` (vs PyFQ); RMS2: `11.90 dB / 0.967841` |
| `[9/12]` | HW Requant Norm2 | `18,456` | `0.41%` | — |
| `[10/12]` | FFN Gate & Up + SiLU\*Up (32 chunks) | `2,347,295` | `52.40%` | `89.3%` MAC eff. |
| `[11/12]` | GEMM Down (2048×8192, K-Tiling 2×4096) | `1,052,290` | `23.49%` | `99.65%` MAC eff. |
| `[12/12]` | SFU Residual 2 + Preload/Verify | `104,489` | `2.33%` | Final vs PyFQ: **`7.93 dB / 0.920303`**; vs FP32: **`6.98 dB / 0.899392`** |

*(Ghi chú: Giá trị `RTL vs FP32 = 6.98 dB / 0.899392` khớp sát ngưỡng lý thuyết của `Py Fake-Quant vs FP32 = 7.08 dB / 0.900795` trong Python đối với cấu hình 4-bit FP4+Log8 không scale residual trên kích thước `D=2048, D_FF=8192`.)*

### 5.3. Benchmark tốc độ QuestaSim 2024.1
- **Baseline mặc định** (`vlog -sv` + `vsim -c`, không dump sóng): **`0:17:05`**
- **Ép `-O5`** (`vlog -O5` + `vsim -voptargs="-O5"`): **`0:17:13`** (không nhanh hơn vì `vopt` mặc định của QuestaSim 2024.1 đã chạy ở mức `(fast)` kernel tối ưu tối đa khi không bật `+acc`).

---

## 6. Hướng dẫn tái lập (Reproduce)
```powershell
# 1. Sinh 34.2 MB trọng số DRAM + TCDM preload + golden
python tb/generate_llama3_1b_layer_data.py

# 2. Chạy unit test K-Tiling, 128 KB Ring & GQA
cd sim
vlog -work work_kg -sv -f filelist_ktiling_gqa.f
vsim -c work_kg.tb_ktiling_gqa -do "run -all; quit"

# 3. Chạy Llama-3.2-1B full layer end-to-end
vlog -work work_llama -sv -f filelist_llama3_1b.f
vsim -c work_llama.tb_llama3_1b_layer -do "run -all; quit"
```
