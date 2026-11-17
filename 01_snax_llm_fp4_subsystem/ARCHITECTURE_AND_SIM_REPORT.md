# Tổng hợp Kiến trúc & Kết quả Simulation (`snax_llm_top_clean`)

Tài liệu này tổng hợp kiến trúc phần cứng của subsystem tăng tốc LLM FP4 + Log8-Scale (`snax_llm_top_clean`), các nâng cấp RTL mới nhất (`Core -> SFU` fused path, `LOG2E_Q16` softmax, `4D Write-Out`), và kết quả mô phỏng xác minh end-to-end trên QuestaSim 2024.1.

---

## 1. Kiến trúc Hệ thống (`snax_llm_top_clean`)

### 1.1. Sơ đồ khối & Luồng dữ liệu chính
- **TCDM & Crossbar (`interleaved_interconnect_72b`, `multi_bank_tcdm_72b`)**:
  - 32 bank SRAM × 72-bit (1024 words/bank), kết nối **22 masters**:
    - `M0`: DMA, `M1`: CPU host
    - `M2`: `u_act_streamer` (`snax_streamer_act_bcast`)
    - `M3..18`: `u_wt_streamer` (`snax_streamer_wt_16ch`, 16 kênh song song zero-conflict)
    - `M19`: `u_write_out` (`snax_streamer_write_out`)
    - `M20`: `u_read_fp16` (`snax_stream_read_fp16` — luồng A cho SFU / Log-Quantizer)
    - `M21`: `u_read_fp16_b` (`snax_stream_read_fp16` — luồng B cho SFU Residual / RoPE sin-cos / SwiGLU Up)
- **Lõi tính toán chính (`u_core` — `snax_fused_llm_core`)**:
  - Hỗ trợ Dense GEMM (FP4+Log8), Fused `V + Transpose`, và Causal FlashAttention (`QK^T -> Softmax/Online-Rescale -> PV`).
  - **Cập nhật Softmax base-2 (`LOG2E_Q16 = 94548`)**: Sửa hằng số chuyển đổi cơ số trước `softmax_requantizer` từ `LN2_Q16 (45426)` sang **`LOG2E_Q16 = 94548`** ($1/\ln 2$ ở Q16.16) bằng bộ nhân CSD 7-term không dùng multiplier: `(<<<16) + (<<<15) - (<<<12) + (<<<8) + (<<<6) + (<<<4) + (<<<2)` để thực thi đúng $e^s = 2^{s/\ln 2}$.
- **Bộ xử lý phi tuyến (`u_sfu` — `snax_llm_sfu_opt`)**:
  - Thực thi `BYPASS`, `RESIDUAL`, `RMSNORM` (2-pass), `ROPE`, `SWIGLU` (FP20 nội bộ, `rsqrt`, `sigmoid PWL`).
  - **Cập nhật đường nối thẳng `Core -> SFU` (`sfu_in_a_src = 1`, `RW_SFU_CTRL[21]`)**: Thêm hàng đợi **`fused_fifo` (4-entry)** nối thẳng `core_fp16_packed` từ `u_core` vào cổng `in_a` của `u_sfu`, đồng thời nối `core_out_fifo_ready_i` của `u_seq` vào `!fused_fifo_full` khi bật chế độ fused.
- **Khối nén & chuyển vị (`u_log_quantizer`, `u_transpose_buffer`)**:
  - `u_log_quantizer` (`log_quantizer` / `slq_encoder`): lượng tử hóa online vector 16×FP16 sang gói 72-bit (64-bit E2M1 FP4 + 8-bit Log8 scale).
  - Nhận đầu vào từ `u_core`, `u_transpose_buffer`, `u_read_fp16`, hoặc `u_sfu` (`sfu_out_dst = 0`).
- **Bộ ghi 4D (`u_write_out` — `snax_streamer_write_out`)**:
  - **Cập nhật `MODE_FP4_4D = 2'd3` (`WRITE_FP4_4D`)**: Hỗ trợ 4 vòng lặp lồng nhau (`dim0..dim3`, mỗi chiều có `cnt`/`stride` độc lập) được điều khiển từ `snax_llm_sequencer`:
    - **FlashAttention (`ST_PREP_DRAIN`, `out_mode = 2'd2`)**: ghi thẳng đầu ra theo layout **token-major** cho `GEMM O`.
    - **Standalone Requant (`ST_RD_LAUNCH`, `out_mode = 2'd3`)**: ghi thẳng layout Attention `Q` và rải 16 bank `0..15` cho `K` mà không cần CPU reorder.

---

## 2. Kết quả Simulation (QuestaSim 2024.1)

### 2.1. System SFU & Fused `Core -> SFU` ([tb_snax_sfu_system.sv](file:///d:/LVThs/NPU_PRJ/NPU_LOG/tb/tb_snax_sfu_system.sv))
- **Kết quả**: `Errors: 0, Warnings: 17` (`Time: 8295 ns`)
- **5/5 test case PASS**:
  - `[PASS] System RESIDUAL`
  - `[PASS] System ROPE`
  - `[PASS] System SWIGLU`
  - `[PASS] System RMSNORM`
  - `[PASS] System FUSED Core->SFU RESIDUAL (16x16 GEMM + Residual, sfu_in_a_src=1)`

### 2.2. End-to-End LLaMA Prefill Layer ([tb_snax_llm_layer_mid.sv](file:///d:/LVThs/NPU_PRJ/NPU_LOG/tb/tb_snax_llm_layer_mid.sv))
- **Cấu hình**: `M_TOKENS = 32`, `D_MODEL = 128`, `D_FF = 256`, `N_HEADS = 4`, `D_K = D_V = 32` (`Q_TILES=4, KV_TILES=2, DV_TILES=2`).
- **Trạng thái**: `Errors: 0`, `Checks: 80 passed, 0 failed`.

#### A. Độ chính xác số học trên dữ liệu thực `NousResearch/Llama-3.2-1B` (Layer 1 slice)
Trích xuất từ residual stream thực sau Layer 0 (`x_real std=0.0409, max_abs=0.2842`), fuse $d_k^{-1/4}$ vào `W_Q, W_K`:

| Cặp so sánh | SQNR (dB) | Cosine Similarity | Ghi chú |
| :--- | :---: | :---: | :--- |
| **`LLM_eval` (`02_eval_llm_edge/quantizers.py`) vs `FP32`** | `38.69 dB` | `0.999932` | Mô hình đánh giá PPL (`log2e`) |
| **`Golden Model` (`PyFQ`) vs `FP32`** | `37.27 dB` | `0.999906` | Bit-exact `u_core` + `u_log_quantizer` |
| **`RTL` (`snax_llm_top_clean`) vs `FP32`** | **`37.59 dB`** | **`0.999913`** | `max_abs = 0.00384`, **`fail = 0 / 4096`** |
| **`RTL` vs `Golden Model` (`PyFQ`)** | **`39.33 dB`** | **`0.999942`** | Độ lệch nhỏ chỉ do SFU `FP20/PWL` |
| **`LLM_eval` vs `Golden Model` (`PyFQ`)** | `39.24 dB` | `0.999940` | Nhất quán giữa PyTorch và RTL fixed-point |

#### B. Phân rã số Cycle từng Phase (`Total = 194,348` cycles)

| Phase | Tác vụ | Khối thực thi | HW Job Cycles (`u_seq`) | Wall Cycles (TB) |
| :---: | :--- | :--- | :---: | :---: |
| **`[1]`** | DMA Preload (12,000 words) | `DMA -> TCDM` | — | `23,169` |
| **`[2]`** | SFU RMSNorm 1 (32 tokens) | `u_sfu` | — | `6,080` |
| **`[3]`** | Requant `Xn` (256 vec) | `u_log_quantizer` | `2,315` | `2,329` |
| **`[4]`** | GEMM `Q` (`32×128×128`) | `u_core` | `2,184` | `2,201` |
| **`[5]`** | SFU RoPE `Q` (256 vec) | `u_sfu` | — | `23,807` |
| **`[6]`** | Requant `Q` **4D** (`QPKT_BANK`) | `u_log_quantizer + 4D WO` | `2,315` | `2,345` |
| **`[7]`** | GEMM `K` (`32×128×128`) | `u_core` | `2,184` | `2,201` |
| **`[8]`** | SFU RoPE `K` (256 vec) | `u_sfu` | — | `23,807` |
| **`[9]`** | Requant `K` **4D** (`banks 0..15`) | `u_log_quantizer + 4D WO` | `2,315` | `2,345` |
| **`[10]`** | Fused `V` + Transpose (16 tiles) | `u_core + transpose` | `3,024` | `3,328` |
| **`[11]`** | Reorder `Q/K` | *(Loại bỏ nhờ 4D WO)* | `0` | `0` |
| **`[12]`** | FlashAttention **4D** (`4H, 2KV, 2DV`) | `u_core + 4D WO` | `2,707` | `2,734` |
| **`[13]`** | GEMM `O` (`32×128×128`) | `u_core` | `2,184` | `2,201` |
| **`[14a]`** | SFU Residual 1 (256 vec) | `u_sfu` | — | `12,543` |
| **`[14b]`** | SFU RMSNorm 2 (32 tokens) | `u_sfu` | — | `6,080` |
| **`[15]`** | Requant `Norm2` (256 vec) | `u_log_quantizer` | `2,315` | `2,329` |
| **`[16]`** | FFN `Gate/Up` (4 GEMM) + `cpu_copy_gate` + `SiLU` | `u_core + cpu + u_sfu` | `8,736` | `43,107` |
| **`[17]`** | Requant `FFN` (`lo+hi`, 512 vec) | `u_log_quantizer` | `4,630` | `4,657` |
| **`[18]`** | GEMM `Down` (`32×256×128`) | `u_core` | `4,232` | `4,249` |
| **`[19]`** | SFU Residual 2 (256 vec) | `u_sfu` | — | `12,543` |
| **Verify** | Đọc đối chiếu golden (TB only) | `CPU` | — | `12,288` |
| **Tổng** | **Toàn layer (+5 cyc reset)** | — | **`39,141`** | **`194,348`** |

- **Cycle mix phần cứng**: `seq_busy = 39,048`, `sfu_pipe = 35,072` (`sfu_core = 13,248`), `core_compute = 22,016` (hiệu suất MAC đạt **93.8% – 96.8%** ở các GEMM lớn), `dma = 11,584`, `tb_verify = 12,288`.
- **Hiệu quả của 4D Write-Out**: `Total job cycles` giữ nguyên `39,141` (0 cycle phát sinh), giảm `3,040` cycles toàn mô phỏng (`197,388 -> 194,348`) và loại bỏ hoàn toàn `cpu_reorder_q/k/attn_o`.

---

## 3. Phân tích Bottleneck & Hướng Tối ưu tiếp theo
1. **Gọi `u_sfu` rời rạc từng vector qua CSR (~`84,000` cycles overhead trong TB)**: Phần cứng `sfu_pipe` chỉ chạy `35,072` cycles nhưng `wall_cycles` của các bước SFU + `cpu_copy_gate` lên tới `119,163` cycles. Tối ưu bằng cách batch `sfu_num_vecs = 256`, dùng đường `Core -> fused_fifo -> SFU` (`sfu_in_a_src = 1`), và trỏ `A_OUT_BASE` của `GEMM Up` sang thẳng `G_STG_BANK`.
2. **Online Requant qua TCDM (`13,890` cycles = `35.5%` của `seq_busy`)**: Tận dụng đường `u_sfu -> u_log_quantizer` (`sfu_out_dst = 0`) để nén trực tiếp sau `RMSNorm` / `RoPE` / `SwiGLU` mà không ghi/đọc FP16 trung gian qua TCDM.
3. **DMA & Fused V tile nhỏ**: Bật `cfg_enable_dma = 1` để overlap DMA double-buffer với `u_core`.
