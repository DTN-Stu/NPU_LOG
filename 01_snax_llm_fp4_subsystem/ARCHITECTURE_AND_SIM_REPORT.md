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

---

## 4. Cập nhật Tối ưu Fused Pipeline & Kết quả mới nhất

Các tối ưu đã triển khai và kiểm chứng trên QuestaSim 2024.1:

### 4.1. Fuse `GEMM Q/K -> SFU RoPE -> LogQ -> FP4` (bỏ ghi/đọc FP16 trung gian)
- **Kiến trúc**: Đẩy trực tiếp đầu ra FP16 của `u_core` qua `fused_fifo` (4-entry) $\rightarrow$ `u_sfu` (RoPE với `cos/sin` từ `u_read_fp16_b`) $\rightarrow$ `u_log_quantizer` $\rightarrow$ TCDM, thay cho chuỗi rời rạc `GEMM -> TCDM(FP16) -> SFU RoPE -> TCDM(FP16) -> Requant -> TCDM(FP4)`.
- **Hoán vị cột trọng số $W_Q', W_K'$ offline**: Sắp xếp trước thứ tự cột cặp $(j, j+d_k/2)$ theo cặp cặp đôi liền kề của `u_sfu` để bỏ hoàn toàn bước CPU repack trước RoPE (không cần hoán vị activation $X$).
- **Kiểm chứng bit-exact (`tb/tb_fused_qk_rope_compare.sv`)**:
  - `Exact 72b FP4 Packet Match : 256 / 256`
  - `Method B (Fused) vs Method A (Rời rạc) : SQNR = 999.00 dB, CosineSim = 1.000000`
- **Phân tích stall trên `u_mac_array` khi chạy Fused**:
  - Trong `snax_llm_sequencer.sv`, tín hiệu `gemm_beat_ok` (`!gemm_drain_active_q || ({1'b0, cnt_gbeat_q} < gdr_idx_q)`) chủ động tạm dừng `gemm_compute_fire` khi slot accumulator chưa drain xong do backpressure từ `u_read_fp16_b` (`snax_stream_read_fp16.sv` đọc `cos/sin` ~9 cycles/vector).
  - Cơ chế này an toàn tuyệt đối (không mất beat hay deadlock nhờ skid buffer `SKID_DEPTH = 4` ở `snax_streamer_act_bcast.sv`), chỉ làm `Fused Q/K+RoPE+LogQ` tốn `3,703` job cycles thay vì `2,184` cycles.

### 4.2. Tự động hóa SFU bằng `snax_sfu_sequencer` & tối ưu FFN (`tb/tb_snax_llm_layer_fused.sv`)
1. **Dùng `snax_sfu_sequencer` (`u_sfu_seq`)** thay vòng lặp CPU CSR cho `RMSNorm 1`, `Residual 1`, `RMSNorm 2`, `SiLU`, `Residual 2`:
   - `RMSNorm 1 / 2`: giảm từ `6,080` xuống `5,103` wall cycles.
   - `Residual 1 / 2`: giảm từ `12,543` xuống `4,654` wall cycles.
2. **Loại bỏ `cpu_copy_gate()` trong phase `[16]`**:
   - Mở rộng `run_gemm` nhận tham số `out_base` (`A_OUT_BASE`), trỏ `GEMM gate chunk` ghi thẳng vào `G_STG_BANK` và `GEMM up chunk` ghi vào `OUT_BASE`.
   - Giảm `8,192` wall cycles (`120,390 -> 112,198`), `cpu` cycles giảm từ `15,021 -> 10,925`.
3. **Fuse `SFU SiLU -> u_log_quantizer -> FP4` trực tiếp vào `FF1_BANK` (bỏ `[17] Requant FFN`)**:
   - Bật `sfu_out_dst = 0` (`A_SFU_CTRL[1] = 0`) và `A_WO_MODE = 0` trong `run_sfu_seq` (`out_fp4 = 1'b1`), đồng thời gộp 2 batch 16+16 token thành 1 lần chạy `M_TOKENS = 32` cho mỗi chunk, ghi packet FP4 thẳng vào `FF1_BANK` (tránh ghi đè `ACT_BANK` đang giữ `norm2` cho `chunk = 1`).
   - Loại bỏ hoàn toàn `4,657` cycles của phase `[17] Requant FFN` và giảm thêm `606` cycles ở phase `[16]`.

### 4.3. Tổng kết tiến trình tối ưu toàn Layer (`32` tokens, `d_model=128`, `d_ff=256`)

| Phiên bản | Tổng Wall Cycles | `Total job cycles` | `Checks` | Thay đổi chính |
| :--- | :---: | :---: | :---: | :--- |
| `tb_snax_llm_layer_mid.sv` (baseline 4D WO) | `194,348` | `39,141` | `80 passed, 0 failed` | Baseline với CPU gọi SFU rời rạc + `cpu_copy_gate()` + Requant rời rạc |
| `tb_snax_llm_layer_fused.sv` (Fused Q/K + `u_sfu_seq`) | `120,390` | `37,549` | `78 passed, 0 failed` | Fuse `Q/K + RoPE + LogQ` + `u_sfu_seq` (`-73,958` cycles) |
| + Bỏ `cpu_copy_gate()` (`A_OUT_BASE` trực tiếp) | `112,198` | `37,549` | `78 passed, 0 failed` | `GEMM gate` ghi thẳng `G_STG_BANK` (`-8,192` cycles) |
| + Fuse `SFU SiLU -> LogQ -> FF1_BANK` (hiện tại) | **`106,936`** | **`32,919`** | **`76 passed, 0 failed`** | Bỏ luôn ghi FP16 trung gian & `[17] Requant FFN` (`-5,262` cycles; tổng giảm **`87,412` cycles ≈ 45.0%**) |

- **Chi tiết kết quả chạy mới nhất (`sim/logs/tb_snax_llm_layer_fused.log`)**:
  - `Cycle mix`: `total=106936`, `seq_busy=32838`, `sfu_pipe=25792`, `sfu_core=13184`, `core_compute=22016`, `core_fp16=2944`, `cpu=10875`, `dma=11584`, `tb_verify=8192`.
  - `Errors: 0, Warnings: 0`, `Checks: 76 passed, 0 failed`.

### 4.4. Xác minh độ chính xác trên dữ liệu thực `NousResearch/Llama-3.2-1B` (Layer 1)
- Trích xuất slice thực từ `NousResearch/Llama-3.2-1B` Layer 1 (`x_real std=0.0416, max_abs=0.2718`) ra `sim/sim_snax_llm_layer_mid/llama32_1b_layer1_slice.npz` và sinh dữ liệu bằng `tb/generate_snax_llm_layer_mid_data.py --real-slice ...` (đồng thời bổ sung tùy chọn `--llm-scale` cho dữ liệu synthetic đúng biên độ LLM `x_std=0.04, w_std=0.03`).
- **Kết quả trên `tb_snax_llm_layer_fused.sv`**:
  - `SFU residual 1`: `max_abs = 0.00291`, **`fail = 0 / 4096`**
  - `SFU RMS2`: `max_abs = 0.02148`, **`fail = 0 / 4096`**
  - `Final residual`: `max_abs = 0.00293`, **`fail = 0 / 4096`**
  - `RTL vs FP32`: **`SQNR = 37.74 dB, CosineSim = 0.999916`**
  - `Py Fake-Quant vs FP32`: `SQNR = 37.68 dB, CosineSim = 0.999915`
  - `RTL vs PyFQ`: **`SQNR = 40.18 dB, CosineSim = 0.999952`**

---

## 5. Thiết kế Simple Gearbox DMA (`512-bit DRAM <-> 4×72-bit TCDM`) & Layer lớn hơn TCDM

### 5.1. Kiến trúc phần cứng
- **`snax_simple_dma.sv`**:
  - Giao tiếp `valid/ready` đơn giản (không dùng AXI): Descriptor Slave (`dma_desc_*`), External DRAM Master (`512-bit` / 64B per beat, phù hợp gắn Ramulator2/DRAMSim3 để ước lượng băng thông & năng lượng), và `TCDM_PORTS = 4` cổng TCDM (`4 × 72-bit = 288 bits/cycle = 36 bytes/cycle`, khớp LPDDR4X/LPDDR5 64-bit ở 500 MHz–1 GHz).
  - **Gearbox `512b <-> 72b`** (tỉ lệ tròn `9 beats × 512b = 64 packets × 72b`) dùng buffer **`1096-bit`** pipeline các lệnh đọc DRAM để che độ trễ DRAM (`6` cycles), kèm 4 holding register ở ngõ ra TCDM để đảm bảo in-order và an toàn khi xung đột bank.
  - Hỗ trợ cả `DRAM -> TCDM` (`dir[0]=0`) và `TCDM -> DRAM` (`dir[0]=1`), cùng 2 kiểu địa chỉ TCDM (`dir[1]=0`: rải 16 bank `0..15` cho trọng số; `dir[1]=1`: tuyến tính `+0x200`).
  - Tích hợp sẵn bộ đếm telemetry: `stat_dram_rd_beats_o`, `stat_dram_wr_beats_o`, `stat_tcdm_words_o`, `stat_busy_cycles_o`.
- **`snax_llm_top_clean.sv` (`NUM_MASTERS = 25`)**:
  - Giữ nguyên cổng `dma_mem_*` (`M0`) và mở rộng thêm 3 cổng `dma_aux_mem_*[2:0]` (`M22..24`, mặc định `'0` để tương thích 100% với các TB cũ).
  - Giữ nguyên cấu trúc arbitration `fixed_priority_arb_n` hiện tại (`M0` qua `HOST_MASTER=0` và `M22..24` có index lớn hơn `M2..21` nên mọi cổng DMA đều nhường quyền ưu tiên cho các streamer).
- **`snax_llm_sequencer.sv`**:
  - Bổ sung **GEMM DMA Prefetch Producer** (`dma_gn_q`, `dma_gm_q`, `dma_pending_q`) tự động phát liên tiếp descriptor trọng số (`144 * cfg_gemm_k_blocks_i` bytes/tile) vào 4-slot ring buffer song song khi `u_core` đang tính.

### 5.2. Kết quả kiểm chứng DMA & Layer lớn hơn TCDM (QuestaSim 2024.1)
1. **`tb/tb_simple_dma_gemm.sv`** (`Errors: 0`):
   - **Round-trip `TCDM -> 512b DRAM -> TCDM`** (`128` words = `1,152 B`): **`128 / 128` exact**.
   - **Streaming GEMM (`M=2, N=8, K=8`, 16 tiles qua 4-slot ring, `DRAM_LATENCY=6`)**: **`1024 / 1024` words bit-exact**; nâng cấp từ 1 cổng (`72b/cyc`) lên 4 cổng (`288b/cyc`) + pipeline DRAM giúp rút ngắn từ `4,602` cycles xuống **`2,315` cycles** (sát `~2,184` non-DMA, `dma_busy = 2,080`, `288` DRAM beats = `18,432 B`).
2. **`tb/tb_large_layer_dma.sv` — Layer lớn hơn dung lượng TCDM (`Errors: 0`)**:
   - **Cấu hình**: `M=16, D_MODEL=256, D_FF=512`, **tổng trọng số = `40,960` words (`368,640` bytes > `32,768` words toàn TCDM)**, stream toàn bộ 7 phép chiếu từ DRAM 512-bit:
     - `GEMM Q / K / V / O / Gate 0 / Gate 1 / Up 0 / Up 1`: mỗi phép `4,096` wt words (`576` beats = `36,864 B`) chạy `4,489` cycles (`dma_busy = 4,097`).
     - `GEMM Down`: `8,192` wt words (`1,152` beats = `73,728 B`) chạy `8,837` cycles (`dma_busy = 8,129`).
   - **Tổng hợp toàn layer**: **`5,760` DRAM 512b read beats (`368,640` bytes)**, **`40,960` TCDM 72b words**, **`40,905` DMA busy cycles**, **`0 errors`** bit-exact.
