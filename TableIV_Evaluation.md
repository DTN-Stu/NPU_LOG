# Nhiệm vụ: Tạo số liệu Stage-by-Stage Verification cho Table IV (RTL vs. PyFQ vs. FP32) trên Real Llama-3.2-1B Layer 1

## 1. Bối cảnh & Mục tiêu
Cần xây dựng bảng số liệu **Table IV (Stage-by-Stage RTL Numerical Verification)** cho bài báo IEEE TCAD/TVLSI trên 1 layer đầy đủ của mô hình thật `NousResearch/Llama-3.2-1B` (`Layer 1`, `D_MODEL=2048, D_FF=8192, H_Q=32, H_KV=8, D_K=D_V=64`, trọng số đã cache tại `sim/sim_llama3_1b_weights/dram_weights_l1.hex`).

**Đầu ra cuối cùng bắt buộc:**
1. Cập nhật `tb/gen_llama3_1b_configurable.py` để mô hình `PyFQ` khớp chặt phần cứng RTL (dùng `CSD_Q5` + `exp_free_softmax_requantizer` với `LOG2E_Q16 = 94548`) và xuất đầy đủ golden cho **7 giai đoạn** (cả `FP32` lẫn `PyFQ`).
2. Cập nhật `tb/tb_llama3_1b_configurable.sv` để đo và in đầy đủ **SQNR (dB), CosineSim, MaxAbs** cho tất cả 7 giai đoạn (`RTL vs. PyFQ`, `RTL vs. FP32`) bằng cách đọc trực tiếp `dram_mem` (0 chu kỳ RTL phát sinh).
3. Chạy mô phỏng Verilator bằng `sim/run_llama3_1b_cfg.sh` cho **3 cấu hình** và tổng hợp kết quả chuẩn bảng paper vào file `tb/TABLE_IV_VERIFICATION_REPORT.md`.

> **Ràng buộc an toàn:** KHÔNG sửa bất kỳ file RTL nào trong `01_snax_llm_fp4_subsystem/`. Nếu quá trình đối chiếu stage-by-stage phát hiện bug thật trong RTL (ví dụ sai địa chỉ/scale ở một stage), hãy dừng lại, ghi rõ bằng chứng và báo cáo thay vì tự ý sửa RTL.

---

## 2. File liên quan cần đọc trước khi sửa
- `tb/gen_llama3_1b_configurable.py` (script sinh dữ liệu & golden hiện tại)
- `tb/tb_llama3_1b_configurable.sv` (testbench Verilator 100% 1D-DMA, ZERO `cpu_mem`)
- `sim/run_llama3_1b_cfg.sh` (script build & chạy Verilator, dùng Python `/home/nguyen/qenv/bin/python3.12`)
- `02_eval_llm_edge/quantizers.py` (tham chiếu hàm `exp_free_softmax_requantizer` và `CSD_Q5`)
- `tb/generate_snax_llm_layer_mid_data.py` (tham chiếu công thức fixed-point Q16.16, `LOG2E_Q16 = 94548`, `P_RND = 4096`, `P_OFFSET = -21`, và `l_accumulator` 32-entry `recip_lut`)
- `01_snax_llm_fp4_subsystem/softmax_requantizer.sv`, `log8_fp4_arith_units.sv`, `l_accumulator.sv`, `snax_sfu_fp20_units.sv` (đối chiếu hằng số phần cứng nếu cần)

---

## 3. Các bước thực hiện chi tiết

### Bước 1: Nâng cấp mô hình `PyFQ` khớp chặt RTL trong `tb/gen_llama3_1b_configurable.py`
Hiện tại `tb/gen_llama3_1b_configurable.py` (dòng 97–108 và 349–426) có các điểm lệch so với RTL cần sửa:

1. **Thay `2^(sc/8)` bằng `CSD_Q5` trong `fq_rtl_vec` (và/hoặc viết hàm `matmul_log8_csd_vec`):**
   - Trong `log8_fp4_arith_units.sv`, thang đo phần dư dùng bảng `CSD_Q5 = [32, 35, 38, 41, 46, 49, 54, 59] / 32.0`.
   - Khuyến nghị tối ưu:
     - Giữ `fq_rtl_vec(mat, use_csd=True)` với `q_s = np.floor(sc / 8.0)`, `r_s = sc & 7`, `scale = (2.0 ** q_s) * (CSD_Q5[r_s] / 32.0)`.
     - Nếu đủ nhanh bằng NumPy/PyTorch vectorized, có thể mô phỏng chính xác hơn cặp `(s_a + s_w)` cho phép nhân block `log8_fp4_mac_array` (hoặc kiểm tra chênh lệch giữa per-operand CSD vs paired CSD trên 1 GEMM để chọn cách nhanh mà vẫn đạt `>35 dB` vs RTL).
2. **Thay FP32 `np.exp` Softmax (dòng 410–415) bằng `exp_free_softmax_requantizer` khớp 100% `softmax_requantizer.sv`:**
   - Nhân score $S$ (ở Q16.16) với hằng số phần cứng **`LOG2E_Q16 = 94548`** (`(s_q16 * 94548) >> 16` hoặc dùng đúng 7-term CSD `(s<<16)+(s<<15)-(s<<12)+(s<<8)+(s<<6)+(s<<4)+(s<<2)`).
   - Dùng đúng causal mask / block max / running max $m$, 7 ngưỡng Q16.16 `[17238, 50961, 82774, 116497, 148310, 196608, 300480]`, `p_blk = clamp(((blk_max - row_max + 4096) >> 13) - 21, -128, 127)`, và `S_blk = 2^floor(p_blk/8) * CSD_Q5[p_blk & 7] / 32`.
   - Tính `p_unnorm = E2M1[mag_idx] * S_blk`, `l_sum = sum(p_unnorm)` (có thể dùng cả bản chia chuẩn và kiểm tra độ lệch so với `recip_lut` 32-entry của `l_accumulator.sv`).
   - Sau đó lượng tử hóa `attn_fq` bằng `fq_rtl_vec` (vì phần cứng xuất `O` qua `final_normalizer` -> `slq_encoder` thành gói 72-bit FP4 tại `DRAM_OPKT`).
3. **Đảm bảo thứ tự dữ liệu & lượng tử hóa khớp RTL cho từng stage:**
   - `Q` & `K`: lưu ý RTL chạy `Fused Q/K + RoPE + LogQ` trên kênh đã hoán vị cặp `(2j, 2j+1) <-> (j, j + D_K/2)` trước khi lượng tử hóa 16 phần tử/block (hàm `quant_rope_qk` ở dòng 358 đã làm đúng chiều này — hãy giữ nguyên và xuất thêm bản đã hoán vị hoặc thống nhất chiều so sánh với TB).
   - `V`: trong Prefill, `snax_transpose_buffer` chuyển vị theo từng block 16 tokens dọc trục `N_KV` (`v_new_fq` ở dòng 389–390 đã làm đúng).
   - Chú ý **tùy chọn phụ kiểm chứng SFU (`--sfu-approx` hoặc in riêng cột đối chứng):** Giữ `PyFQ` chính dùng RMSNorm/RoPE/SwiGLU chuẩn FP32 (để phân tách rõ sai số của `Log8-FP4 + Exp-Free Softmax` với sai số của khối phụ trợ `u_sfu` FP20/PWL), đồng thời **in trong log Python** một báo cáo nhanh chênh lệch nếu bật thêm `sfu_sigmoid_pwl_fp20` (4 đoạn hệ số góc `0.25, 0.125, 0.03125`) cho `SwiGLU` để chẩn đoán chính xác nguồn gốc mọi sai số ở FFN!
4. **Xuất thêm file golden trung gian (FP16 hex) & in đủ bảng `PyFQ vs. FP32` cho mọi stage:**
   - Xuất thêm vào `out_dir`:
     - `g_q_fp32.hex` / `g_q_fq.hex` (shape `[H_Q, m_tokens, D_K]`, lưu theo dạng FP16 hex cho gọn hoặc FP32 hex cho chính xác)
     - `g_k_fp32.hex` / `g_k_fq.hex` (shape `[H_KV, m_tokens, D_K]`)
     - `g_v_fp32.hex` / `g_v_fq.hex` (shape `[H_KV, m_tokens, D_V]`)
     - `g_attn_o_fp32.hex` / `g_attn_o_fq.hex` (shape `[H_Q, m_tokens, D_V]`)
   - In bảng tổng hợp **`PyFQ vs. FP32`** (`SQNR dB`, `CosSim`, `MaxAbs`) cho **tất cả 7 stage** (`RMSNorm 1`, `Fused Q+RoPE`, `Fused K+RoPE`, `Fused V+Transpose`, `Exp-Free FlashAttn O`, `GEMM O + Res1`, `RMSNorm 2`, `FFN + Res2 Final`) ra stdout và lưu thành `out_dir/pyfq_vs_fp32.json` (hoặc `.csv`).

---

### Bước 2: Bổ sung đo 7 Stage trong `tb/tb_llama3_1b_configurable.sv` (0 cycle phát sinh)
Trong `tb/tb_llama3_1b_configurable.sv`, tất cả tensor trung gian đều **đã được DMA ghi ra `dram_mem`** ở cuối các phase:
- `DRAM_DBG` (sau RMSNorm 1) -> `RMSNorm 1` (đã có)
- `DRAM_QPKT` (layout `[h][t_abs][kp]` với `kp = 0..3`, mỗi word là gói 72-bit `Log8-FP4` của 16 kênh đã hoán vị RoPE) -> **THÊM MỚI**
- `DRAM_K` (layout `[hkv][kj][kp][l]` với `kj = KV_TILES_PAST + mb`, `l = 0..15` là 16 tokens của chunk, mỗi word 72-bit là 16 kênh đã hoán vị RoPE) -> **THÊM MỚI**
- `DRAM_V` (layout `[hkv][dvi][kj][l]` với `l = 0..15` là 16 kênh trong `dvi`, mỗi word 72-bit chứa 16 tokens dọc `N_KV`) -> **THÊM MỚI**
- `DRAM_OPKT` (layout `[h][t_abs][dvi]` với `dvi = 0..3`, mỗi word 72-bit là 16 kênh output của FlashAttention) -> **THÊM MỚI**
- `DRAM_RES1` -> `GEMM O + Residual 1` (đã có)
- `DRAM_DBG` (sau RMSNorm 2) -> `RMSNorm 2` (đã có)
- `DRAM_OUT` -> `FFN (Gate/Up/SwiGLU/Down) + Residual 2 (Final)` (đã có)

**Yêu cầu cài đặt trong `tb/tb_llama3_1b_configurable.sv`:**
1. Viết hàm `decode_log8_pkt_elem(input logic [71:0] pkt, input int elem_idx)` giải nén gói 72-bit `Log8-FP4` sang `real` dùng đúng `CSD_Q5` (`[32,35,38,41,46,49,54,59]/32.0`) và `E2M1_LUT = {0, 0.5, 1, 1.5, 2, 3, 4, 6}`.
2. Đọc `DRAM_QPKT`, `DRAM_K` (lưu ý un-permute cặp RoPE `(2j, 2j+1) -> (j, j+32)` nếu golden không permute, hoặc so trực tiếp trên tensor cùng thứ tự), `DRAM_V` (chú ý chiều transpose `l` = channel, `elem_idx` = token trong block 16), và `DRAM_OPKT` ngay sau khi hoàn tất phase tương ứng (hoặc ở cuối simulation) — **tuyệt đối không gọi thêm DMA hay `cpu_mem`**, chỉ dùng `read_dram_9b(...)` trong testbench nên `cyc_total` và `cpu_mem_active_cyc == 0` giữ nguyên 100%!
3. In đầy đủ cho cả **Prefill** và **Decode** bảng `[METRIC]` có đủ `SQNR (dB)`, `CosineSim`, `MaxAbs` cho:
   - `Stage 1: RMSNorm 1` (`RTL vs FP32`)
   - `Stage 2: Fused Q + RoPE + LogQ` (`RTL vs PyFQ` & `RTL vs FP32`)
   - `Stage 3: Fused K + RoPE + LogQ` (`RTL vs PyFQ` & `RTL vs FP32`)
   - `Stage 4: Fused V + Transpose + LogQ` (`RTL vs PyFQ` & `RTL vs FP32`)
   - `Stage 5: Exp-Free FlashAttention O` (`RTL vs PyFQ` & `RTL vs FP32`)
   - `Stage 6: GEMM O + Residual 1` (`RTL vs PyFQ` & `RTL vs FP32`)
   - `Stage 7: RMSNorm 2` (`RTL vs PyFQ` & `RTL vs FP32`)
   - `Stage 8: FFN (Gate/Up/SwiGLU/Down) + Residual 2 (Final)` (`RTL vs PyFQ` & `RTL vs FP32`)
4. **Chẩn đoán bắt buộc (Diagnostic):**
   - Trong log cũ (`m32_kv32`), `RMSNorm 2 vs PyFQ` chỉ đạt `~14.94 dB` và `Final vs FP32` có `MaxAbs = 26.98` (nghi do token 0 `<|begin_of_text|>` có massive activation ở Layer 1 Llama-3.2-1B, hoặc do `RMSNorm 2` có khác biệt về `inv_n`/rounding/scaling).
   - Hãy in thêm dòng chẩn đoán: (a) **không tính token 0** (nếu `n_past == 0`) cho `RMSNorm 1`, `Res1`, `RMSNorm 2`, `Final`; và (b) **kiểm tra ngay lý do** nếu `RTL vs PyFQ` ở bất kỳ stage nào (đặc biệt `Q/K/V/Attn O/Res1/RMSNorm 2`) nhỏ hơn `25 dB`!

---

### Bước 3: Chạy Verilator cho 3 cấu hình & lưu log
Chạy trong WSL / bash tại `sim/` (tự động tái sử dụng `sim_llama3_1b_weights/dram_weights_l1.hex` đã cache, không cần tạo lại 34.2 MB weights):
```bash
cd sim
# 1) Prefill chuẩn từ đầu prompt (M=32, N_KV=32) — ~100s
./run_llama3_1b_cfg.sh --mode prefill --m-tokens 32 --n-kv 32 --rebuild 2>&1 | tee sim_llama3_1b_prefill_m32_kv32/table4_sim.log

# 2) Chunked Prefill không dính BOS token 0 (M=32, N_KV=128) — ~100s
./run_llama3_1b_cfg.sh --mode prefill --m-tokens 32 --n-kv 128 2>&1 | tee sim_llama3_1b_prefill_m32_kv128/table4_sim.log

# 3) Autoregressive Decode (M=1, N_KV=128) — ~21s
./run_llama3_1b_cfg.sh --mode decode --n-kv 128 2>&1 | tee sim_llama3_1b_decode_m1_kv128/table4_sim.log
```
*(Lưu ý: Nếu chạy trên Windows PowerShell thì gọi qua `wsl bash -lc "cd /mnt/c/Users/nguyen/Desktop/NPU_LOG/sim && ..."`).*

---

### Bước 4: Xuất báo cáo chuẩn cho Paper
Tạo file `tb/TABLE_IV_VERIFICATION_REPORT.md` gồm:
1. **Bảng Table IV chính** (cho cả `Prefill M=32, N_KV=32`, `Chunked Prefill M=32, N_KV=128`, và `Decode M=1, N_KV=128`) với các cột:
   `| Stage | Tensor Shape | RTL vs. PyFQ (SQNR dB / CosSim / MaxAbs) | RTL vs. FP32 (SQNR dB / CosSim / MaxAbs) | PyFQ vs. FP32 (SQNR dB / CosSim) |`
2. **Phân tích nguyên nhân sai lệch (Error Attribution):** Tách bạch rõ phần nào đến từ lượng tử hóa 4-bit `Log8-FP4 + Exp-Free Softmax` (`PyFQ vs. FP32`), phần nào đến từ sai số xấp xỉ `FP20/PWL` của khối phụ trợ `u_sfu` hoặc `Q16.16/recip_lut` (`RTL vs. PyFQ`), và hiệu ứng BOS token 0 nếu có.
3. **Xác nhận toàn vẹn hệ thống:** `Checks passed/failed (0 fail)`, `cpu_mem_active_cyc == 0`, `max_inflight_seen <= 4`, `DRAM canaries PASS`, `Total cycles`, `MAC active cycles`, `DRAM read/write beats`.