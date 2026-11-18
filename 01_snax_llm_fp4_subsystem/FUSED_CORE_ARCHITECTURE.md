# Kiến trúc Tái cấu trúc 7 Khối của `snax_fused_llm_core`

Tài liệu này mô tả chi tiết cấu trúc RTL sau khi tách bạch hoàn toàn các khối chức năng bên trong [snax_fused_llm_core.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv), đảm bảo:
1. **Pure Compute (`log8_fp4_mac_array`)**, **Storage (`rmw_latch_acc_buffer`)**, và **FlashAttention Rescaler (`attn_o_rescaler_bank`)** tách riêng thành 3 module độc lập.
2. Loại bỏ bộ chuyển đổi FP16 trùng lặp (`logscale_lazy_drain_norm`), hợp nhất toàn bộ đường ra FP16 (cho cả GEMM và Attention $O$-drain) qua duy nhất **`final_normalizer`**.
3. Giữ nguyên 100% giao tiếp top-level của `snax_fused_llm_core` để [snax_llm_top_clean.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv) và các testbench hiện có không bị thay đổi.

---

## 1. Sơ đồ Phân cấp Module (`snax_fused_llm_core`)

```text
snax_fused_llm_core
├── u_mac_array            (log8_fp4_mac_array.sv)       : 100% Pure Compute (256 MACs/cycle)
├── u_acc_buffer           (rmw_latch_acc_buffer.sv)     : 100% Storage (1W2R D-Latch + Clear + Drain)
├── u_attn_rescaler        (attn_o_rescaler_bank.sv)     : 16x CSD Attention O-Rescaler + Delay + Mux
├── u_softmax_requantizer  (softmax_requantizer.sv)      : Exp-free Base-2 Softmax -> FP4 P + l_contrib
├── u_p_fifo               (p_fifo.sv)                   : Transient 64-block Ring Buffer cho P & resc_*
├── u_l_accumulator        (l_accumulator.sv)            : Tích lũy mẫu số l & sinh 1/l (LZC + LUT)
└── u_final_normalizer     (final_normalizer.sv)         : Unified FP16 Drain (GEMM & O-slot)
```

---

## 2. Chức năng & Giao tiếp chi tiết của các Khối đã chỉnh sửa

### 2.1. `log8_fp4_mac_array` ([log8_fp4_mac_array.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/log8_fp4_mac_array.sv))
* **Vai trò:** Khối tính toán số học thuần túy (16 lanes × 16 elements = **256 MACs/cycle**), không chứa bộ nhớ `latch_array`, không chứa logic rescale của Attention, và không chứa bộ chuẩn hóa FP16.
* **Pipeline 3 tầng:**
  * **Stage 1A:** `dot_product_16` (`fp4_mult_lut` + `compressor_4to2`) tính tích vô hướng 16 phần tử E2M1 (13-bit Q2) song song với `scale_adder` tách `(q, r)`.
  * **Stage 1B:** `logscale_frac_scaler` (nhân phần dư $m_r/32$ bằng 3-term CSD không dùng bộ nhân) $\rightarrow$ `bidirectional_shifter` căn chỉnh về Q16.16 (32-bit bão hòa).
  * **Stage 2 (RMW Adder):** 16 bộ `accum_adder` (32-bit saturating add) cộng `aligned_prod_q1` với `effective_old_res_i` (nhận từ `u_attn_rescaler`), xuất kết quả ghi lại `acc_wdata_o` sang `u_acc_buffer`.
* **Giao tiếp chính:**
  * *In:* `act_broadcast_i [71:0]`, `wt_in_i [15:0][71:0]`, `lane_valid_i [15:0]`, `lane_acc_idx_i`, `effective_old_res_i [15:0][31:0]`.
  * *Out:* `lane_valid_q1_o [15:0]`, `acc_waddr_o`, `acc_wdata_o [511:0]`.

---

### 2.2. `rmw_latch_acc_buffer` ([rmw_latch_acc_buffer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/rmw_latch_acc_buffer.sv))
* **Vai trò:** Khối lưu trữ tích lũy 1-Write 2-Read thuần túy dựa trên `latch_array` (mặc định `ACC_USE_FF = 0` dùng D-Latch; có thể đổi sang `1` cho ablation Flip-Flop).
* **Cơ chế hoạt động:**
  * **Port A (RMW cùng chu kỳ):** Đọc giá trị cũ `old_res_o` (tự động che về `0` nếu slot đang nằm trong `slot_cleared`) để đưa sang `u_attn_rescaler` / `u_mac_array`, và ghi ngược `acc_wdata_i` vào cuối chu kỳ Stage 2.
  * **Port B (Non-blocking Rolling Drain):** Đọc slot `tile_drain_acc_idx_i` (có bypass trực tiếp `acc_wdata_i` nếu trùng slot đang hoàn thành Stage 2), chốt qua thanh ghi đầu ra `raw_drained_word_o` (512-bit, độ trễ 1 chu kỳ) và `raw_drained_valid_o`.
* **Giao tiếp chính:**
  * *In:* `lane_valid_q1_i`, `acc_rescale_req_q1_i`, `acc_addr_q1_i`, `acc_wdata_i [511:0]`, `tile_drain_req_i`, `tile_drain_acc_idx_i`, `tile_clear_on_drain_i`.
  * *Out:* `old_res_o [15:0][31:0]`, `raw_drained_word_o [511:0]`, `raw_drained_valid_o`.

---

### 2.3. `attn_o_rescaler_bank` ([attn_o_rescaler_bank.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/attn_o_rescaler_bank.sv))
* **Vai trò:** Cô lập hoàn toàn phần cứng *Online Rescaling* của FlashAttention (pha $P \times V$ khi running max $m$ tăng) ra khỏi mảng MAC GEMM.
* **Cơ chế hoạt động:**
  * Giữ 2 tầng thanh ghi trễ (`s1a -> q1`) cho `{acc_rescale_req_i, resc_q_i, resc_r_i}` (lấy từ `p_fifo`) để đồng bộ chính xác với Stage 2 của `u_mac_array`.
  * Dùng 16 bộ tổ hợp `vsq_rescaler_32b` (3-term CSD + arithmetic right shifter) nhân `old_res_i` với $2^{\Delta m}$.
  * Qua Mux `acc_rescale_req_q1`: nếu đang ở mode GEMM hoặc $Q \cdot K^T$ (`acc_rescale_req_q1 = 0`), đi thẳng `old_res_i` ra `effective_old_res_o` với 0 sai số.
* **Giao tiếp chính:**
  * *In:* `acc_rescale_req_i`, `resc_q_i [7:0]`, `resc_r_i [2:0]`, `old_res_i [15:0][31:0]`.
  * *Out:* `acc_rescale_req_q1_o`, `effective_old_res_o [15:0][31:0]`.

---

### 2.4. Hợp nhất ngõ ra FP16 qua `final_normalizer` trong `snax_fused_llm_core`
* Trước đây, GEMM dùng 16 bộ `logscale_lazy_drain_norm` trong `vsq_matmul_engine`, còn Attention dùng `final_normalizer` (với chân `bypass_inv_l`), gây dư thừa 16 bộ 32-bit LZC.
* Sau khi cập nhật [snax_fused_llm_core.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv):
  * `norm_drain_req = raw_drained_valid && ((cfg_mode_i != 2'b01) || drain_is_o_q);`
  * `norm_bypass_inv_l = (cfg_mode_i == 2'b00);`
  * Cả **GEMM** (`bypass_inv_l = 1`) và **FlashAttention $O$-drain** (`bypass_inv_l = 0`) đều dùng chung duy nhất `u_final_normalizer` $\rightarrow$ Output Skid Register $\rightarrow$ `out_vec_fp16_packed_o`.

---

## 3. Lưu ý khi Compile / Synthesis
* Khi biên dịch `snax_fused_llm_core.sv` (hoặc `snax_llm_top_clean.sv`), thay `vsq_matmul_engine.sv` bằng các file đã tách:
  * `log8_fp4_arith_units.sv` (chứa `logscale_lazy_drain_norm`, `scale_adder`, `logscale_frac_scaler`, `bidirectional_shifter`, `accum_adder`)
  * `log8_fp4_mac_array.sv`
  * `rmw_latch_acc_buffer.sv`
  * `attn_o_rescaler_bank.sv`
  *(Không cần include `vsq_matmul_engine.sv` trong filelist của `snax_llm_top_clean`; chỉ giữ `vsq_matmul_engine.sv` cho `04_ablation_rtl`.)*

---

## 4. SFU Integration (ngoài fused core)

SFU là khối **ngoài** `snax_fused_llm_core`, được tích hợp ở cấp `snax_llm_top_clean`:

* `u_sfu = snax_llm_sfu_opt #(.NUM_LANES(16), .ALU_LANES(8))`.
* TCDM crossbar tăng **21 → 22 masters**:
  * Master 20: `u_read_fp16` cho operand A.
  * Master 21: `u_read_fp16_b` cho operand B (gamma / cos-sin / up / x).
* Output SFU đi vào `log_quantizer` (FP4, `out_dst=0`) hoặc `snax_streamer_write_out` (FP16, `out_dst=1`).
* CSR SFU dùng RW 46..51, START=53; RO base=54, ID=61. Chi tiết xem `SFU_OPTIMIZATION_REPORT.md` mục 7.
* Khi `sfu_en=1`, core output bị park (`core_fp16_ready=0`) để tránh tranh logq/write-out.
* Filelist của top cần thêm `snax_sfu_fp20_units.sv` và `snax_llm_sfu_opt.sv`.

Lưu ý: tích hợp này không thay đổi cấu trúc 7 block bên trong fused core; các testbench core/GEMM/attention cũ vẫn chạy với top khi SFU disabled (`sfu_en=0`).
