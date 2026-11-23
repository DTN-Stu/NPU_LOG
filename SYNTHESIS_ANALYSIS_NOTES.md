# Synthesis Analysis Notes — System, Sequencer & Quantizer

Tài liệu tổng hợp số liệu synthesis sơ bộ và phân tích RTL/kiến trúc phục vụ tối ưu phần cứng và trình bày số liệu trong paper.

---

## 1. Kết quả Synthesis sơ bộ toàn hệ thống

| Block | Levels of Logic | Critical Path (ps) | WNS (ps) | TNS (ps) | Cell Area ($\mu\text{m}^2$) | Ghi chú |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`u_core` (`snax_fused_llm_core`)** | `1.00` | `96.26` | `0.00` | `0.00` | `17,339.86` | Clean |
| **`u_seq` (`snax_llm_sequencer`)** | `36.00` | `916.70` | `+0.02` | `0.00` | `5,081.86` | **Lớn bất thường (~2.4× `u_sfu`)** |
| **`u_sfu` (`snax_llm_sfu_opt_pipe`)** | `52.00` | `1233.99` | `+0.07` | `0.00` | `2,125.22` | Đã tối ưu pipeline + share ALU |
| **`u_write_out` (`snax_streamer_write_out`)** | `52.00` | `1231.42` | `+3.37` | `0.00` | `2,517.22` | Clean |
| **`u_wt_streamer` (`snax_streamer_wt_16ch`)** | `48.00` | `1249.96` | `+0.10` | `0.00` | `2,620.77` | Clean |
| **`u_csr` (`snax_flash_csr_snax`)** | `10.00` | `539.93` | `+1039.55` | `0.00` | `512.36` | Đã thu gọn mạnh từ mức `1,762 µm²` |

---

## 2. Phân tích `u_seq` (`snax_llm_sequencer`) & Đánh giá `snax_llm_sequencer_area_opt`

### 2.1 Nguyên nhân `snax_llm_sequencer.sv` lớn (`5,081.86 µm²`)
Trong [01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv), diện tích phình to chủ yếu do **hàng chục bộ nhân tổ hợp 32-bit (`*`) chạy song song** để tính lại địa chỉ từ chỉ số vòng lặp (`index * stride`) thay vì dùng con trỏ cộng dồn:
- **Khối địa chỉ TCDM (`dòng 305, 360–414`):** nhiều phép nhân 2–3 thừa số nối tiếp (`q_blk_per_head`, `total_k_blocks`, `preload_wt_addr`, `act_tile_tcdm_addr`, `attn_q_addr`, `attn_k/v_pin_addr`, `attn_k/v_preload_addr`).
- **Khối DMA Arbiter (`dòng 1435–1532`):** 11 nhánh `if / else if` đều chứa biểu thức nhân 2–4 thừa số 32-bit (`* cfg_stride_tile_i`, `* cfg_attn_dv_tiles_i * cfg_attn_kv_tiles_i * 144`, ...).
- **Khối Drain Write-back (`dòng 1680–1688`), `wo_num_batches_o` (`dòng 704–736`), `auto_causal_mask` (`dòng 446–448`):** dùng phép nhân 32-bit và ép kiểu `32'(...)` dư bit-width.

### 2.2 Đánh giá [snax_llm_sequencer_area_opt.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer_area_opt.sv)
Module tối ưu diện tích áp dụng đúng hướng thiết kế:
- Chỉ giữ **1 bộ nhân `16×16` duy nhất** (`mul_p = mul_a_q * mul_b_q`, `dòng 386`) chia sẻ theo thời gian qua `CMP_PARAM_INIT` (`PINIT_N = 11`) và `CMP_O_INIT`.
- Dùng dịch-cộng (`mul_tile_bytes`, `mul_u32_u5`, `mul_u32_u3`) và hệ thống con trỏ cộng dồn `*_ptr_q`.

**Các điểm cần lưu ý khi tích hợp `snax_llm_sequencer_area_opt.sv`:**
1. **`last_chunk_len` (`dòng 248–255`):**
   - Dùng vòng lặp trừ `for (k = 0; k < 32; k = k + 1)` -> tạo chuỗi 32 bộ so sánh/trừ 16-bit nối tiếp (rủi ro timing) và sai nếu `macro_m_total > 32 * eff_m_sub`.
   - *Không có ở module cũ:* module cũ dùng `dma_a_cur_m_sub = min(macro_m_total - dma_a_m_base_q, eff_m_sub)`. Trong `snax_llm_sequencer_area_opt.sv` (`dòng 1567`) vẫn có sẵn `dma_a_cur_m_sub`, nên có thể bỏ `last_chunk_len` và dùng thẳng `mul_u32_u5(a_step_m, dma_a_cur_m_sub[4:0])`.
2. **Cắt bit `[4:0]` ở `eff_m_sub[4:0]` / `last_a_m[4:0]` (`dòng 436–437`):** giới hạn `eff_m_sub <= 31`.
3. **Cắt bit `[15:0]` ở `dòng 470, 477–478`:** an toàn cho cấu hình `Llama-3.2-1B` (`< 65,536`), cần kiểm tra nếu mở rộng sang model lớn hơn.
4. **Overhead `~22–32` chu kỳ mỗi lần `start_i`:** rất nhỏ (**`~0.27%`**, ước tính `~3,650 / 1,371,606` chu kỳ trong Decode `M=1, N_KV=128`), không ảnh hưởng đáng kể vì bottleneck Decode là DRAM bandwidth (`dma = 757,479` chu kỳ).
5. **Tên module (`dòng 33`):** đang là `snax_llm_sequencer_area_opt`, cần đồng bộ tên/instantiation và chạy lại regression Verilator trước khi lấy số synthesis cuối.

---

## 3. Kết quả Synthesis Quantizer & Lựa chọn bảng cho Paper

### 3.1 Số liệu gốc đầy đủ (lưu nội bộ)

| Quantizer | Block Size | WNS (ps) | Crit Path (ps) | Levels | Comb ($\mu\text{m}^2$) | Seq ($\mu\text{m}^2$) | Total Area ($\mu\text{m}^2$) | Pwr Unif SAIF (mW) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`slq_encoder` (`log_quantizer`)** | `16` | `+353.20` | `1177.17` | `42` | `136.53` | `85.67` | `222.20` | `0.146` |
| **`nvfp4_dyn_quantizer`** | `16` | `+1.88` | `1521.92` | `65` | `659.58` | `127.37` | `786.96` | `0.218` |
| `MiX_MX_Quantizer (iso-16)` | `16` | `+2.80` | `1517.53` | `56` | `222.04` | `27.29` | `249.33` | `0.0463` |
| `MiX_MX_Quantizer (default-32)` | `32` | `+0.01` | `1523.84` | `70` | `474.05` | `51.55` | `525.61` | `0.0888` |

### 3.2 Quyết định trình bày trong Paper (chỉ giữ `slq_encoder` vs `nvfp4_dyn_quantizer`)

**Lý do loại `MiX_MX_Quantizer` khỏi bảng chính của paper:**
- Toàn bộ bài viết không đề cập hay so sánh MAC/System của `MiX_MX`; nếu đột ngột đưa `MiX_MX_Quantizer` vào bảng quantizer dễ bị reviewer chất vấn về tính nhất quán của baseline (cherry-picking).
- Ngoài ra, `MiX_MX_Quantizer (iso-16)` có `Seq Area = 27.29 µm²` (ít FF pipeline hơn hẳn `85.67 µm²` của `slq_encoder`), làm cột Power (`0.0463 mW` vs `0.146 mW`) bị lệch khung so sánh nếu không chuẩn hóa lại pipeline I/O.

**Bảng đối chứng sạch cho Paper (`Block Size = 16`, Iso-Bitwidth):**

| Quantizer (`B=16`) | Crit Path (ps) | Levels | WNS (ps) | Comb ($\mu\text{m}^2$) | Seq ($\mu\text{m}^2$) | Total Area ($\mu\text{m}^2$) | Power (mW) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nvfp4_dyn_quantizer`** | `1521.92` | `65` | `+1.88` | `659.58` | `127.37` | `786.96` | `0.218` |
| **`slq_encoder` (Ours)** | **`1177.17`** | **`42`** | **`+353.20`** | **`136.53`** | **`85.67`** | **`222.20`** | **`0.146`** |
| **Cải thiện (Ours vs. NVFP4)** | **-22.7%** | **-23** | **+351.32** | **-79.3% (4.83×)** | **-32.7%** | **-71.8% (3.54×)** | **-33.0%** |

> **Gợi ý viết text:** Chọn `nvfp4_dyn_quantizer` làm đối chứng trực tiếp vì cùng kích thước block `B=16` và thuộc nhóm bộ lượng tử hóa động độ chính xác cao (`E4M3` + `tensor scale`), tương xứng trực tiếp với `Log8-FP4` (`SLQ`) trên bảng đánh giá thuật toán PPL/Zero-shot.
