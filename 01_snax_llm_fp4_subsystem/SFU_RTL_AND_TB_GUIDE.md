# SNAX LLM Special Function Unit (`snax_llm_sfu`) — RTL & Testbench Guide

Tài liệu này mô tả kiến trúc RTL, giao tiếp phần cứng, các phép toán hỗ trợ và testbench kiểm chứng của khối **SFU 16-lane SIMD** trong `01_snax_llm_fp4_subsystem/` để các agent/kỹ sư sau này có thể nhanh chóng nắm bắt và tích hợp vào `snax_llm_top_clean.sv`.

---

## 1. Tổng quan các file liên quan

| File | Vai trò |
| :--- | :--- |
| `01_snax_llm_fp4_subsystem/snax_sfu_fp_units.sv` | Thư viện số học IEEE-754 synthesizable (FP16 $\leftrightarrow$ FP32, Adder, Multiplier, `rsqrt`, PWL `sigmoid`). |
| `01_snax_llm_fp4_subsystem/snax_llm_sfu.sv` | Top-level SFU 16-lane SIMD (bus 256-bit `valid/ready` tương thích hệ thống). |
| `01_snax_llm_fp4_subsystem/tb_snax_llm_sfu.sv` | Testbench tự kiểm tra (self-checking) cho cả 4 phép toán chính. |
| `sim/filelist_sfu.f` | Filelist chạy simulation trên QuestaSim. |

---

## 2. Kiến trúc RTL & Các Module con (`snax_sfu_fp_units.sv`)

Mọi phép toán bên trong SFU đều chuyển từ **FP16 $\rightarrow$ FP32** để tránh tràn số (ví dụ $x^2$ trong RMSNorm dễ vượt quá giới hạn `65504` của FP16), sau đó đóng gói **FP32 $\rightarrow$ FP16** (làm tròn RNE) ở ngõ ra:

* `sfu_fp16_to_fp32` / `sfu_fp32_to_fp16`: Chuyển đổi IEEE-754 FP16 $\leftrightarrow$ FP32 (hỗ trợ số subnormal FP16 và bão hòa/flush-to-zero).
* `sfu_fp32_addsub` (`sub_i`: `0` cộng, `1` trừ) & `sfu_fp32_mul`: Khối cộng/trừ và nhân FP32 tổ hợp dùng chung cho 16 lanes.
* `sfu_rsqrt_fp32`: Tính $1/\sqrt{x}$ cho RMSNorm gồm 2 bước:
  1. Chia đôi số mũ + tra bảng LUT 32-entry (`{e_lsb, m[22:19]}`) lấy seed $y_0$.
  2. Tinh chỉnh 1 bước Newton-Raphson: $y_1 = y_0 \cdot (1.5 - 0.5 x y_0^2)$.
* `sfu_pwl_sigmoid_fp32`: Tính $\sigma(x)$ cho SwiGLU bằng xấp xỉ tuyến tính từng đoạn đối xứng (4 đoạn trên $|x|$ với các mốc `0, 1.0, 2.5, 3.5, 5.0`), loại bỏ hoàn toàn mạch `exp()`.

---

## 3. Giao tiếp & Chế độ Hoạt động (`snax_llm_sfu.sv`)

### 3.1. Parameters & Ports chính
* `NUM_LANES = 16` (256-bit vector = `16 × 16-bit FP16`).
* `MAX_VEC_BLOCKS = 256` (bộ đệm tối đa $256 \times 16 = 4096$ phần tử/token cho RMSNorm).
* **Luồng dữ liệu (`valid / ready`):**
  * `in_a_fp16_i [255:0]`, `in_a_valid_i`, `in_a_ready_o`: Nối từ `core_fp16_packed` (`u_core`) hoặc `rd_fp16_data` (`u_read_fp16`).
  * `in_b_fp16_i [255:0]`, `in_b_valid_i`, `in_b_ready_o`: Mang toán hạng phụ (Residual $Y$, $\gamma$ của RMSNorm, cặp $\{\sin,\cos\}$ của RoPE, hoặc `up_proj` của SwiGLU).
  * `out_fp16_o [255:0]`, `out_valid_o`, `out_ready_i`: Nối vào `snax_logq_shared` hoặc `u_write_out`.
* **Điều khiển (CSR):**
  * `start_i`, `busy_o`, `done_o`
  * `cfg_op_i [2:0]`, `cfg_num_vecs_i [15:0]`, `cfg_inv_n_fp32_i [31:0]` (hằng số $1/D$ FP32), `cfg_eps_fp32_i [31:0]` ($\epsilon$ FP32, mặc định `1e-5` = `32'h3727C5AC`).

### 3.2. Bảng 5 chế độ (`cfg_op_i`)

| `cfg_op_i` | Tên chế độ | Phép toán thực hiện | Độ trễ / Giao thức |
| :---: | :--- | :--- | :--- |
| `3'd0` | `SFU_BYPASS` | `out = in_a` (không cần `in_b`) | 1 chu kỳ (streaming) |
| `3'd1` | `SFU_RESIDUAL` | `out[i] = in_a[i] + in_b[i]` | 1 chu kỳ (streaming) |
| `3'd2` | `SFU_RMSNORM` | 1-pass buffered token: tích lũy $\sum x^2$ qua `cfg_num_vecs_i` vector $\rightarrow$ `RMS_RSQRT` (1 chu kỳ) $\rightarrow$ `out[i] = x_buf[i] * inv_rms * in_b[i]` | FSM 4 trạng thái (`RMS_IDLE`, `RMS_ACCUM`, `RMS_RSQRT`, `RMS_SCALE`) |
| `3'd3` | `SFU_ROPE` | Xoay 8 cặp/vector: `in_b[2k]=cos_k`, `in_b[2k+1]=sin_k`; `out[2k]=a[2k]*cos - a[2k+1]*sin`, `out[2k+1]=a[2k]*sin + a[2k+1]*cos` | 1 chu kỳ (streaming) |
| `3'd4` | `SFU_SWIGLU` | `out[i] = (in_a[i] * sigmoid_pwl(in_a[i])) * in_b[i]` | 1 chu kỳ (streaming) |

---

## 4. Testbench & Cách chạy Simulation (`tb_snax_llm_sfu.sv`)

Testbench `tb_snax_llm_sfu` dùng hàm `fp16_to_real` / `real_to_fp16` và toán học `real` chuẩn (`$cos`, `$sin`, `$exp`, `$sqrt`) để kiểm tra tự động 4 bài test:
1. `RESIDUAL_ADD` (sai số cho phép `0.02`)
2. `ROPE` 8 cặp (sai số cho phép `0.02`)
3. `SWIGLU` 16 lanes từ `-3.0` đến `+3.0` (sai số cho phép `0.08`)
4. `RMSNORM` với `4` vector = `64` phần tử, `1/64 = 32'h3C800000`, `eps = 1e-5` (sai số cho phép `0.03`)

**Lệnh chạy trên QuestaSim (`sim/run_questa.bat`):**
```bat
c:\Users\nguyen\Desktop\NPU_LOG\sim\run_questa.bat -Filelist filelist_sfu.f -Top tb_snax_llm_sfu
```
Kết quả chuẩn đã kiểm chứng trên QuestaSim 2024.1: `Errors: 0, Warnings: 0`, in ra `=== ALL SFU TESTS PASSED (0 errors) ===`.

---

## 5. Gợi ý cho Agent tiếp theo khi tích hợp vào `snax_llm_top_clean.sv`
1. Cổng `in_a` của SFU nhận từ Mux giữa `core_fp16_packed` và `rd_fp16_data`; cổng `out` đi vào Mux đầu vào của `snax_logq_shared` (hoặc `u_write_out`).
2. Cần thêm 1 bộ đệm/streamer nhỏ (hoặc dùng `snax_stream_read_fp16` / scratchpad riêng) để cấp luồng `in_b_fp16_i` ($\gamma$, $\sin/\cos$, `up_proj`, residual $Y$).
3. Mở rộng thanh ghi cấu hình trong `snax_flash_csr_snax.sv` cho `cfg_op_i`, `cfg_num_vecs_i`, `cfg_inv_n_fp32_i`, `cfg_eps_fp32_i`.
