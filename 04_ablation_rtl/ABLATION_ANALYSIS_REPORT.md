# Báo cáo Phân tích Kết quả Ablation RTL & Synthesis (ASAP7 7nm @ 500 MHz)

Tài liệu này tổng hợp toàn bộ số liệu thực nghiệm đã kiểm chứng (RTL simulation trên QuestaSim 2024.1, Logic Synthesis bằng Synopsys Design Compiler `O-2018.06-SP1` với thư viện ASAP7 7nm RVT TT `PVT_0P7V_25C`, Gate-Level Simulation xuất `.saif` bằng VCS `O-2018.09-SP2`, và tính công suất SAIF bằng `dc_shell`) cho **6 cấu hình** ở chuẩn **Iso-Throughput `256 MACs/cycle @ 500 MHz` (`0.256 TOPS`, `ACC_DEPTH_P = 16`)**.

---

## 1. Bảng Kết quả Tổng hợp Toàn chip (`#1`–`#6`)

| Config | Top Module | Kiến trúc | Cờ Compile | Slack (ps) | Comb / Seq Area ($\mu\text{m}^2$) | Total Area ($\mu\text{m}^2$) | Switch / Int Power (mW) | Power SAIF (mW) | Area Eff. ($\text{TOPS/mm}^2$) | Power Eff. ($\text{TOPS/W}$) | Composite Eff. |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | `cfg1_mxfp4_g16_fp32_ff` | `MXFP4` (`E8M0`) + FP32, FF | `-no_autoungroup` | `+0.04` | `6244.77 / 3502.96` | `9747.74` | `3.194 / 3.770` | `6.971` | `26.26` | `36.72` | `964.5` |
| **#2** | `cfg2_nvfp4_fp32_ff` | `NVFP4` (`E4M3`) + FP32, FF | `-no_autoungroup` | `+0.02` | `7165.06 / 3527.22` | `10692.29` | `3.683 / 3.765` | `7.457` | `23.94` | `34.33` | `821.9` |
| **#3** | `cfg3_nvfp4_int32_lazy_ff` | `NVFP4` + INT32 Lazy, FF | `-no_autoungroup` | `+0.07` | `6855.62 / 3832.38` | `10688.00` | `3.814 / 4.273` | `8.095` | `23.95` | `31.62` | `757.5` |
| **#4** | `cfg4_log8_int32_lazy_ff` | `Log8` + INT32 Lazy, FF | `autoungroup` | `+0.13` | `6060.37 / 3749.57` | `9809.93` | `3.588 / 4.392` | `7.987` | `26.10` | `32.05` | `836.4` |
| **#5** | `cfg5_log8_int32_lazy_latch` | `Log8` + INT32 Lazy, **D-Latch** | `autoungroup` | `0.00` | `5788.81 / 2629.82` | **`8418.64`** | `3.657 / 3.784` | `7.448` | **`30.41`** | `34.37` | **`1045.2`** |
| **#6** | `cfg6_log8_fp32_ff` *(LOG8_MUL)* | `Log8` + FP32, FF | `-no_autoungroup` | `0.00` | `6698.21 / 3521.16` | `10219.37` | `3.424 / 3.564` | `6.996` | `25.05` | `36.59` | `916.7` |

*(Ghi chú: `#4` và `#5` đã đặt `ENABLE_RESCALE = 0` và `ENABLE_PACKED_OUT = 0` trong `vsq_matmul_engine.sv` để loại bỏ `vsq_rescaler_32b` và `core_vec_packed` không dùng. `#6` dùng module `LOG8_MUL` có cùng cấu trúc cây cộng `10→11→12→13b` và 1 tầng thanh ghi output giống `MXFP4_g16_MUL` / `NVFP4_MUL`.)*

---

## 2. Bảng Phân tách Datapath vs. Accumulator (`u_ff_acc` / `u_latch_array`)

Với `ACC_DEPTH_P = 16` (`16 slots × 16 lanes × 32 bit = 8 Kbit`), riêng mảng lưu trữ accumulator chiếm **`42%–54%` tổng diện tích** toàn chip. Khi tách riêng phần **Datapath** (`Total - Accumulator`):

| Config | Kiến trúc | Total Area ($\mu\text{m}^2$) | Acc Area ($\mu\text{m}^2$) | **Datapath Area ($\mu\text{m}^2$)** | Total Power (mW) | Acc Power (mW) | **Datapath Power (mW)** | **DP Area Eff. ($\text{TOPS/mm}^2$)** | **DP Power Eff. ($\text{TOPS/W}$)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | `MXFP4` + FP32 | `9747.74` | `5300.73` | **`4447.01`** | `6.971` | `2.168` | **`4.803`** | `57.57` | `53.30` |
| **#6** | `Log8` + FP32 *(LOG8_MUL)* | `10219.37` | `5285.19` | **`4934.18`** | `6.996` | `1.763` | **`5.233`** | `51.88` | `48.92` |
| **#2** | `NVFP4` + FP32 | `10692.29` | `5276.66` | **`5415.63`** | `7.457` | `1.783` | **`5.674`** | `47.27` | `45.12` |
| **#3** | `NVFP4` + INT32 Lazy | `10688.00` | `5377.64` | **`5310.36`** | `8.095` | `2.452` | **`5.643`** | `48.21` | `45.37` |
| **#4** | `Log8` + INT32 Lazy (FF) | `9809.93` | `4950.42` | **`4859.51`** | `7.987` | `2.608` | **`5.379`** | `52.68` | `47.59` |
| **#5** | `Log8` + INT32 Lazy (Latch) | `8418.64` | `3561.09` | **`4857.55`** | `7.448` | `2.333` | **`5.115`** | `52.70` | `50.05` |

---

## 3. Đối chiếu với Baseline `MiX` (28nm, `results_28nm_iso512.csv`)

Trong `03_baselines/MiX-main/hardware_rtl/results_28nm_iso512.csv` (512 MACs/cycle @ 500 MHz, systolic array `4×4`, mỗi PE chỉ có 1 thanh ghi tích lũy tương đương `ACC_DEPTH = 1`):

| Nguồn | `MXFP4_g16` | `NVFP4` | Chênh lệch (`NVFP4` vs `MXFP4_g16`) |
| :--- | :---: | :---: | :---: |
| **`MiX` (28nm, `ACC_DEPTH=1`)** | `57,422.48 µm²`, `33.858 mW` | `70,390.66 µm²`, `36.472 mW` | **`+22.6%` Area**, **`+7.7%` Power** |
| **Thực nghiệm của chúng ta (7nm, `ACC_DEPTH=16`)** | `9,747.74 µm²`, `6.971 mW` | `10,692.29 µm²`, `7.457 mW` | `+9.7%` Area, **`+7.0%` Power** |
| **Chỉ riêng Datapath của chúng ta (7nm)** | `4,447.01 µm²` | `5,415.63 µm²` | **`+21.8%` Area** |

- Tỷ lệ tăng công suất (**`+7.0%`** vs **`+7.7%`**) và tỷ lệ tăng diện tích Datapath (**`+21.8%`** vs **`+22.6%`**) khớp gần như tuyệt đối với `MiX`.
- Phần chênh lệch `+944.55 µm²` giữa `#1` và `#2` đến từ đúng 2 điểm của NVFP4:
  1. Bộ nhân `4b × 4b → 8b` trong `NVFP4_MUL` (`+29.5 µm²` cho 16 lanes).
  2. 16 bộ nhân `13b × 8b → 21b` (`+644.8 µm²`) và mở rộng `fp32_norm_add_2in` từ `MAG_W=13` lên `MAG_W=21` (`+294.3 µm²`).

---

## 4. Các Phát hiện Kỹ thuật Chính

1. **Log8-Scale + FP32 (`#6` với `LOG8_MUL`) nằm chính giữa `MXFP4` (`#1`) và `NVFP4` (`#2`)**:
   - **Area**: `#6` (`10,219.37 µm²`) cắt đúng **50%** phần overhead của NVFP4 so với MXFP4 (`+471.6 µm²` vs `+944.6 µm²`); giảm **`-4.4%` toàn chip** và **`-8.9%` riêng Datapath** (`4934.18` vs `5415.63 µm²`) so với `#2`.
   - **Power**: `#6` (`6.996 mW`) gần như bằng `#1 MXFP4` (`6.971 mW`, chỉ `+0.36%`) và giảm **`-6.2%` toàn chip** (`-7.8%` riêng Datapath) so với `#2 NVFP4` (`7.457 mW`).

2. **Vì sao `dot_product_16` cũ lớn hơn `NVFP4_MUL` (`~+680 µm²`)**:
   - `dot_product_16.sv` sign-extend lên `13 bit` ngay trước Level 1 và ép cứng `91` bộ `compressor_4to2` (kể cả trên các bit dấu `[12:9]`), đồng thời `fp4_mult_lut` dùng LUT 6→8b + negator `9-bit`.
   - `NVFP4_MUL` / `LOG8_MUL` chỉ nhân `2b×2b`, lấy bù-2 `5-bit`, dịch trái `<< shamt`, và dùng cây cộng tăng bit dần (`10→11→12→13b`) để DesignWare tự tối ưu.

3. **Vì sao Power của INT32 Lazy FF (`#3`: `8.095 mW`, `#4`: `7.987 mW`) cao hơn nhóm FP32 (`6.971–7.457 mW`)**:
   - Bản thân `u_drain_norm` cực nhẹ (chỉ **`~0.022–0.029 mW`**, `< 0.4%` tổng công suất), và **Datapath-only** của `#4` (`5.379 mW`) và `#5` (`5.115 mW`) đều **thấp hơn `#2` (`5.674 mW`) và `#3` (`5.643 mW`)**.
   - Phần công suất tăng ở `#3`/`#4` nằm tại **`u_ff_acc` / `u_latch_array` (`2.452 / 2.608 mW` so với `1.763–1.783 mW` ở `#6`/`#2`)** do:
     - Lưu **INT32 bù-2 (`Q16.16`)**: khi tổng đổi dấu qua `0`, 16 bit cao mở rộng dấu `[31:16]` lật hàng loạt trong mảng 8 Kbit.
     - Kích thích mặc định trong `tb_ablation_saif.sv` (`pkt[71:64] = 50..75`) gây bão hòa liên tục ở `Log8-Scale` (`2^12..2^18`).
     - `#3`/`#4`/`#5` mang thêm tầng thanh ghi pipeline `aligned_prod_q1` 512-bit.

4. **Vai trò quyết định của D-Latch (`#5`: `8,418.64 µm²`, `7.448 mW`)**:
   - Thay FF bằng D-Latch giảm `u_latch_array` từ `4950.42 → 3561.09 µm²` (**`-28.1%`**) và giảm công suất mảng nhớ từ `2.608 → 2.333 mW`, giúp `#5` đạt **Area nhỏ nhất (`-21.3%` so với `#2` NVFP4)** và **Composite Efficiency cao nhất (`1045.2`)**.

5. **Mở rộng lên 32-Lane PE (`NUM_LANES_P = 32`, `ACC_DEPTH_P = 16`)**:
   - Cả Datapath lẫn Accumulator đều tăng tuyến tính `~2×`, nên **tỷ lệ `%Area` và `%Power` giữa các khối gần như giữ nguyên (< 1% sai khác)**; `Area Eff.` và `Power Eff.` giữ nguyên khi Throughput tăng gấp đôi (`0.256 → 0.512 TOPS`).

---

## 5. Khuyến nghị Tối ưu Tiếp theo cho Đợt Chạy Paper
1. **Đồng nhất cờ tổng hợp**: Chạy lại toàn bộ `#1`–`#6` với cùng một thiết lập (`compile_ultra` + `+define+IGC_CELL +define+LATCH_CELL` cho ASAP7).
2. **Tối ưu `latch_array.sv`**:
   - Bỏ `negedge rst_n` trong module `dff` (`USE_FF=1`) để map sang `DFFHQx*` (đã có `slot_cleared` bảo vệ).
   - Tăng `ICG_GROUP = 64` (giảm từ 256 xuống 128 `icg_cell`, tải vẫn dưới giới hạn 92.16 fF của `ICGx4_ASAP7_75t_R`).
   - Cô lập `raddr2` / mux đọc Port B khi `ren2 = 0` để tránh cây mux 16:1 (512-bit) chuyển mạch vô ích 15/16 chu kỳ.
3. **Chuẩn hóa kích thích `tb_ablation_saif.sv` cho Log8**: Dùng dải scale có dấu hợp lý (ví dụ `[-16..+16]`) hoặc trace LLM thật (`+define+STIM_LLM_HEX`) để tránh bão hòa 32-bit giả tạo ở `#4`/`#5`.
