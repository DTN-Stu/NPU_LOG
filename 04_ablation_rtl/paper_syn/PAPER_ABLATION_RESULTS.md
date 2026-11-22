# Kết quả Synthesis & SAIF Power Chuẩn cho Paper (5-Config Ablation)

- **Công nghệ & Công cụ:** ASAP7 7nm RVT TT (`PVT_0P7V_25C`), Synopsys Design Compiler `O-2018.06-SP1`, Synopsys VCS `O-2018.09-SP2` (Gate-Level Simulation).
- **Ràng buộc đồng nhất:** `500 MHz` (`2000 ps`), `compile_ultra -no_autoungroup`, `+define+IGC_CELL +define+LATCH_CELL` (`ICGx4_ASAP7_75t_R`, `DHLx2_ASAP7_75t_R`), `set_max_time_borrow 500.0`, `set_clock_gating_check -setup 50.0 -hold 50.0`, `FF_ASYNC_RST = 0`, `ACC_DEPTH_P = 16` (`256 MACs/cycle = 0.256 TOPS`).
- **Tất cả 5 cấu hình đạt:** `TNS = 0.00 ps`, `WNS >= 0.00 ps`, **0 unconstrained endpoints**.
- **Kích thích Power:**
  - `Real Llama-3.2-1B`: Lượng tử hóa bit-exact từ cùng tensor `real_llama32_1b_layer1_slice.npz` sang từng định dạng (`MXFP4`, `NVFP4`, `Log8`), 4000 capture cycles sau 64 cycles warmup.
  - `Uniform`: Ngẫu nhiên phân bố đều (worst-case toggle, đối chứng kiểu `MiX`).

## 1. Bảng PPA Toàn chip (500 MHz, ASAP7 7nm)

| Config | WNS (ps) | TNS (ps) | Comb ($\mu\text{m}^2$) | Seq ($\mu\text{m}^2$) | Total Area ($\mu\text{m}^2$) | Acc Area ($\mu\text{m}^2$) | Datapath Area ($\mu\text{m}^2$) | Pwr Real (mW) | Pwr Unif (mW) | Area Eff. ($\text{TOPS/mm}^2$) | Pwr Eff. Real ($\text{TOPS/W}$) | Composite Eff. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1: MXFP4 + FP32 + FF** | `+0.01` | `0.00` | `5814.82` | `2827.72` | **`8642.54`** | `4226.76` | `4415.79` | **`5.932`** | `7.225` | `29.62` | `43.16` | **`1278.3`** |
| **C2: NVFP4 + FP32 + FF** | `+0.03` | `0.00` | `6788.73` | `2854.47` | **`9643.20`** | `4219.06` | `5424.14` | **`7.893`** | `8.530` | `26.55` | `32.43` | **`861.0`** |
| **C3: Log8 + FP32 + FF** | `+0.01` | `0.00` | `6269.01` | `2836.60` | **`9105.60`** | `4223.87` | `4881.73` | **`7.316`** | `8.013` | `28.11` | `34.99` | **`983.8`** |
| **C4: Log8 + INT32 Lazy + FF** | `+0.01` | `0.00` | `5162.97` | `2729.87` | **`7892.84`** | `4116.49` | `3776.35` | **`6.397`** | `6.525` | `32.43` | `40.02` | **`1298.0`** |
| **C5: Log8 + INT32 Lazy + D-Latch** | `+0.00` | `0.00` | `4868.90` | `2326.76` | **`7195.67`** | `3427.03` | `3768.64` | **`5.856`** | `5.982` | **`35.58`** | **`43.72`** | **`1555.3`** |

## 2. Phân rã Công suất (Real Llama-3.2-1B SAIF) & Hiệu suất Riêng Datapath

| Config | Sw (mW) | Int (mW) | Leak (mW) | Total (mW) | Acc Pwr (mW) | DP Pwr (mW) | DP Area Eff. ($\text{TOPS/mm}^2$) | DP Pwr Eff. ($\text{TOPS/W}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1: MXFP4 + FP32 + FF** | `2.976` | `2.949` | `0.0071` | `5.932` | `1.424` | `4.508` | `57.97` | `56.79` |
| **C2: NVFP4 + FP32 + FF** | `4.097` | `3.788` | `0.0080` | `7.893` | `1.812` | `6.081` | `47.20` | `42.10` |
| **C3: Log8 + FP32 + FF** | `3.774` | `3.535` | `0.0074` | `7.316` | `1.736` | `5.580` | `52.44` | `45.88` |
| **C4: Log8 + INT32 Lazy + FF** | `3.092` | `3.298` | `0.0064` | `6.397` | `2.214` | `4.183` | `67.79` | `61.20` |
| **C5: Log8 + INT32 Lazy + D-Latch** | `3.037` | `2.812` | `0.0060` | `5.856` | `1.772` | `4.084` | `67.93` | `62.68` |

## 3. Luận điểm chính rút ra cho Paper
1. **Hiệu quả của `Log8` (C3 vs C1, C2):** Trên cùng nền `FP32 + FF`, `Log8` cắt **53.8%** phần datapath area overhead của `NVFP4` so với `MXFP4` (`+465.94 µm²` vs `+1008.35 µm²`), đồng thời giảm `-7.3%` công suất thực tế (`7.316` vs `7.893 mW`).
2. **Hiệu quả của `INT32 Lazy Drain` (C4 vs C3):** Đẩy bộ chuẩn hóa FP32 ra khỏi vòng lặp RMW giúp giảm `-22.6%` Datapath Area (`4881.73 -> 3776.35 µm²`) và `-25.0%` Datapath Power (`5.580 -> 4.183 mW`), đồng thời mở khóa khả năng dịch bit cho FlashAttention $O$-rescaling.
3. **Hiệu quả của `Clock-Gated D-Latch` (C5 vs C4):** Giảm `-16.7%` Acc Area (`4116.49 -> 3427.03 µm²`) và `-20.0%` Acc Power (`2.214 -> 1.772 mW`).
4. **Tổng thể (`C5` vs `NVFP4 C2` & `MXFP4 C1`):** `C5` giảm **`-25.4%` Area** và **`-25.8%` Power** so với `NVFP4` (`1.81×` Composite Efficiency: `1555.3` vs `861.0`), và thậm chí nhỏ hơn **`-16.7%` Area** và tiết kiệm điện hơn cả baseline `MXFP4` (`C1`).
