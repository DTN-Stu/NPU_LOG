# 07_pnr — ASAP7 7nm Physical Implementation (P&R @ 500 MHz)

Thư mục này lưu toàn bộ script, ràng buộc, LEF đã vá tương thích, log, report signoff và hình ảnh layout sau Place & Route cho `snax_llm_top_clean` bằng **Cadence Innovus `v15.20-p005_1`** trên công nghệ **ASAP7 7.5T Multi-Vt (RVT/LVT/SLVT) + 32 SRAM macros (`srambank_256x4x72_6t122`)**.

---

## 1. Cấu trúc thư mục

| File / Thư mục | Mô tả |
| :--- | :--- |
| [mmmc.tcl](mmmc.tcl) | Cấu hình MMMC: 15 file `.lib` TT (RVT/LVT/SLVT) + SRAM `.lib` + `qrcTechFile_typ03_scaled4xV06` |
| [snax_500mhz.sdc](snax_500mhz.sdc) | Ràng buộc timing `500 MHz` (`2000.0 ps`), giữ nguyên exception & I/O budget từ synthesis (`+333.3 ps` margin) |
| [01_floorplan.tcl](01_floorplan.tcl) | `init_design`, floorplan chữ C 32 SRAM (`14 Top + 4 Right + 14 Bottom`), blockage khe SRAM, `createGuide`, `editPin` (M4, cạnh Trái), `addWellTap` |
| [02_power.tcl](02_power.tcl) | `sroute`, `m2followRail.tcl`, `addStripe` (`M3..M6`), `verifyConnectivity`, `verify_drc` |
| [03_place.tcl](03_place.tcl) | `place_opt_design`, `checkPlace`, `timeDesign -preCTS` |
| [04_cts.tcl](04_cts.tcl) | `ccopt_design -cts`, `optDesign -postCTS` (setup & hold), `report_ccopt_*` |
| [05_route.tcl](05_route.tcl) | `routeDesign`, `optDesign -postRoute` (setup & hold), chèn `DECAP`/`TAPCELL`/`FILLER`, `extractRC` (`.spef`), `verifyConnectivity`, `verify_drc`, xuất `.def.gz` & netlist post-P&R |
| [highlight_blocks.tcl](highlight_blocks.tcl) | Script tô màu 11 nhóm khối kiến trúc trong Innovus GUI |
| `lef_fixed/` | Bản vá cục bộ của `asap7_tech_4x_201209.lef` & `srambank_256x4x72_6t122.lef.4x.lef` tương thích Innovus `15.2` |
| `reports/` | Toàn bộ report từng chặng (`01_fplan`, `02_pwr_*`, `03_preCTS_*`, `04_postCTS_*`, `05_postRoute_*`, `05_final_*`) |
| `logs/` | Toàn bộ log & `.cmd` của các lần chạy trên `eda` |
| [layout_clean.png](layout_clean.png), [layout_clean.gif](layout_clean.gif), [layout_legend.png](layout_legend.png) | Ảnh layout sạch (tắt IO pins) và bảng chú thích màu từng khối |

> **Lưu ý dung lượng lớn:** Các file nhị phân/nặng trên `eda:~/sys_pnr/out/` gồm `05_final.enc.dat`, `snax_llm_top_clean_500m.spef` (`344 MB`), `snax_llm_top_clean_500m.def.gz` (`70 MB`), `snax_llm_top_clean_pnr.v` (`54 MB`) được lưu tại server `eda`.

---

## 2. Kết quả Post-Route Signoff (`500 MHz` / `2000.0 ps`)

### 2.1. Timing (sau `extractRC` SPEF/RCDB từ Quantus QRC)
| Giai đoạn | Setup WNS / TNS (All `36,851` paths) | Hold WNS / TNS (`reg2reg`: `35,800` paths) | Ghi chú |
| :--- | :---: | :---: | :--- |
| **Pre-CTS (`03_place`)** | `0.000 ns` / `0.000 ns` (`0` viols) | — | Overflow `0.49% H / 0.75% V` |
| **Post-CTS (`04_cts`)** | `0.000 ns` / `0.000 ns` (`0` viols) | `+0.003 ns` / `0.000 ns` (`0` viols) | `reg2cgate`: `+0.178 ns`, Overflow `0.55% H / 0.48% V` |
| **Post-Route (`05_route`)** | **`0.000 ns` / `0.000 ns` (`0` viols)** | **`+0.003 ns` / `0.000 ns` (`0` viols)** | Tổng `default` hold còn `1` path I/O `-0.002 ns` (`-2 ps`) |

### 2.2. Diện tích (`4x` Innovus → `1x` 7nm thật, chia `4` cho cạnh và `16` cho diện tích)
* **Die Boundary:** `2505.6 × 2268.0 µm` (`4x`) → **`626.4 × 567.0 µm` (`0.355 mm²`)** ở `1x` thật.
* **Total Cell Area (trước filler):** `2,052,734.76 µm²` (`4x`) → **`128,295.92 µm²`** (`1x` thật, tăng `+6.2%` so với `120,764.90 µm²` sau synthesis).
* **Phân bổ diện tích các khối chính (`1x` thật, từ `reports/05_final_area.rpt`):**
  * `u_tcdm` (32 SRAMs): `83,937.34 µm²`
  * `u_core` (MAC + D-Latch + Softmax + Normalizer): `17,783.97 µm²`
  * `u_interconnect` (25×32 Crossbar): `11,486.75 µm²`
  * `u_wt_streamer`: `2,568.75 µm²` | `u_write_out`: `2,488.56 µm²` | `u_sfu`: `2,166.00 µm²` | `u_seq`: `1,590.14 µm²`
* **Density:** `17.651%` (trước filler) → `100.000%` (sau `DECAPx{10,6,4}`, `TAPCELL`, `FILLER`).

### 2.3. Công suất (`reports/05_final_power.rpt`, `0.70V`, `500 MHz`, default switching activity + SPEF)
* **Total Power:** **`137.0 mW`** (`Internal = 102.5 mW`, `Switching = 32.36 mW`, `Leakage = 2.095 mW`).
* **Clock Network (`clk`):** `1.386 mW` (`1.012%` tổng công suất).

### 2.4. Connectivity & DRC (`reports/05_final_conn.rpt`, `reports/05_final_drc.rpt`)
* **Power Grid (`02_pwr_drc.rpt`):** `0` DRC violations.
* **Post-Route Connectivity:** `1,000` cảnh báo `IMPVFC-200` là các đoạn stub `M1/M2` của `m2followRail.tcl` nằm trong khe hở giữa 32 SRAM (không phải đứt net tín hiệu).
* **Post-Route DRC:** `1,000` lỗi `MAR` (Minimum Area Rule) trên `M1` tại via pad của `Net clk` do `routeBottomRoutingLayer 2` trên Innovus `15.2` thả pad `M1` nhỏ hơn ngưỡng Minimum Area của `asap7_tech_4x_201209.lef`.

---

## 3. Các điểm hiệu chỉnh tương thích ASAP7 4x trên Innovus `15.2`
1. **Đơn vị thư viện (`IMPTS-432`):** SRAM `.lib` dùng `1ns`/`1pf` còn ASAP7 `.lib` dùng `1ps`/`1ff` → bắt buộc gọi `setLibraryUnit -time 1ps -cap 1ff` **trước** `init_design`.
2. **SRAM LEF & Tech LEF (`IMPLF-40`, `IMPLF-82`, `NRFL-2`):** Lưu bản vá tại `lef_fixed/`:
   * Đổi `SITE coreSite` → `SITE asap7sc7p5t` trong `srambank_256x4x72_6t122.lef.4x.lef` và snap `174.5608` → `174.56`.
   * Đặt `MANUFACTURINGGRID 0.002` và bỏ `LEF58_RIGHTWAYONGRIDONLY` trong `asap7_tech_4x_201209.lef` để `verify_drc` chạy trên Innovus `15.2`.
3. **Hướng đặt SRAM (`IMPFP-3929`):** `srambank_256x4x72_6t122` chỉ có `SYMMETRY X Y` (không cho xoay `R90`) → bố cục chữ C giữ thẳng cả 32 bank (`14 Top R0 + 4 Right MY + 14 Bottom MX`).
4. **Biến `var` trong `m2followRail.tcl`:** Gọi `catch {unset var}` trước và sau khi `source m2followRail.tcl` để `saveDesign` không lỗi.
5. **Tắt `SIAware` (`IMPOPT-6080`):** Gọi `setDelayCalMode -SIAware false` trước `optDesign -postRoute` khi chạy phân tích Non-OCV.

---

## 4. Hướng dẫn sử dụng

### Mở lại layout đã P&R xong để xem / chụp ảnh (`eda:~/sys_pnr`)
```bash
cd ~/sys_pnr
/EDA/Cadence/INNOVUS152/bin/innovus
```
Trong `innovus>`:
```tcl
setLibraryUnit -time 1ps -cap 1ff
restoreDesign out/05_final.enc.dat snax_llm_top_clean
source highlight_blocks.tcl
setPreference DisplayPtnPin 0
catch {setLayerPreference term -isVisible 0}
fit
dumpToGIF layout_clean.gif
```

### Chạy lại toàn bộ flow P&R từ đầu (`eda:~/sys_pnr`)
```bash
cd ~/sys_pnr
/EDA/Cadence/INNOVUS152/bin/innovus -no_gui -init run_stage1_2.tcl -log logs/stage1_2
/EDA/Cadence/INNOVUS152/bin/innovus -no_gui -init 03_place.tcl     -log logs/stage3
/EDA/Cadence/INNOVUS152/bin/innovus -no_gui -init 04_cts.tcl       -log logs/stage4
/EDA/Cadence/INNOVUS152/bin/innovus -no_gui -init 05_route.tcl     -log logs/stage5
```
