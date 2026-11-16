# Hướng dẫn Chạy Simulation & Synthesis (PC Local + Server `eda`, PDK ASAP7 7nm)

Tài liệu này ghi lại quy trình thực tế đã được kiểm chứng trên PC này và server `eda` cho bộ RTL trong `04_ablation_rtl/`, giúp các agent hoặc người dùng sau có thể chạy lại ngay mà không gặp các lỗi môi trường đã xử lý.

---

## 1. Thông tin Môi trường & Đường dẫn

### 1.1. Trên PC cục bộ (Windows)
- **Workspace RTL**: `d:\LVThs\NPU_PRJ\NPU_LOG\04_ablation_rtl`
- **QuestaSim**: `C:\questasim64_2024.1\win64\` (`vlib.exe`, `vlog.exe`, `vsim.exe`)
- **PDK ASAP7 7nm (nguồn)**: `C:\Users\nguye\Desktop\NPU_LOG\lib\ASAP7\`
  - Liberty (`.lib`): `C:\Users\nguye\Desktop\NPU_LOG\lib\ASAP7\lib\`
  - Verilog stdcell (`.v`): `C:\Users\nguye\Desktop\NPU_LOG\lib\ASAP7\verilog\*_RVT_TT_*.v`
- **Script synthesis tham chiếu**: `C:\Users\nguye\Desktop\NPU_LOG\syn\`
  - `scripts\pe_syn.tcl`, `setup\dc_setup.tcl`, `libs\compile_libs_asap7.tcl`, `constraints\vsq_feba_core_top.sdc`

### 1.2. Trên Server EDA (`ssh eda`)
- **Công cụ đã xác nhận có sẵn**:
  - Synopsys Design Compiler: `/EDA/Synopsys/syn_v201806/bin/dc_shell` (`O-2018.06-SP1`)
  - Synopsys Library Compiler: `/EDA/Synopsys/libcompiler_v201806/bin/lc_shell`
  - Synopsys VCS: `/EDA/Synopsys/vcsmx_v201809/bin/vcs` (`O-2018.09-SP2`)
  - Synopsys PrimeTime / `vcd2saif`: `/EDA/Synopsys/pt_v201806/bin/pt_shell`, `/EDA/Synopsys/syn_v201806/bin/vcd2saif`
  - Cadence Genus: `/EDA/Cadence/GENUS152/bin/genus`
  - QuestaSim (Linux): `/EDA/Mentor/Questasim10.7c/questasim/bin/vsim`
- **Thư viện ASAP7 `.db` đã compile sẵn trên `eda`**:
  - Thư mục: `/home/ICer/vsq_syn/syn/libs/db/asap7/`
  - Các file RVT TT: `AO_RVT_TT.db`, `OA_RVT_TT.db`, `SIMPLE_RVT_TT.db`, `INVBUF_RVT_TT.db`, `SEQ_RVT_TT.db` (và `fakeram7_64x256.db`).

> **Lưu ý quan trọng khi truyền file lên `eda`:**
> **KHÔNG dùng `scp`** vì `.bashrc` trên `eda` in log `lmgrd` ra stdout khi mở phiên non-interactive, làm `scp` lỗi (`Received message too long`).
> Thay vào đó, hãy đóng gói bằng `tar` và pipe qua `ssh`:
> ```powershell
> tar -C <LOCAL_STAGING_DIR> -cf - . | ssh eda "mkdir -p ~/ablation_test && tar -xf - -C ~/ablation_test"
> ```

---

## 2. Bước 1 — Chạy RTL Simulation trên PC bằng QuestaSim

*(Nếu chạy trong sandbox gặp lỗi `{Illegal System DLL Relocation}`, cần chạy ngoài sandbox.)*

```powershell
cd d:\LVThs\NPU_PRJ\NPU_LOG\04_ablation_rtl
if (Test-Path work) { Remove-Item -Recurse -Force work }
vlib work
vlog -sv -f filelist.f tb_ablation_saif.sv +define+DUT_CFG1
vsim -c tb_ablation_saif -do "run -all; quit"
```

- Thay `+define+DUT_CFG1` bằng `DUT_CFG2` .. `DUT_CFG5` tương ứng với 5 cấu hình.
- Kết quả đạt khi terminal in: `[tb_ablation_saif] Completed 4000 capture cycles cleanly.` và `Errors: 0`.

---

## 3. Bước 2 — Upload RTL & Chạy Logic Synthesis (`dc_shell`) trên `eda`

### 3.1. Các lưu ý bắt buộc khi viết script TCL cho ASAP7 + `04_ablation_rtl`
1. **Đơn vị thời gian**: Thư viện ASAP7 dùng `1 ps`. Với mục tiêu **500 MHz (`2.0 ns`)**, đặt `PERIOD_NS = 2.0` và `TSCALE = 1000.0` (tương đương `2000.0 ps`) nếu không gọi `set_units -time ns`.
2. **Tên port Reset**: Script mẫu `pe_syn.tcl` dùng `rst_n`, nhưng cả 5 top module trong `04_ablation_rtl/` đều dùng **`reset_n`**. Phải đặt:
   ```tcl
   set_false_path -from [get_ports reset_n]
   ```
3. **Driving cell ASAP7**: Dùng `NAND2x1_ASAP7_75t_R` (pin `Y`), điều kiện hoạt động `PVT_0P7V_25C`.
4. **Danh sách RTL tối thiểu cho từng Config**:
   - Chung (`+incdir+$RTL_DIR`): `pe_defines.svh`
   - Config #1 (`cfg1_mxfp4_g16_fp32_ff`): `latch_array.sv`, `MXFP4_g16_MUL.sv`, `fp32_lane_backend.sv`, `cfg1_mxfp4_g16_fp32_ff.sv`
   - Config #2 (`cfg2_nvfp4_fp32_ff`): `latch_array.sv`, `NVFP4_MUL.sv`, `fp32_lane_backend.sv`, `cfg2_nvfp4_fp32_ff.sv`
   - Config #3 (`cfg3_nvfp4_int32_lazy_ff`): `fp4_mult_lut.sv`, `compressor_4to2.sv`, `dot_product_16.sv`, `latch_array.sv`, `vsq_rescaler_32b.sv`, `vsq_matmul_engine.sv`, `nvfp4_scale_mac_lane.sv`, `cfg3_nvfp4_int32_lazy_ff.sv`
   - Config #4 / #5 (`cfg4_log8_int32_lazy_ff`, `cfg5_log8_int32_lazy_latch`): giống Config #3 nhưng bỏ `nvfp4_scale_mac_lane.sv` và thay top file tương ứng. *(Lưu ý: nếu muốn map ICG/D-Latch sang cell ASAP7 tường minh ở Config #5, kiểm tra define `IGC_CELL` / `LATCH_CELL` trong `latch_array.sv`.)*

### 3.2. Script TCL mẫu (`syn_cfg1_test.tcl`)
```tcl
set BASE_DIR   [file normalize [file dirname [info script]]]
set RTL_DIR    [file join $BASE_DIR rtl]
set REPORT_DIR [file join $BASE_DIR reports]
set OUTPUT_DIR [file join $BASE_DIR output]
file mkdir $REPORT_DIR
file mkdir $OUTPUT_DIR

set DB_ASAP7 "/home/ICer/vsq_syn/syn/libs/db/asap7"
set CELL_DBS [list \
    [file join $DB_ASAP7 AO_RVT_TT.db] \
    [file join $DB_ASAP7 OA_RVT_TT.db] \
    [file join $DB_ASAP7 SIMPLE_RVT_TT.db] \
    [file join $DB_ASAP7 INVBUF_RVT_TT.db] \
    [file join $DB_ASAP7 SEQ_RVT_TT.db]]

set_app_var search_path [concat [list $RTL_DIR $DB_ASAP7] [get_app_var search_path]]
set_app_var synthetic_library [list dw_foundation.sldb]
set_app_var target_library $CELL_DBS
set_app_var link_library   [concat [list "*"] $CELL_DBS]

set_app_var hdlin_auto_save_templates true
set_app_var verilogout_no_tri         true
set_app_var bus_naming_style          {%s[%d]}

set DESIGN_NAME "cfg1_mxfp4_g16_fp32_ff"
set PERIOD_NS   2.0
set TSCALE      1000.0

set files [list \
    [file join $RTL_DIR latch_array.sv] \
    [file join $RTL_DIR MXFP4_g16_MUL.sv] \
    [file join $RTL_DIR fp32_lane_backend.sv] \
    [file join $RTL_DIR cfg1_mxfp4_g16_fp32_ff.sv]]

analyze -format sverilog -vcs [list "+incdir+$RTL_DIR"] $files
elaborate $DESIGN_NAME
current_design $DESIGN_NAME
link

set CLK [expr {$PERIOD_NS * $TSCALE}]
create_clock -name clk -period $CLK [get_ports clk]
set_clock_uncertainty [expr {0.05 * $TSCALE}] [get_clocks clk]

set in_ports [remove_from_collection [all_inputs] [get_ports clk]]
set_input_transition [expr {0.05 * $TSCALE}] $in_ports
set_driving_cell -lib_cell NAND2x1_ASAP7_75t_R -pin Y $in_ports
set_input_delay  -clock clk -max [expr {0.2 * $CLK}] $in_ports
set_input_delay  -clock clk -min 0                 $in_ports
set_output_delay -clock clk -max [expr {0.2 * $CLK}] [all_outputs]
set_output_delay -clock clk -min 0                 [all_outputs]
set_load 0.02 [all_outputs]
set_false_path -from [get_ports reset_n]
set_max_area 0

compile_ultra -no_autoungroup

redirect [file join $REPORT_DIR ${DESIGN_NAME}_area.rpt]   { report_area -hierarchy }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_power.rpt]  { report_power -hierarchy }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_timing.rpt] { report_timing -max_paths 5 }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_qor.rpt]    { report_qor }

write -format verilog -hierarchy -output [file join $OUTPUT_DIR ${DESIGN_NAME}_netlist.v]
write -format ddc     -hierarchy -output [file join $OUTPUT_DIR ${DESIGN_NAME}.ddc]
write_sdc -nosplit                 [file join $OUTPUT_DIR ${DESIGN_NAME}.sdc]
exit
```

Chạy trên `eda`:
```bash
cd ~/ablation_test && dc_shell -f syn_cfg1_test.tcl > syn_cfg1.log 2>&1
```

---

## 4. Bước 3 — Gate-Level Simulation (`vcs`) xuất `.saif` trên `eda`

### 4.1. Hai lỗi phổ biến đã xử lý
1. **Lỗi `Error-[IIPC] Illegal implicit port connection`**: Netlist Verilog do DC xuất tách port mảng 2D thành dạng concat, nên `.*` trong testbench sẽ lỗi. `tb_ablation_saif.sv` đã được sửa sang kết nối port tường minh để dùng chung cho cả RTL và Gate-Level.
2. **Lỗi `Error-[URMI] Unresolved modules` cho `altos_dff_sr_0` / `altos_latch`**: Khi dùng `-v asap7sc7p5t_SEQ_RVT_TT_220101.v`, `vcs` không nạp các `primitive` UDP nằm ở cuối file. **Giải pháp**: truyền trực tiếp các file `.v` vào `vcs` (không dùng `-v`), kèm `+nospecify +notimingchecks`.

### 4.2. Chuẩn bị & chạy `vcs`
Upload 5 file Verilog RVT TT từ `C:\Users\nguye\Desktop\NPU_LOG\lib\ASAP7\verilog\` (`AO`, `INVBUF`, `OA`, `SEQ`, `SIMPLE`) và `tb_ablation_saif.sv` vào `~/ablation_test/sim/`, sau đó chạy trên `eda`:

```bash
cd ~/ablation_test/sim
vcs -sverilog -full64 +v2k -timescale=1ns/10ps +nospecify +notimingchecks \
    asap7_v/asap7sc7p5t_AO_RVT_TT_201020.v \
    asap7_v/asap7sc7p5t_INVBUF_RVT_TT_201020.v \
    asap7_v/asap7sc7p5t_OA_RVT_TT_201020.v \
    asap7_v/asap7sc7p5t_SEQ_RVT_TT_220101.v \
    asap7_v/asap7sc7p5t_SIMPLE_RVT_TT_201020.v \
    ../output/cfg1_mxfp4_g16_fp32_ff_netlist.v \
    tb_ablation_saif.sv \
    +define+DUT_CFG1 +define+DUMP_SAIF \
    -o simv_cfg1 > vcs_cfg1.log 2>&1
./simv_cfg1
```

Kết quả thành công khi log in `[tb_ablation_saif] Completed 4000 capture cycles cleanly.` và tạo ra `ablation_dut.saif` (~16 MB cho Config #1).

---

## 5. Bước 4 — Đọc `.saif` vào `dc_shell` để xuất `power_saif.rpt`

Script `read_saif_cfg1.tcl` trên `eda`:
```tcl
set BASE_DIR   [file normalize [file dirname [info script]]]
set REPORT_DIR [file join $BASE_DIR reports]
set OUTPUT_DIR [file join $BASE_DIR output]
set SIM_DIR    [file join $BASE_DIR sim]

set DB_ASAP7 "/home/ICer/vsq_syn/syn/libs/db/asap7"
set CELL_DBS [list \
    [file join $DB_ASAP7 AO_RVT_TT.db] \
    [file join $DB_ASAP7 OA_RVT_TT.db] \
    [file join $DB_ASAP7 SIMPLE_RVT_TT.db] \
    [file join $DB_ASAP7 INVBUF_RVT_TT.db] \
    [file join $DB_ASAP7 SEQ_RVT_TT.db]]

set_app_var search_path [concat [list $DB_ASAP7] [get_app_var search_path]]
set_app_var target_library $CELL_DBS
set_app_var link_library   [concat [list "*"] $CELL_DBS]

set DESIGN_NAME "cfg1_mxfp4_g16_fp32_ff"
read_ddc [file join $OUTPUT_DIR ${DESIGN_NAME}.ddc]
current_design $DESIGN_NAME
link

read_saif -input [file join $SIM_DIR ablation_dut.saif] -instance_name "tb_ablation_saif/dut"
redirect [file join $REPORT_DIR ${DESIGN_NAME}_power_saif.rpt] { report_power -hierarchy }
exit
```

Chạy trên `eda`:
```bash
cd ~/ablation_test && dc_shell -f read_saif_cfg1.tcl > read_saif_cfg1.log 2>&1
```

---

## 6. Kết quả Tham chiếu Đã Kiểm chứng (Config #1 @ 500 MHz, ASAP7 7nm RVT TT)

| Hạng mục | Giá trị đo được | Nguồn |
| :--- | :--- | :--- |
| **RTL Sim (QuestaSim 2024.1)** | `4000` capture cycles, `0` lỗi | Local PC |
| **Setup Slack (500 MHz / `2000 ps`)** | **`+0.04 ps` (MET)**, `0` path vi phạm | `cfg1_mxfp4_g16_fp32_ff_qor.rpt` |
| **Total Cell Area** | **`9747.74 µm²`** (Comb `6244.77`, Seq `3502.96`, `96,318` leaf cells) | `cfg1_mxfp4_g16_fp32_ff_area.rpt` |
| **Power (Default DC, không SAIF)** | `4.118 mW` | `cfg1_mxfp4_g16_fp32_ff_power.rpt` |
| **Power (Gate-Level SAIF, 4000 cyc)** | **`6.971 mW`** (Switch `3.194`, Int `3.770`, Leak `7.67e+06 pW`) | `cfg1_mxfp4_g16_fp32_ff_power_saif.rpt` |
| **Accumulator (`u_ff_acc`) Power** | `2.168 mW` (`31.1%` tổng công suất) | `cfg1_mxfp4_g16_fp32_ff_power_saif.rpt` |
