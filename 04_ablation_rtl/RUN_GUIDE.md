# Hướng dẫn Chạy Simulation, Synthesis (500 MHz) & Thu thập Số liệu SAIF (Configs #1–#5)

Tài liệu này hướng dẫn chi tiết quy trình từ **kiểm tra RTL**, **tổng hợp ASIC (Synopsys Design Compiler)**, **mô phỏng Gate-Level xuất SAIF (Synopsys VCS)** đến **xử lý số liệu** để lập bảng kết quả cho paper **IEEE TCAD / TVLSI** (Iso-Throughput **256 MACs/cycle @ 500 MHz**, `ACC_DEPTH = 16`).

---

## 1. Tổng quan 5 Cấu hình Ablation

| Config | Top Module | Macro Testbench | Scale & Format | Accumulator |
| :--- | :--- | :--- | :--- | :--- |
| **#1** | `cfg1_mxfp4_g16_fp32_ff` | `+define+DUT_CFG1` | `MXFP4_g16` (E8M0, `MiX` RTL) | FP32 In-Loop, FF (`DEPTH=16`) |
| **#2** | `cfg2_nvfp4_fp32_ff` | `+define+DUT_CFG2` | `NVFP4` (E4M3, `MiX` RTL) | FP32 In-Loop, FF (`DEPTH=16`) |
| **#3** | `cfg3_nvfp4_int32_lazy_ff` | `+define+DUT_CFG3` | `NVFP4` (`nvfp4_scale_mac_lane`) | INT32 + Lazy FP16 Drain, FF (`DEPTH=16`) |
| **#4** | `cfg4_log8_int32_lazy_ff` | `+define+DUT_CFG4` | `Log8-Scale` (`vsq_matmul_engine`) | INT32 + Lazy FP16 Drain, FF (`DEPTH=16`) |
| **#5** | `cfg5_log8_int32_lazy_latch` | `+define+DUT_CFG5` | `Log8-Scale` (`vsq_matmul_engine`) | INT32 + Lazy FP16 Drain, **D-Latch** (`DEPTH=16`) |

---

## 2. Bước 1 — Chạy RTL Simulation (Kiểm tra chức năng)

Đứng tại thư mục `04_ablation_rtl/`, chạy VCS cho từng config (ví dụ `DUT_CFG5`):

```bash
# Chế độ mặc định: phân phối giống LLM (bell-shaped E2M1)
vcs -sverilog -full64 +v2k -timescale=1ns/10ps \
    -f filelist.f tb_ablation_saif.sv \
    +define+DUT_CFG5 -R

# Hoặc dùng trace thật từ 1 layer LLM (cần có stim_act.hex và stim_wt.hex cùng thư mục)
vcs -sverilog -full64 +v2k -timescale=1ns/10ps \
    -f filelist.f tb_ablation_saif.sv \
    +define+DUT_CFG5 +define+STIM_LLM_HEX -R

# Hoặc dùng Uniform-Random (đối chiếu kiểu MiX)
vcs -sverilog -full64 +v2k -timescale=1ns/10ps \
    -f filelist.f tb_ablation_saif.sv \
    +define+DUT_CFG5 +define+STIM_UNIFORM -R
```

Kết quả đạt yêu cầu khi log in ra:
`[tb_ablation_saif] Completed 4000 capture cycles cleanly.`

---

## 3. Bước 2 — Chạy Logic Synthesis bằng Synopsys Design Compiler (`dc_shell`)

### 3.1. Chuẩn bị script `syn_ablation.tcl`
Tạo file `syn_ablation.tcl` trong `04_ablation_rtl/` (cập nhật 4 biến thư viện PDK của bạn ở đầu file):

```tcl
# ==============================================================================
# syn_ablation.tcl — Chạy: dc_shell -f syn_ablation.tcl -x "set TOP_NAME cfg5_log8_int32_lazy_latch"
# ==============================================================================
set PDK_DB_PATH "<PATH_TO_PDK_DB>"
set TARGET_DB   "<YOUR_STDCELL_LIB>.db"
set OPCOND      "<OPERATING_CONDITION>"
set DRIVE_CELL  "<INPUT_DRIVE_CELL>"

if {![info exists TOP_NAME]} { set TOP_NAME "cfg5_log8_int32_lazy_latch" }

set_host_options -max_cores 16
set search_path [list "." "../01_snax_llm_fp4_subsystem" "../03_baselines/MiX-main/hardware_rtl/src/systolic_arrays/MXFP4_g16" "../03_baselines/MiX-main/hardware_rtl/src/systolic_arrays/NVFP4" $PDK_DB_PATH]
set_app_var target_library [list $TARGET_DB]
set_app_var link_library   [concat "*" $TARGET_DB]

# Đọc toàn bộ RTL từ filelist
set rtl_files [list \
    "../01_snax_llm_fp4_subsystem/fp4_mult_lut.sv" \
    "../01_snax_llm_fp4_subsystem/compressor_4to2.sv" \
    "../01_snax_llm_fp4_subsystem/dot_product_16.sv" \
    "../01_snax_llm_fp4_subsystem/latch_array.sv" \
    "../01_snax_llm_fp4_subsystem/vsq_rescaler_32b.sv" \
    "../01_snax_llm_fp4_subsystem/vsq_matmul_engine.sv" \
    "../01_snax_llm_fp4_subsystem/nvfp4_scale_mac_lane.sv" \
    "../03_baselines/MiX-main/hardware_rtl/src/systolic_arrays/MXFP4_g16/MXFP4_g16_MUL.sv" \
    "../03_baselines/MiX-main/hardware_rtl/src/systolic_arrays/NVFP4/NVFP4_MUL.sv" \
    "fp32_lane_backend.sv" \
    "cfg1_mxfp4_g16_fp32_ff.sv" \
    "cfg2_nvfp4_fp32_ff.sv" \
    "cfg3_nvfp4_int32_lazy_ff.sv" \
    "cfg4_log8_int32_lazy_ff.sv" \
    "cfg5_log8_int32_lazy_latch.sv" \
]

# Lưu ý: Nếu dùng thư viện ASAP7, thêm -define {IGC_CELL LATCH_CELL}
analyze -format sverilog $rtl_files
elaborate $TOP_NAME
current_design $TOP_NAME
link

# Ràng buộc Clock 500 MHz (2.0 ns)
create_clock -name clk -period 2.0 [get_ports clk]
set_clock_uncertainty 0.01 [get_clocks clk]
set_clock_transition  0.10 [get_clocks clk]
set_driving_cell -lib_cell $DRIVE_CELL [remove_from_collection [all_inputs] [get_ports clk]]
set_load 0.01 [all_outputs]
set_operating_conditions $OPCOND

# Cho phép clock-gating tự động nếu dùng cell thư viện chuẩn
compile_ultra -gate_clock

file mkdir "syn_out/${TOP_NAME}"
write -format ddc     -hierarchy -output "syn_out/${TOP_NAME}/${TOP_NAME}.ddc"
write -format verilog -hierarchy -output "syn_out/${TOP_NAME}/${TOP_NAME}.syn.v"
write_sdc "syn_out/${TOP_NAME}/${TOP_NAME}.sdc"

report_area   -hierarchy > "syn_out/${TOP_NAME}/area.rpt"
report_timing -max_paths 5 > "syn_out/${TOP_NAME}/timing.rpt"
report_power  -hierarchy > "syn_out/${TOP_NAME}/power_default.rpt"
exit
```

### 3.2. Chạy Synthesis cho cả 5 Configs
```bash
for TOP in \
    cfg1_mxfp4_g16_fp32_ff \
    cfg2_nvfp4_fp32_ff \
    cfg3_nvfp4_int32_lazy_ff \
    cfg4_log8_int32_lazy_ff \
    cfg5_log8_int32_lazy_latch
do
    dc_shell -f syn_ablation.tcl -x "set TOP_NAME $TOP" | tee "syn_${TOP}.log"
done
```

> **Kiểm tra bắt buộc:** Mở `syn_out/<TOP>/timing.rpt`, đảm bảo **`slack (MET) >= 0.00`** ở xung nhịp `2.0 ns` (500 MHz) cho cả 5 config.

---

## 4. Bước 3 — Gate-Level Simulation xuất SAIF & Đo Power thực tế

**Không dùng `power_default.rpt`** (vì DC ước lượng toggle rate mặc định không chính xác). Thực hiện 2 bước sau cho mỗi config:

### 4.1. Chạy Gate-Level Simulation trên VCS để dump `.saif`
*(Cần trỏ thêm file Verilog model `.v` của thư viện Standard Cell khi chạy gate-level)*

```bash
# Ví dụ với Config #5 (thay DUT_CFG1..5 và đường dẫn netlist tương ứng)
vcs -sverilog -full64 +v2k -timescale=1ns/10ps +neg_tchk \
    -v <PATH_TO_STDCELL_VERILOG_MODEL>.v \
    syn_out/cfg5_log8_int32_lazy_latch/cfg5_log8_int32_lazy_latch.syn.v \
    tb_ablation_saif.sv \
    +define+DUT_CFG5 +define+DUMP_SAIF \
    -o simv_cfg5
./simv_cfg5
mv ablation_dut.saif syn_out/cfg5_log8_int32_lazy_latch/power.saif
```

*(Nếu dùng **trace thật từ LLM**, thêm `+define+STIM_LLM_HEX` vào lệnh `vcs` ở trên.)*

### 4.2. Đọc SAIF vào Design Compiler để tính `Power (mW)`
Tạo file `read_saif_power.tcl`:
```tcl
if {![info exists TOP_NAME]} { set TOP_NAME "cfg5_log8_int32_lazy_latch" }
set TARGET_DB "<YOUR_STDCELL_LIB>.db"
set PDK_DB_PATH "<PATH_TO_PDK_DB>"
set search_path [list "." $PDK_DB_PATH]
set_app_var target_library [list $TARGET_DB]
set_app_var link_library   [concat "*" $TARGET_DB]

read_ddc "syn_out/${TOP_NAME}/${TOP_NAME}.ddc"
current_design $TOP_NAME
read_saif -input "syn_out/${TOP_NAME}/power.saif" -instance_name "tb_ablation_saif/dut"
report_power -hierarchy > "syn_out/${TOP_NAME}/power_saif.rpt"
exit
```

Chạy cho cả 5 config:
```bash
for TOP in cfg1_mxfp4_g16_fp32_ff cfg2_nvfp4_fp32_ff cfg3_nvfp4_int32_lazy_ff cfg4_log8_int32_lazy_ff cfg5_log8_int32_lazy_latch; do
    dc_shell -f read_saif_power.tcl -x "set TOP_NAME $TOP"
done
```

---

## 5. Bước 4 — Thu thập, Xử lý Dữ liệu & Tính các Chỉ số cho Paper

### 5.1. Vị trí trích xuất số liệu từ Report
1. **Diện tích (`Area_um2`):** Mở `syn_out/<TOP>/area.rpt`, lấy dòng **`Total cell area:`** (đơn vị $\mu\text{m}^2$).
   * Để vẽ **Area Breakdown** (Datapath vs. Accumulator vs. Lazy Drain), xem phần phân cấp module con ngay bên dưới trong `area.rpt` (`u_ff_acc` / `u_latch_array`, `GEN_DRAIN_NORM`, `u_fp32_acc`, `u_mix_*_mul`).
2. **Công suất (`Power_mW`):** Mở `syn_out/<TOP>/power_saif.rpt`, kiểm tra tỷ lệ SAIF annotation (~100%), lấy dòng **`Total Power`** (quy đổi về đơn vị **mW**; lưu ý DC đôi khi xuất `uW` hoặc `mW`, cần kiểm tra đơn vị ở header của report!).
3. **Timing (`Slack` / $f_{\max}$):** Lấy từ `syn_out/<TOP>/timing.rpt` (`slack (MET)`).

### 5.2. Công thức quy đổi Hiệu suất (chuẩn `256 MACs/cycle @ 500 MHz`)
* $1\text{ MAC} = 2\text{ OPs} \Rightarrow 256\text{ MACs/cycle} = 512\text{ OPs/cycle}$
* Tại $f = 500\text{ MHz} = 0.5\text{ GHz}$:
  $$\text{Throughput} = 512 \times 0.5 = 256\text{ GOPS} = \mathbf{0.256\text{ TOPS}}$$
* **Area Efficiency ($\text{TOPS/mm}^2$):**
  $$\text{Area Eff.} = \frac{0.256\text{ TOPS}}{\text{Area\_}\mu\text{m}^2 \times 10^{-6}} = \frac{256{,}000}{\text{Area\_}\mu\text{m}^2}$$
* **Power Efficiency ($\text{TOPS/W}$):**
  $$\text{Power Eff.} = \frac{0.256\text{ TOPS}}{\text{Power\_mW} \times 10^{-3}} = \frac{256}{\text{Power\_mW}}$$
* **Composite PE Efficiency (giống Fig. 10 của MiX):**
  $$\text{Composite Eff.} = \text{Area Eff.} \times \text{Power Eff.}$$

### 5.3. Script Python tổng hợp bảng tự động (`parse_syn_reports.py`)
Bạn có thể lưu đoạn script sau vào `04_ablation_rtl/parse_syn_reports.py` để tự động đọc `area.rpt` và `power_saif.rpt` xuất ra bảng CSV/Markdown:

```python
import re
from pathlib import Path

CONFIGS = [
    ("Config #1 (MXFP4_g16 + FP32 + FF)",     "cfg1_mxfp4_g16_fp32_ff"),
    ("Config #2 (NVFP4 + FP32 + FF)",         "cfg2_nvfp4_fp32_ff"),
    ("Config #3 (NVFP4 + INT32 Lazy + FF)",   "cfg3_nvfp4_int32_lazy_ff"),
    ("Config #4 (Log8 + INT32 Lazy + FF)",    "cfg4_log8_int32_lazy_ff"),
    ("Config #5 (Log8 + INT32 Lazy + Latch)", "cfg5_log8_int32_lazy_latch"),
]

TOPS = 0.256  # 256 MACs/cycle * 2 OPs/MAC * 0.5 GHz

def parse_area(path: Path) -> float:
    txt = path.read_text()
    m = re.search(r"Total cell area:\s+([\d.]+)", txt)
    return float(m.group(1)) if m else 0.0

def parse_power_mw(path: Path) -> float:
    txt = path.read_text()
    m = re.search(r"Total\s+[\d.eE+-]+\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)\s*(mW|uW|W)?", txt)
    if not m:
        return 0.0
    val, unit = float(m.group(1)), (m.group(2) or "mW")
    if unit == "uW": val /= 1e3
    elif unit == "W": val *= 1e3
    return val

print("| Architecture | Area (um2) | Power (mW) | Area Eff. (TOPS/mm2) | Power Eff. (TOPS/W) |")
print("| :--- | :---: | :---: | :---: | :---: |")
for label, top in CONFIGS:
    area = parse_area(Path(f"syn_out/{top}/area.rpt"))
    pwr  = parse_power_mw(Path(f"syn_out/{top}/power_saif.rpt"))
    aeff = (TOPS * 1e6 / area) if area > 0 else 0.0
    peff = (TOPS * 1e3 / pwr)  if pwr  > 0 else 0.0
    print(f"| {label} | {area:.2f} | {pwr:.3f} | {aeff:.2f} | {peff:.2f} |")
```
