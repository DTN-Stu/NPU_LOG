# System-Level PPA & Hybrid Bottom-Up Synthesis (`snax_llm_top_clean`)

## 1. Môi trường & Cấu hình Tổng hợp
- **Top Module:** `snax_llm_top_clean` (from `snax_llm_top_clean_bu.sv` / `snax_llm_top_clean.sv`)
- **Công nghệ:** ASAP7 7nm Multi-Vt (`RVT + LVT + SLVT`, `PVT_0P7V_25C`, 15 `.db` libraries) + 32× SRAM macro `srambank_256x4x72_6t122` (`256×4×72b` = `72 Kb` / bank, tổng `288 KB` TCDM)
- **Công cụ:** Synopsys Design Compiler `O-2018.06-SP1`
- **Tần số mục tiêu:** **`600 MHz`** (`CLK_PERIOD = 1666.70 ps`)
- **Constraints (`scripts/constraints_sys_600mhz.sdc`):**
  - `set_clock_uncertainty -setup 83.3`, `-hold 20.0`, `set_clock_transition 25.0`
  - `set_input_delay -max 333.34` (20% period), `set_output_delay -max 333.34` (20% period)
  - `set_max_transition 100.0`, `set_max_fanout 48`, `set_max_time_borrow 416.68` (25% period)
  - `set_multicycle_path -setup 2 -hold 1` từ thanh ghi cấu hình CSR (`*sb_cfg_q*reg*`, `*act_cfg_q*reg*`)

---

## 2. Quy trình Hybrid Bottom-Up (2 Stage)

1. **Stage 1 — Sub-Block Synthesis (`scripts/syn_subblock.tcl`, chạy song song):**
   Tổng hợp độc lập 6 khối nặng nhất (`u_core`, `u_sfu`, `u_seq`, `u_write_out`, `u_wt_streamer`, `u_csr`) tại `600 MHz`, xuất `.ddc` (`ddc_bu/`) và Verilog gate-level netlist (`netlist_bu/*.v` với `change_names -rules verilog -hierarchy`).
2. **Stage 2 — Top-Level Integration (`scripts/syn_sys_netlist_bu.tcl` & `scripts/syn_sys_bottomup.tcl`):**
   - **Flow Netlist `.v` (Khuyến nghị — `syn_sys_netlist_bu.tcl`):**
     1. Đọc 6 gate-level netlists (`read_verilog netlist_bu/*.v`).
     2. `analyze` + `elaborate` phần RTL còn lại (Top, `interleaved_interconnect_72b`, `multi_bank_tcdm_72b`, `snax_streamer_act_bcast`, `snax_stream_read_fp16`, `slq_encoder`, `snax_transpose_buffer`, `snax_transpose_ctrl`, `snax_sfu_sequencer`) qua `scripts/snax_llm_top_clean_bu.sv`.
     3. **Stage A:** `set_dont_touch` 6 sub-block, chạy `compile_ultra -no_autoungroup` để map phần Top/Interconnect/TCDM/glue.
     4. **Stage B:** Mở `set_dont_touch`, chạy `compile_ultra -incremental -no_autoungroup` để tối ưu xuyên biên và thu hồi leakage (HSVT pass).
   - **Flow `.ddc` (`syn_sys_bottomup.tcl`):** Dùng `work_empty` + `remove_design` + `read_ddc` + `change_link` + `link`, cho kết quả tương đương.

### Hướng dẫn chạy trên `eda` (`~/sys_syn/`)
```bash
# Stage 1: Tổng hợp song song 6 sub-block -> ddc_bu/
bash run_bottomup_eda.sh

# (Tùy chọn cho Netlist BU) Xuất 6 .ddc sang netlist_bu/*.v
dc_shell -f export_netlists.tcl > logs/export_netlists.log 2>&1

# Stage 2 (Netlist Bottom-Up @ 600 MHz)
dc_shell -f syn_sys_netlist_bu.tcl > logs/syn_sys_netlist_bu.log 2>&1
```

---

## 3. Kết quả Stage 1 (6 Sub-Blocks @ 600 MHz)

| Sub-Block | Top Module | Leaf Cells | Levels | Crit Path (ps) | WNS (ps) | TNS (ps) | Cell Area ($\mu\text{m}^2$) |
|---|---|---:|---:|---:|---:|---:|---:|
| `u_core` | `snax_fused_llm_core` | `180,435` | `1.00` | `96.26` | **`0.00`** | `0.00` | `17,339.86` |
| `u_seq` (`2c80ee9` 3-FSM) | `snax_llm_sequencer` | `15,528` | `56.00` | `1234.50` | **`+0.11`** | `0.00` | **`1,645.18`** |
| `u_sfu` | `snax_llm_sfu_opt_pipe` | `21,318` | `52.00` | `1233.99` | **`+0.07`** | `0.00` | `2,125.22` |
| `u_write_out` | `snax_streamer_write_out` | `18,605` | `52.00` | `1231.42` | **`+3.37`** | `0.00` | `2,517.22` |
| `u_wt_streamer` | `snax_streamer_wt_16ch` | `18,155` | `48.00` | `1249.96` | **`+0.10`** | `0.00` | `2,620.77` |
| `u_csr` | `snax_flash_csr_snax` | `3,460` | `10.00` | `539.93` | **`+1039.55`** | `0.00` | `512.36` |

---

## 4. Kết quả Stage 2 Toàn Hệ thống (`snax_llm_top_clean` @ 600 MHz)

### 4.1. So sánh A/B: Netlist `.v` Bottom-Up (Latest `2c80ee9` `u_seq`) vs bản cũ (`5ea3774`)
| Metric | Netlist `.v` BU (`2c80ee9` 3-FSM `u_seq`) | Netlist `.v` BU (cũ `5ea3774`) | `.ddc` BU (cũ `5ea3774`) |
|---|---:|---:|---:|
| **Setup WNS / TNS** | **`0.00 ps` / `0.00 ps`** (`0` violating) | `0.00 ps` / `0.00 ps` (`0` violating) | `0.00 ps` / `0.00 ps` (`0` violating) |
| **Critical Path** | `96.26 ps` (`Levels = 1.00`, `slack = 0.00`) | `96.26 ps` (`Levels = 1.00`, `slack = 0.00`) | `96.26 ps` (`Levels = 1.00`, `slack = 0.00`) |
| **Hold (pre-CTS, ideal clk)** | `-9.99 ps` / `-177.99 ps` (`30` paths) | `-9.99 ps` / `-175.64 ps` (`29` paths) | `-9.99 ps` / `-176.44 ps` (`29` paths) |
| **Leaf Cell Count** | **`366,249`** (`332,881` comb + `33,336` seq + `32` SRAM) | `394,049` | `393,806` |
| **Combinational Area** | **`26,553.87 um²`** | `29,663.75 um²` | `29,657.75 um²` |
| **Noncombinational Area** | `10,304.88 um²` | `10,022.23 um²` | `10,022.23 um²` |
| **SRAM Macro Area (32×)** | `83,906.15 um²` | `83,906.15 um²` | `83,906.15 um²` |
| **Total Cell Area** | **`120,764.90 um²`** (`~0.121 mm²`) | `123,592.14 um²` | `123,586.13 um²` |
| **DRC Violations** | **`143` nets** (`139` trans, `3` cap, `16` fanout) | `183` nets (`179` trans, `3` cap, `16` fanout) | `187` nets |
| **Compile Wall Time / Peak RAM** | `1005.37s` | `979.52s` / `2,440 MB` | `958.38s` / `2,498 MB` |

### 4.2. Phân rã Diện tích Theo Khối (Netlist `.v` Bottom-Up — Latest `2c80ee9`)
| Instance | Module | Area ($\mu\text{m}^2$) | Tỉ trọng (%) |
|---|---|---:|---:|
| `u_tcdm` | `multi_bank_tcdm_72b` (32× SRAM macros) | `83,927.19` | `69.5%` |
| `u_core` | `snax_fused_llm_core` | `17,280.17` | `14.3%` |
| `u_interconnect` | `interleaved_interconnect_72b` (29×32) | `7,146.50` | `5.9%` |
| `u_wt_streamer` | `snax_streamer_wt_16ch` | `2,506.74` | `2.1%` |
| `u_write_out` | `snax_streamer_write_out` | `2,426.17` | `2.0%` |
| `u_sfu` | `snax_llm_sfu_opt_pipe` | `2,113.63` | `1.8%` |
| `u_seq` (`2c80ee9` 3-FSM) | `snax_llm_sequencer` | **`1,560.85`** | **`1.3%`** |
| `u_transpose_buffer` | `snax_transpose_buffer` | `1,387.56` | `1.1%` |
| `u_csr` | `snax_flash_csr_snax` | `511.90` | `0.4%` |
| `u_read_fp16` + `u_read_fp16_b` | `snax_stream_read_fp16` (2×) | `564.90` | `0.5%` |
| `u_act_streamer` | `snax_streamer_act_bcast` | `236.65` | `0.2%` |
| `u_log_quantizer` | `log_quantizer` (`slq_encoder`) | `224.36` | `0.2%` |
| `u_sfu_seq` | `snax_sfu_sequencer` | `210.56` | `0.2%` |
| `u_transpose_ctrl` | `snax_transpose_ctrl` | `41.48` | `0.0%` |
| Top glue (`snax_llm_top_clean`) | Top-level logic & regs | `626.23` | `0.5%` |
| **Total** | **`snax_llm_top_clean`** | **`120,764.90`** | **`100.0%`** |

### 4.3. Ghi chú Critical Path, Power & DRC
- **Critical Path (`snax_llm_top_clean_timing.rpt`):** `u_core/u_acc_buffer/u_latch_array/gen_latch_storage.wdata_q_reg[31]` $\rightarrow$ `u_dlatch/u_dhl` (`data arrival = 96.26 ps`, `time borrowed = 179.59 ps`, `slack (MET) = 0.00 ps`).
- **Default Switching Power (`snax_llm_top_clean_power_default.rpt`, vectorless `analysis_effort low`):** `99.280 mW` total (`Switch = 1.223 mW`, `Internal = 97.979 mW`, `Leakage = 77.4 uW`), trong đó `u_core` chiếm `7.411 mW` (`7.5%`) — phần lớn con số vectorless này đến từ toggle mặc định của 32 SRAM macro chưa gắn SAIF thực tế.
- **DRC (`183 / 407,028 nets ≈ 0.045%`) & Hold (`-9.99 ps`, `29` paths):** Pre-CTS trên ideal clock, xử lý chuẩn trong giai đoạn P&R / CTS.

---

## 5. Danh sách File trong `05_system_ppa/`
- `SYSTEM_PPA_RESULTS.md`: Tài liệu tổng hợp kết quả & hướng dẫn chạy này.
- `scripts/syn_subblock.tcl`, `scripts/syn_sys_netlist_bu.tcl`, `scripts/syn_sys_bottomup.tcl`, `scripts/constraints_sys_600mhz.sdc`, `scripts/run_bottomup_eda.sh`, `scripts/snax_llm_top_clean_bu.sv`
- `reports/reports_netlist_bu/*`, `reports/reports/*`, `reports/reports_bu/*`
