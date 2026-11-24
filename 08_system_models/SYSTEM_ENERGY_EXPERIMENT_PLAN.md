# Master Execution Plan: System Energy Breakdown & 4-Mode Attention Comparison (Section V-E)

> **Mục đích tài liệu:** Bản đặc tả thực thi khép kín (*Zero-Ambiguity Runbook*) dành cho agent/kỹ sư triển khai các thí nghiệm đo đạc, mô phỏng và xuất số liệu **System Energy Breakdown** & **4-Mode Attention Comparison** cho bài báo **IEEE TCAS-I / TVLSI / TCAD** (`NPU_LOG`).
>
> **NGUYÊN TẮC TỐI THƯỢNG (ANTI-HALLUCINATION):** Mọi con số đưa vào bảng/biểu đồ **BẮT BUỘC** phải truy xuất trực tiếp từ log RTL, báo cáo EDA, file `.lib`, hoặc script có assertion kiểm chứng sai số `0 B` / `< 2%`. **CẤM TUYỆT ĐỐI** tự suy diễn, nội suy bừa, hoặc điền hằng số không có nguồn trong `energy_params.json`.
>
> **QUY ƯỚC TẦN SỐ ĐỒNG NHẤT (LOCKED FREQUENCY):**
> - **Synthesis Sign-off Netlist (`05_system_ppa`):** Tổng hợp ở **`600 MHz`** (`1666.7 ps`, `WNS = 0.00 ps`) để giữ `+20%` timing headroom.
> - **System Evaluation & GLS Power (`06_gls`, `V-C`, `V-D`, `V-E`):** Đánh giá toàn hệ thống đồng nhất ở **`500 MHz`** (`CLK_PERIOD = 2.0 ns`, `Peak = 256 GOPS`, đồng bộ 100% với `Table IV C1–C5` và `Table V-A`). Khi chạy `report_power` trên netlist `600 MHz`, dùng constraint `500 MHz` (`05_system_ppa/eda_backup/constraints_sys_500mhz.sdc` hoặc `create_clock -period 2000 [get_ports clk]`) để khớp tuyệt đối với SAIF `500 MHz` (`0.001000 toggles/ps`).

---

## 1. Phạm vi So sánh & Các Chế độ Chuẩn (Locked Scope)

### 1.1. Bốn chế độ Attention chuẩn (loại bỏ `ONLINE_2B` khỏi headline)
| ID | Tên hiển thị trong Paper | Mô tả phần cứng (khớp `tb/tb_attn_comparison.sv`) | S/P Spill (`M=256`) | Tổng DRAM (`M=256`) |
| :--- | :--- | :--- | :---: | :---: |
| `NAIVE` | **Unfused 3-Phase** | `QK^T -> DRAM S -> 2-pass Softmax + SLQ -> DRAM P -> PV` | `16.52 MB` | `27.53 MB` (`36.1×`) |
| `ONLINE_2A` | **Unfused Online SM** | Giống `NAIVE`, nhưng Phase 2 Softmax đọc `S` **1 lượt** | `11.80 MB` | `22.81 MB` (`29.9×`) |
| `FLASH_RING` | **FlashAttn (No KV-Pin)** | Fused `D-Latch + Exp-Free SM + P-FIFO` (`m_q=8, kv_pin=0`) | **`0 B`** | `50.92 MB` (`66.8×`) |
| `FLASH_KVPIN` | **FlashAttn + KV-Pin (Ours)** | Fused + **GQA 4:1 KV-Pinning** trong `Banks 0–15` (`m_q=8, kv_pin=1`) | **`0 B`** | **`0.76 MB`** (`1.0×`) |

- **LƯU Ý CẤM:** **KHÔNG** dùng `ONLINE_2B` (`m_q = 1, 403.24 MB, 529×`) làm headline hoặc đưa vào biểu đồ chính (tránh bị reviewer bắt lỗi *strawman baseline*).

### 1.2. Bốn cấu hình So sánh Toàn Layer (Full Transformer Layer — trả lời phản biện *"KV nhỏ hơn Weights"*)
Đánh giá trên **3 Edge LLMs** khớp `Table III` (`Llama-3.2-1B`, `SmolLM2-1.7B`, `Qwen3-1.7B-Base`) tại `M = N_KV ∈ {256, 512, 1024}`:
1. **`FP16 Unfused`**: Weights FP16 (`121.6 MB/layer` trên `Llama-3.2-1B`) + `NAIVE` Attention.
2. **`NVFP4 Unfused`**: Weights 4.5-bit (`34.2 MB/layer`) + `NAIVE` Attention + PE Power từ `C2` (`04_ablation_rtl/paper_syn/PAPER_ABLATION_RESULTS.md`).
3. **`Log8-FP4 + FLASH_RING`**: Weights 4.5-bit (`34.2 MB/layer`) + `FLASH_RING` Attention + PE Power từ `C5`.
4. **`Log8-FP4 + FLASH_KVPIN (Ours)`**: Weights 4.5-bit (`34.2 MB/layer`) + `FLASH_KVPIN` + Fused SFU (`Core -> fused_fifo -> SFU -> LogQ`).

---

## 2. Ba Bẫy Sai Số Nghiêm Trọng CẤM MẮC PHẢI (Hard Guardrails)

| # | Bẫy đã phát hiện trong Repo | Nguyên nhân gốc rễ (đã kiểm chứng trên đĩa) | Quy tắc xử lý BẮT BUỘC |
| :-: | :--- | :--- | :--- |
| **1** | **GLS Power bị sai tần số `5×` (`100 MHz` thay vì `500 MHz`)** | `06_gls/tb_gemm_os_gls.sv:60` đặt `always #5 clk = ~clk;` (`10 ns = 100 MHz`), làm DC báo `PWR-12` và ép `snax_llm_top_clean_power_gls_test1.rpt` xuống `16.545 mW`. Đồng thời `TEST=1` quá ngắn (`738 cycles`), không kích hoạt `u_softmax_requantizer` và `u_sfu`. | **CẤM** dùng số `16.545 mW` từ `test1.rpt`. Phải sửa clock về **`500 MHz`** (`always #1 clk = ~clk;` với `` `timescale 1ns/1ps ``, `CLK_PERIOD = 2.0 ns`), dump VCD/SAIF chỉ trên **cửa sổ active compute**, dùng constraint `500 MHz` khi `report_power`, và kiểm tra log DC **không còn `PWR-12` sai lệch tần số**. |
| **2** | **SRAM `u_tcdm` bị tính điện ảo `78.16 mW` (tăng khống `10×–30×`)** | `05_system_ppa/eda_backup/sram_lib/srambank_256x4x72_6t122.lib:259-322` ghi `clk` `internal_power` với `when : "!write"` mà **thiếu điều kiện `banksel`**. Do đó DC tính trọn năng lượng đọc (`4.291 pJ/cycle`) cho **cả 32 bank ngay cả khi idle** (`32 × 4.291 pJ × 0.6 GHz ≈ 82 mW`). | **CẤM** lấy `u_tcdm` power từ `snax_llm_top_clean_power_default.rpt` (`78.16 mW`) hay `test1.rpt` (`13.46 mW`). Phải tách riêng **`P_logic = P_top - P_tcdm`** từ DC, còn **`E_sram`** tính theo công thức vật lý `N_rd * e_rd + N_wr * e_wr + P_leak * T` từ đúng thông số `.lib`! |
| **3** | **File `sim/attn_comparison_results.csv` bị dừng dở ở `M=256`** | `sim/attn_comparison_results.csv` dừng tại dòng 38 (`FLASH_RING, 256, 256, 32`), còn thiếu `ONLINE_2A/NAIVE` ở `BW=32` và thiếu **toàn bộ 4 mode** ở `BW=16` và `BW=8 B/cyc`. | Phải chạy hoàn tất các điểm còn thiếu trên Verilator RTL trước khi vẽ Bandwidth Sensitivity và tính Energy cho `BW = 16, 8 B/cyc`. **Không được tự ước lượng cycle khi chưa chạy RTL.** |

---

## 3. Lộ Trình Thực Thi Chi Tiết (4 Workstreams Tuần Tự)

### Workstream 1 — Hoàn tất RTL Attention Sweep (`M = 256`, `BW = 32, 16, 8 B/cyc`)
- **Công cụ & file nguồn:** `sim/run_attn_comparison.sh`, `sim/sweep_attn_comparison.sh`, `tb/tb_attn_comparison.sv`.
- **Các điểm BẮT BUỘC chạy bổ sung (lưu tiếp vào `sim/attn_comparison_results.csv`):**
  1. `M=256, N_KV=256, BW=32`: chạy nốt `ONLINE_2A`, `NAIVE`.
  2. `M=256, N_KV=256, BW=16` (`8 GB/s @ 500 MHz`): chạy đủ 4 mode `FLASH_KVPIN`, `FLASH_RING`, `ONLINE_2A`, `NAIVE`.
  3. `M=256, N_KV=256, BW=8` (`4 GB/s @ 500 MHz`, `tb_attn_comparison.sv:229` đã hỗ trợ sẵn `DRAM_BW=8`): chạy đủ 4 mode `FLASH_KVPIN`, `FLASH_RING`, `ONLINE_2A`, `NAIVE`.
  4. *(Tùy chọn nếu đủ thời gian)*: chạy `M=512, N_KV=512` tại `BW=64` và `BW=16` cho 4 mode (Stage C của `sweep_attn_comparison.sh`).
- **Tiêu chí nghiệm thu (PASS Gate):**
  - Mọi dòng CSV mới bắt buộc có: `checks_fail == 0`, `cpu_mem_cyc == 0`, `max_inflight <= 4`.
  - Xác nhận tính nhất quán vật lý: ở cùng `(mode, M, N_KV)`, `rd_bytes`, `wr_bytes`, `sp_rd_bytes`, `sp_wr_bytes` phải **bất biến** khi đổi `DRAM_BW` (`64 / 32 / 16 / 8`).

---

### Workstream 2 — Đo Chính Xác On-Chip Logic Power (`500 MHz` SAIF trên Netlist `600 MHz`) & Trích Thông Số SRAM `.lib`

#### Bước 2A: Sửa GLS Clock về `500 MHz` & Dump SAIF đúng cửa sổ Active Compute (`06_gls/`)
- **File cần chỉnh (tạo bản copy hoặc sửa trực tiếp có kiểm soát):** `06_gls/tb_gemm_os_gls.sv` và `06_gls/run_gls_questasim.ps1`.
- **Sửa bắt buộc:**
  1. Giữ `` `timescale 1ns/1ps `` và sửa dòng 60 thành `always #1 clk = ~clk;` để chu kỳ clock chính xác là **`2.000 ns` (`500.0 MHz`)** *(số nguyên tuyệt đối, không sai số làm tròn)*.
  2. Chỉ bật `$dumpon` khi `busy_o == 1'b1` (trong lõi compute sustained) và `$dumpoff` ngay khi `done_irq` lên mức 1 — **loại bỏ hoàn toàn chu kỳ reset và nạp khởi tạo**.
  3. Chạy tối thiểu **3 workload đại diện** trên netlist 7nm (`snax_llm_top_clean`, tổng hợp ở 600 MHz):
     - **Workload A (GEMM Sustained):** Prefill GEMM tile lớn (ví dụ Test GEMM multi-tile trong `tb_gemm_os_gls.sv`) $\rightarrow$ thu `P_logic_gemm` (mW).
     - **Workload B (Fused FlashAttention):** Fused Attention test (kích hoạt `u_mac_array`, `u_acc_buffer`, `u_softmax_requantizer`, `u_p_fifo`, `u_attn_rescaler`, `u_l_accumulator`, `u_final_normalizer`) $\rightarrow$ thu `P_logic_attn` (mW).
     - **Workload C (SFU + LogQ Epilogue):** Test kích hoạt `u_sfu` + `u_log_quantizer` $\rightarrow$ thu `P_logic_sfu` (mW).
- **Trích Power từ Synopsys Design Compiler / PrimeTime PX (`500 MHz`):**
  - Chuyển VCD $\rightarrow$ SAIF (`vcd2saif`), đọc vào DC với constraint **`500 MHz`** (`05_system_ppa/eda_backup/constraints_sys_500mhz.sdc` hoặc `create_clock -period 2000 [get_ports clk]`), chạy `report_power -hier -analysis_effort medium`.
  - **Bắt buộc kiểm tra log DC:**
    - `clk` toggle rate annotated phải là **`0.001000 toggles/ps`** (`500 MHz`), KHÔNG có cảnh báo `PWR-12` lệch tần số.
    - Trích **`P_logic = P_total(snax_llm_top_clean) - P_total(u_tcdm)`** (tách riêng `Switch`, `Internal`, `Leakage` của từng khối con cho `Table VI` và cho `energy_params.json`).

#### Bước 2B: Trích Năng lượng SRAM chính xác từ `srambank_256x4x72_6t122.lib`
- **File nguồn:** `05_system_ppa/eda_backup/sram_lib/srambank_256x4x72_6t122.lib` (`time_unit=1ns`, `current_unit=1mA`, `voltage_unit=1V` $\Rightarrow$ đơn vị `internal_power` là **`pJ`**).
- **Cách tính chính xác (viết script parser Python đọc trực tiếp `.lib`, không nhập tay):**
  1. **`e_sram_rd_pJ` (1 lần đọc 72-bit word từ 1 bank 9 KB):**
     - Từ `pin(clk)` với `when : "!write"` (dòng 291–322): `VDD (1.44236 + 0.373495) + VSS (2.11495 + 0.360267) = 4.291 pJ` *(cộng thêm công suất chuyển mạch chân `rd[71:0]` tại slew/load danh định ở dòng 1884–1944: `~72 × 0.5 × 0.0022 ≈ 0.08 pJ` $\rightarrow$ tổng `~4.37 pJ/read`, tương đương `0.061 pJ/bit`)*.
  2. **`e_sram_wr_pJ` (1 lần ghi 72-bit word vào 1 bank 9 KB):**
     - Parse `pin(clk)` với `when : "write"` (dòng 259–290: `0.186927 + 0.132678 = 0.320 pJ`) cộng tổng `internal_power` trên bus `wd[71:0]` và `addr`/`write` khi ghi. *(Lưu ý: nếu `mk_SRAM_lib.py` mô hình hóa năng lượng ghi thấp bất thường so với đọc, **BẮT BUỘC** ghi chú minh bạch và chạy kiểm chứng chéo với **CACTI 7.0** tại 7nm cho 1 bank `9 KB, 72-bit` để đảm bảo `e_sram_wr` nằm trong dải vật lý hợp lý `~4.0–6.0 pJ/access`, tránh báo cáo năng lượng ghi quá thấp gây mất tin cậy!)*
  3. **`P_sram_leak_mW`:** Trong `.lib` hiện `cell_leakage_power : 0` (dòng 115). Nếu dùng leakage của 32 bank 7nm, trích giá trị bảo thủ từ CACTI 7.0 / ASAP7 FakeRAM2.0 và ghi rõ nguồn trong `energy_params.json`.

---

### Workstream 3 — Hoàn thiện Bộ Tiling Models 4 Chế Độ (`08_system_models/`) & Kiểm Định 0-Byte Error

- **Nền tảng đã hoàn thành:** `08_system_models/naive_attn_tiling_model.py` đã đạt **sai số `0 B`** trên DRAM và **sai số `0.00 dB`** trên SQNR so với RTL `NAIVE` (`M = N_KV = 256`).
- **Nhiệm vụ cần triển khai:**
  1. Tạo `08_system_models/attn_4modes_tiling_model.py` (tái sử dụng `load_natural_dram_tensors`, `DLatchMacCore`, `MemoryProfiler` từ `naive_attn_tiling_model.py`) để mô phỏng tường minh cả **4 chế độ**: `NAIVE`, `ONLINE_2A`, `FLASH_RING`, `FLASH_KVPIN`.
  2. Giữ nguyên quy ước **ma trận gốc 3D trong DRAM** (`DRAM_Q: (32, M, 64)`, `DRAM_K: (8, N_KV, 64)`, `DRAM_V: (8, N_KV, 64)`) và mô tả tường minh từng lệnh cắt tile `DRAM <-> TCDM <-> Core`.
  3. **Cổng Kiểm Định Tự Động (BẮT BUỘC PASS 100% trước khi chuyển sang Workstream 4):**
     Chạy trên `sim/sim_attn_cmp_m256_kv256` (và kiểm tra chéo thêm `m64_kv64`, `m128_kv128`) với `assert` đối chiếu trực tiếp `sim/attn_comparison_results.csv`:

     | Mode (`M=256, N_KV=256`) | `rd_bytes` (Exact) | `wr_bytes` (Exact) | `sp_rd_bytes` (Exact) | `sp_wr_bytes` (Exact) | `SQNR_fq` Target |
     | :--- | :---: | :---: | :---: | :---: | :---: |
     | `NAIVE` | `20,840,448` | `6,684,672` | `10,616,832` | `5,898,240` | `18.04 dB` (`±0.05`) |
     | `ONLINE_2A` | `16,121,856` | `6,684,672` | `5,898,240` | `5,898,240` | `18.04 dB` (`±0.05`) |
     | `FLASH_RING` | `50,626,560` | `294,912` | `0` | `0` | `16.91 dB` (`±0.20`) |
     | `FLASH_KVPIN` | `466,944` | `294,912` | `0` | `0` | `16.91 dB` (`±0.20`) |

  4. Xuất bảng `08_system_models/results/tcdm_dram_4modes.csv` chứa đủ `dram_rd_bytes`, `dram_wr_bytes`, `sp_rd_bytes`, `sp_wr_bytes`, `tcdm_rd_words`, `tcdm_wr_words`, `mac_ops` cho `M = N_KV ∈ {64, 128, 256, 512, 1024}`.

---

### Workstream 4 — Tính `System Energy Breakdown` & Xuất Bảng/Biểu Đồ Paper

#### Bước 4A: Khóa toàn bộ tham số vào `08_system_models/energy_params.json`
Mọi script tính năng lượng **CHỈ ĐƯỢC ĐỌC** từ file JSON duy nhất này:
```json
{
  "syn_freq_hz": 600000000,
  "eval_freq_hz": 500000000,
  "logic_power_mw": {
    "gemm_active": "<FROM_WS2_GLS_500MHZ>",
    "attn_fused_active": "<FROM_WS2_GLS_500MHZ>",
    "sfu_active": "<FROM_WS2_GLS_500MHZ>",
    "pe_c2_nvfp4_ratio": 1.3478,
    "pe_c1_mxfp4_ratio": 1.0130,
    "source": "06_gls 500MHz SAIF (on 600MHz netlist) + 04_ablation_rtl/paper_syn/PAPER_ABLATION_RESULTS.md"
  },
  "sram_72b": {
    "e_rd_pJ_per_word": "<FROM_WS2B_LIB_PARSER>",
    "e_wr_pJ_per_word": "<FROM_WS2B_LIB_OR_CACTI7>",
    "p_leak_mw_32banks": "<FROM_WS2B_CACTI7>",
    "source": "05_system_ppa/eda_backup/sram_lib/srambank_256x4x72_6t122.lib + CACTI 7.0 7nm"
  },
  "dram_offchip": {
    "primary_standard": "LPDDR4X-3200",
    "e_dram_pJ_per_bit": 16.0,
    "sensitivity_lpddr5_pJ_per_bit": 9.0,
    "source": "Micron LPDDR4X datasheet / DRAMSim3 standard edge model"
  }
}
```
*(Ghi chú: `pe_c2_nvfp4_ratio = 7.893 / 5.856 = 1.3478` lấy chính xác từ `Table IV` Real SAIF của `C2` vs `C5`.)*

#### Bước 4B: Công thức tính Năng lượng (`08_system_models/compute_system_energy.py`)
Với mỗi điểm thí nghiệm `(mode, M, N_KV, BW)`:
1. **Thời gian thực thi:** $T_{\text{exec}} = \frac{\text{cyc\_total\_w\_sfu}}{f_{\text{clk}}}$ *(dùng trực tiếp cột `cyc_total_w_sfu` từ `sim/attn_comparison_results.csv` cho `M ≤ 256`, và mô hình cycle hiệu chỉnh sai số `< 2%` cho `M = 512, 1024`)*.
2. **Năng lượng Logic ($E_{\text{logic}}$):**
   - Với `FLASH_RING`, `FLASH_KVPIN`: $E_{\text{logic}} = P_{\text{logic, attn}} \times T_{\text{exec}}$ *(chú ý: vì `FLASH_*` chạy $2.5\times$ MAC cycles so với `NAIVE`/`ONLINE_2A`, nên lấy `P_logic` tách theo tỷ lệ `cyc_mac / cyc_total` hoặc đo trực tiếp SAIF của từng mode để phản ánh trung thực chi phí recomputation!)*.
   - Với `NAIVE`, `ONLINE_2A`: $E_{\text{logic}} = P_{\text{logic, gemm}} \times \frac{\text{cyc\_hw}}{f_{\text{clk}}} + P_{\text{logic, sfu}} \times \frac{\text{p2\_sfu\_est}}{f_{\text{clk}}}$.
3. **Năng lượng On-Chip SRAM ($E_{\text{sram}}$):**
   $$E_{\text{sram}} = N_{\text{tcdm\_rd}} \cdot e_{\text{sram\_rd}} + N_{\text{tcdm\_wr}} \cdot e_{\text{sram\_wr}} + P_{\text{sram\_leak}} \cdot T_{\text{exec}}$$
4. **Năng lượng Off-Chip DRAM ($E_{\text{dram}}$):**
   $$E_{\text{dram}} = (\text{rd\_bytes} + \text{wr\_bytes}) \times 8 \times e_{\text{DRAM}}$$
5. **Tổng & Phân rã (%):** $E_{\text{total}} = E_{\text{logic}} + E_{\text{sram}} + E_{\text{dram}}$, xuất cả giá trị tuyệt đối (`mJ` hoặc `µJ`) lẫn chuẩn hóa (`Normalized Energy` so với baseline `= 1.00`), cho cả `LPDDR4X (16 pJ/bit)` và sensitivity check `LPDDR5 (9 pJ/bit)`.

#### Bước 4C: Mở rộng sang Full Transformer Layer (`08_system_models/full_layer_energy_model.py`)
- **Kiểm định bắt buộc với RTL thật (`Llama-3.2-1B`, `LLAMA3_1B_PREFILL_REPORT.md`):**
  - Tổng trọng số 1 layer (4.5-bit): `34,209,792 B` (`34.21 MB`) — model phải khớp **0 B**.
  - Tổng chu kỳ Prefill 1 layer: `4,479,514` cycles — model phải khớp sai số **`< 2.0%`**.
- Sau khi PASS kiểm định, chạy cho **3 model** (`Llama-3.2-1B`, `SmolLM2-1.7B`, `Qwen3-1.7B-Base`) tại `M = 256, 512, 1024` trên 4 cấu hình hệ thống (`FP16 Unfused`, `NVFP4 Unfused`, `Log8-FP4 + FLASH_RING`, `Log8-FP4 + FLASH_KVPIN`) để xuất:
  - **Full-Layer DRAM Traffic Breakdown** (`Weights GEMM` vs. `Attention`)
  - **Full-Layer System Energy Breakdown** (`DRAM` + `SRAM` + `Logic`, đơn vị `mJ/layer` và `µJ/token`).

---

## 4. Danh Sách Deliverables & Tiêu Chuẩn Hình Vẽ Paper (IEEE 2-Column, 300 DPI)

Tất cả output lưu trong `08_system_models/results/`:
1. `attn_4modes_energy_summary.csv` & `full_layer_energy_summary.csv`: Bảng số liệu gốc đã qua assertion.
2. **`fig_attn_bw_sensitivity.pdf/.png`**: Độ trễ chuẩn hóa (`Normalized Cycles`) theo `DRAM_BW ∈ {64, 32, 16, 8} B/cyc` tại `M = 256` cho 4 mode (`NAIVE`, `ONLINE_2A`, `FLASH_RING`, `FLASH_KVPIN`) — chứng minh `FLASH_KVPIN` đi ngang phẳng lì.
3. **`fig_attn_energy_breakdown.pdf/.png`**: Cột chồng **Attention System Energy Breakdown** (`Off-Chip DRAM` / `288-KB SRAM` / `Compute Logic`) cho 4 mode tại `M = 64, 128, 256, 512, 1024` (kèm chú thích tỷ lệ giảm `×`).
4. **`fig_full_layer_energy_breakdown.pdf/.png`**: Cột chồng **Full-Layer System Energy Breakdown** trên 3 Edge LLMs (`Llama-3.2-1B`, `SmolLM2-1.7B`, `Qwen3-1.7B` + `Geo. Mean`, đúng chuẩn `HiKV Fig. 19` trên IEEE TCAS-I) so sánh `FP16 Unfused` vs. `NVFP4 Unfused` vs. `Log8-FP4 + FLASH_RING` vs. `Log8-FP4 + FLASH_KVPIN (Ours)`.

---

## 5. Checklist Nghiệm Thu Cuối (Sign-Off trước khi đưa vào Paper)

- [ ] `sim/attn_comparison_results.csv` có đủ 4 mode (`NAIVE`, `ONLINE_2A`, `FLASH_RING`, `FLASH_KVPIN`) ở `M=256, BW ∈ {64, 32, 16, 8}`, toàn bộ `checks_fail == 0`, `cpu_mem_cyc == 0`.
- [ ] Log DC của `06_gls` xác nhận tần số evaluate **`500 MHz`** (`0.001000 toggles/ps`, dùng `constraints_sys_500mhz.sdc`, không có `PWR-12`), đã loại `u_tcdm` power ảo ra khỏi `P_logic`.
- [ ] `attn_4modes_tiling_model.py` PASS assertion **0 B error** trên toàn bộ 4 mode so với `sim/attn_comparison_results.csv`.
- [ ] `full_layer_energy_model.py` PASS assertion **`< 2%` cycle error** và **0 B weight error** so với `LLAMA3_1B_PREFILL_REPORT.md`.
- [ ] `energy_params.json` khóa cứng mọi hằng số kèm nguồn trích dẫn; đã chạy sensitivity check ở cả `LPDDR4X (16 pJ/bit)` và `LPDDR5 (9 pJ/bit)` và xác nhận xu hướng kết luận không thay đổi.
- [ ] Cập nhật số `Power (mW)` thật (`500 MHz` SAIF trên netlist `600 MHz`) và `TOPS/W` (`256 GOPS @ 500 MHz`) vào `Table VI` & `Table VII` trong `PAPER_BLUEPRINT_TCAD_TVLSI.md`.
