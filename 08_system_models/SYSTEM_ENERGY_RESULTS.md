# System Energy & Bandwidth Sensitivity Results (`500 MHz`, ASAP7 7nm)

Báo cáo tổng hợp toàn bộ kết quả thực nghiệm đo đạc trực tiếp theo [SYSTEM_ENERGY_EXPERIMENT_PLAN.md](SYSTEM_ENERGY_EXPERIMENT_PLAN.md). Mọi con số đều có nguồn truy vết trực tiếp từ RTL log, Gate-Level SAIF (`100.00%` annotated tại `500 MHz`, `0` cảnh báo `PWR-12`), file `.lib`, và script Python đã qua assertion tự động.

---

## 1. Workstream 1 — RTL Attention Bandwidth Sweep (`M = N_KV = 256`, `Llama-3.2-1B` Layer 1)

Nguồn: [sim/attn_comparison_results.csv](../sim/attn_comparison_results.csv) (`checks_fail == 0`, `cpu_mem_cyc == 0`, `max_inflight == 3`, bất biến byte tuyệt đối qua cả 4 băng thông):

| Mode | `DRAM_BW = 64` (32 GB/s) | `DRAM_BW = 32` (16 GB/s) | `DRAM_BW = 16` (8 GB/s) | `DRAM_BW = 8` (4 GB/s) | Slowdown (`8` vs `64`) | Total DRAM Bytes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`NAIVE`** | `4,495,372` | `5,227,538` | `5,818,386` | `7,139,350` | `1.59×` | `27,525,120 B` |
| **`ONLINE_2A`** | `3,812,876` | `4,528,146` | `4,972,562` | `6,000,662` | `1.57×` | `22,806,528 B` |
| **`FLASH_RING`** | `3,853,304` | `3,854,072` | `4,266,132` | `7,417,000` | `1.92×` | `50,921,472 B` |
| **`FLASH_KVPIN` (Ours)** | **`3,810,837`** | **`3,810,894`** | **`3,814,292`** | **`3,819,448`** | **`1.002× (+0.23%)`** | **`761,856 B`** |

> **Phát hiện Bug RTL minh bạch (`commit 2c80ee9`):** Trong [snax_seq_dma_fsm.sv](../01_snax_llm_fp4_subsystem/snax_seq_dma_fsm.sv#L736-L744) sau đợt refactor 3-FSM, hai con trỏ `dma_wk_dram_ptr_q` và `dma_wg_v_dram_ptr_q` chỉ tăng tiến mà không reset về đầu KV-head khi `dma_kv_q` wrap cho `(qi, dvi)` kế tiếp ở chế độ `FLASH_RING` (`kv_pin = 0`), làm sai dữ liệu `FLASH_RING` (`SQNR = -70.63 dB`), trong khi `FLASH_KVPIN` (`kv_pin = 1`) không bị ảnh hưởng. Hai điểm `FLASH_RING` (`BW=16, 8`) đã được chạy trên bản tiền refactor `2c80ee9~1` (trùng khớp phiên bản sinh các dòng `2–38` trước đó của CSV) và PASS `0` lỗi (`SQNR = 15.92 dB`, `CosineSim = 0.988881`, `rd = 50,626,560 B`, `wr = 294,912 B`). Cần bổ sung bản vá reset con trỏ về `kv_head_base` khi wrap `dma_kv_q` trong `snax_seq_dma_fsm.sv`.

---

## 2. Workstream 2A & 2B — Real `500 MHz` Gate-Level SAIF Power & SRAM `.lib` Energy

### 2.1. Công suất Logic Thực (`snax_llm_top_clean`, ASAP7 7nm TT `0.7V 25C`, `500 MHz` trên netlist `600 MHz`)
Nguồn: [06_gls/reports_500mhz/](../06_gls/reports_500mhz/) (`100.00%` SAIF annotated trên toàn bộ `8,421` nets / `1,887` ports / `20,010` pins, **`0` cảnh báo `PWR-12`**, đã loại trừ công suất ảo của `u_tcdm` khỏi `P_logic = P_top - P_u_tcdm`):

| Workload (`500 MHz` Active Window) | Active Cycles | `P_top` (mW) | `P_u_tcdm` ảo (mW) | **`P_logic` Thật (mW)** | `u_core` (mW) | `u_sfu` (mW) | Logic Eff. (`256 GOPS`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: GEMM Sustained (`+TEST=2A`)** | `1,131` | `84.565` | `67.395` | **`17.170`** | `5.280` | `1.024` | **`14.91 TOPS/W`** |
| **B: Fused FlashAttention (`+TEST=6A-1`)** | `4,981` | `85.884` | `67.954` | **`17.930`** | `6.249` | `1.024` | **`14.28 TOPS/W`** |
| **C: SFU + Core→SFU (`tb_sfu_gls_500mhz`)** | `336` | `84.014` | `68.233` | **`15.781`** | `3.897` | `1.126` | — |

- Chi tiết `u_core` tăng `+0.969 mW` từ GEMM (`5.280 mW`) sang Fused FlashAttention (`6.249 mW`) phản ánh trung thực hoạt động của `u_softmax_requantizer` (`0.615 -> 0.674 mW`), `u_attn_rescaler` (`0.034 -> 0.175 mW`), `u_p_fifo` (`0.649 -> 0.685 mW`), `u_l_accumulator` (`0.182 -> 0.197 mW`) và `u_mac_array` (`1.589 -> 1.806 mW`).

### 2.2. Năng lượng 288-KB 72-bit SRAM (`32 × 9 KB`)
Nguồn: [parse_sram_lib.py](parse_sram_lib.py) đọc [srambank_256x4x72_6t122.lib](../05_system_ppa/eda_backup/sram_lib/srambank_256x4x72_6t122.lib) + CACTI 7.0 7nm:
- **`e_sram_rd = 4.3311 pJ/word`** (`0.0602 pJ/bit`) — trích trực tiếp từ `.lib` (`clk !write = 4.2911 pJ` + `bus(dataout) 72b @ 50% = 0.0400 pJ`).
- **`e_sram_wr = 4.8000 pJ/word`** — dùng giá trị vật lý bảo thủ CACTI 7.0 7nm (`4.0–6.0 pJ`) do `.lib` sinh bởi `mk_SRAM_lib.py` có artifact ghi thấp bất thường (`0.3252 pJ`).
- **`P_sram_leak = 1.6000 mW`** (`50 µW/bank × 32 banks`, do `.lib` để `cell_leakage_power : 0`).

---

## 3. Workstream 3 & 4B — Attention 4-Mode Energy Breakdown (`M = 64..1024`, `BW = 64`)

- Kiểm định [attn_4modes_tiling_model.py](attn_4modes_tiling_model.py): **PASS 12/12 assertion sai số `0 B`** so với RTL (`M = 64, 128, 256`).
- Kiểm định nội suy chu kỳ [compute_system_energy.py](compute_system_energy.py) (fit `M=64, 128` $\rightarrow$ dự đoán `M=256`): sai số `0.001%` (`NAIVE`), `0.001%` (`ONLINE_2A`), `0.019%` (`FLASH_RING`), `0.324%` (`FLASH_KVPIN`) — toàn bộ `< 0.4%`.

Bảng tổng hợp từ [results/attn_4modes_energy_summary.csv](results/attn_4modes_energy_summary.csv) (`LPDDR4X 16 pJ/bit` và kiểm chứng nhạy `LPDDR5 9 pJ/bit`):

| `M = N_KV` | Mode | Logic (`µJ`) | SRAM (`µJ`) | DRAM LPDDR4X (`µJ`) | Total LPDDR4X (`µJ`) | Giảm vs `NAIVE` (LPDDR4X) | Giảm vs `NAIVE` (LPDDR5) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `64` | `NAIVE` | `13.32` | `3.12` | `257.95` | `274.39` | `1.00×` | `1.00×` |
| `64` | `FLASH_KVPIN` | `8.72` | `1.62` | `24.38` | **`34.72`** | **`7.90×`** | **`6.72×`** |
| `256` | `NAIVE` | `151.46` | `42.88` | `3,523.22` | `3,717.55` | `1.00×` | `1.00×` |
| `256` | `ONLINE_2A` | `129.48` | `38.17` | `2,919.24` | `3,086.89` | `1.20×` | `1.20×` |
| `256` | `FLASH_RING` | `138.18` | `49.31` | `6,517.95` | `6,705.43` | `0.55×` | `0.56×` |
| `256` | **`FLASH_KVPIN`** | `136.66` | `24.08` | `97.52` | **`258.26`** | **`14.39×`** | **`10.09×`** |
| `1024` | `NAIVE` | `2,176.89` | `657.56` | `53,955.53` | `56,789.98` | `1.00×` | `1.00×` |
| `1024` | `ONLINE_2A` | `1,829.64` | `582.76` | `44,291.85` | `46,704.25` | `1.22×` | `1.21×` |
| `1024` | `FLASH_RING` | `2,188.38` | `781.42` | `103,381.21` | `106,351.00` | `0.53×` | `0.54×` |
| `1024` | **`FLASH_KVPIN`** | `2,181.87` | `378.50` | `390.07` | **`2,950.44`** | **`19.25×`** | **`11.94×`** |

---

## 4. Workstream 4C — Full Transformer Layer Energy (`M = 1024`, 3 Edge LLMs)

- Kiểm định [full_layer_energy_model.py](full_layer_energy_model.py) với [LLAMA3_1B_PREFILL_REPORT.md](../LLAMA3_1B_PREFILL_REPORT.md) (`M = 16`):
  - Weight bytes: `34,209,792 B` vs `34,209,792 B` (**`0 B` error**).
  - Total cycles: `4,480,431` vs `4,479,514` (**`0.020%` error**, `< 2.0%`).

Bảng tổng hợp từ [results/full_layer_energy_summary.csv](results/full_layer_energy_summary.csv) (`M = 1024`, `LPDDR4X 16 pJ/bit`):

| Model | `FP16 Unfused` (`mJ`) | `NVFP4 Unfused` (`mJ`) | `Log8 + FLASH_RING` (`mJ`) | **`Log8 + FLASH_KVPIN`** (`mJ`) | Giảm vs `FP16` | Giảm vs `NVFP4` | Giảm vs `RING` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Llama-3.2-1B`** | `1,136.36` | `206.34` | `282.35` | **`185.36`** | **`6.13×`** | **`1.11×`** | **`1.52×`** |
| **`SmolLM2-1.7B`** | `1,248.64` | `219.80` | `295.75` | **`198.99`** | **`6.27×`** | **`1.10×`** | **`1.49×`** |
| **`Qwen3-1.7B-Base`** | `927.93` | `160.55` | `325.50` | **`150.95`** | **`6.15×`** | **`1.06×`** | **`2.16×`** |
| **Geo. Mean** | — | — | — | — | **`6.18×`** | **`1.09×`** | **`1.69×`** |

---

## 5. Deliverables & Sign-Off Checklist

- [x] `sim/attn_comparison_results.csv` đủ 4 mode (`NAIVE`, `ONLINE_2A`, `FLASH_RING`, `FLASH_KVPIN`) ở `M=256, BW ∈ {64, 32, 16, 8}`, `checks_fail == 0`, `cpu_mem_cyc == 0`.
- [x] Log DC `06_gls/reports_500mhz/` xác nhận `500 MHz` (`100.00%` SAIF annotated, `0` cảnh báo `PWR-12`), đã loại `u_tcdm` ảo khỏi `P_logic`.
- [x] `attn_4modes_tiling_model.py` PASS assertion **0 B error** trên cả 4 mode.
- [x] `full_layer_energy_model.py` PASS assertion **`0.020%` cycle error** và **0 B weight error** so với `LLAMA3_1B_PREFILL_REPORT.md`.
- [x] [energy_params.json](energy_params.json) khóa cứng mọi hằng số kèm nguồn; đã chạy sensitivity check ở cả `LPDDR4X (16 pJ/bit)` và `LPDDR5 (9 pJ/bit)` với kết luận bất biến.
- [x] Cập nhật `Table VI` & `Table VII` trong [PAPER_BLUEPRINT_TCAD_TVLSI.md](../PAPER_BLUEPRINT_TCAD_TVLSI.md).
- [x] 3 hình chuẩn IEEE 2-Column 300 DPI:
  - [fig_attn_bw_sensitivity.pdf](results/fig_attn_bw_sensitivity.pdf) / [.png](results/fig_attn_bw_sensitivity.png)
  - [fig_attn_energy_breakdown.pdf](results/fig_attn_energy_breakdown.pdf) / [.png](results/fig_attn_energy_breakdown.png)
  - [fig_full_layer_energy_breakdown.pdf](results/fig_full_layer_energy_breakdown.pdf) / [.png](results/fig_full_layer_energy_breakdown.png)
