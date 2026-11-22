# Báo cáo Thực nghiệm & Thẩm định So sánh Kiến trúc Attention (`Standard` vs. `Online Softmax` vs. `FlashAttention`)

Tài liệu này ghi nhận đầy đủ **cách thiết kế Testbench, bộ Testcase, phương pháp chạy/debug, toàn bộ 37 kết quả đo RTL đã hoàn thành**, và **phân tích thẩm định độ tin cậy** cho thí nghiệm đề xuất trong [ATTENTION_COMPARISON_FLASH_VS_ONLINE_VS_NAIVE.md](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/ATTENTION_COMPARISON_FLASH_VS_ONLINE_VS_NAIVE.md).

- **DUT phần cứng (giữ nguyên 100%, không chỉnh sửa RTL):**
  - Core & TCDM: [snax_llm_top_clean.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L38-L48) (`BANK_DEPTH = 1024`, `NUM_BANKS = 32`, `DATA_WIDTH = 72`, tổng dung lượng SRAM on-chip `TCDM = 294,912 B = 288 KB`).
  - DMA 1D: [snax_simple_dma.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_simple_dma.sv#L14-L19) (`TCDM_PORTS = 8`, tức `8 × 72-bit = 576b/cyc`; DRAM `512-bit`, byte-strobe `64 B`). *(Ghi chú: tài liệu thiết kế ban đầu ghi ước lượng gearbox `4×72b = 36 B/cyc`, trong khi RTL thực tế triển khai `8×72b`.)*
- **Workload thực:** `NousResearch/Llama-3.2-1B` Layer 1 (`H_Q = 32`, `H_KV = 8` — GQA 4:1, `d_k = d_v = 64`, Causal Mask).
- **Trình mô phỏng:** Verilator `5.032` (WSL2, chạy tuần tự 1 tiến trình với `nice -n 19` ở chế độ hạ xung chống nóng).
- **Tổng trạng thái đã đo:** **37/37 lượt chạy hoàn tất đều PASS** (`checks_fail = 0`, `cpu_mem_cyc = 0`, `max_inflight <= 3`, `DRAM canaries PASS`), dữ liệu gốc lưu tại [attn_comparison_results.csv](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/attn_comparison_results.csv).

---

## 1. Kiến trúc Testbench, Danh sách File & Nguyên tắc Zero-Cheat

### 1.1. Danh sách các file cấu thành hạ tầng thí nghiệm

| File | Vai trò |
| :--- | :--- |
| [tb_attn_comparison.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_attn_comparison.sv) | Testbench SystemVerilog (`906` dòng): mô hình hóa DRAM 512-bit 16 MB có bộ bóp băng thông `DRAM_BW` (`64 / 32 / 16 B/cyc`), điều phối đủ 5 chế độ Attention, đếm chính xác beat DRAM, chu kỳ MAC/DMA/Stall và xuất `CSV_ROW`. |
| [gen_attn_comparison.py](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/gen_attn_comparison.py) | Script Python (`191` dòng): hook kích thích thật từ `NousResearch/Llama-3.2-1B` Layer 1, áp dụng `RMSNorm1 + RoPE + qk_scale`, đóng gói Log8-FP4 chuẩn phần cứng (`q_words.hex`, `k_words.hex`, `v_words.hex`) và xuất đồng thời 2 golden độc lập (`g_attn_fp32.hex` và `g_attn_fq.hex`). |
| [filelist_attn_comparison.f](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/filelist_attn_comparison.f) | Danh sách `34` file RTL trong [01_snax_llm_fp4_subsystem](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem) + `../tb/tb_attn_comparison.sv` (`40` dòng). |
| [run_attn_comparison.sh](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/run_attn_comparison.sh) | Script biên dịch & chạy 1 cấu hình (`108` dòng): hỗ trợ `+MODE`, `+M_TOKENS`, `+N_KV`, `+DRAM_BW`, `--skip-gen`, `--rebuild` và `--dump` (`.fst`). |
| [sweep_attn_comparison.sh](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/sweep_attn_comparison.sh) | Script điều phối chạy sweep tuần tự (`124` dòng): chia `smoke / A / B / C / D`, dùng `nice -n 19`, ghi từng dòng `CSV_ROW` vào CSV ngay sau mỗi lượt chạy và mặc định tự bỏ qua các case đã có (`RESUME=1`). |
| [attn_comparison_results.csv](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/attn_comparison_results.csv) | Bảng kết quả gốc dạng CSV (`38` dòng = 1 header `24` cột + `37` dòng đo đạc thực tế). |

### 1.2. Các chốt kiểm soát Zero-Cheat được thực thi trong Testbench
1. **Cấm tuyệt đối `cpu_mem`:** Cổng `cpu_mem_req_valid` được nối cứng `1'b0` ([tb_attn_comparison.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_attn_comparison.sv#L106-L110)) và kiểm tra mỗi chu kỳ clock ([dòng 276–281](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_attn_comparison.sv#L276-L281)): chỉ cần phát sinh 1 chu kỳ active là báo `[FAIL]` (`cpu_mem_active_cyc == 0`).
2. **100% luồng dữ liệu qua 1D-DMA thực:** Toàn bộ việc nạp `Q, K, V`, xả/nạp `S, P` (ở `NAIVE`/`ONLINE_2A`) và ghi kết quả `O` đều thực hiện qua `snax_simple_dma` 8-port và `snax_llm_sequencer`.
3. **Bảo vệ biên bộ nhớ & chống treo:**
   - Vùng `DRAM_O` được bọc 64 B canary `8'hA5` ở đầu và cuối, đồng thời điền poison `8'h5A` trước khi chạy ([dòng 461–486](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_attn_comparison.sv#L461-L486)).
   - Kiểm soát thanh ghi `dut.u_seq.inflight_out_cnt_q <= 4` ở mọi chu kỳ ([dòng 284–287](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_attn_comparison.sv#L284-L287)).
4. **Kiểm tra 100% đầu ra với Dual Golden:** Đọc lại toàn bộ `[H_Q][M_TOKENS][D_V]` từ `DRAM_O` và so đồng thời với:
   - `g_attn_fq.hex` (`PyFQ`: `Q, K, V, P` lượng tử hóa Log8-FP4 + Softmax `np.exp()` FP32) với ngưỡng `Cos vs PyFQ >= 0.96` để xác nhận RTL tính đúng mô hình FP4.
   - `g_attn_fp32.hex` (PyTorch FP32 thuần) với ngưỡng `Cos vs FP32 >= 0.80` để kiểm tra độ hợp lý đầu cuối ([dòng 865–866](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_attn_comparison.sv#L865-L866)).

### 1.3. Chiến lược biên dịch chống OOM & chạy chống nóng
- Binary [Vtb_attn_comparison](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/run_attn_comparison.sh#L72-L96) được biên dịch **1 lần duy nhất** trên ext4 (`/tmp/obj_attn_cmp/Vtb_attn_comparison`) qua 3 pha:
  1. `verilator --cc --main --exe --timing` sinh mã C++.
  2. Build Precompiled Header (`.gch`) ở `OPT_FAST="-O2"`, biên dịch riêng các file coroutine `initial` của TB (`_eval_initial__TOP__Vtiming`) ở `-O0` để tránh phình RAM/OOM của GCC.
  3. Biên dịch phần RTL ở `OPT_FAST="-O2" -j 4` và link.
- Khi mô phỏng, mọi lượt chạy dùng lại binary đã cache với mức RAM ổn định **`221 MB`**, chạy tuần tự **từng tiến trình một** dưới `nice -n 19`.

---

## 2. Các Chế độ Testcase & Bộ Dữ liệu Thật đã sinh

### 2.1. Định nghĩa 5 chế độ (`+MODE`) trong [tb_attn_comparison.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_attn_comparison.sv#L715-L826)

| Mode | Đại diện kiến trúc | Cấu hình điều phối & tham số CSR | Cách thực thi trên RTL |
| :--- | :--- | :--- | :--- |
| **`FLASH_KVPIN`** | FlashAttention hoàn chỉnh (đề xuất) | `macro=1`, `M_Q=8`, `Q_TILES=M/8`, `Q_SUB=min(Q_TILES, 8)`, `KV_PIN=1`, `Sep-KV=1`, `GQA_M1=3`, `causal=1` | 1 lệnh CSR duy nhất quét toàn bộ 32 heads; mỗi cụm 4 Q-heads ghim chung KV vào 2 slot TCDM, không ghi `S/P` ra DRAM (`sp = 0 B`). |
| **`FLASH_RING`** | FlashAttention + Ring Buffer (tắt KV-Pin) | Giống `FLASH_KVPIN`, chỉ khác **`KV_PIN=0`** | Vẫn giữ 2D tile fusion (`M_Q=8`) và `p_fifo`/`O`-rescaler on-chip (`sp = 0 B`), nhưng dùng 4-slot Ring DMA nên phải đọc lại KV từ DRAM cho từng cụm Q/dvi. |
| **`ONLINE_2B`** | Online Softmax ghép `PV` không có 2D tile fusion | `macro=1`, **`M_Q=1`**, `Q_TILES=M`, `Q_SUB=min(Q_TILES, 64)`, `KV_PIN=0`, `Sep-KV=1`, `GQA_M1=3` | Tương ứng Kịch bản 2B: không ghi `S/P` ra DRAM, nhưng vì `M_Q=1` nên mỗi token query phải đọc lại KV riêng và không tái sử dụng được dữ liệu theo tile 2D. |
| **`ONLINE_2A`** | Unfused Online Softmax (tách pha `QK^T -> S -> P -> PV`) | 3 pha/chunk `M_BATCH=16` × `32` heads; P2 đọc `S` qua DMA **1 pass**; `p2_sfu_est = 4 cyc / 16 elem` | Pha 1 GEMM tính `S` FP16 ghi `DRAM_S`; Pha 2 đọc `S` 1 lần + SLQ requant + ghi `P` ra `DRAM_P` (cộng ước lượng `p2_sfu_est`, SW `$exp()` điền `P`); Pha 3 đọc `P` và chạy GEMM `P·V` -> `DRAM_O`. |
| **`NAIVE`** | Standard Attention 3 pha (không Online Softmax, không Flash) | Giống `ONLINE_2A`, nhưng P2 đọc `S` qua DMA **2 pass** (`P2 Rd S1`, `P2 Rd S2`); `p2_sfu_est = 8 cyc / 16 elem` | Mô phỏng Kịch bản 1: buộc phải duyệt `S` từ DRAM 2 pass (max + exp/sum) trước khi chuẩn hóa ghi `P` ra DRAM rồi nạp lại ở Pha 3. |

### 2.2. 7 bộ dữ liệu thật `Llama-3.2-1B` Layer 1 đã sinh sẵn

Tất cả 7 bộ đều được [gen_attn_comparison.py](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/gen_attn_comparison.py) xuất thành công vào `sim/sim_attn_cmp_m<M>_kv<N_KV>/`, kèm giá trị trần lý thuyết của mô hình lượng tử hóa `PyFQ` (FP4 `Q, K, V, P` + Softmax `np.exp()` FP32) so với PyTorch FP32 thuần:

| Thư mục trong `sim/` | `M_TOKENS` | `N_KV` (`n_past`) | `PyFQ vs FP32` (`SQNR / Cos / MaxAbs`) | Thời gian sinh | Trạng thái mô phỏng RTL |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `sim_attn_cmp_m64_kv64` | `64` | `64` (`0`) | `6.18 dB` / `0.892174` / `0.5478` | `9.69 s` | **Đã đo đủ 5 modes × 3 BW (15/15)** |
| `sim_attn_cmp_m128_kv128` | `128` | `128` (`0`) | `5.94 dB` / `0.888450` / `0.5478` | `5.76 s` | **Đã đo đủ 5 modes × 3 BW (15/15)** |
| `sim_attn_cmp_m256_kv256` | `256` | `256` (`0`) | `6.01 dB` / `0.889786` / `0.8769` | `7.14 s` | **Đã đo `BW=64` (5/5) + `BW=32` (2/5: `FLASH_*`)** |
| `sim_attn_cmp_m64_kv256` | `64` | `256` (`192`) | `6.36 dB` / `0.899511` / `0.8769` | `6.52 s` | Đã sinh hex, chưa chạy mô phỏng RTL |
| `sim_attn_cmp_m64_kv512` | `64` | `512` (`448`) | `7.90 dB` / `0.924947` / `0.5599` | `8.81 s` | Đã sinh hex, chưa chạy mô phỏng RTL |
| `sim_attn_cmp_m512_kv512` | `512` | `512` (`0`) | `6.43 dB` / `0.899067` / `0.8769` | `9.43 s` | Đã sinh hex, chưa chạy mô phỏng RTL |
| `sim_attn_cmp_m1024_kv1024` | `1024` | `1024` (`0`) | `6.96 dB` / `0.908431` / `0.8769` | `14.51 s` | Đã sinh hex, chưa chạy mô phỏng RTL |

---

## 3. Cách chạy Test, Debug Waveform & Phân tích Thời gian Chạy

### 3.1. Chạy 1 testcase đơn lẻ và xuất waveform `.fst` để debug
Thực hiện trong WSL2 tại thư mục `sim/`:

```bash
# 1) Chạy 1 cấu hình bất kỳ (dùng lại bộ hex và binary đã biên dịch sẵn)
nice -n 19 ./run_attn_comparison.sh --mode FLASH_KVPIN --m-tokens 256 --n-kv 256 --dram-bw 16 --skip-gen

# 2) Nếu chỉnh sửa TB và muốn biên dịch lại từ đầu:
nice -n 19 ./run_attn_comparison.sh --mode FLASH_KVPIN --m-tokens 64 --n-kv 64 --dram-bw 64 --skip-gen --rebuild

# 3) Xuất waveform .fst cho riêng testcase đó để mở bằng GTKWave/Surfer
#    (tự động biên dịch riêng ở /tmp/obj_attn_cmp_fst và xuất file tb_attn_<MODE>_m<M>_kv<N_KV>_bw<BW>.fst)
nice -n 19 ./run_attn_comparison.sh --mode FLASH_KVPIN --m-tokens 64 --n-kv 64 --dram-bw 64 --skip-gen --dump

# 4) Sinh mới dữ liệu ở kích thước khác (bỏ --skip-gen)
nice -n 19 ./run_attn_comparison.sh --mode NAIVE --m-tokens 64 --n-kv 256 --dram-bw 16
```

### 3.2. Chạy Sweep tuần tự (chống OOM & chống nóng)
Sử dụng [sweep_attn_comparison.sh](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/sweep_attn_comparison.sh):

```bash
# Các stage định nghĩa sẵn: smoke | A (64,128) | B (256 + 64x256/512) | C (512) | D (1024)
./sweep_attn_comparison.sh --stage A

# Cờ hữu ích:
#   --init-csv   : tạo lại header mới cho attn_comparison_results.csv (cẩn thận: ghi đè CSV cũ)
#   --rebuild    : ép biên dịch lại ở lượt đầu tiên của stage
#   --no-resume  : chạy lại cả những dòng đã có trong attn_comparison_results.csv (mặc định RESUME=1 tự bỏ qua)
```

### 3.3. Tại sao Stage B chạy lâu & thời gian đo thực tế
Trên máy đã hạ xung CPU chống nóng, tốc độ mô phỏng Verilator 1 luồng đo được là **`~124,000–128,000` chu kỳ RTL/giây** (`~1.21–1.28 ms/s` với `CLK_PERIOD = 10 ns`):
- Ở **`M=64, BW=64`**: `ONLINE_2A` mất `2.83 s`, `NAIVE` mất `2.92 s`, riêng `ONLINE_2B` (`1.39M` cyc) mất `11.27 s`.
- Ở **`M=256, BW=64` (đo thực tế từ log)**:
  - `FLASH_KVPIN` (`3.81M` cyc): **`30.65 s`** | `FLASH_RING` (`3.85M` cyc): **`31.37 s`**
  - `ONLINE_2A` (`3.29M` cyc HW): **`27.76 s`** | `NAIVE` (`3.45M` cyc HW): **`28.86 s`**
  - Riêng **`ONLINE_2B`** (`22.09M` cyc): **`174.55 s` (~2.9 phút)** cho một lượt duy nhất!
- Vì lệnh `--stage B` gom 25 lượt vào một lệnh nền và độ phức tạp Attention tăng bậc hai ($O(M \cdot N_{KV})$ -> gấp đôi độ dài chuỗi là **4×** thời gian):
  - 3 lượt `ONLINE_2B` ở `M=256` (`BW=64, 32, 16`) tốn khoảng `3 + 3 + 4.5 ≈ 10.5 phút`. Lúc lệnh bị dừng ở phút thứ 8, script đã hoàn tất 7 lượt đầu của Stage B và đang chạy lượt thứ 8 (`ONLINE_2B, M=256, BW=32`).
  - **Ước lượng thời gian nếu chạy tiếp ở tốc độ ~125k cyc/s:**
    - `M=512`: ~2 phút/lượt cho 4 mode chính (`FLASH_KVPIN`, `FLASH_RING`, `ONLINE_2A`, `NAIVE`); riêng `ONLINE_2B` ~11–12 phút/lượt.
    - `M=1024`: ~8 phút/lượt cho 4 mode chính (~64 phút nếu chạy đủ 8 lượt ở `BW=64` và `BW=16`).

---

## 4. Toàn bộ Kết quả Thực nghiệm Đã Đo (`37/37` Lượt PASS)

Mọi giá trị dưới đây được đối chiếu nguyên văn từ [attn_comparison_results.csv](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/attn_comparison_results.csv). Tất cả 37 lượt đều có **`cpu_mem_cyc = 0`** và **`checks_fail = 0`**.

### 4.1. So sánh Kiến trúc theo Chiều dài Chuỗi tại `BW = 64 B/cyc` (`M = 64, 128, 256`)

| `M=N_KV` | Mode | `cyc_hw` | `p2_sfu_est` | `cyc_total_w_sfu` | `cyc_mac` | `cyc_dma` | DRAM Read (B) | DRAM Write (B) | Total DRAM (B) | Tỉ lệ DRAM vs `KVPIN` | Interm `S+P` (B) | `MAC Util` |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | :---: |
| **`64`** | **`FLASH_KVPIN`** | `243,093` | `0` | **`243,093`** | `163,840` | `24,882` | `116,736` | `73,728` | **`190,464`** | **1.0×** | **`0`** | **67.40%** |
| `64` | `FLASH_RING` | `250,808` | `0` | `250,808` | `163,840` | `74,376` | `3,219,456` | `73,728` | `3,293,184` | 17.3× | `0` | 65.32% |
| `64` | `ONLINE_2A` | `315,660` | `32,768` | `348,428` | `65,536` | `36,736` | `1,155,072` | `565,248` | `1,720,320` | 9.0× | `737,280` | 18.81% |
| `64` | `NAIVE` | `327,564` | `65,536` | `393,100` | `65,536` | `41,344` | `1,449,984` | `565,248` | `2,015,232` | 10.6× | `1,032,192` | 16.67% |
| `64` | `ONLINE_2B` | `1,393,308` | `0` | `1,393,308` | `163,840` | `442,336` | `25,239,552` | `73,728` | `25,313,280` | 132.9× | `0` | 11.76% |
| **`128`** | **`FLASH_KVPIN`** | `957,205` | `0` | **`957,205`** | `655,360` | `52,034` | `233,472` | `147,456` | **`380,928`** | **1.0×** | **`0`** | **68.47%** |
| `128` | `FLASH_RING` | `976,504` | `0` | `976,504` | `655,360` | `263,464` | `12,730,368` | `147,456` | `12,877,824` | 33.8× | `0` | 67.11% |
| `128` | `ONLINE_2A` | `968,972` | `131,072` | `1,100,044` | `262,144` | `115,456` | `4,227,072` | `1,867,776` | `6,094,848` | 16.0× | `2,949,120` | 23.83% |
| `128` | `NAIVE` | `1,011,212` | `262,144` | `1,273,356` | `262,144` | `133,888` | `5,406,720` | `1,867,776` | `7,274,496` | 19.1× | `4,128,768` | 20.59% |
| `128` | `ONLINE_2B` | `5,539,036` | `0` | `5,539,036` | `655,360` | `1,736,640` | `100,810,752` | `147,456` | `100,958,208` | 265.0× | `0` | 11.83% |
| **`256`** | **`FLASH_KVPIN`** | `3,810,837` | `0` | **`3,810,837`** | `2,621,440` | `106,754` | `466,944` | `294,912` | **`761,856`** | **1.0×** | **`0`** | **68.79%** |
| `256` | `FLASH_RING` | `3,853,304` | `0` | `3,853,304` | `2,621,440` | `985,704` | `50,626,560` | `294,912` | `50,921,472` | 66.8× | `0` | 68.03% |
| `256` | `ONLINE_2A` | `3,288,588` | `524,288` | `3,812,876` | `1,048,576` | `399,872` | `16,121,856` | `6,684,672` | `22,806,528` | 29.9× | `11,796,480` | 27.50% |
| `256` | `NAIVE` | `3,446,796` | `1,048,576` | `4,495,372` | `1,048,576` | `473,600` | `20,840,448` | `6,684,672` | `27,525,120` | 36.1× | `16,515,072` | 23.33% |
| `256` | `ONLINE_2B` | `22,088,028` | `0` | `22,088,028` | `2,621,440` | `6,881,152` | `402,948,096` | `294,912` | `403,243,008` | 529.3× | `0` | 11.87% |

### 4.2. Độ nhạy theo Băng thông DRAM (`BW = 64 -> 32 -> 16 B/cyc`)

Bảng dưới liệt kê toàn bộ các lượt đo `BW = 32` và `BW = 16 B/cyc` đã hoàn tất (gồm đủ 100% ở `M=64`, `M=128` và 2 lượt `BW=32` đầu tiên ở `M=256`):

| `M=N_KV` | Mode | `BW=64` Total cyc | `BW=32` `cyc_hw` / Total (`stalls`) | `BW=16` `cyc_hw` / Total (`stalls`) | Độ trễ tăng thêm (`BW=64 -> 16`) |
| :---: | :--- | ---: | ---: | ---: | :---: |
| **`64`** | **`FLASH_KVPIN`** | `243,093` | `243,150` / `243,150` (`1,585`) | `243,772` / **`243,772`** (`6,287`) | **`+0.28%`** |
| `64` | `FLASH_RING` | `250,808` | `251,000` / `251,000` (`21,857`) | `280,180` / `280,180` (`126,910`) | `+11.71%` |
| `64` | `ONLINE_2A` | `348,428` | `325,133` / `357,901` (`13,441`) | `358,923` / `391,691` (`69,631`) | `+12.42%` |
| `64` | `NAIVE` | `393,100` | `337,421` / `402,957` (`13,825`) | `380,427` / `445,963` (`83,327`) | `+13.45%` |
| `64` | `ONLINE_2B` | `1,393,308` | `1,458,910` / `1,458,910` (`165,121`) | `2,114,396` / `2,114,396` (`987,969`) | `+51.75%` |
| **`128`** | **`FLASH_KVPIN`** | `957,205` | `957,262` / `957,262` (`3,161`) | `958,396` / **`958,396`** (`12,775`) | **`+0.12%`** |
| `128` | `FLASH_RING` | `976,504` | `976,888` / `976,888` (`84,641`) | `1,084,532` / `1,084,532` (`499,582`) | `+11.06%` |
| `128` | `ONLINE_2A` | `1,100,044` | `999,949` / `1,131,021` (`42,241`) | `1,118,219` / `1,249,291` (`260,863`) | `+13.57%` |
| `128` | `NAIVE` | `1,273,356` | `1,042,957` / `1,305,101` (`43,009`) | `1,198,091` / `1,460,235` (`315,903`) | `+14.68%` |
| `128` | `ONLINE_2B` | `5,539,036` | `5,801,310` / `5,801,310` (`657,889`) | `8,423,004` / `8,423,004` (`3,942,049`) | `+52.07%` |
| **`256`** | **`FLASH_KVPIN`** | `3,810,837` | `3,810,894` / `3,810,894` (`6,313`) | *(chưa chạy)* | `+0.00%` *(tại BW=32)* |
| `256` | `FLASH_RING` | `3,853,304` | `3,854,072` / `3,854,072` (`333,089`) | *(chưa chạy)* | `+0.02%` *(tại BW=32)* |

*(Ghi chú: Giá trị `cyc_dma` ở `BW=32` / `BW=16` cũng có thay đổi nhẹ ở một số lượt — ví dụ ở `M=128`: `FLASH_KVPIN` `51,994` / `51,992`, `FLASH_RING` `263,400` / `231,077`, `ONLINE_2B` `1,736,640` / `1,605,632`, `ONLINE_2A` `116,480` / `113,920`, `NAIVE` `134,912` / `132,352`; toàn bộ chi tiết nằm trong [attn_comparison_results.csv](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/attn_comparison_results.csv).)*

### 4.3. Độ chính xác & Kiểm định Phần cứng (`RTL vs PyFQ`, `RTL vs FP32`, Invariants)

Vì thay đổi `DRAM_BW` (`64 / 32 / 16`) không làm thay đổi giá trị toán học bit-exact của phép tính, kết quả độ chính xác giống hệt nhau trên mọi mức `BW`:

| `M=N_KV` | Nhóm Mode | **RTL vs `PyFQ`** (`SQNR` / `Cos`) | **RTL vs `FP32`** (`SQNR` / `Cos` / `MaxAbs`) | Trần **`Python PyFQ vs FP32`** | `cpu_mem_cyc` | `max_inflight` | `checks_fail` |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `64` | `FLASH_KVPIN` / `FLASH_RING` | `16.54 dB` / `0.990459` | `5.61 dB` / `0.886991` / `0.4900` | `6.18 dB` / `0.892174` (`0.5478`) | `0` | `3` | `0` |
| `64` | `ONLINE_2B` | `16.54 dB` / `0.990459` | `5.61 dB` / `0.886991` / `0.4900` | `6.18 dB` / `0.892174` (`0.5478`) | `0` | `1` | `0` |
| `64` | `NAIVE` / `ONLINE_2A` *(lai SW `$exp`)* | `17.65 dB` / `0.991385` | `6.07 dB` / `0.887422` / `0.4900` | `6.18 dB` / `0.892174` (`0.5478`) | `0` | `3` | `0` |
| `128` | `FLASH_KVPIN` / `FLASH_RING` | `16.73 dB` / `0.991087` | `5.33 dB` / `0.883348` / `0.4986` | `5.94 dB` / `0.888450` (`0.5478`) | `0` | `3` | `0` |
| `128` | `ONLINE_2B` | `16.73 dB` / `0.991087` | `5.33 dB` / `0.883348` / `0.4986` | `5.94 dB` / `0.888450` (`0.5478`) | `0` | `1` | `0` |
| `128` | `NAIVE` / `ONLINE_2A` *(lai SW `$exp`)* | `17.79 dB` / `0.991653` | `5.82 dB` / `0.883918` / `0.4900` | `5.94 dB` / `0.888450` (`0.5478`) | `0` | `3` | `0` |
| `256` | `FLASH_KVPIN` / `FLASH_RING` | `16.91 dB` / `0.991625` | `5.32 dB` / `0.882623` / `0.9498` | `6.01 dB` / `0.889786` (`0.8769`) | `0` | `3` | `0` |
| `256` | `ONLINE_2B` | `16.91 dB` / `0.991625` | `5.32 dB` / `0.882623` / `0.9498` | `6.01 dB` / `0.889786` (`0.8769`) | `0` | `1` | `0` |
| `256` | `NAIVE` / `ONLINE_2A` *(lai SW `$exp`)* | `18.04 dB` / `0.992115` | `5.84 dB` / `0.884022` / `0.8819` | `6.01 dB` / `0.889786` (`0.8769`) | `0` | `3` | `0` |

### 4.4. Trạng thái phạm vi Sweep hiện tại
- **Đã hoàn tất (37 lượt):** toàn bộ Stage `smoke` (`64x64 BW=64`), toàn bộ Stage `A` (`64x64` và `128x128` đủ `5 modes × 3 BW`), và `7/25` lượt đầu Stage `B` (`256x256 BW=64` đủ 5 modes + `256x256 BW=32` cho `FLASH_KVPIN` & `FLASH_RING`).
- **Chưa chạy:** `M=256 BW=32` từ `ONLINE_2B` trở đi, `M=256 BW=16`, chunked prefill `(64, 256)` / `(64, 512)`, `M=512` (Stage `C`) và `M=1024` (Stage `D`).

---

## 5. Phân tích Kỹ thuật & Thẩm định Độ Tin Cậy (Critical Review)

### 5.1. Những kết luận ĐÁNG TIN TUYỆT ĐỐI (100% từ RTL thật)

1. **`FLASH_KVPIN` chạm sát cận dưới lý thuyết về lưu lượng DRAM:**
   - Với `M=256, N_KV=256`, dung lượng vật lý tối thiểu của `Q + K + V + O` ở định dạng 9 B/16 phần tử là `294,912 + 73,728 + 73,728 + 294,912 = 737,280 B` (`737.3 KB`).
   - RTL `FLASH_KVPIN` thực đo được **`761,856 B`** (chỉ dư `+3.3%` do căn biên beat 64 B ở phần đọc; riêng `wr_bytes = 294,912 B` khớp **1.000×** tới từng byte).
   - Lưu lượng DRAM giảm **29.9×** so với `ONLINE_2A`, **36.1×** so với `NAIVE`, **66.8×** so với `FLASH_RING` và **529.3×** so với `ONLINE_2B`. Tổng lưu lượng trung gian `S + P` giảm từ `11.80 MB` / `16.52 MB` về **`0 B`**.

2. **`FLASH_KVPIN` gần như miễn nhiễm trước nghẽn băng thông DRAM:**
   - Khi giảm băng thông 4× từ `BW=64` xuống `BW=16 B/cyc`, độ trễ toàn cục tại `M=128` của `FLASH_KVPIN` chỉ tăng **`+0.12%`** (`957,205 -> 958,396` chu kỳ, `dram_stalls = 12,775`).
   - Cùng điều kiện đó, `FLASH_RING` tăng `+11.06%` (`499,582` stalls), `ONLINE_2A` tăng `+13.57%` (`260,863` stalls), `NAIVE` tăng `+14.68%` (`315,903` stalls), còn `ONLINE_2B` tăng vọt **`+52.07%`** (`3,942,049` stalls).

3. **Phát hiện nhược điểm kiến trúc thật của RTL [snax_llm_sequencer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L1062-L1085):**
   - Trong khối điều khiển Attention của sequencer, bộ đếm `cnt_dvi_q` (`dvi = 0..3`, tương ứng `DV_TILES = 4` cho `D_V = 64`) được đặt **bên ngoài** vòng lặp `cnt_kj_q` (`kj = 0..KV_TILES-1`), đồng thời chế độ FlashAttention chạy ở `M_Q = 8` (một nửa chiều cao `16` của GEMM `M_BATCH = 16`).
   - Hệ quả thực tế: để hoàn thành 4 `DV_TILES` của `O`, phần cứng đang **tính lại toàn bộ `Q·K^T` và Online Softmax tới 4 lần** cho mỗi tile `Q`!
   - Bằng chứng trực tiếp trên CSV ở `M=256`: **`cyc_mac` của `FLASH_KVPIN` / `FLASH_RING` / `ONLINE_2B` là `2,621,440` chu kỳ — gấp đúng `2.50×` so với `1,048,576` chu kỳ của `NAIVE` / `ONLINE_2A`**.
   - Điều này cũng giải thích hai hệ quả quan trọng đã đo:
     - Nếu tắt `KV_PIN` (`FLASH_RING`), việc lặp `4 dvi × 2` (`M_Q=8` vs `16`) buộc DMA phải đọc lại `K` gấp **8×**, đẩy tổng DRAM của `FLASH_RING` lên **`50.92 MB`** — tệ hơn cả `NAIVE` (`27.53 MB`) ở `BW=64`.
     - Khi bật `KV_PIN`, `FLASH_KVPIN` triệt tiêu hoàn toàn lưu lượng đọc lại KV nhưng vẫn gánh `2.50×` chu kỳ MAC, khiến ở băng thông rộng (`BW=64`, `M=256`), tổng chu kỳ của nó (`3,810,837`) chỉ tương đương `ONLINE_2A` (`3,812,876`) chứ chưa bỏ xa về tốc độ tuyệt đối.

4. **Giải thích chính xác hai thước đo độ chính xác (`vs PyFQ` cao vs `vs FP32` thấp):**
   - **`RTL vs PyFQ` cao (`16.54–18.04 dB`, `Cos = 0.9905–0.9921`):** Khẳng định phần cứng RTL thực hiện đúng toán học Log8-FP4 và Online Softmax LUT/FP20 của kiến trúc.
   - **`RTL vs FP32` thấp (`5.32–6.07 dB`, `Cos = 0.8826–0.8874`) không phải lỗi TB hay lỗi RTL:** Chính trần lý thuyết Python `PyFQ` (dù dùng Softmax `np.exp()` FP32 tuyệt đối) cũng chỉ đạt **`5.94–6.18 dB` / `Cos = 0.8885–0.8922`** trên workload này, vì 3 lý do vật lý:
     1. *BOS outlier ở Token 0 của `V`:* Theo [gen_attn_comparison.py](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/gen_attn_comparison.py#L160-L164), `V` đóng gói theo nhóm 16 token dọc trục context dùng chung 1 scale Log8. Token 0 (BOS) của Layer 1 có biên độ lớn, làm mất độ phân giải FP4 của 15 token cùng nhóm. Khi dịch cửa sổ né vùng BOS ở chunked prefill `M=64, N_KV=512 (n_past=448)`, chính `PyFQ vs FP32` tăng ngay lên **`7.90 dB / Cos = 0.924947`**.
     2. *Khuếch đại lượng tử hóa qua `exp()` 2 tầng:* Sai số FP4 của `Q, K` đi qua hàm mũ phi tuyến rồi xác suất `P` lại bị lượng tử hóa FP4 trước khi nhân `V` FP4.
     3. *Đo trực tiếp đầu ra Attention trước `W_O` và Residual:* Ở báo cáo full Layer 1 ([LLAMA3_1B_LAYER1_SWEEP_REPORT.md](file:///c:/Users/nguyen/Desktop/NPU_LOG/LLAMA3_1B_LAYER1_SWEEP_REPORT.md#L96-L100)), `Residual 1 vs FP32` đạt `12–23 dB` (`Cos 0.97–0.997`) là do được cộng nhánh residual `x_in` có năng lượng lớn hơn nhiều so với nhiễu Attention; riêng mạch Online Softmax RTL khi chạy dữ liệu không có BOS outlier ở `Test 6A-1 / 6B` ([dòng 53–55](file:///c:/Users/nguyen/Desktop/NPU_LOG/LLAMA3_1B_LAYER1_SWEEP_REPORT.md#L53-L55)) đạt tới **`20.59–23.87 dB` (`Cos = 0.9972–0.9988`)**.

---

### 5.2. Những giới hạn & bất đối xứng của TB hiện tại (cần minh bạch, không tin mù quáng)

Nếu trích dẫn trực tiếp **tổng chu kỳ `Total Cycles`** hoặc **độ lệch độ chính xác `+0.5 dB`** của `NAIVE`/`ONLINE_2A`, cần lưu ý 3 điểm bất đối xứng trong [tb_attn_comparison.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_attn_comparison.sv):

1. **Phase 2 của `NAIVE` & `ONLINE_2A` là mô phỏng lai (HW DMA + SW ước lượng + SW `$exp()`):**
   - Do DUT `snax_llm_top_clean` chỉ có khối Softmax tích hợp bên trong đường FlashAttention, ở [dòng 743–761](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_attn_comparison.sv#L743-L761), `NAIVE`/`ONLINE_2A` chạy **DMA đọc `S` (`1`/`2` pass), SLQ requant và DMA ghi `P` bằng RTL thật** (nên `sp_rd_bytes`/`sp_wr_bytes` phản ánh đúng chi phí traffic), nhưng:
     - Chu kỳ tính Softmax (`p2_sfu_est_cyc` = `+524,288` cho `ONLINE_2A` và `+1,048,576` cho `NAIVE` ở `M=256`, theo công thức `M_BATCH × KV_TILES × 4` hoặc `8`) là **ước lượng giải tích cộng thêm**, không phải chu kỳ đo từ một khối SFU unfused thật.
     - Khối `run_requant_wrap` lúc chạy ghi kết quả lượng tử hóa `S` thô ra `DRAM_P`, sau đó hàm SW `compute_softmax_s_to_p()` ([dòng 587–634](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_attn_comparison.sv#L587-L634)) dùng **`$exp()` FP64 chuẩn IEEE-754** ghi đè `P` thật vào `DRAM_P` trong **0 chu kỳ**. Điều này tạo ra **lợi thế giả khoảng `+0.46..+0.52 dB` so với `FP32`** (và `+1.1 dB` so với `PyFQ`) cho `NAIVE`/`ONLINE_2A` so với mạch `exp2` LUT/FP20 của `FLASH_*`.

2. **Chi phí điều khiển CSR và căn biên DMA đang "phạt nặng" `NAIVE` & `ONLINE_2A`:**
   - Ở [dòng 728–731](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_attn_comparison.sv#L728-L731) và [dòng 774–777](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_attn_comparison.sv#L774-L777), TB nạp từng token `Q` và xả từng token `O` bằng `16` lệnh `run_dma_1d` tách lẻ (`36 B`/lệnh trên bus 64 B), kéo theo ở `M=256` phải phát tới **`19,968` lệnh job CSR** cho `ONLINE_2A` và **`20,480` lệnh** cho `NAIVE` (thay vì `1` lệnh macro duy nhất ở `FLASH_*`). Điều này cộng thêm đáng kể overhead ghi CSR/bắt tay DMA và làm `MAC Util` (`27.50%` / `23.33%`) thấp hơn kịch bản hợp nhất DMA.
   - Ngoài ra, `S` được ghi ra DRAM nguyên dạng packet 72-bit của TCDM (`9 B` cho 4 phần tử FP16 thay vì `8 B` FP16 đóng gói chặt), làm lưu lượng `S` dư thêm `+12.5%` so với hệ thống ghi FP16 thuần.

3. **Mô hình DRAM băng thông cố định lý tưởng:**
   - Cơ chế `dram_req_ready` tại [dòng 226–254](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_attn_comparison.sv#L226-L254) dùng độ trễ đọc cố định `6` chu kỳ và nhịp `1/1` (`BW=64`), `1/2` (`BW=32`), `1/4` (`BW=16`), phù hợp để khảo sát độ nhạy băng thông định hướng nhưng chưa mô phỏng row-buffer conflict hay refresh của LPDDR thực tế.

---

### 5.3. Hướng tối ưu đề xuất (cho cả Testbench và RTL)

1. **Nâng cấp Testbench cho so sánh công bằng tuyệt đối (nếu cần số liệu công bố):**
   - Gom cụm 1D-DMA hoặc báo tách riêng phần `CSR launch overhead` khỏi `compute/DMA time` cho `NAIVE`/`ONLINE_2A`, chuẩn hóa kích thước ghi `S` theo FP16 `8 B/4 elem`.
   - Thay SW `$exp()` FP64 trong `compute_softmax_s_to_p()` bằng mô hình số học tương đương `exp2` LUT/FP20 của RTL để tách bạch lỗi thuật toán Online vs Softmax unfused khỏi sai số bộ tính `exp`.
2. **Tối ưu kiến trúc RTL [snax_llm_sequencer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv) & [snax_fused_llm_core.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv):**
   - Trong vòng lặp FlashAttention, giữ xác suất `P_j` (hoặc tái sử dụng hàng đợi `p_fifo` ở độ sâu `KV_TILES`/`tile` hoặc mở rộng số cột song song) để **quét qua toàn bộ `dvi = 0..3` mà không tính lại `Q·K^T` và Online Softmax 4 lần**. Khi loại bỏ được hệ số nhân `4×` của vòng `dvi`, `FLASH_KVPIN` sẽ giảm mạnh `cyc_mac` và chiến thắng áp đảo cả về **tốc độ (latency)** lẫn **lưu lượng DRAM**.
