# Báo cáo Kiểm chứng Hệ thống & Sweep 1 Layer Llama-3.2-1B (Zero-Cheat / 100% 1D-DMA)

Tài liệu này ghi lại đầy đủ kết quả thực nghiệm sau khi chuẩn hóa bộ testbench sang tiêu chuẩn **zero-cheat** (cấm hoàn toàn `cpu_mem`, mọi luồng dữ liệu qua 100% 1D-DMA, golden model phần mềm độc lập, kiểm tra 100% đầu ra kèm DRAM canary + backpressure) và chạy sweep **1 layer hoàn chỉnh của mô hình thực `NousResearch/Llama-3.2-1B`** trên nhiều kích thước `M` và `N_KV` (`32 / 64 / 128 / 256` tokens).

- **RTL DUT (giữ nguyên, không chỉnh sửa):** [snax_llm_top_clean.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv) (`BANK_DEPTH=1024`, `TCDM 288 KB`, 32 banks × 72-bit) + [snax_simple_dma.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_simple_dma.sv) (`TCDM_PORTS=8`, DRAM 512-bit byte-strobe 64-bit).
- **Trình mô phỏng:** Verilator 5.032 trên WSL2.

---

## 1. Kiến trúc Testbench Chuẩn mực & Chiến lược Build Chống OOM

### 1.1. Các nguyên tắc Zero-Cheat được áp dụng
1. **Cấm tuyệt đối `cpu_mem` di chuyển dữ liệu:** `cpu_mem_req_valid_i` (hoặc `cpu_req_valid`) được nối cứng `1'b0` và giám sát mọi chu kỳ (`cpu_mem_active_cyc == 0`). Toàn bộ nạp `A/Q/X`, `gamma1/2`, `RoPE`, `V_tail`, stream `W/KV`, writeback `O/V` và đo đạc kết quả đều đi qua **1D-DMA** (`snax_simple_dma` 8-port + [snax_llm_sequencer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv)).
2. **Golden Model độc lập (không tự so HW-vs-HW):**
   - Trong [tb_gemm_os_all_sizes.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_gemm_os_all_sizes.sv): toàn bộ GEMM, SLQ (`sw_logq_vec`), V-Transpose và FlashAttention (`sw_flash_attn_vec`) được tính độc lập bằng SW và đối chiếu chéo với chuẩn dấu phẩy động `real` / `$exp()` (`sat=0`, xác minh `resc_valid > 0`).
   - Trong [tb_llama3_1b_configurable.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_llama3_1b_configurable.sv): kích thích thực & trọng số (`34.2 MB`) trích xuất từ hook PyTorch của `NousResearch/Llama-3.2-1B` Layer 1 qua [gen_llama3_1b_configurable.py](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/gen_llama3_1b_configurable.py), đối chiếu đồng thời với **PyTorch FP32** và **PyFQ (FP4+Log8 Fake-Quant)**.
3. **Kiểm tra 100% đầu ra, không dùng `force` giả & có bảo vệ biên DRAM:**
   - Kiểm tra đủ 100% tile/token (bao gồm 32/32 tiles ở `4B` trong `tb_gemm_os_all_sizes.sv` và toàn bộ vector đầu ra trên `DRAM_OUT` trong `tb_llama3_1b_configurable.sv`).
   - Canaries `8'hA5` ở 64 B đầu/cuối cùng poison `8'h5A` trước khi chạy, cộng thêm monitor `inflight_out_cnt_q <= 4`.

### 1.2. Tối ưu chống OOM & chạy từng testcase riêng lẻ
- **Trong [run_llama3_1b_cfg.sh](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/run_llama3_1b_cfg.sh):**
  - Biên dịch **1 lần duy nhất** trên Linux ext4 (`/tmp/obj_llama3_cfg/Vtb_llama3_1b_cfg`), tách 3 pha: (1) `verilator --cc` → (2) prebuild PCH ở `-O2` và compile các file coroutine `initial` TB (`_eval_initial__TOP__Vtiming`) ở `-O0` không xung đột PCH → (3) compile RTL với `OPT_FAST="-O2" -j 4`.
  - Đo thực tế bằng `/usr/bin/time -v` cho cả build + chạy `Prefill M=32, N_KV=32`: đỉnh RAM **`1,883,388 KB` (~`1.80 GB`), `Swaps: 0`**; mọi lượt chạy mô phỏng dùng lại binary cache tiêu thụ cố định **`277 MB` RAM** (`Nice = 19`, chạy tuần tự).
- **Trong [run_gemm_os.sh](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/run_gemm_os.sh):**
  - Hỗ trợ chọn từng testcase/subtest lúc runtime (`+TEST=...`), xuất waveform `.fst` riêng (`tb_gemm_os_test_<CASE>.fst`) và gắn tín hiệu marker `tb_test_id` (`logic [7:0]`).

---

## 2. Phần 1 — Kết quả Kiểm chứng Hệ thống `tb_gemm_os_all_sizes.sv` (16/16 Subtests PASS)

- **File TB & Script:** [tb_gemm_os_all_sizes.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_gemm_os_all_sizes.sv) (`2232` dòng) | [filelist_gemm_os_all_sizes.f](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/filelist_gemm_os_all_sizes.f) | [run_gemm_os.sh](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/run_gemm_os.sh)
- **Kết quả chạy gộp (`./run_gemm_os.sh ALL`):**
  - **Waveform:** [tb_gemm_os_all_sizes.fst](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/tb_gemm_os_all_sizes.fst) (`124M`) | **Walltime:** `49.678 s` (`alloced 341 MB`)
  - **`cpu_mem_req_valid active cyc : 0`** | **`Total errors : 0`**
  - **DMA active / rd / wr:** `41,107` cyc / `27,918` rd beats / `744` wr beats
  - **DRAM stalls / TCDM WO stalls:** `10,631` / `1,138` | **`Max inflight_out_cnt : 3` (`<= 4`)**

| Test | Kịch bản kiểm tra | Toán học (`SW-vs-Real / exp()`) & Stress | Kết quả bit-exact |
| :--- | :--- | :--- | :--- |
| **Test 1** | Decode `M=1, N=32, K=128`, `K_TILES=4` (`+9B/+18B` unaligned) | `SQNR = 30.12 dB`, `Cos = 0.999536`, `sat = 0` | **PASS** (`8/8` FP16 + `2/2` FP4 words) |
| **Test 2A** | Prefill `M=32, N=32, K=64`, `out=0`, `act_lin`, `dma=1` | `SQNR = 49.21 dB`, `Cos = 0.999995`, `sat = 0` | **PASS** (`256/256` words) |
| **Test 2B** | Prefill `out=0`, `act_16bank`, `dma=1` | *(chung bộ stimulus Test 2)* | **PASS** (`256/256` words) |
| **Test 2C** | Prefill `out=1` (FP4 linear), `dma=1` | *(chung bộ stimulus Test 2)* | **PASS** (`64/64` pkts) |
| **Test 2D** | Prefill `out=2` (FP4 16-bank scatter), `dma=1` | *(chung bộ stimulus Test 2)* | **PASS** (`64/64` pkts) |
| **Test 2E** | Prefill `out=3` (Fused-V Transpose + SLQ, `K_TILES=2`), `dma=1` | Đối chiếu SW Transpose + SLQ | **PASS** (`64/64` cols) |
| **Test 3** | Weight Ring Reuse `M=64, N=64, K=128` (4x reuse) | `SQNR = 32.59 dB`, `Cos = 0.999729`, `sat = 0` | **PASS** (`512` W-words fetched, `1024/1024` words) |
| **Test 4A** | Macro-DMA Ping-Pong có phần dư `M=48, N=48, K=1024` | `SQNR = 54.30 dB`, `Cos = 0.999999`, `sat = 0` | **PASS** (`144/144` pkts, `9/9` tiles) |
| **Test 4B** | `>288 KB` TCDM Scale `M=128, N=64, K=4096` (`432 KB`) | `SQNR = 54.38 dB`, `Cos = 0.999999`, `sat = 0` | **PASS** (**ALL `32/32` tiles / `512/512` pkts**, `rd=23040, wr=80` beats) |
| **Test 5A** | Backpressure thật (~50% DRAM stall + TCDM conflict), `out=0` | `SQNR = 36.42 dB`, `Cos = 0.999886`, `sat = 0`; `DRAM stalls=520, WO stalls=908` | **PASS** (`512/512` words) |
| **Test 5B** | Backpressure thật, `out=2` (FP4 scatter) | *(chung suite Test 5, không `force`)* | **PASS** (`128/128` pkts) |
| **Test 5C** | Backpressure thật, `out=3` (Fused-V `K_TILES=2`) | *(chung suite Test 5, không `force`)* | **PASS** (`128/128` cols) |
| **Test 6A-1** | FlashAttn `KV_Pin=1` + GQA 2:1 + Sep-KV (Non-Causal) | `SQNR = 20.59 dB`, `Cos = 0.997182`, `sat = 0`; `182` online rescale events | **PASS** (`768/768` words, `DRAM rd=66` beats) |
| **Test 6A-2** | FlashAttn Chunked Causal (`kv_past=16`) | `98` online rescale events | **PASS** (`768/768` words) |
| **Test 6B** | FlashAttn Macro-DMA Q/O + KV-Pin + Causal + GQA + Remainder + 4D FP4 | `SQNR = 23.87 dB`, `Cos = 0.998764`, `sat = 0` | **PASS** (`320/320` pkts) |
| **Test 6C** | HW Decode V-Transpose (`op_mode=2'b10`, `V_NB=4`, 16-bank & 1-bank) | Đối chiếu SW V-Transpose + SLQ | **PASS** (`64/64` pkts) |

### Kết quả chạy từng Subtest độc lập (không compile lại)

| Lệnh `+TEST` | Kết quả (`errors`, `cpu_mem`) | Walltime | File `.fst` xuất ra | Dung lượng |
| :--- | :---: | :---: | :--- | :---: |
| `./run_gemm_os.sh 1` | `0`, `0` | `0.110 s` | [tb_gemm_os_test_1.fst](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/tb_gemm_os_test_1.fst) | `343K` |
| `./run_gemm_os.sh 2D` | `0`, `0` | `0.134 s` | [tb_gemm_os_test_2D.fst](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/tb_gemm_os_test_2D.fst) | `867K` |
| `./run_gemm_os.sh 2E` | `0`, `0` | `0.138 s` | [tb_gemm_os_test_2E.fst](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/tb_gemm_os_test_2E.fst) | `907K` |
| `./run_gemm_os.sh 4A` | `0`, `0` | `3.024 s` | [tb_gemm_os_test_4A.fst](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/tb_gemm_os_test_4A.fst) | `7.2M` |
| `./run_gemm_os.sh 5A` | `0`, `0` | `0.316 s` | [tb_gemm_os_test_5A.fst](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/tb_gemm_os_test_5A.fst) | `1.6M` |
| `./run_gemm_os.sh 5B` | `0`, `0` | `0.262 s` | [tb_gemm_os_test_5B.fst](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/tb_gemm_os_test_5B.fst) | `1.5M` |
| `./run_gemm_os.sh 5C` | `0`, `0` | `0.268 s` | [tb_gemm_os_test_5C.fst](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/tb_gemm_os_test_5C.fst) | `1.5M` |
| `./run_gemm_os.sh 6A-2` | `0`, `0` | `1.409 s` | [tb_gemm_os_test_6A-2.fst](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/tb_gemm_os_test_6A-2.fst) | `5.3M` |
| `./run_gemm_os.sh 6B` | `0`, `0` | `1.449 s` | [tb_gemm_os_test_6B.fst](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/tb_gemm_os_test_6B.fst) | `7.7M` |
| `./run_gemm_os.sh 6C` | `0`, `0` | `0.151 s` | [tb_gemm_os_test_6C.fst](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/tb_gemm_os_test_6C.fst) | `333K` |

---

## 3. Phần 2 — Sweep 1 Layer `NousResearch/Llama-3.2-1B` (`32 / 64 / 128 / 256` tokens)

- **File TB & Script:** [tb_llama3_1b_configurable.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_llama3_1b_configurable.sv) | [gen_llama3_1b_configurable.py](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/gen_llama3_1b_configurable.py) | [run_llama3_1b_cfg.sh](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/run_llama3_1b_cfg.sh) | [filelist_llama3_1b_cfg.f](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/filelist_llama3_1b_cfg.f)
- **Kiến trúc Layer 1 thực:** `D_MODEL=2048`, `D_FF=8192`, `H_Q=32`, `H_KV=8` (GQA 4:1), `D_K=D_V=64`.
- **Trạng thái chung toàn bộ 11/11 lượt chạy:** **`failed = 0`**, **`cpu_mem_req_valid active cyc = 0`**, **`DRAM canaries PASS`**, **`alloced = 277 MB`**.

### 3.1. Prefill Sweep (`M` input tokens & Chunked Causal `N_KV` context)

#### Hiệu năng phần cứng & lưu lượng DRAM

| Cấu hình (`PAST`) | `Q_T` / `Q_SUB` | `Pre-Attn` cyc | `Attn` cyc | `Total` cyc | `MAC` cyc | `DMA` cyc | DRAM rd beats (`Bytes`) | DRAM wr beats (`Bytes`) | MaxInflight | Checks (`Pass/Fail`) | Walltime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M=32, KV=32** (`0`) | `4 / 4` | `1,030,892` | `62,293` | `8,676,577` | `7,643,136` | `1,231,570` | `1,078,776` (`69,041,664`) | `11,664` (`746,496`) | `3` | `2366 / 0` | `99.333 s` |
| **M=32, KV=128** (`96`) | `4 / 4` | `1,030,892` | `241,813` | `8,856,097` | `7,766,016` | `1,232,572` | `1,079,784` (`69,106,176`) | `11,664` (`746,496`) | `3` | `2366 / 0` | `102.681 s` |
| **M=32, KV=256** (`224`) | `4 / 4` | `1,030,892` | `481,173` | `9,095,457` | `7,929,856` | `1,234,044` | `1,081,128` (`69,192,192`) | `11,664` (`746,496`) | `3` | `2366 / 0` | `103.722 s` |
| **M=64, KV=64** (`0`) | `8 / 8` | `2,061,784` | `243,093` | `17,471,657` | `15,368,192` | `2,464,146` | `2,157,552` (`138,083,328`) | `23,328` (`1,492,992`) | `3` | `4720 / 0` | `196.138 s` |
| **M=128, KV=128** (`0`) | `16 / 8` | `4,123,568` | `957,205` | `35,414,329` | `31,064,064` | `4,930,562` | `4,315,104` (`276,166,656`) | `46,656` (`2,985,984`) | `3` | `9428 / 0` | `387.763 s` |
| **M=256, KV=256** (`0`) | `32 / 8` | `8,247,136` | `3,810,837` | `72,725,081` | `63,438,848` | `9,863,810` | `8,630,208` (`552,333,312`) | `93,312` (`5,971,968`) | `3` | `18844 / 0` | `793.886 s` |

#### Độ chính xác từng tầng qua chuỗi Prefill (`SQNR` / `CosineSim` / `MaxAbs`)

| Cấu hình | `PyFQ` vs `FP32` | `RMSNorm 1` vs FP16 | `Residual 1` vs PyFQ | `Residual 1` vs FP32 | `RMSNorm 2` vs PyFQ | `RMSNorm 2` vs FP32 | `Final` vs PyFQ | `Final` vs FP32 | Log |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **M=32, KV=32** | *(pre-generated)* | `35.63 dB` / `0.999863` (`1.1621`) | `25.81 dB` / `0.998690` (`0.1684`) | `22.98 dB` / `0.997497` (`0.1730`) | `14.94 dB` / `0.984052` (`0.5264`) | `12.17 dB` / `0.969766` (`0.5225`) | `36.69 dB` / `0.999951` (`2.6250`) | `19.29 dB` / `0.998860` (`26.9808`) | [sim.log](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/sim_llama3_1b_prefill_m32_kv32/sim.log) |
| **M=32, KV=128** | `11.09 dB` / `0.960773` | `40.57 dB` / `0.999956` (`0.1672`) | `16.79 dB` / `0.989837` (`0.1313`) | `12.07 dB` / `0.971493` (`0.1914`) | `15.51 dB` / `0.986070` (`0.5137`) | `11.11 dB` / `0.961430` (`0.6436`) | `13.35 dB` / `0.977966` (`0.1658`) | `10.32 dB` / `0.954554` (`0.1915`) | [sim.log](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/sim_llama3_1b_prefill_m32_kv128/sim.log) |
| **M=32, KV=256** | `20.38 dB` / `0.998995` | `38.01 dB` / `0.999925` (`0.5664`) | `23.68 dB` / `0.997872` (`0.1802`) | `19.78 dB` / `0.994824` (`0.1577`) | `15.57 dB` / `0.986263` (`0.7171`) | `11.68 dB` / `0.966348` (`0.7175`) | `41.14 dB` / `0.999963` (`0.7500`) | `20.50 dB` / `0.998978` (`21.5077`) | [sim.log](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/sim_llama3_1b_prefill_m32_kv256/sim.log) |
| **M=64, KV=64** | `20.04 dB` / `0.998866` | `37.48 dB` / `0.999911` (`1.1621`) | `23.24 dB` / `0.997641` (`0.1684`) | `19.97 dB` / `0.995036` (`0.2321`) | `15.14 dB` / `0.984805` (`0.5264`) | `11.99 dB` / `0.968474` (`0.5225`) | `35.53 dB` / `0.999918` (`2.6250`) | `19.26 dB` / `0.998798` (`26.9808`) | [sim.log](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/sim_llama3_1b_prefill_m64_kv64/sim.log) |
| **M=128, KV=128** | `19.98 dB` / `0.998772` | `38.17 dB` / `0.999924` (`1.1621`) | `21.19 dB` / `0.996234` (`0.1684`) | `17.48 dB` / `0.991324` (`0.2321`) | `15.17 dB` / `0.984957` (`0.5264`) | `11.59 dB` / `0.965569` (`0.6611`) | `34.00 dB` / `0.999857` (`2.6250`) | `19.20 dB` / `0.998676` (`26.9808`) | [sim.log](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/sim_llama3_1b_prefill_m128_kv128/sim.log) |
| **M=256, KV=256** | `20.11 dB` / `0.998800` | `38.09 dB` / `0.999923` (`1.1621`) | `20.22 dB` / `0.995305` (`0.1873`) | `16.59 dB` / `0.989353` (`0.4388`) | `14.96 dB` / `0.984212` (`0.7589`) | `11.47 dB` / `0.964599` (`0.6863`) | `34.50 dB` / `0.999839` (`2.6250`) | `19.63 dB` / `0.998694` (`26.9808`) | [sim.log](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/sim_llama3_1b_prefill_m256_kv256/sim.log) |

---

### 3.2. Decode Sweep (`M=1, N_KV = 32, 64, 128, 256` + `DRAM_BP=1`)

#### Hiệu năng phần cứng & lưu lượng DRAM (`MaxInflight = 1`, `Checks = 157 / 0`)

| Cấu hình (`BP`) | `Attn` cyc | `Total` cyc | `MAC` cyc | `DMA` cyc | DRAM stalls | DRAM rd beats (`Bytes`) | DRAM wr beats (`Bytes`) | Walltime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M=1, KV=32** (`BP=0`) | `4,890` | `1,347,510` | `238,848` | `755,895` | `0` | `535,440` (`34,268,160`) | `416` (`26,624`) | `20.884 s` |
| **M=1, KV=32** (`BP=1`) | `4,938` | `1,461,451` | `238,848` | `755,076` | `196,748` | `535,440` (`34,268,160`) | `416` (`26,624`) | `21.458 s` |
| **M=1, KV=64** (`BP=0`) | `9,690` | `1,352,310` | `240,128` | `756,423` | `0` | `535,776` (`34,289,664`) | `416` (`26,624`) | `20.753 s` |
| **M=1, KV=128** (`BP=0`) | `19,290` | `1,361,910` | `242,688` | `757,479` | `0` | `536,448` (`34,332,672`) | `416` (`26,624`) | `21.015 s` |
| **M=1, KV=256** (`BP=0`) | `38,490` | `1,381,110` | `247,808` | `759,591` | `0` | `537,792` (`34,418,688`) | `416` (`26,624`) | `21.263 s` |

#### Độ chính xác từng tầng qua chuỗi Decode (`SQNR` / `CosineSim` / `MaxAbs`)

| Cấu hình | `PyFQ` vs `FP32` | `RMSNorm 1` vs FP16 | `Residual 1` vs PyFQ | `Residual 1` vs FP32 | `RMSNorm 2` vs PyFQ | `RMSNorm 2` vs FP32 | `Final` vs PyFQ | `Final` vs FP32 | Log |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **KV=32** (`BP=0`) | `8.25 dB` / `0.925717` | `30.58 dB` / `0.999609` (`0.4897`) | `13.42 dB` / `0.977623` (`0.0527`) | `9.83 dB` / `0.948074` (`0.0874`) | `11.62 dB` / `0.966894` (`0.2303`) | `8.37 dB` / `0.926257` (`0.3220`) | `11.59 dB` / `0.965947` (`0.0492`) | `8.77 dB` / `0.935324` (`0.1258`) | [sim.log](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/sim_llama3_1b_decode_m1_kv32/sim.log) |
| **KV=32** (`BP=1`) | `8.25 dB` / `0.925717` | `30.58 dB` / `0.999609` (`0.4897`) | `13.42 dB` / `0.977623` (`0.0527`) | `9.83 dB` / `0.948074` (`0.0874`) | `11.62 dB` / `0.966894` (`0.2303`) | `8.37 dB` / `0.926257` (`0.3220`) | `11.59 dB` / `0.965947` (`0.0492`) | `8.77 dB` / `0.935324` (`0.1258`) | [sim_bp.log](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/sim_llama3_1b_decode_m1_kv32/sim_bp.log) |
| **KV=64** (`BP=0`) | `11.04 dB` / `0.960045` | `48.04 dB` / `0.999996` (`0.0331`) | `15.67 dB` / `0.988243` (`0.0372`) | `11.39 dB` / `0.967750` (`0.0605`) | `15.66 dB` / `0.986623` (`0.1991`) | `11.43 dB` / `0.964296` (`0.3383`) | `12.62 dB` / `0.974925` (`0.0513`) | `9.85 dB` / `0.949816` (`0.0672`) | [sim.log](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/sim_llama3_1b_decode_m1_kv64/sim.log) |
| **KV=128** (`BP=0`) | `12.19 dB` / `0.971014` | `38.78 dB` / `0.999939` (`0.1335`) | `19.60 dB` / `0.994523` (`0.0289`) | `13.23 dB` / `0.977230` (`0.0722`) | `18.03 dB` / `0.992350` (`0.1062`) | `11.90 dB` / `0.969276` (`0.2849`) | `15.43 dB` / `0.985577` (`0.0737`) | `11.60 dB` / `0.966174` (`0.1049`) | [sim.log](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/sim_llama3_1b_decode_m1_kv128/sim.log) |
| **KV=256** (`BP=0`) | `12.37 dB` / `0.970596` | `39.09 dB` / `0.999985` (`0.0673`) | `17.86 dB` / `0.992866` (`0.0430`) | `13.21 dB` / `0.980892` (`0.0781`) | `14.64 dB` / `0.982828` (`0.2153`) | `10.40 dB` / `0.953804` (`0.4517`) | `12.40 dB` / `0.973386` (`0.0380`) | `10.73 dB` / `0.958538` (`0.0483`) | [sim.log](file:///c:/Users/nguyen/Desktop/NPU_LOG/sim/sim_llama3_1b_decode_m1_kv256/sim.log) |

---

## 4. Phân tích Tổng hợp Xu hướng Scale & Sai số (`32 → 256` tokens)

1. **Tính đúng đắn & độ ổn định Macro-DMA / FlashAttention:**
   - Ở `Prefill M=256, N_KV=256`, toàn bộ pha FlashAttention chạy trọn vẹn trong **1 lần launch** (`Q_T=32, Q_SUB=8`, tự chia 4 macro Q/O slice ping-pong vượt xa `TCDM 288 KB`), tổng toàn layer luân chuyển `552.33 MB` đọc / `5.97 MB` ghi DRAM với `0` lỗi canary và `Max inflight_out_cnt = 3`.
   - Khi bật DRAM backpressure (`Decode KV=32, BP=1`), dù phát sinh `196,748` DRAM stall cycles, toàn bộ 7 chỉ số sai số khớp **bit-exact tuyệt đối** so với `BP=0`.
2. **Quy luật mở rộng chu kỳ tính toán:**
   - **Decode Attention:** tăng tuyến tính chính xác `4,800` cycles cho mỗi lần tăng thêm 32 context (`4,890 → 9,690 → 19,290 → 38,490`).
   - **Prefill Self-Attention:** tăng đúng quy luật `O(M · N_KV)` (`62,293 → 243,093 → 957,205 → 3,810,837`); phần **Chunked Prefill (`M=32`, `N_KV = 32 / 128 / 256`)** tăng tuyến tính theo context (`62,293 → 241,813 → 481,173`).
3. **Ổn định sai số số học:**
   - Trên chuỗi Prefill `M = 32 → 64 → 128 → 256`, `Final vs PyFQ CosineSim` duy trì ở **`0.999951 → 0.999918 → 0.999857 → 0.999839`** và `Final vs FP32` ổn định **`0.998860 → 0.998798 → 0.998676 → 0.998694`** (`~19.20–19.63 dB`), khớp trần lý thuyết của Fake-Quant PyTorch (`~0.9988`).
   - Trên các token riêng lẻ có biên độ nhỏ không chứa token BOS (`Prefill M=32, KV=128` có `max_abs=0.3667`, các điểm `Decode M=1` có `max_abs=0.21–0.30`), sai số gốc của bản thân định dạng FP4+Log8 là `Cos = 0.9257..0.9710` (`PyFQ vs FP32`), còn RTL bám sát PyFQ (`Cos = 0.9659..0.9856` ở Decode, `0.9780` ở `M=32, KV=128`).

---

## 5. Lệnh tái lập nhanh

```bash
cd /mnt/c/Users/nguyen/Desktop/NPU_LOG/sim

# 1) System Zero-Cheat Suite (full hoặc từng subtest + .fst)
./run_gemm_os.sh ALL
./run_gemm_os.sh 6A-2

# 2) Llama-3.2-1B Configurable Layer Sweep (dùng binary cache trong /tmp/obj_llama3_cfg)
./run_llama3_1b_cfg.sh --mode prefill --m-tokens 32 --n-kv 32 --skip-gen
./run_llama3_1b_cfg.sh --mode prefill --m-tokens 256 --n-kv 256 --skip-gen
./run_llama3_1b_cfg.sh --mode decode --n-kv 256 --skip-gen
./run_llama3_1b_cfg.sh --mode decode --n-kv 32 --skip-gen --dram-bp 1
```
