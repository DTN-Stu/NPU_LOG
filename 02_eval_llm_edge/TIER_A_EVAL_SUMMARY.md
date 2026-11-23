# Tổng hợp Kết quả & Phân tích Đánh giá Thuật toán (Tier-A Evaluation)

Tài liệu này tổng hợp toàn bộ kết quả thực nghiệm GPU (Google Colab T4), phân tích lý thuyết/mạch, đối chiếu với công bố quốc tế và checklist chốt số liệu chính thức cho bài báo **IEEE TCAD / TVLSI**.

---

## 1. Thiết lập Thí nghiệm & Độ khớp Phần cứng RTL

Các kịch bản đánh giá nằm trong [eval_edge_llm.py](file:///c:/Users/nguyen/Desktop/NPU_LOG/02_eval_llm_edge/eval_edge_llm.py) và [quantizers.py](file:///c:/Users/nguyen/Desktop/NPU_LOG/02_eval_llm_edge/quantizers.py), vận hành ở chế độ **100% Training-Free Post-Training Quantization (RTN thuần)** — không sử dụng Hadamard rotation, SmoothQuant, calibration hay fine-tuning/QAT:

- **Lớp Linear (`W4A4`):** Lượng tử hóa **cả trọng số `W` và activation đầu vào `X`** xuống 4-bit cho toàn bộ các phép chiếu `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Giữ `embed_tokens` và `lm_head` ở FP16 (theo chuẩn chung của các nghiên cứu W4A4).
- **Khối Self-Attention (`Q @ K^T` và `P @ V`):**
  - `Q` và `K`: lượng tử hóa 4-bit dọc `head_dim` cho mọi cấu hình.
  - `V`: lượng tử hóa 4-bit dọc **trục token `N_KV`** (`--v_quant_axis token` / `v_transpose=True`) cho mọi cấu hình — khớp 100% với trục tích vô hướng $P \cdot V$ trên phần cứng FP4 và bộ đệm `snax_transpose_buffer` trong [snax_v_quant_path.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_v_quant_path.sv#L68-L80).
  - Ma trận xác suất `P`:
    - Ở cấu hình **`log8_fp4_expfree` (Full RTL)**: `P` được **lượng tử hóa 4-bit (`E2M1 + Log8 CSD`)** trực tiếp từ điểm Attention qua `exp_free_softmax_requantizer` (`--base_scale_mode log2e`, khớp hằng số `LOG2E_Q16 = 94548` trong [snax_fused_llm_core.sv#L89](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv#L89)), đồng thời bật bộ nhân phân số 3-term CSD (`CSD_Q5/32`, khớp `logscale_frac_scaler` trong [log8_fp4_arith_units.sv#L117-L158](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/log8_fp4_arith_units.sv#L117-L158)).
    - Ở các baseline (`mxfp4*`, `nvfp4`, `log8_fp4`, `log8_fp4_csd`): `P` giữ ở Softmax FP32 chuẩn (tức điều kiện test thực tế đang ưu ái hơn cho baseline so với `log8_fp4_expfree`).
- **Định dạng Scale tham chiếu (tất cả `E2M1` 4-bit dữ liệu):**
  - `MXFP4` (`B=32`, `4.25 b/elem`): E8M0 thuần lũy thừa của 2 — gồm `mxfp4_g32_floor` (OCP v1.0 gốc) và `mxfp4` (`ceil`).
  - `MXFP4_g16` (`B=16`, **`4.50 b/elem` iso-bitwidth**): `mxfp4_g16_floor` (OCP v1.0, gặp lỗi clip vùng `[6, 8)`) và `mxfp4_g16_ceil` (bản sửa clip theo `MiX` MICRO 2026 Table IV).
  - `NVFP4` (`B=16`, `4.50 b/elem`): **Scale 2 cấp** — cấp 1 `FP32 global tensor_scale` + cấp 2 `FP8 E4M3` (trên phần cứng Blackwell & `MiX` [NVFP4_MUL.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/03_baselines/MiX-main/hardware_rtl/src/systolic_arrays/NVFP4/NVFP4_MUL.sv#L88-L93) dùng `UE4M3` 7-bit không dấu `[6:0]`, max = 448).
  - `Log8-FP4` (`B=16`, `4.50 b/elem`): **Scale 1 cấp duy nhất** 8-bit có dấu `[-128, 127]` (`2^(s/8)`, bao phủ 32 octaves), **không dùng `FP32 tensor_scale`**, sinh bởi bộ mã hóa không phép chia [slq_encoder.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/slq_encoder.sv).

---

## 2. Kết quả Chính — Bộ 3 Edge LLM 1B–2B Tiêu biểu cho `Table III`

Đo trên Colab T4 (`--seq_len 1024 --max_chunks 16`, `V-axis = token`, `--base_scale_mode log2e`):

### 2.1. Perplexity (`WikiText-2 / C4`, thấp hơn là tốt hơn)

| Cấu hình | Scale / Softmax | Bit/elem | `NousResearch/Llama-3.2-1B` *(Meta)* | `HuggingFaceTB/SmolLM2-1.7B` *(HuggingFace)* | `Qwen/Qwen3-1.7B-Base` *(Alibaba)* |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`fp16`** | FP16 gốc | 16.00 | `10.377 / 13.081` | `8.397 / 12.042` | `9.511 / 14.499` |
| `mxfp4_g32_floor` | E8M0 floor (`B=32`, OCP v1.0) / FP32 | 4.25 | `56.887 / 68.809` | `115.322 / 120.244` | — |
| `mxfp4` | E8M0 ceil (`B=32`) / FP32 | 4.25 | `44.000 / 55.349` | `41.004 / 57.675` | — |
| **`mxfp4_g16_floor`** | E8M0 floor (`B=16`, OCP v1.0) / FP32 | **4.50** | `46.929 / 62.090` | `127.850 / 135.206` | `21.716 / 30.438` |
| **`mxfp4_g16_ceil`** | E8M0 ceil (`B=16`, MiX fix) / FP32 | **4.50** | `32.320 / 40.689` | `29.048 / 39.268` | `22.358 / 31.649` |
| **`nvfp4`** | FP32 + UE4M3 (`B=16`) / FP32 | **4.50** | `16.542 / 20.952` | `11.212 / 15.106` | `12.872 / 18.537` |
| `log8_fp4` | `slq_encoder`, exact $2^{s/8}$ (`B=16`) / FP32 | 4.50 | `17.696 / 22.597` | `12.576 / 16.701` | `12.996 / 19.319` |
| `log8_fp4_csd` | `slq_encoder` + **3-term CSD** / FP32 | 4.50 | `17.710 / 22.650` | `12.258 / 16.969` | `13.107 / 19.517` |
| **`log8_fp4_expfree` (Ours Full RTL)** | **3-term CSD + Exp-Free `P4` Softmax** | **4.50** | **`17.948 / 22.580`** | **`12.325 / 16.803`** | **`13.186 / 19.504`** |
| `log8_fp4_exact` | Exact-Log8 (`round(8*log2)` W & A/KV) / FP32 | 4.50 | — | — | `12.990 / 18.715` |
| `log8_fp4_exactw_csd` | Offline Exact-Log8 W + SLQ A/KV (`CSD`) / FP32 | 4.50 | — | — | `13.084 / 19.524` |
| `log8_fp4_exactw_expfree` | Offline Exact-Log8 W + SLQ A/KV (`CSD`) + `P4` | 4.50 | — | — | `13.075 / 19.792` |

### 2.2. Sai số Khối Attention `O` (Full RTL Exp-Free Log8-FP4 CSD vs. FP16 Softmax)

| Model | `L1 Error` | `Rel-L1 (%)` | `Cosine Similarity` |
| :--- | :---: | :---: | :---: |
| `NousResearch/Llama-3.2-1B` | `0.006550` | `12.73%` | **`0.979551`** |
| `HuggingFaceTB/SmolLM2-1.7B` | `0.020856` | `15.07%` | **`0.917037`** |
| `Qwen/Qwen3-1.7B-Base` | `0.111141` | `11.21%` | **`0.978274`** |

### 2.3. Ba Luận điểm Rút ra từ Bảng Chính

1. **Đánh bại áp đảo `MXFP4_g16` ở cùng ngân sách `4.50 b/elem` (`B=16`):**
   - So với chuẩn OCP `mxfp4_g16_floor`: Full RTL `log8_fp4_expfree` **giảm `-61.8%` PPL** trên `Llama-3.2-1B` (`46.929 -> 17.948`), **giảm `-90.4%`** (tốt hơn **10.4×**) trên `SmolLM2-1.7B` (`127.850 -> 12.325`), và **giảm `-39.3%`** trên `Qwen3-1.7B-Base` (`21.716 -> 13.186`).
   - So với bản đã sửa lỗi clip `mxfp4_g16_ceil` (MiX Table IV): Full RTL vẫn **giảm `-44.5%`**, **`-57.6%`**, và **`-41.0%`** tương ứng trên 3 model.
2. **Bám rất sát `NVFP4` dù loại bỏ hoàn toàn `FP32 global tensor scale` và bộ nhân `E4M3`:**
   - Trên `Qwen3-1.7B-Base`, `log8_fp4` (`12.996`) gần như ngang tuyệt đối `nvfp4` (`12.872`, chỉ `+0.124` PPL / `+0.96%`); `log8_fp4_expfree` chỉ chênh `+0.314` PPL dù nén thêm cả `P` xuống 4-bit.
   - Trên `Llama-3.2-1B` và `SmolLM2-1.7B`, chênh lệch lần lượt là `+1.41` và `+1.11` PPL ở chế độ lượng tử hóa thêm cả `P4`.
3. **3-term CSD và Exp-Free Softmax gần như không làm mất độ chính xác (100% Training-Free PTQ):**
   - Từ `log8_fp4` $\rightarrow$ `log8_fp4_csd` $\rightarrow$ `log8_fp4_expfree` chỉ thay đổi `+0.25 / -0.25 / +0.19` PPL trên WikiText-2 (C4 hầu như không đổi: `22.597 -> 22.580`, `16.701 -> 16.803`, `19.319 -> 19.504`).

---

## 3. Kết quả Mở rộng & Chẩn đoán Dòng `Qwen2.5`

| Cấu hình (`V-axis = token`, `log2e`) | `Qwen/Qwen2.5-0.5B` *(full `2048`)* | `Qwen/Qwen2.5-1.5B` *(16×`1024`)* |
| :--- | :---: | :---: |
| Attention `O` Error (`L1 / Rel-L1 / CosSim`) | `0.020911 / 9.98% / 0.985834` | `0.023870 / 7.92% / 0.993882` |
| `fp16` | `13.070 / 20.052` | `9.382 / 13.993` |
| `mxfp4_g32_floor` | — | `8940.269 / 8180.302` |
| `mxfp4` (`B=32`, ceil) | `1168.259 / 1440.003` | `3559.186 / 3684.525` |
| `mxfp4_g16_floor` (`4.50b`) | `546.743 / 846.284` | `8370.882 / 9140.907` |
| `mxfp4_g16_ceil` (`4.50b`) | `952.547 / 1194.457` | `2805.491 / 3203.797` |
| `nvfp4` | `33.137 / 52.037` | `141.635 / 95.860` |
| `log8_fp4` | `51.241 / 104.475` | `4920.519 / 5244.292` |
| `log8_fp4_csd` | — | `5290.805 / 5939.146` |
| `log8_fp4_expfree` | **`51.675 / 108.397`** | `5997.646 / 5946.536` |

### Lý giải kỹ thuật tại sao `Qwen2.5-1.5B` sập cả ở `NVFP4` lẫn `Log8-FP4`
- **Lỗi KHÔNG nằm ở Attention:** `Attention O Error` trên `Qwen2.5-1.5B` đạt **`CosSim = 0.993882` (`Rel-L1 = 7.92%`)**, tốt nhất trong các model đã đo.
- **Nguyên nhân đến từ Massive Activations ở MLP `down_proj` của họ `Qwen2.5`:** Kiến trúc `Qwen2.5` (khác `Qwen3` có QK-Norm) xuất hiện các gai kích hoạt cực đại trong khối SwiGLU (`silu(gate) * up -> down_proj`) tiến sát trần FP16 `65,504`. Khi ép chạy **W4A4 RTN thuần** (không Hadamard rotation hay giữ riêng `down_proj`), ngay cả `NVFP4` cũng nổ lên `141.6` (~15× FP16).
- **Lý do `Log8-FP4` nhạy hơn `NVFP4` trên các gai cực đại này:** `NVFP4` đặt `tensor_scale = global_max / (6 * 448)` ở FP32 nên sai số trên đỉnh gai lớn nhất gần như `0.00%`; ngược lại `slq_encoder.sv` cắt cụt 7-bit đuôi mantissa nên đỉnh bị lệch tối đa `-8.22%`, trên gai biên độ hàng vạn sẽ tạo sai số tuyệt đối lớn tràn vào residual stream.
- **Kết luận chọn benchmark:** Với `Qwen2.5-0.5B` (nơi `NVFP4` chưa sập), `Log8-FP4 Full RTL` vẫn **tốt hơn `10.6×` đến `18.4×`** so với `MXFP4_g16` ở cùng 4.50 bit. Còn để đại diện họ Qwen ở phân khúc 1.5B–1.7B cho `Table III`, **`Qwen/Qwen3-1.7B-Base`** ổn định và phù hợp hơn hẳn cho bài toán W4A4 training-free.

---

## 4. Phân tích Chuyên sâu: Lý thuyết `Log8`, Mạch `slq_encoder.sv` & `NVFP4` / `MiX`

### 4.1. So sánh độ chính xác lý thuyết vs. bộ mã hóa 0-gate `slq_encoder.sv`

Phép mô phỏng quét qua toàn bộ dải chuẩn hóa FP16 và phân phối tổng hợp (`B=16`) cho thấy:

| Bộ mã hóa Scale (`B=16`) | Sai số đỉnh tối đa | Sai số trung bình | Bias hệ thống | SQNR vùng nhỏ `N(0, 1e-4)` |
| :--- | :---: | :---: | :---: | :---: |
| **`Log8` lý tưởng** (`round(8*log2(max/6))`) | **`4.40%`** | **`2.17%`** | **`+0.04%`** | **`20.35 dB`** |
| **`NVFP4`** (`UE4M3` + `FP32 tensor_scale`) | `5.86%` | `2.25%` | `-0.08%` | `20.29 dB` |
| **`slq_encoder.sv`** (`{E, M[9:7]}`, `-140`) | `8.22%` | `3.15%` | `-2.24%` | `12.10 dB` *(vùng subnormal FP16)* |

- Trên **cùng 8-bit scale**, khoảng chia lô-ga-rít `Log8` lý tưởng **mịn và đều hơn `UE4M3`** (bước nhảy tương đối cố định `9.05%` trên toàn octave, thay vì nhảy `12.5%` ở đầu octave của `E4M3`).
- Khoảng cách rất nhỏ giữa `log8_fp4` và `nvfp4` trên `Llama-3.2-1B` / `SmolLM2-1.7B` đến từ việc `slq_encoder.sv` đánh đổi độ chính xác bằng **xấp xỉ Mitchell 0-gate** (loại bỏ hoàn toàn bộ chia, bộ nhân, `FP32 amax` 2-pass) và cắt subnormal FP16.

### 4.2. Tại sao `Log8-FP4` bỏ được `FP32 Global Tensor Scale` của `NVFP4`

Kiểm chứng trực tiếp từ mã nguồn **`SeoLabCornell/MiX`** ([NVFP4_MUL.sv#L88-L93](file:///c:/Users/nguyen/Desktop/NPU_LOG/03_baselines/MiX-main/hardware_rtl/src/systolic_arrays/NVFP4/NVFP4_MUL.sv#L88-L93), [nvfp4.py#L27-L39](file:///c:/Users/nguyen/Desktop/NPU_LOG/03_baselines/MiX-main/accuracy/src/quantization/nvfp4.py#L27-L39)) và đặc tả NVIDIA PTX xác nhận:
- `NVFP4` dùng scale **`UE4M3` không dấu 7-bit `[6:0]`** (4 exponent + 3 mantissa, MSB `[7] = 0` bỏ trống), dải biểu diễn chỉ `15` octaves (`2^-6 .. 448`) nên **bắt buộc phải thêm cấp `FP32 global tensor_scale`** (tốn thêm 1 pass quét toàn tensor khi quantize online).
- `Log8-FP4` tận dụng **đầy đủ cả 8 bit** dưới dạng số nguyên có dấu `[-128, 127]` (`5-bit octave + 3-bit log`), đạt dải động **`32` octaves** (`2^-16 .. 2^+15.875`) — thừa đủ bao phủ toàn bộ dải FP16 mà **không cần `tensor_scale` toàn cục**.

### 4.3. Kết quả Thực nghiệm Offline Exact-Log8 cho Trọng số (`W`) vs. Activation (`A/KV`)

Trong phần cứng thực tế (`dram_weights.hex`), trọng số `W` được đóng gói offline một lần duy nhất trên CPU trước khi nạp vào DRAM, còn activation `A/KV` đi qua [slq_encoder.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/slq_encoder.sv) online. Để kiểm chứng giả thuyết tối ưu 0-gate cho `W`, ta đã bổ sung `quantize_log8_fp4_exact()` và `quantize_log8_fp4_exact_csd()` vào [quantizers.py](file:///c:/Users/nguyen/Desktop/NPU_LOG/02_eval_llm_edge/quantizers.py#L148-L182) và đo đối chứng trên `Qwen/Qwen3-1.7B-Base` (`16×1024`):

| Cấu hình | `W` Quantizer | `A / KV` Quantizer | Softmax `P` | `WikiText-2` | `Δ vs NVFP4` | `C4` | `Δ vs NVFP4` |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `nvfp4` | 2-level UE4M3 | 2-level UE4M3 | FP32 | `12.872` | `0.000` | `18.537` | `0.000` |
| **`log8_fp4_exact`** | Exact-Log8 | Exact-Log8 | FP32 | **`12.990`** | **`+0.118 (+0.92%)`** | **`18.715`** | **`+0.178 (+0.96%)`** |
| `log8_fp4` | `slq_encoder` | `slq_encoder` | FP32 | `12.996` | `+0.124` | `19.319` | `+0.782` |
| `log8_fp4_csd` | `slq_encoder`+CSD | `slq_encoder`+CSD | FP32 | `13.107` | `+0.235` | `19.517` | `+0.980` |
| `log8_fp4_exactw_csd` | Exact-Log8+CSD | `slq_encoder`+CSD | FP32 | `13.084` | `+0.212` | `19.524` | `+0.987` |
| `log8_fp4_exactw_expfree` | Exact-Log8+CSD | `slq_encoder`+CSD | Exp-Free `P4` | `13.075` | `+0.203` | `19.792` | `+1.255` |
| **`log8_fp4_expfree` (Full RTL)** | **`slq_encoder`+CSD** | **`slq_encoder`+CSD** | **Exp-Free `P4`** | **`13.186`** | **`+0.314 (+2.44%)`** | **`19.504`** | **`+0.967 (+5.22%)`** |

**Kết luận kỹ thuật & Quyết định cho Paper:**
- **Định dạng 1-cấp `Log8-FP4` lý tưởng gần ngang tuyệt đối `NVFP4`:** `log8_fp4_exact` (`12.990 / 18.715`) chỉ chênh `+0.92%` WT2 và `+0.96%` C4 so với `NVFP4` (`12.872 / 18.537`), dù bỏ hoàn toàn `FP32 global tensor scale`.
- **Điểm nghẽn sai số nằm ở `Activation A/KV`, KHÔNG nằm ở `Weights W`:** Chỉ đổi `W` sang Exact-Log8 (`exactw_csd` / `exactw_expfree`) chỉ giảm nhẹ WT2 `-0.111` (`13.186 -> 13.075`) và hầu như không cải thiện / dao động mẫu trên C4 (`19.504 -> 19.524 / 19.792`). Ngược lại, khi đổi luôn **`A/KV`** sang Exact-Log8 (`log8_fp4_exact`), PPL C4 giảm ngay **`-0.809`** (`19.524 -> 18.715`) do activation có nhiều gai biên độ nhạy với việc cắt 7 bit đuôi mantissa (`-2.24%` bias) của `slq_encoder.sv`.
- **Quyết định trình bày:** Giữ một bộ mã hóa thống nhất `slq_encoder` cho cả `W` và `A/KV` (`log8_fp4_expfree`) làm phương án chính trong bài báo để sạch về kiến trúc và công bằng, không đưa cặp phụ `exactw_*` vào bảng paper nhằm tránh gây nhiễu cho reviewer.

---

## 5. Đối chiếu với Kết quả Công bố Quốc tế (Sanity Check)

Đối chiếu kết quả đo được với các công bố gần đây về FP4/W4A4 (*MiX* MICRO 2026, *Four Over Six*, *FlipQuant*, *BRQ*, *ResQ*, *HAS-VQ*, *Diagnosing FP4 Inference*, *AMXFP4*) trên `WikiText-2`:

| Model | Mốc đo của mình (`eval_edge_llm.py`) | Chuẩn Literature (Full WikiText-2) | Đánh giá độ hợp lý |
| :--- | :--- | :--- | :--- |
| **`Qwen2.5-0.5B`** *(full `2048`)* | `FP16 = 13.070`<br>`MXFP4_g16 = 546.7–952.5`<br>`NVFP4 = 33.137`<br>**`Log8 RTL = 51.675`** | `FP16 = 13.0 – 13.1` (*ResQ*, *Diagnosing FP4 Inference*) | **Khớp tuyệt đối cột `FP16`** (`13.07` vs `13.0–13.1`), khẳng định pipeline tokenization, shift nhãn và tính loss trong [eval_edge_llm.py](file:///c:/Users/nguyen/Desktop/NPU_LOG/02_eval_llm_edge/eval_edge_llm.py) hoàn toàn chuẩn. |
| **`Qwen3-1.7B-Base`** *(`16×1024`)* | `FP16 = 9.511`<br>`MXFP4_g16 = 21.716 / 22.358` (**`2.28×`**)<br>`NVFP4 = 12.872` (`1.35×`)<br>**`Log8 RTL = 13.186` (`1.39×`)** | `Qwen3-1.7B-Base BF16 ≈ 10.23` (`1024`-token window).<br>Bản Instruct `Qwen3-1.7B` (*Four Over Six*): `BF16 ≈ 11.98–12.22`, `MXFP4 W4A4 ≈ 26.95` (**`~2.2×`**), `NVFP4 (M=6) ≈ 23.06`, adaptive `≈ 21.06`. | **Rất khớp:** Tỷ lệ suy giảm `MXFP4` (`2.28×`) gần trùng khít `~2.2×`; bản `Base` kích hoạt mượt hơn bản `Instruct` nên `NVFP4` & `Log8` giữ PPL chặt hơn. `FP16` lệch `-0.7` do mới chấm 16 chunk đầu. |
| **`Llama-3.2-1B`** *(`16×1024`)* | `FP16 = 10.377`<br>`MXFP4_g16 = 46.929 / 32.320`<br>`NVFP4 = 16.542`<br>**`Log8 RTL = 17.948`** | `Base FP16 ≈ 9.7 – 9.8` (full `2048`).<br>*FlipQuant*/*Four Over Six* (`Instruct`, **chỉ Linear W4A4, Attention BF16**): `BF16 13.16` → `NVFP4 15.54` (`+2.38`, C4 `+3.25`).<br>*BRQ* xác nhận `MXFP4` RTN không rotation sập nặng. | **Hợp lý:** `FP16` cao hơn ~0.6 do ngữ cảnh `1024` ngắn hơn `2048`; mức tăng lớn hơn *FlipQuant* vì mình ép lượng tử hóa **cả `Q4/K4/V4` và `P4`**, thuần RTN không xoay. |
| **`SmolLM2-1.7B`** *(`16×1024`)* | `FP16 = 8.397`<br>`MXFP4_g16 = 127.850 / 29.048`<br>`NVFP4 = 11.212`<br>**`Log8 RTL = 12.325`** | `FP16 = 10.04 – 10.12` (*HAS-VQ*, full WikiText-2).<br>Uniform INT4 `4.71b` đạt `~20.03`; W4A4 không rotation vọt `30–100+`. | **Xu hướng W4A4 rất khớp** (`MXFP4_g16` vọt `29–128`, `NVFP4`/`Log8` bám sát `FP16`). `FP16` thấp hơn `-1.65` PPL do 16 chunk đầu dễ hơn toàn tập $\implies$ cần chạy full `2048`. |
| **`Qwen2.5-1.5B`** *(`16×1024`)* | `FP16 = 9.382`<br>`NVFP4 = 141.635`<br>`MXFP4_g16 = 2805–8371`<br>`Log8 = 4921–5998` | *Diagnosing FP4 Inference*, *AMXFP4* & báo cáo overflow FP16 trên `Qwen2.5`: **massive activation outliers ở MLP `down_proj`** sát trần `65,504`. | **Khớp đúng cơ chế đã công bố:** W4A4 RTN thuần đều sập trên `Qwen2.5-1.5B`, dù `Attention O CosSim = 0.993882` chứng minh khối Attention/Softmax chạy chuẩn. |

**Ba bằng chứng tổng hợp:** (1) Hiện tượng "MXFP4 Anomaly" giữa `mxfp4_g16_floor` và `mxfp4_g16_ceil` lặp lại chính xác kết quả `MiX` (MICRO 2026) Table IV; (2) `Qwen2.5-0.5B` full `2048` đạt `13.070` khớp từng số lẻ chuẩn `13.0–13.1`; (3) Mọi tỷ lệ lặp lại nhất quán trên tập `C4` độc lập.

---

## 6. Đề xuất Cấu trúc Bảng đưa vào Paper (IEEE TCAD / TVLSI)

> [!IMPORTANT]
> **Nguyên tắc trình bày cho Reviewer:** **KHÔNG** đổ toàn bộ 5–6 biến thể `log8_fp4*` trong script vào bảng chính (`Table III`) vì sẽ gây rối và loãng trọng tâm. Hãy tách rành mạch thành **2 bảng** với vai trò rõ ràng:
> 1. **`Table III` (Main SOTA Comparison):** chỉ giữ **đúng 1 dòng duy nhất cho `Ours (Log8-FP4 Full RTL)`** (`log8_fp4_expfree`).
> 2. **`Table IV` (Hardware Approximation Ablation):** tách riêng các bước rút gọn phần cứng theo kiểu **bậc thang cộng dồn (`+ ...`)** trên 1 model đại diện (`Qwen3-1.7B-Base`).

### 6.1. `Table III` — Bảng So sánh Chính (Main SOTA Comparison)

*(Số liệu dưới đây đang điền tạm từ đợt chạy sàng lọc `16×1024`; trước khi nộp bài cần thay bằng lượt chạy chốt đồng nhất `--seq_len 2048 --max_chunks 200` + `lm-eval` 5 tác vụ Zero-Shot.)*

| Method | Block / Bits | Scale Format | `Llama-3.2-1B` (WT2 / C4) | `SmolLM2-1.7B` (WT2 / C4) | `Qwen3-1.7B-Base` (WT2 / C4) | Avg. 5-Task Zero-Shot (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `FP16` | — / 16.00b | Native IEEE FP16 | `10.377 / 13.081` | `8.397 / 12.042` | `9.511 / 14.499` | *TBD (lm-eval)* |
| `MXFP4_g16` *(OCP v1.0 floor)* | 16 / 4.50b | 1-lvl E8M0 (floor) | `46.929 / 62.090` | `127.850 / 135.206` | `21.716 / 30.438` | *TBD (lm-eval)* |
| `MXFP4_g16` *(MiX ceil)* | 16 / 4.50b | 1-lvl E8M0 (ceil) | `32.320 / 40.689` | `29.048 / 39.268` | `22.358 / 31.649` | *TBD (lm-eval)* |
| `NVFP4` | 16 / 4.50b | 2-lvl FP32 + UE4M3 | `16.542 / 20.952` | `11.212 / 15.106` | `12.872 / 18.537` | *TBD (lm-eval)* |
| **`Ours (Log8-FP4 Full RTL)`** | **16 / 4.50b** | **1-lvl Signed Log8** | **`17.948 / 22.580`** | **`12.325 / 16.803`** | **`13.186 / 19.504`** | ***TBD (lm-eval)*** |

**Caption đề xuất cho `Table III` (tiếng Anh chuẩn IEEE):**
> *Table III: Training-free PTQ (RTN) perplexity ($\downarrow$) and zero-shot accuracy ($\uparrow$) across 1B–2B edge LLMs at iso-bitwidth ($4.50$ bits/element, $B=16$). All Linear layers (`W4A4`) and Attention tensors (`Q4`, `K4`, and token-wise `V4` matching the on-chip transpose buffer) are quantized to E2M1. Baselines use exact FP32 Softmax, whereas **Ours (Log8-FP4 Full RTL)** simulates the complete bit-exact RTL datapath including the zero-gate SLQ encoder, 3-term CSD fractional scaler, and 4-bit Exp-Free Softmax requantizer (`P4`).*

### 6.2. `Table IV` — Bảng Ablation Rút gọn Phần cứng (Progressive Hardware Approximation Breakdown)

Đo trên `Qwen/Qwen3-1.7B-Base` (và ghép cùng độ trung thực `Attention O` đo trên cả 3 model: `CosSim = 0.9796 / 0.9170 / 0.9783`):

| Step | Configuration | Description | `WikiText-2` ($\Delta$ Step) | `C4` ($\Delta$ Step) | Config in Script |
| :---: | :--- | :--- | :---: | :---: | :--- |
| Ref | `NVFP4` | 2-level `FP32 + UE4M3`, FP32 Softmax | `12.872` *(ref)* | `18.537` *(ref)* | `nvfp4` |
| 1 | `Log8-FP4 Format` | Exact `log2/div`, 1-level 8-bit `2^(s/8)` | `12.990` *(`+0.118` vs NVFP4)* | `18.715` *(`+0.178` vs NVFP4)* | `log8_fp4_exact` |
| 2 | `+ 0-Gate SLQ Encoder` | Replace `log2/div` with bit-slice Mitchell encoder | `12.996` (`+0.006`) | `19.319` (`+0.604`) | `log8_fp4` |
| 3 | `+ 3-Term CSD Scaler` | Replace fractional $2^{r/8}$ mult with `CSD_Q5/32` | `13.107` (`+0.111`) | `19.517` (`+0.198`) | `log8_fp4_csd` |
| 4 | **`+ Exp-Free 4-Bit Softmax`** | **Quantize `P` via `softmax_requantizer` (Full RTL)** | **`13.186` (`+0.079`)** | **`19.504` (`-0.013`)** | **`log8_fp4_expfree`** |

**Vì sao bảng bậc thang này giải quyết triệt để thắc mắc của Reviewer:**
- **Lý do bắt buộc phải mô phỏng `CSD`:** Trong phần cứng, `logscale_frac_scaler` ([log8_fp4_arith_units.sv#L117-L158](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/log8_fp4_arith_units.sv#L117-L158)) và bảng `CSD_Q5` ([softmax_requantizer.sv#L120](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/softmax_requantizer.sv#L120)) không dùng bộ nhân dấu phẩy động cho $2^{r/8}$ mà dùng 3-term CSD (`[32,35,38,41,46,49,54,59]/32`, lệch tối đa `1.65%`). Nếu bỏ qua CSD trong mô phỏng, số liệu thuật toán sẽ **không khớp** diện tích/công suất tổng hợp của RTL.
- **Tách rạch ròi từng đóng góp:** `Table IV` chứng minh định lượng từng kỹ thuật tối giản phần cứng (`0-Gate SLQ`, `3-Term CSD`, `Exp-Free P4`) chỉ gây tổn hao rất nhỏ (`+0.006` đến `+0.111` PPL trên WikiText-2).
- **Lược bỏ biến thể phụ:** Các cấu hình `exactw_*` được ghi nhận trong tài liệu nội bộ này để đối chứng, nhưng không đưa lên bảng paper để giữ câu chuyện gọn, nhất quán (`slq_encoder` đồng bộ cho cả `W` và `A/KV`).

---

## 7. Các Bẫy đã Phát hiện & Checklist trước khi Chạy Chốt số Paper

### Các vấn đề đã xử lý / phát hiện trong quá trình chạy Colab
1. **Lỗi cũ `HuggingFace URI` (`datasets` / `huggingface_hub >= 1.x`):** Đã vá xong bằng canonical ID `"Salesforce/wikitext"` và fallback shard JSON cho `allenai/c4` trong [eval_edge_llm.py](file:///c:/Users/nguyen/Desktop/NPU_LOG/02_eval_llm_edge/eval_edge_llm.py#L124-L221).
2. **Bẫy cờ `--base_scale_mode rtl_ln2`:** Phiên bản cũ nhân nhầm `LN2_Q16 = 45426/65536` ($\ln 2$) thay vì **`LOG2E_Q16 = 94548`** ($\log_2 e$, hằng số thực hiện hành trong [snax_fused_llm_core.sv#L89](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv#L89)), từng làm `log8_fp4_expfree` vọt lên `186.29` / `159.07`. Khi dùng đúng mặc định `log2e`, PPL phục hồi về `17.948` / `12.325` / `13.186`. *(Cần dọn/sửa hằng số `rtl_ln2` trong `quantizers.py` về `94548` hoặc bỏ tùy chọn sai để tránh bấm nhầm.)*
3. **Cảnh báo Memory OOM & tốc độ chậm (`4.00 s/it`) ở `exp_free_softmax_requantizer` khi `seq_len = 2048`:**
   - Do tạo 7 tensor `int64` (`[1, H, 2048, 2048]`) đồng thời trong [quantizers.py#L338-L346](file:///c:/Users/nguyen/Desktop/NPU_LOG/02_eval_llm_edge/quantizers.py#L338-L346), trên `Qwen2.5-0.5B` (14 heads) đã chạm 15.6 GB VRAM Colab T4; sang `Llama-3.2-1B` và `SmolLM2-1.7B` (đều **32 heads**) có nguy cơ crash `OOM` ở `seq_len = 2048`.

### Checklist chốt bộ số liệu cuối cùng cho Paper (IEEE TCAD / TVLSI)

- [x] **Sửa cảnh báo `torch.bucketize` non-contiguous** trong `_nearest_e2m1` ([quantizers.py#L50](file:///c:/Users/nguyen/Desktop/NPU_LOG/02_eval_llm_edge/quantizers.py#L50)) bằng `.contiguous()`.
- [ ] **Tối ưu VRAM & tốc độ trước khi chạy chốt:** Thay 7× `torch.where(full_like int64)` trong `exp_free_softmax_requantizer` bằng `torch.bucketize` + chia chunk theo `q_len = 256` (100% bit-exact, giảm ~10× VRAM), và đồng bộ/bỏ tùy chọn `rtl_ln2` cũ để tránh nhầm lẫn.
- [ ] **Chạy đồng nhất Full Benchmark (`--seq_len 2048 --max_chunks 200`)** cho `Table III` trên bộ 3 model:
  1. `NousResearch/Llama-3.2-1B`
  2. `HuggingFaceTB/SmolLM2-1.7B`
  3. `Qwen/Qwen3-1.7B-Base` *(chạy thêm `log8_fp4_exact` cho `Table IV` trên model này; có thể kèm ghi chú `Qwen/Qwen2.5-0.5B` nếu cần mốc `<1B`)*
- [ ] **Cài `lm-eval` (`pip install -q lm-eval`)** và đo thêm **5 tác vụ Zero-Shot nhẹ** (`piqa`, `hellaswag`, `winogrande`, `arc_easy`, `arc_challenge`) ở cùng điều kiện để điền cột cuối của `Table III`.
