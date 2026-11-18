# Bản thiết kế Chi tiết Paper IEEE TCAD / TVLSI: End-to-End Log8-FP4 Fused LLM Accelerator Subsystem

Tài liệu này là **khung xương chuẩn (Paper Blueprint)** cho bài báo gửi **IEEE TCAD** hoặc **IEEE TVLSI** dựa trên hệ thống hoàn chỉnh [snax_llm_top_clean.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv), kiến trúc 7 khối của [snax_fused_llm_core.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv), các số liệu RTL/Synthesis đã đo trong [ABLATION_ANALYSIS_REPORT.md](file:///c:/Users/nguyen/Desktop/NPU_LOG/04_ablation_rtl/ABLATION_ANALYSIS_REPORT.md), và kết quả End-to-End trong [ARCHITECTURE_AND_SIM_REPORT.md](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/ARCHITECTURE_AND_SIM_REPORT.md).

---

## Phần 1: Định vị Bài báo (TCAD vs. TVLSI) & Tên đề tài gợi ý

* **Nếu nộp IEEE TCAD:** Nhấn mạnh **Algorithm–Architecture–System Co-Design** (đồng thiết kế định dạng `Log8-FP4`, chứng minh toán học *Exp-free Base-2 Softmax*, mô hình hóa xung đột crossbar 22×32, pipeline `Core -> fused_fifo -> SFU`, và `WRITE_FP4_4D` scheduling).
* **Nếu nộp IEEE TVLSI:** Nhấn mạnh **Microarchitecture & Circuit-Level PPA** (mạch CSD 3-term, `log8_fp4_mac_array` không bộ nhân, 1W2R D-Latch RMW accumulator, operand isolation, `LOG2E_Q16` CSD, và kết quả tổng hợp ASIC + SAIF power).

**Gợi ý Title:**
> *"An End-to-End 72-bit Log8-FP4 Transformer Accelerator Subsystem with Exp-Free Fused FlashAttention and Hardware-Offloaded Non-Linear SFU"*

---

## Phần 2: Cấu trúc Chi tiết Từng Section (Tổng 6 Sections, ~11–13 trang IEEE 2-cột)

### Section I: Introduction (~1.5 trang)
* **Nội dung cần viết:**
  1. *Bối cảnh:* Bùng nổ Edge LLM (1B–3B) bị nghẽn bởi cả **Memory Wall** (trọng số + KV-cache + activation trung gian) lẫn **Compute Wall** (tính toán GEMM + Softmax `exp()`/`div` + các phép phi tuyến RoPE/RMSNorm/SwiGLU).
  2. *Hạn chế của SOTA hiện tại:*
     * Weight-only (W4A16) vẫn để KV-cache & activation ở FP16.
     * `MXFP4` (E8M0, 1 octave) sụp đổ độ chính xác trên Edge LLM nhạy cảm với outlier.
     * `NVFP4` (E4M3) đạt độ chính xác tốt nhưng phải gánh bộ nhân scale `4×4` + `13×8`, scale 2 cấp (FP32 tensor scale), bộ nén tốn chia/làm tròn FP8, và **không thể làm Exp-free Softmax** nếu không có mạch tính $2^x$.
     * Các accelerator FP4 trước đây chỉ làm GEMM hoặc bỏ lại Softmax / RoPE / RMSNorm / SwiGLU cho CPU hoặc SFU rời rạc.
  3. *Tóm tắt 4 đóng góp chính (gạch đầu dòng kèm số liệu định lượng):*
     * **C1:** Định dạng **72-bit Log8-FP4** (4.50 bit/elem) + online `slq_encoder` không dùng bộ nhân/chia + TCDM 32×1024×72b (0% phân mảnh scale).
     * **C2:** **Decoupled LogScale-Lazy MAC + D-Latch RMW (`log8_fp4_mac_array` + `rmw_latch_acc_buffer`)** — cắt `8.9%–10.3%` Datapath Area chỉ từ Log8-Scale và `-21.3%` toàn engine khi kết hợp D-Latch trên ASAP7 7nm.
     * **C3:** **Exp-Free Fused FlashAttention** (`softmax_requantizer` + `LOG2E_Q16` CSD + `p_fifo` + `attn_o_rescaler_bank` + `l_accumulator`) — triệt tiêu `exp()` và traffic ma trận $P$ ra SRAM/DRAM, đạt `37.59 dB` SQNR (`CosSim = 0.999913`) trên `Llama-3.2-1B`.
     * **C4:** **End-to-End Subsystem (`snax_llm_top_clean`)** tích hợp `snax_llm_sfu_opt` (`RMSNorm`, `RoPE`, `SwiGLU`, `Residual`), `Core -> fused_fifo -> SFU`, và `WRITE_FP4_4D` (triệt tiêu hoàn toàn CPU reorder, đạt `93.8%–96.8%` MAC utilization).
* **Hình ảnh:** *Fig. 1:* Motivation chart — so sánh khoảng cách SQNR/PPL (`MXFP4` sụp đổ vs `NVFP4` vs `Log8-FP4`) và chi phí phần cứng của từng format.

---

### Section II: Background & Mathematical Co-Design (~1.5 trang)
* **Nội dung cần viết:**
  1. **Định dạng 72-bit Log8-FP4 (`{scale_log8[7:0], 16×E2M1[63:0]}`):** Giải thích công thức giải mã $x_i = (-1)^{s_i} \cdot \text{E2M1}(m_i) \cdot 2^{\text{scale}/8}$ và tại sao bước $1/8$ octave (`S4.3`) vừa đủ độ mịn ngang NVFP4, vừa khớp 1 word SRAM 72-bit.
  2. **Thuật toán Online Encoding không cần `log2()` (`slq_encoder`):** Giải thích phép trích `{E_fp16, M_fp16[9:7]}` + trừ `140` + 7 ngưỡng so sánh.
  3. **Chứng minh Toán học Exp-Free Base-2 Softmax Requantizer:**
     * Chuyển cơ số $e^s = 2^{s \cdot \log_2 e}$ với `LOG2E_Q16 = 94548`.
     * Chứng minh sự triệt tiêu $1/l$ và $m$ trong phép lượng tử hóa $p_t / S_{\text{blk}} = 6 \cdot 2^{-(s_{\text{bmax}} - s_t)}$, và công thức log-domain trực tiếp cho block scale $p_{\text{blk}} = \text{round}(8(s_{\text{bmax}} - m)) - 21$.
     * **Luận điểm phản biện quan trọng:** Chứng minh tại sao *chỉ `Log8-Scale` mới thực hiện được Exp-free Softmax triệt để 100%* (NVFP4 cần mạch $2^x$ để đổi ra E4M3 tuyến tính, còn MXFP4 E8M0 bị sai số scale tới 1 octave).
* **Hình & Bảng:** *Fig. 2:* Cấu trúc gói 72-bit Log8-FP4 so sánh với MXFP4 & NVFP4; *Table I:* Bảng ngưỡng lượng tử hóa 7 mức E2M1 và bảng hằng số CSD `m_r`.

---

### Section III: Proposed Hardware Microarchitecture (~3.5 trang)
Chia làm 2 phần rõ ràng khớp với cây RTL đã tái cấu trúc:
* **III-A. Top-Level Subsystem (`snax_llm_top_clean`):**
  * Crossbar **22 Masters × 32 Banks** + TCDM 72-bit (288 KB).
  * Đường dữ liệu hợp nhất: `u_core` $\leftrightarrow$ `u_sfu` (`fused_fifo` 4-entry, `sfu_in_a_src=1`) $\rightarrow$ `u_log_quantizer` (`slq_encoder`) / `u_transpose_buffer` (16×16) $\rightarrow$ `u_write_out` (`MODE_FP4_4D`).
* **III-B. Fused LLM Core 7 Khối (`snax_fused_llm_core`):**
  1. `u_mac_array` (`log8_fp4_mac_array`): Stage 1A (`dot_product_16` + `scale_adder`) $\rightarrow$ Stage 1B (`logscale_frac_scaler` CSD 3-term + `bidirectional_shifter`) $\rightarrow$ Stage 2 (`accum_adder` 32-bit saturating).
  2. `u_acc_buffer` (`rmw_latch_acc_buffer`): 1W2R D-Latch (`ACC_DEPTH = 16`: 8 O-slots + 8 S-slots) + `slot_cleared` + Port B non-blocking rolling drain.
  3. `u_attn_rescaler` (`attn_o_rescaler_bank`): 16× `vsq_rescaler_32b` + 2-stage delay + operand isolation (`acc_rescale_req_q1`).
  4. `u_softmax_req` (`softmax_requantizer`) + CSD 7-term `LOG2E_Q16 = 94548` (operand-isolated by `sm_in_valid`).
  5. `u_p_fifo` (transient 64-entry) & `u_l_accumulator` (1 shared LZC + `Shifter << LZC` + LUT `1/(1+f)` cho `M_Q` slots).
  6. `u_final_norm` (`final_normalizer`): Unified FP16 drain cho cả GEMM (`bypass_inv_l=1`) và FlashAttention ($O \times 1/l$), operand-isolated.
* **III-C. Non-Linear SFU (`snax_llm_sfu_opt`):** Tóm tắt kiến trúc FP20 nội bộ (`RMSNorm` 2-pass, `RoPE`, `SwiGLU` PWL sigmoid, `Residual`) và cơ chế batching `sfu_num_vecs`.
* **Hình ảnh:**
  * *Fig. 3:* Sơ đồ toàn Subsystem `snax_llm_top_clean` (22 Masters, TCDM, Core, SFU, `fused_fifo`, Transpose, Quantizer, 4D Write-Out).
  * *Fig. 4:* Sơ đồ chi tiết bên trong `snax_fused_llm_core` (7 khối).
  * *Fig. 5:* Chi tiết vi kiến trúc 1 Lane của `PE Lane` (với 2 đường pipeline `Stage 1A | Stage 1B | Stage 2`) và sơ đồ `l_accumulator`.

---

### Section IV: End-to-End Transformer Layer Mapping & Dataflow (~1.5 trang)
* **Nội dung cần viết:**
  1. Cách ánh xạ trọn vẹn **1 LLaMA Transformer Layer** (Norm1 $\rightarrow$ QKV + RoPE + V-Transpose $\rightarrow$ FlashAttention $\rightarrow$ O-proj + Res1 $\rightarrow$ Norm2 $\rightarrow$ Gate/Up/SwiGLU/Down + Res2) lên `snax_llm_sequencer` + `snax_llm_sfu_opt`.
  2. Cơ chế **4D Write-Out (`WRITE_FP4_4D`)** tự động sắp xếp layout `QPKT_BANK` cho $Q$ và rải 16 bank cho $K$, xóa bỏ `cpu_reorder`.
  3. Cơ chế **Credit-based Ring Buffer DMA Prefetch** ẩn độ trễ DRAM.
* **Hình & Bảng:** *Fig. 6:* Biểu đồ Timeline thực thi (Gantt chart) cho thấy sự gối đầu (overlap) giữa `u_core`, `u_sfu` (qua `fused_fifo`), `u_log_quantizer` (`4D WO`), và `DMA`.

---

## Phần 3: Thiết kế Chi tiết Section V — Evaluation (~4 trang, 5 Tiểu mục)

Đây là phần quyết định bài báo có được nhận hay không. Bạn nên chia **Section V** thành **5 tiểu mục (V-A đến V-E)** đi từ **Algorithm $\rightarrow$ RTL Accuracy $\rightarrow$ MAC Microarchitecture $\rightarrow$ Subsystem Breakdown $\rightarrow$ End-to-End System & SOTA**:

---

### V-A. Experimental Setup (~0.4 trang)
* **Algorithmic:** PyTorch/CUDA bit-exact model (`02_eval_llm_edge`) khớp 100% RTL (`LOG2E_Q16`, `slq_encoder`, `softmax_requantizer`). Đánh giá trên Edge LLMs: `Qwen2.5-0.5B/1.5B/3B`, `Llama-3.2-1B/3B` (hoặc `SmolLM2-1.7B`); tập dữ liệu: WikiText-2, C4, PIQA, HellaSwag, WinoGrande, ARC-E/C, MMLU.
* **Hardware Synthesis & Power:** ASAP7 7nm RVT TT (`PVT_0P7V_25C`) @ **500 MHz (`2.0 ns`)**, Synopsys Design Compiler (`compile_ultra`), Gate-level SAIF simulation trên VCS (`4,000` capture cycles), phân tích cả kích thích chuẩn (`LLM-like` / `real LLM trace`) lẫn uniform-random.
* **System Simulation:** QuestaSim 2024.1 cycle-accurate trên `tb_snax_llm_layer_mid.sv` / `tb_snax_llm_layer_fused.sv` với lát cắt Layer 1 thực từ `NousResearch/Llama-3.2-1B`.

---

### V-B. Algorithmic Accuracy & RTL Bit-Exact Verification (~0.8 trang)
* **Bảng & Đồ thị cần đưa ra:**
  1. *Table II (PPL & Zero-shot Ablation):* So sánh `FP16` vs. `MXFP4_g16 (floor)` vs. `MXFP4_g16 (ceil)` vs. `NVFP4` vs. `Log8-FP4` vs. **`Log8-FP4 + Exp-Free Softmax (Full RTL)`** (có thể thêm cột `+ K-Smoothing` giống `MiX` để tăng tính khách quan).
  2. *Table III (RTL End-to-End Verification trên dữ liệu thực `Llama-3.2-1B`):* Lấy trực tiếp số liệu bạn đã đo:
     * `LLM_eval vs FP32`: `38.69 dB`, `CosSim = 0.999932`
     * `Golden (PyFQ) vs FP32`: `37.27 dB`, `CosSim = 0.999906`
     * **`RTL (snax_llm_top_clean) vs FP32`**: **`37.59 dB`**, **`CosSim = 0.999913`**, `max_abs = 0.00384`, `0/4096 fail`
     * `RTL vs Golden (PyFQ)`: `39.33 dB`, `CosSim = 0.999942`
  3. *Fig. 7:* Đường **Per-Block SQNR (dB) theo Block Dynamic Range Ratio (`Max/Mean`)** (học từ Fig. 5 của `MiX`) cho `Log8-FP4` vs `MXFP4_g16` vs `NVFP4`.
* **Ý nghĩa & Biện luận chống bẻ:**
  * Chứng minh `MXFP4` sụp đổ trên Edge LLM nhỏ (`1280.16` vs `35.56` PPL trên `Qwen2.5-0.5B`), trong khi `Log8-FP4` giữ chất lượng sát `NVFP4` ở cùng mức **4.50 bit/elem**.
  * Chứng minh `Exp-Free Softmax` hầu như không gây mất mát độ chính xác (`35.56 -> 35.92` PPL trên `Qwen2.5-0.5B`, `CosSim = 0.9952` cho riêng Attention $O$, và `37.59 dB` end-to-end trên `Llama-3.2-1B`).

---

### V-C. Microarchitectural MAC & Accumulator Ablation (Iso-Throughput 256 MACs/cycle @ 500 MHz) (~0.8 trang)
* **Bảng & Đồ thị cần đưa ra:**
  1. *Table IV (6-Config Ablation từ `04_ablation_rtl`):* Báo cáo đủ 2 lớp thông số: **Total Engine** (`Area`, `Power SAIF`, `Area Eff.`, `Power Eff.`, `Composite Eff.`) và **Datapath-Only (tách riêng Accumulator)**.
  2. *Table V (Cross-Validation với `MiX` 28nm):* Bảng đối chiếu chéo tỷ lệ tăng của `NVFP4` so với `MXFP4_g16` (`MiX`: `+22.6%` Area, `+7.7%` Power vs. Của bạn: `+21.8%` Datapath Area, `+7.0%` Power).
  3. *Fig. 8:* Biểu đồ **PPA–Accuracy Pareto Frontier** (trục hoành: `Composite PE Efficiency = TOPS/mm² × TOPS/W`; trục tung: `SQNR / Avg. Accuracy`) cho `#1`–`#6`.
* **Ý nghĩa & Biện luận chống bẻ (Rất quan trọng!):**
  * **Chặn trước câu hỏi *"Area giảm nhờ D-Latch chứ không phải Log8"*:** Chỉ thẳng vào cột **Datapath-Only**: ngay cả khi không dùng D-Latch, `Log8` (`#6` / `#4`) đã tự cắt **`-8.9% đến -10.3%` Area** và **`-7.8% đến -9.9%` Power** so với `NVFP4` (`#2`), và cắt **50% overhead** so với `MXFP4` (`#1`).
  * **Chủ động giải thích vai trò D-Latch (`#5`):** Vì FlashAttention yêu cầu `ACC_DEPTH = 16` (chiếm `42%–54%` diện tích), việc kết hợp `Log8 CSD` + `1W2R D-Latch` (`-28.1%` acc area) mới đẩy mức tiết kiệm toàn engine lên **`-21.3%` Area (`30.41 TOPS/mm²`, `Composite = 1045.2`)**.
  * **Lưu ý:** Trước khi chốt số liệu cuối, chạy lại `#1`–`#6` với **cùng 1 bộ cờ compile** và dùng **dải scale LLM hợp lý / trace LLM thật** trong `tb_ablation_saif.sv` để tránh bão hòa INT32 giả tạo ở `#4`/`#5`.

---

### V-D. Module-Level Overhead & Full Subsystem Breakdown (~0.8 trang)
* **Bảng & Đồ thị cần đưa ra:**
  1. *Table VI (Module-Level Comparison):*
     * `expfree_softmax_requantizer` (kèm `LOG2E_Q16` CSD) vs. **Standard LUT/PWL FP16 Softmax** (chứng minh tiết kiệm diện tích & năng lượng khi bỏ `exp()` + `1/l` tại $P$).
     * `slq_encoder` (`u_log_quantizer`) vs. **`MiX_MX_Quantizer`** (và/hoặc NVFP4 quantizer).
  2. *Table VII & Fig. 9 (Full Subsystem Area & SAIF Power Breakdown của `snax_llm_top_clean`):*
     * Tách bạch **2 phần**:
       * **Total Logic Area & Power:** `u_mac_array`, `u_acc_buffer`, `u_attn_rescaler`, `u_softmax_req + p_fifo + l_acc`, `u_final_norm`, `u_sfu`, `u_log_quantizer + transpose`, `interconnect_22x32 + sequencer + write_out`.
       * **On-chip Memory:** `TCDM 288 KB (32×1024×72b)`.
* **Ý nghĩa & Biện luận chống bẻ:**
  * Không giấu TCDM, nhưng **tách riêng Logic vs. TCDM SRAM** (giống Table V của `MiX`) để reviewer thấy rõ phần đóng góp của từng khối trong core (`u_sfu`, `u_softmax_req`, `u_log_quantizer` chỉ chiếm tỷ lệ nhỏ nhưng giải quyết toàn bộ bottleneck hệ thống).
  * Chỉ ra `slq_encoder` tuy nhỏ (~vài % Logic Area) nhưng giúp **nén 3.56× toàn bộ TCDM/DRAM traffic** (từ 16b FP16 xuống 4.5b Log8-FP4).

---

### V-E. End-to-End Performance, Memory Traffic & SOTA Comparison (~1.2 trang)
* **Bảng & Đồ thị cần đưa ra:**
  1. *Fig. 10 (Cycle Waterfall & Optimization Ablation trên `tb_snax_llm_layer_mid / fused`):*
     * So sánh tổng Wall Cycles toàn layer LLaMA qua 4 mốc:
       1. *Base (CSR per-vec SFU, CPU reorder)*: `197,388` cycles
       2. *+ `WRITE_FP4_4D`*: `194,348` cycles (`-3,040` cyc, `0` CPU reorder, `HW Job = 39,141` cyc, MAC util `93.8%–96.8%`)
       3. *+ SFU Batching (`256`) & `Core -> fused_fifo -> SFU`* *(cần chạy thêm)*
       4. *+ `SFU -> u_log_quantizer` & `DMA Prefetch (cfg_enable_dma=1)`* *(cần chạy thêm)*
  2. *Fig. 11 (Memory Traffic Reduction):* So sánh tổng lượng byte đọc/ghi **TCDM & DRAM** của 1 layer FlashAttention + FFN giữa: *FP16 Baseline* vs. *Unfused FP4 (ghi $P$ và FP16 trung gian qua TCDM)* vs. ***Ours (Fused `p_fifo` + `fused_fifo` + `4D WO` + 72-bit Log8-FP4)***.
  3. *Table VIII (So sánh với các nghiên cứu SOTA):* Đối chiếu `snax_llm_top_clean` với các kiến trúc tiêu biểu (cần chuẩn hóa hoặc ghi chú rõ công nghệ nm):

| Nhóm đối thủ | Kiến trúc SOTA cần so sánh | Điểm khác biệt & Lợi thế của thiết kế chúng ta |
| :--- | :--- | :--- |
| **1. FP4 / Microscaling Accelerators** | **MiX** *(MICRO 2026)*<br>**VSQ** *(JSSC 2023)*<br>**BitMoD** *(HPCA 2025)*<br>**OliVe** *(ISCA 2023)* | Các thiết kế này chủ yếu chỉ tăng tốc GEMM (bỏ ngỏ Softmax/SFU); thiết kế của ta **tích hợp trọn vẹn GEMM + Exp-Free FlashAttention + SFU** trên cùng 1 subsystem, và **#5** đạt `Area Eff.` / `Composite Eff.` vượt `NVFP4` & `MXFP4_g16` ở chế độ Iso-Throughput. |
| **2. FlashAttention / Exp-free Softmax** | **FACT** *(ISCA 2023)*<br>**Softermax** *(ASPLOS 2021)*<br>**SpAtten** *(HPCA 2021)* | `Softermax` vẫn phải tính $1/l$ và lưu $P$ ở độ chính xác cao; thiết kế của ta **triệt tiêu cả $1/l$ khi sinh $P$** nhờ Log8-FP4 và **truyền thẳng $P$ qua `p_fifo`** (0 traffic SRAM/DRAM cho $P$). |
| **3. RISC-V / SNAX Subsystems** | **SNAX / Snitch** *(Colleman et al., TCAS-I / TVLSI)*<br>**Gemmini** *(DAC 2021)* | `SNAX` gốc dùng TCDM 64-bit (bị phân mảnh nếu chứa scale riêng) và phụ thuộc CPU cho non-linear/reorder; ta tùy biến **TCDM 72-bit aligned (0% fragmentation)** + **`snax_llm_sfu_opt` + `4D Write-Out`** cho 0 chu kỳ CPU reorder. |

#### Đối chiếu Mạch số Chi tiết: `Softermax` (ASPLOS 2021) vs. Thiết kế của chúng ta

| Tiêu chí mạch số | **Softermax (ASPLOS 2021)** | **Circuit của chúng ta (`softmax_requantizer` + Fused Core)** |
| :--- | :--- | :--- |
| **1. Tính $2^x$ cho từng phần tử** | Tách phần nguyên (shift) + dùng **LUT (`clut`) nội suy phần lẻ** $2^{\text{frac}}$ cho **từng phần tử** $K$. | **Không có LUT $2^x$:** Thu gọn phép tính $p_t/S = 6\cdot 2^{-(s_{\text{bmax}}-s_t)}$ thành **1 phép trừ nguyên + so sánh 7 ngưỡng cố định**. |
| **2. Vị trí phép chuẩn hóa $1/l$** | Chuẩn hóa/chia $1/l$ trên **cả $N_{\text{seq}}$ phần tử $p_t$** của mỗi hàng ma trận $P$. | **Bỏ hoàn toàn $1/l$ khi sinh $P$:** Chỉ nhân $1/l$ **1 lần ở cuối trên vector $O$ ($D_V$ phần tử)** tại `final_normalizer`. |
| **3. Lưu trữ & Định dạng $P$** | Xuất xác suất tuyến tính (INT8/FP16), vẫn cần lưu đệm $P$ và bộ nhân 8b/16b cho $P \cdot V$. | Xuất trực tiếp **E2M1 FP4 + Log8 scale** đẩy thẳng vào `p_fifo` cho `log8_fp4_mac_array` (`fp4_mult_lut`, không bộ nhân), **0 traffic SRAM/DRAM cho $P$**. |
| **4. Yêu cầu Fine-tuning (QAT)** | Thay thẳng $e^x \rightarrow 2^x$ không có hệ số chuyển cơ số, thường **cần fine-tuning**. | Có bộ chuyển cơ số **CSD 7-term `LOG2E_Q16 = 94548`** $\rightarrow$ **100% Training-Free (PTQ)**, đạt `CosSim = 0.999913` trên `Llama-3.2-1B`. |

> **Luận điểm phản biện:** *"Softermax mới chỉ xấp xỉ `exp()` bằng bảng LUT $2^{\text{frac}}$ nhưng vẫn phải tính $2^x$ và chuẩn hóa $1/l$ trên toàn bộ $N_{\text{seq}}$ phần tử của ma trận $P$ (và thường cần fine-tuning). Thiết kế của chúng tôi tiến thêm một bước nhờ đồng thiết kế với **Log8-FP4**: triệt tiêu cả bảng LUT $2^x$ lẫn phép chia $1/l$ ở tầng sinh $P$ bằng mạch so sánh 7 ngưỡng đơn giản, đồng thời giữ tính **training-free** nhờ bộ chuyển cơ số CSD `LOG2E_Q16`."*

---

## Phần 4: Tóm tắt Lộ trình Việc cần chạy trước khi bắt đầu viết

1. **Chạy Simulation End-to-End tối ưu (`tb_snax_llm_layer_fused.sv`):** Bật SFU batching + `Core -> fused_fifo -> SFU` + `SFU -> Log-Quantizer` + `cfg_enable_dma = 1` để lấy con số Wall Cycles mới (kéo từ `194k` xuống sát `30k–45k` cycles).
2. **Chạy Synthesis đồng nhất & Full Subsystem (`snax_llm_top_clean`):**
   * Đồng nhất cờ compile cho `#1`–`#6` trong `04_ablation_rtl` (dùng kích thích LLM thật).
   * Tổng hợp `snax_llm_top_clean` (bỏ `vsq_matmul_engine.sv`) để lấy bảng Breakdown Table VII.
   * Tổng hợp thêm `softmax_requantizer` vs `LUT FP16 Softmax` (Table VI).
3. **Chạy Algorithmic Eval trên Edge LLM 1B–3B (`02_eval_llm_edge`):** Chạy `Llama-3.2-1B/3B` & `Qwen2.5-1.5B/3B` (PPL + Zero-shot) để hoàn thiện Table II và Fig. 7.
