# Phân tích Hệ thống Toàn diện: Vì sao Tách `Readback + DMA-WB` và Chia 3 Sequencer là Sai lầm Kiến trúc — Lộ trình Hợp nhất 1 Unified Sequencer End-to-End Cho SNAX-FP4 LLM Subsystem

> **Phạm vi & Cam kết kỹ thuật:** Tài liệu này chỉ phân tích trực tiếp dựa trên mã nguồn RTL và testbench hiện hành trong thư mục `01_snax_llm_fp4_subsystem/` và `tb/`. Mọi tên module, tên cổng, độ rộng bit, tham số, thanh ghi CSR, địa chỉ TCDM và hành vi chu kỳ đều trích xuất đối chiếu 1-1 từ RTL thực tế — không suy đoán, không độn chữ rỗng.

---

## MỤC LỤC TỔNG THỂ

1. [Chương 1. Bóc trần Nguồn gốc Câu hỏi: Vì sao Tách `Readback + DMA-WB` là Sai lầm Kiến trúc](#chương-1-bóc-trần-nguồn-gốc-câu-hỏi-vì-sao-tách-readback--dma-wb-là-sai-lầm-kiến-trúc)
2. [Chương 2. Giải phẫu Hạ tầng Dữ liệu Cố định: 25-Master Crossbar & TCDM 288 KB (`32 Banks × 1024 Words × 72b`)](#chương-2-giải-phẫu-hạ-tầng-dữ-liệu-cố-định-25-master-crossbar--tcdm-288-kb-32-banks--1024-words--72b)
3. [Chương 3. Giải phẫu Sâu `Readback` (`op_mode = 2'b11`): Trùng lặp Mạch với `SFU_BYPASS` & Vì sao Phải Xóa Bỏ Khối Độc lập](#chương-3-giải-phẫu-sâu-readback-op_mode--2b11-trùng-lặp-mạch-với-sfu_bypass--vì-sao-phải-xóa-bỏ-khối-độc-lập)
4. [Chương 4. Giải phẫu Sâu `DMA-WB` (`cfg_dma_wb_i = 1`) & Cơ chế Gearbox `snax_simple_dma.sv`](#chương-4-giải-phẫu-sâu-dma-wb-cfg_dma_wb_i--1--cơ-chế-gearbox-snax_simple_dmasv)
5. [Chương 5. Giải phẫu Hệ thống SFU (`snax_llm_sfu_opt.sv` + `snax_sfu_sequencer.sv`) & Điểm Chết ở Prefill Lớn](#chương-5-giải-phẫu-hệ-thống-sfu-snax_llm_sfu_optsv--snax_sfu_sequencersv--điểm-chết-ở-prefill-lớn)
6. [Chương 6. Giải phẫu `V-Transpose` (`snax_transpose_buffer.sv` + `snax_transpose_ctrl.sv`) & Gộp Flush KV Cache](#chương-6-giải-phẫu-v-transpose-snax_transpose_buffersv--snax_transpose_ctrlsv--gộp-flush-kv-cache)
7. [Chương 7. Truy vết End-to-End 1 Layer LLM (`Llama-3.2-1B`) — Hiện trạng Rời rạc vs Kiến trúc Hợp nhất](#chương-7-truy-vết-end-to-end-1-layer-llm-llama-32-1b--hiện-trạng-rời-rạc-vs-kiến-trúc-hợp-nhất)
8. [Chương 8. Audit Toàn bộ Bảng CSR (`RW 0..53`, `RO 0..9`), Aliasing & Thiết kế Lại CSR Không Chồng chéo](#chương-8-audit-toàn-bộ-bảng-csr-rw-053-ro-09-aliasing--thiết-kế-lại-csr-không-chồng-chéo)
9. [Chương 9. Đặc tả Kiến trúc `1 Unified Sequencer Duy nhất` (`snax_llm_unified_sequencer`)](#chương-9-đặc-tả-kiến-trúc-1-unified-sequencer-duy-nhất-snax_llm_unified_sequencer)
10. [Chương 10. Blueprint RTL Chi tiết cho Sequencer Hợp nhất & Dọn Sạch Top-Level `snax_llm_top_clean.sv`](#chương-10-blueprint-rtl-chi-tiết-cho-sequencer-hợp-nhất--dọn-sạch-top-level-snax_llm_top_cleansv)
11. [Chương 11. Phân tích Định lượng Diện tích, Băng thông DRAM, Chu kỳ Chết (Control Overhead) & Timing](#chương-11-phân-tích-định-lượng-diện-tích-băng-thông-dram-chu-kỳ-chết-control-overhead--timing)
12. [Chương 12. Kế hoạch Refactor An toàn & Ma trận Kiểm chứng End-to-End Không Regression](#chương-12-kế-hoạch-refactor-an-toàn--ma-trận-kiểm-chứng-end-to-end-không-regression)

---

## Chương 1. Bóc trần Nguồn gốc Câu hỏi: Vì sao Tách `Readback + DMA-WB` là Sai lầm Kiến trúc

### 1.1. Đặt lại đúng câu hỏi cốt lõi của hệ thống

Trong một bộ tăng tốc NPU cho mô hình ngôn ngữ lớn (LLM Accelerator) thực thi **end-to-end một Transformer Layer** (gồm: `RMSNorm 1` $\rightarrow$ `Q/K/V Projection` + `RoPE` + `V Transpose` $\rightarrow$ `FlashAttention` $\rightarrow$ `O Projection` + `Residual 1` $\rightarrow$ `RMSNorm 2` $\rightarrow$ `FFN Gate/Up` + `SwiGLU` $\rightarrow$ `FFN Down` + `Residual 2`), vai trò của **Sequencer** là điều phối đồng bộ **toàn bộ hệ thống** (`DRAM <-> DMA <-> TCDM 288 KB <-> Streamers <-> Core <-> SFU <-> Log-Quantizer`) để hoàn thành phép toán với **hai tiêu chí tối thượng**:

1. **Tiêu chí 1 (Độ phủ kích thước tuyệt đối trên bộ nhớ hữu hạn):** Chạy được **mọi kích thước ma trận và mọi độ dài ngữ cảnh / batch** (`N_Q`, `N_KV`, `M`, `N`, `K`, `D_MODEL`, `D_FF`) dù toàn bộ tensor lớn hơn rất nhiều lần `288 KB` TCDM vật lý, không bao giờ tràn TCDM hay bắt CPU phải nhảy vào cắt thủ công từng mẩu rồi tự gọi DMA chép qua chép lại giữa các bước.
2. **Tiêu chí 2 (Tối ưu băng thông DRAM & tính khép kín trên chip):** Dữ liệu một khi đã được nạp lên TCDM hoặc tính ra từ `Core` / `SFU` phải được tái sử dụng tối đa hoặc đưa thẳng qua tầng kế tiếp theo luồng 1 chiều (`fused stream`), triệt tiêu các chu kỳ ghi ra TCDM rồi đọc ngược lại (`intermediate spill`) hay ghi ra DRAM rồi kéo ngược lên TCDM (`round-trip bounce`).

Khi soi hệ thống hiện tại trong [snax_llm_top_clean.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv) dưới đúng lăng kính 2 tiêu chí này, câu hỏi:
> *"`Readback + DMA-WB` — tại sao 2 khối này cần tách riêng ra?"*

chạm đúng vào **căn bệnh kiến trúc lớn nhất** của subsystem hiện nay. Nếu chỉ trả lời hời hợt ở bề mặt *"Readback là nén FP16 $\rightarrow$ FP4 trên TCDM, còn DMA-WB là copy TCDM $\leftrightarrow$ DRAM, nên chúng là 2 chế độ riêng"*, câu trả lời đó **mô tả hiện trạng chứ không suy luận về bản chất hệ thống**, và bỏ sót 3 sự thật kỹ thuật cực kỳ quan trọng:

1. **Sự thật thứ nhất — Hệ thống đang bị xé làm 3 Sequencer + 1 Sub-Controller rời rạc**, làm phát sinh hàng trăm dòng dây mux chồng chéo ngay tại [snax_llm_top_clean.sv#L630-L997](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L630-L997).
2. **Sự thật thứ hai — `Readback` bị trùng lặp hoàn toàn trên phần cứng với chính đường `u_sfu(SFU_BYPASS) -> u_log_quantizer`**, và phần lớn lý do gọi `Readback` trong lịch sử (`Q+RoPE`, `K+RoPE`, `SwiGLU`, `RMSNorm`) vốn là **hệ quả chắp vá** khi đường Fused hoặc `u_sfu_seq` chưa hoàn chỉnh.
3. **Sự thật thứ ba — `DMA-WB` là chiếc nạng (band-aid) dùng để vá lỗ hổng thiếu Macro-DMA ở `FlashAttention`, `V-Transpose`, và đặc biệt là `SFU Sequencer`**. Chỉ riêng trong một layer `Llama-3.2-1B` tại [tb_llama3_1b_layer.sv#L905-L1110](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_llama3_1b_layer.sv#L905-L1110), CPU testbench phải dừng luồng tính toán để gọi `run_dma_writeback` tới **7 lần rời rạc**!

---

### 1.2. Kiểm kê Toàn bộ Các Bộ Điều khiển Đang Tồn tại Song song trong `snax_llm_top_clean.sv`

Bảng dưới đây liệt kê **đầy đủ mọi khối FSM điều khiển (Controllers / Sequencers)** hiện đang được instantiate cùng lúc trong [snax_llm_top_clean.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv):

| STT | Instance trong Top | File RTL | Dòng code | Phạm vi công việc hiện tại | Giao diện bị cắt vụn / Trùng lặp |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **1** | `u_gemm_seq` | [snax_gemm_os_sequencer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_gemm_os_sequencer.sv) | `943` | GEMM 4-Loop OS + Macro-DMA `A/W/O` + `Fused V` trigger | Chỉ làm `op_mode = 2'b00` khi `!cfg_dma_wb && !use_legacy_gemm`. Phải mux toàn bộ `dma_*`, `act_*`, `wt_*`, `wo_*`, `core_*` với `u_seq`. |
| **2** | `u_seq` | [snax_llm_sequencer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv) | `1,244` | FlashAttention (`2'b01`), V-Transpose (`2'b10`), **Readback (`2'b11`)**, **DMA-WB (`cfg_dma_wb_i=1`)**, GEMM fallback (`2'b00`) | Chứa logic DMA Ring riêng, engine Drain `nrm` riêng, và giữ 2 khối dịch vụ nhỏ `ST_RD_*` + `ST_DMA_WB_*` khiến Top không thể gỡ bỏ `u_seq` ngay cả khi đã có `u_gemm_seq`. |
| **3** | `u_sfu_seq` | [snax_sfu_sequencer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_sfu_sequencer.sv) | `256` | RMSNorm 2-pass (`op=2`), Residual (`op=1`), SwiGLU (`op=4`) 2D trên TCDM | **Chỉ chạy khi `csr_src_sel == 1`** ([snax_llm_top_clean.sv#L918](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L918)); **0 có cổng DMA**; không biết gọi `SFU_BYPASS -> LogQ` hay 4D write-out; phải ăn ké CSR `RW_SEQ_GEMM_MT/NT` và nhét 2 stride vào 1 thanh ghi `32-bit`. |
| **4** | `u_transpose_ctrl` | [snax_transpose_ctrl.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_transpose_ctrl.sv) | `147` | Đọc 16 hàng `FP16` từ Core vào `u_transpose_buffer`, xoay xuất 16 cột sang `u_log_quantizer` | Bản chất chỉ là adapter thu-phát 16x16 cho `V_proj`; nhưng bị biến thành 1 job `op_mode = 2'b10` tách rời ở `u_seq` (và khi fused thì cũng chỉ chạy tới TCDM, không đẩy về DRAM `KV cache`). |
| **5** | Mux `snax_llm_top_clean` | [snax_llm_top_clean.sv#L845-L996](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L845-L996) | `~152` | Mux 2 tầng (`gseq` vs `leg`) + mux (`sfu_seq` vs `csr` vs `seq`) | Thêm logic tranh chấp `gemm_os_active`, `sfu_seq_active`, `wo_mode_fp16_q`, `src_v_sel`, `src_rd_sel`, `src_core_sel` dễ sinh bug lệch state. |

**Tổng cộng hiện tại đang tốn tới `2,590+` dòng RTL thuần điều khiển**, chia làm 3 bộ não tách rời không hề nói chuyện trực tiếp với nhau!

---

### 1.3. Đối chiếu 2 Quan điểm: "Gom cơ học vào 1 file" vs "Hợp nhất đúng bản chất Datapath"

Đây chính là ranh giới giữa **"copy code ghép chung"** (tạo thêm rác) với **"tư duy thiết kế lại kiến trúc"**:

| Khía cạnh | Cách hiểu Sai ("Gom 4 chế độ vào 1 file") | Cách hiểu Đúng ("Hợp nhất theo Datapath End-to-End") |
| :--- | :--- | :--- |
| **Số lượng Sequencer ở Top** | Gom `u_gemm_seq` + `u_seq` thành 1, nhưng vẫn giữ `u_sfu_seq` chạy ngoài theo `csr_src_sel=1` | **ĐÚNG 1 Sequencer duy nhất** quản lý toàn bộ vòng đời 1 layer: `Compute Core (GEMM / Attn)` + `SFU / Post-Proc` + `Shared 3-Tag Macro-DMA`. |
| **Số phận của khối `Readback`** | Giữ nguyên `op_mode = 2'b11` + nhánh mux `src_rd_sel` song song trong Top như một khối riêng biệt | **Xóa tư cách "khối/engine riêng" của `Readback`.** Bản chất `Readback` chỉ là đường `TCDM Read FP16 -> SFU (chế độ BYPASS 3'd0) -> LogQ -> WriteOut`! Dùng chung 1 đường `SFU/Post-Proc` duy nhất thay vì đấu 2 đường đọc-nén trùng nhau. |
| **Số phận của khối `DMA-WB`** | Giữ nguyên `cfg_dma_wb_i` tách rời để CPU tự bưng dữ liệu giữa các pha compute và SFU | **Biến `Macro-DMA` thành hạ tầng nền cho MỌI Engine (`GEMM`, `FlashAttention`, `SFU`).** Khi mọi Engine đều tự kéo input từ DRAM và flush output ra DRAM theo Ping-Pong, 7 lần gọi `DMA-WB` thủ công trong layer biến mất. Chế độ DMA thô (`DRAM <-> TCDM` không compute) nếu giữ lại cho host init/debug chỉ là **1 flag passthrough (`MODE_DMA_BYPASS`) trên cùng DMA Arbiter** (~10 dòng), không phải 1 khối kiến trúc tách riêng. |
| **Số phận của `V-Transpose`** | Vẫn giữ opcode `2'b10` standalone tách rời khỏi GEMM | `u_transpose_ctrl` chỉ là **bộ lọc xoay trục hậu kỳ (Epilogue Shuffler)** bật bằng 1 cấu hình output (`OUT_MODE_V_TRANSPOSE`), dùng cho `GEMM V` (và xả thẳng qua `TAG_O_WB` về DRAM KV-cache khi bật Macro-DMA). |
| **Hệ quả trên CPU Host** | CPU phải lật qua lật lại `csr_src_sel=0` (GEMM/Attn/RB/DMA) và `csr_src_sel=1` (SFU), nhớ nhiều bảng CSR alias nhau | Mọi lệnh đều bắn qua **1 chuẩn Descriptor thống nhất trên 1 Sequencer**, không còn khái niệm `csr_src_sel=1` để chạy `u_sfu_seq`. |

---

### 1.4. Chứng minh từ Lịch sử Tiến hóa Code Trong Chính Repo Này

Nếu kiểm tra chuỗi các testbench từ cũ đến mới trong thư mục `tb/`, ta thấy rõ **mỗi lần kiến trúc tiến hóa hoàn chỉnh hơn thì vai trò của `Readback` và `DMA-WB` lại teo nhỏ đi**:

1. **Thế hệ 1 — [tb_snax_llm_layer_smoke.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_snax_llm_layer_smoke.sv):**
   - Core chưa nối với SFU; CPU tự tính RMSNorm, RoPE, Residual, SiLU từng vector trên TCDM qua `Master Port 1`.
   - Sau mỗi bước CPU xong, bắt buộc phải gọi **`Readback` (`A_SEQ_CTRL = 3`)** ở dòng `892` (`requant Xn`), `949` (`requant Q`), `966` (`requant K`), `1100` (`requant norm2`), `1132` (`requant ffn act`) $\rightarrow$ gọi `Readback` tới **5 lần/layer** chỉ làm "máy chạy vặt nén FP4" cho CPU!
2. **Thế hệ 2 — [tb_fused_qk_rope_compare.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_fused_qk_rope_compare.sv):**
   - **Method A (`L333-413`):** Chạy `GEMM Q` xuất `FP16` ra TCDM $\rightarrow$ CPU đọc/repack + `u_sfu` tính RoPE từng vector $\rightarrow$ **gọi `Readback (0x13)`** đọc lại FP16 nén sang FP4.
   - **Method B (`L417-440`):** Dùng thẳng đường `Core -> fused_fifo -> u_sfu (RoPE) -> u_log_quantizer -> u_write_out` trong **1 lần launch duy nhất (`A_START = 0x183`)** $\rightarrow$ **`Readback` biến mất hoàn toàn**, bỏ toàn bộ chu kỳ ghi/đọc FP16 trung gian trên TCDM!
3. **Thế hệ 3 — [tb_llama3_1b_layer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_llama3_1b_layer.sv) (Hiện trạng):**
   - Đã xóa được `Readback` ở `Q+RoPE`, `K+RoPE`, `GEMM V`, `SwiGLU`.
   - Nhưng **vẫn còn kẹt lại**:
     * `run_requant` / `run_requant_wrap` (`Readback`) ở bước `[2/12]` (sau RMSNorm 1) và `[9/12]` (sau RMSNorm 2) — vì `u_sfu_seq` đứng ngoài và bài test xuất FP16 ra check SQNR trước khi nén FP4!
     * **7 lần `run_dma_writeback` (`DMA-WB`)** ở bản Prefill (`L920`, `L1007`, `L1013`, `L1014`, `L1030`, `L1043`, `L1086`, `L1110`) — vì `u_seq` (FlashAttention, V) và `u_sfu_seq` (RMSNorm, Residual) **chưa có Macro-DMA tự chủ**!
4. **Thế hệ 4 — [tb_gemm_os_all_sizes.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_gemm_os_all_sizes.sv):**
   - Cho riêng phép toán GEMM khi nâng cấp lên `snax_gemm_os_sequencer.sv` với 4-Loop + 3-Tag Macro-DMA (`TAG_A`, `TAG_W`, `TAG_O_WB`): chỉ cần **1 lệnh duy nhất** cho mọi kích thước `M=64..128+`, `N=64..128+`, `K=128..512+`, **0 lần nào phải gọi `DMA-WB` rời trước hay sau lệnh GEMM**!

Kết luận đanh thép từ lịch sử phát triển của chính repo:
> **`Readback` tách riêng là tàn dư của thời chưa fuse Epilogue Quantize vào SFU/Core.**
> **`DMA-WB` tách riêng là tàn dư của thời chưa tích hợp 3-Tag Macro-DMA vào FlashAttention & SFU.**
> Còn việc để **3 Sequencer (`u_gemm_seq`, `u_seq`, `u_sfu_seq`)** cùng sống chung trong `snax_llm_top_clean.sv` chính là lý do khiến hai tàn dư đó vẫn bị đóng băng thành 2 khối riêng rẽ không dọn sạch được!

---

## Chương 2. Giải phẫu Hạ tầng Dữ liệu Cố định: 25-Master Crossbar & TCDM 288 KB (`32 Banks × 1024 Words × 72b`)

Để hiểu vì sao có thể gộp sạch mọi khối điều khiển mà không gây xung đột bus hay tắc nghẽn FIFO, ta phải giải phẫu chính xác 100% hạ tầng bộ nhớ và crossbar bên dưới trong [snax_llm_top_clean.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv), [interleaved_interconnect_72b.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/interleaved_interconnect_72b.sv), [fixed_priority_arb_n.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/fixed_priority_arb_n.sv), và [multi_bank_tcdm_72b.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/multi_bank_tcdm_72b.sv).

### 2.1. Thông số Hình học Vật lý của TCDM (`u_tcdm`)

* **Số lượng SRAM Banks (`NUM_BANKS`):** `32` macro Single-Port SRAM đồng bộ (`sram_bank_72b`, [multi_bank_tcdm_72b.sv#L10-L53](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/multi_bank_tcdm_72b.sv#L10-L53)).
* **Chiều sâu mỗi Bank (`BANK_DEPTH`):** `1024` words (`BANK_ADDR_W = $clog2(1024) = 10 bits`).
* **Độ rộng mỗi Word (`DATA_WIDTH`):** `72 bits = 9 bytes` (`STRB_WIDTH = 9 bits`, write-enable từng byte).
  * Ở định dạng nén **FP4 E2M1 + Log8**: `1 word 72-bit` = `{scale_log8[7:0], fp4_lane[15:0][3:0]}` (chứa trọn 16 phần tử).
  * Ở định dạng **FP16** (`u_write_out` / `u_read_fp16`): mỗi word chứa `4 × 16b = 64 bits` dữ liệu ở `word[63:0]` và `8'h00` padding ở `word[71:64]` ([snax_streamer_write_out.sv#L175-L184](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_streamer_write_out.sv#L175-L184), [snax_stream_read_fp16.sv#L8](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_stream_read_fp16.sv#L8)). Do đó `1 vector FP16` gồm `16 phần tử (256 bits)` chiếm **`4 words 72-bit`** trên TCDM!
* **Dung lượng TCDM tổng cộng:**
  $$\text{TCDM\_Bytes} = 32 \times 1024 \times 9 = 294,912\text{ bytes} = \mathbf{288\text{ KB}} \quad (32,768\text{ words})$$
  * Nửa thấp (`Banks 0..15`): `16 × 1024 × 9 = 147,456 B = 144 KB`.
  * Nửa cao (`Banks 16..31`): `16 × 1024 × 9 = 147,456 B = 144 KB`.
  * Không gian payload FP16 tối đa (chỉ tính `64b = 8B/word`): `32,768 × 8 B = 262,144 B = 256 KB` dữ liệu FP16 thuần (chứa tối đa `8,192` vector 16-lane FP16).

### 2.2. Giải mã Địa chỉ Interleaved Crossbar (`m_req_addr -> {bank_addr, bank_sel}`)

Theo [interleaved_interconnect_72b.sv#L98-L103](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/interleaved_interconnect_72b.sv#L98-L103) với `BYTE_OFFSET_W = 4`, `BANK_SEL_W = 5`, `BANK_ADDR_W = 10`:

```text
31                 19 18            9 8       4 3      0
+--------------------+---------------+---------+--------+
|   unused / 0       | bank_addr[9:0]| bank[4:0]| align4 |
+--------------------+---------------+---------+--------+
                          (<< 9)        (<< 4)   (4'b0000)
```

* **`addr[3:0]`**: Sub-word byte offset (`16-byte` aligned cho mỗi word 72-bit).
* **`addr[8:4]` (`m_target_bank`)**: Chọn **1 trong 32 Banks** (`0..31`). Tăng `0x10` (+16) là nhảy sang **bank kế tiếp cùng word index**.
* **`addr[18:9]` (`m_bank_addr`)**: Chọn **word `0..1023`** bên trong bank đó. Tăng `0x200` (+512) là nhảy sang **word kế tiếp cùng bank**.
* **Độ trễ SRAM & Crossbar:** Crossbar `u_interconnect` chuyển tiếp combinational trong chu kỳ chấp nhận grant (`m_req_ready_o[i]` là combinational từ `fixed_priority_arb_n`), sau đó `sram_bank_72b` chốt yêu cầu tại cạnh lên clock và trả `b_rvalid_i / b_rdata_i` ở **chu kỳ kế tiếp (`1-cycle read latency`)**. Tracker trong `u_interconnect` (`read_in_flight_q`, `grant_master_id_q`) đẩy `m_rsp_valid_o / m_rsp_rdata_o` combinational từ output SRAM về đúng Master đã thắng grant.

### 2.3. Bản đồ Đầy đủ 25 Masters trên Crossbar (`NUM_MASTERS = 25`)

Tại [snax_llm_top_clean.sv#L371-L454](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L371-L454), 25 Master được đấu vào `u_interconnect` như sau:

| Master Index | Cổng / Module sở hữu | Chiều truy cập | Băng thông đỉnh (Words/cyc) | Vai trò trong các chế độ (`GEMM` / `ATTN` / `SFU` / `RB` / `DMA`) |
| :---: | :--- | :---: | :---: | :--- |
| **`0`** | `dma_mem_req` (`DMA Port 0`) | **R / W** | `1` (hoặc port duy nhất khi TCDM $\rightarrow$ DRAM!) | Đọc/Ghi TCDM trong `snax_simple_dma` (chiều `TCDM -> DRAM` chỉ dùng DUY NHẤT port này!) |
| **`1`** | `cpu_mem_req` (RV32I Port) | **R / W** | `1` | CPU host đọc/ghi debug, init hằng số hoặc can thiệp thủ công |
| **`2`** | `u_act_streamer` (`snax_streamer_act_bcast`) | **Read** | `1` | Đọc `Activation A` (GEMM) hoặc `Query Q` (Attention) phát broadcast vào Core |
| **`3..18`** | `u_wt_streamer` (`snax_streamer_wt_16ch`) | **Read** | **`16`** (`1,152 b/cyc`) | Đọc song song 16 lanes `Weight W` (GEMM) hoặc `Key/Value K,V` (Attention) từ 16 banks liên tiếp |
| **`19`** | `u_write_out` (`snax_streamer_write_out`) | **Write** | `1` | Ghi kết quả FP4 (1w), FP16 (4w), V-transpose (16w/batch), hoặc FP4 4D từ `Core / SFU / LogQ` xuống TCDM |
| **`20`** | `u_read_fp16` (`snax_stream_read_fp16`) | **Read** | `1` (`1 vec / 5+ cyc`) | Đọc FP16 từ TCDM cho **`Readback`** HOẶC cho **SFU Operand A** |
| **`21`** | `u_read_fp16_b` (`snax_stream_read_fp16`) | **Read** | `1` (`1 vec / 5+ cyc`) | Đọc FP16 từ TCDM cho **SFU Operand B** (`Gamma` của RMSNorm, `X` của Residual, `sin/cos` của RoPE, `Up` của SwiGLU) |
| **`22..24`** | `dma_aux_mem_req[0..2]` (`DMA Ports 1..3`) | **Write** | **`3`** | Kết hợp với Master 0 thành **4 words/cycle (`288 b/cyc`)** cho chiều `DRAM -> TCDM` (`snax_simple_dma`) |

### 2.4. Phát hiện Kỹ thuật Trực tiếp từ `fixed_priority_arb_n` trong `u_interconnect`

Kiểm tra [snax_llm_top_clean.sv#L457-L464](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L457-L464) đối chiếu với [fixed_priority_arb_n.sv#L21-L49](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/fixed_priority_arb_n.sv#L21-L49):

```systemverilog
    interleaved_interconnect_72b #(
        .NUM_MASTERS   (NUM_MASTERS),
        .NUM_BANKS     (NUM_BANKS),
        .DATA_WIDTH    (DATA_WIDTH),
        .ADDR_WIDTH    (ADDR_WIDTH),
        .BANK_DEPTH    (BANK_DEPTH)
    ) u_interconnect ( ... );
```

Nhìn vào tham số mặc định của `interleaved_interconnect_72b`:
* `.HOST_MASTER (0)`
* `.HOST_MASK ('0)` $\Rightarrow$ Trong `fixed_priority_arb_n`: `low_mask = 25'b0_0000_0000_0000_0000_0000_0001` (chỉ có **Master 0** nằm ở mức ưu tiên thấp!).
* Với mọi master còn lại trong `non_host_req = req_i & ~low_mask`, điều thức `non_host_grant = non_host_req & ~(non_host_req - 1'b1)` chọn **bit có chỉ số NHỎ NHẤT thắng tuyệt đối!**

Hệ quả thứ tự ưu tiên **trên cùng 1 bank** hiện đang là:
$$\text{Master } 1\text{ (CPU)} > 2\text{ (Act)} > 3..18\text{ (Wt)} > 19\text{ (WO)} > 20\text{ (RD\_A)} > 21\text{ (RD\_B)} > 22..24\text{ (DMA Aux 1..3)} > 0\text{ (DMA Port 0)}$$

Từ phương trình ưu tiên này rút ra **3 hệ quả kiến trúc sống còn** cho việc hợp nhất Sequencer + bỏ `Readback`/`DMA-WB`:

1. **Bug tiềm ẩn ở `DMA 4-port` khi đụng cùng bank với `Act/Wt/WO/RD`:**
   - Do `Masters 22..24` không được đặt vào `HOST_MASK` (`{3'b111, 21'd0, 1'b1}` mới đúng nếu chỉ muốn DMA thua toàn bộ streamers, hoặc `{3'b111, 19'd0, 2'b11}` nếu muốn cả CPU lẫn DMA 4-port cùng nhường streamers), hiện tại `DMA Port 0` ở mức thấp nhất, còn `DMA Ports 1..3` (`22..24`) tuy vẫn thua `1..21` nhờ có số index lớn hơn, nhưng lại **thắng `DMA Port 0`** nếu lỡ chạm cùng bank (không xảy ra trong 1 chu kỳ từ `snax_simple_dma` vì 4 port luôn ghi vào 4 bank khác nhau liên tiếp: `b, b+1, b+2, b+3` hoặc 4 word ở bank tuyến tính không bao giờ đụng 1 bank? **CẨN THẬN!** Trong chế độ **`dir[1] = 1` (linear_mode)**, `next_tcdm_addr` cộng `0x200` (tức **CÙNG 1 BANK** `cur[8:4]`, chỉ tăng `word[18:9]`)!).
   - Kiểm chứng ngay dòng `snax_simple_dma.sv#L96-L97` và `#L270-L283`:
     * Trong `dir[1] = 1` (linear `DRAM -> TCDM`), `snax_simple_dma` nạp `hold_addr_q[0..3]` bằng `cur_taddr`, `cur_taddr + 0x200`, `cur_taddr + 0x400`, `cur_taddr + 0x600` $\rightarrow$ **CẢ 4 HOLD PORTS ĐỀU TRỎ VÀO CÙNG 1 BANK**!
     * Khi đó trên crossbar, `Masters 22..24` (`DMA Aux 1..3`) thắng từng chu kỳ trước `Master 0` (`DMA Port 0`): cycle 0 ghi Port 1 (`22`), cycle 1 ghi Port 2 (`23`), cycle 2 ghi Port 3 (`24`), cycle 3 ghi Port 0 (`0`) $\rightarrow$ **đảo thứ tự ghi các word?** Không sao vì mỗi `hold_addr_q[p]` mang địa chỉ word riêng của nó, nhưng hiệu suất linear `DRAM -> TCDM` rơi từ `4 words/cyc` xuống **`1 word/cyc`** (bị nghẽn cổ chai single-port của đúng 1 bank đó)!
     * **Bài học thiết kế TCDM Layout cực kỳ quan trọng:** Bất cứ khi nào Macro-DMA đọc/ghi giữa DRAM và TCDM tốc độ cao (`Q_buf`, `A_buf`, `O_buf`, `SFU_buf`), **BẮT BUỘC dùng layout dàn đều qua 16 bank (`Banks 0..15` hoặc `Banks 16..31`)** bằng `dir[1] = 0` (hoặc địa chỉ `16-bank interleaved`) thay vì `linear` nhồi vào 1 bank — nhờ vậy 4 port DMA bắn song song 4 bank khác nhau đạt đúng **4 words = 36 bytes/cycle**, không bị thắt cổ chai ở single-port SRAM!

2. **Bắt buộc phân tách Bank Disjoint giữa Đường Tính (`Compute / SFU`) và Đường DMA Ping-Pong:**
   - Khi `u_wt_streamer` đang đọc `Banks 0..15` (cả 16 banks bị chiếm kín 100% ở các beat active), nếu `DMA` ghi slot ring kế tiếp **nằm trong cùng `Banks 0..15`**, `DMA` sẽ thua priority (`22..24 > 0` đều `< Wt`? Khoan: `Wt` là `3..18`, nhỏ hơn `22..24`, nên `Wt` thắng `DMA`!). Kết quả là trong các pha `Wt` đọc liên tục, `DMA` ghi vào nửa Ring bị stalled từng chu kỳ xen kẽ (hoặc trong FlashAttention khi `QK` replay `8` chu kỳ: `Wt` chỉ đọc TCDM **1 trong 8 chu kỳ**, `7` chu kỳ replay còn lại `Wt` KHÔNG request crossbar $\Rightarrow$ `DMA` có `7/8 = 87.5%` băng thông TCDM trống để bơm KV ngầm không mất một chu kỳ compute!).
   - Tương tự ở nửa trên **`Banks 16..31` (`144 KB`)**:
     * Nếu một lệnh **SFU Macro-DMA** đang chạy trên `Banks 16..23` (Buffer 0) trong lúc DMA bơm Buffer 1 ở `Banks 24..31` (hoặc chia theo dải `word [0..511]` vs `[512..1023]`), dù chia theo word hay theo bank: chỉ cần khi `u_read_fp16` / `u_write_out` chỉ dùng `1` bank mỗi chu kỳ (tối đa 2 masters đọc + 1 master ghi = 3/16 banks), băng thông dư trên 13 banks còn lại cho DMA là dồi dào — và nếu đặt **Buffer 0 trên 8/16 banks hoặc word disjoint** kết hợp fixed priority `RD/WO > DMA`, tính toán không bao giờ bị DMA làm stall dù chỉ 1 chu kỳ!

3. **Tại sao `u_write_out` (`Master 19`) và `u_read_fp16` (`Master 20`) KHÔNG BAO GIỜ nên chạy tách thành 2 job riêng nếu có thể ghép dòng?**
   - Trong chế độ `Readback`, `u_read_fp16` (Master 20) đọc 4 words FP16 từ TCDM (`>= 5` chu kỳ vì trạng thái FSM `IDLE -> REQ -> RSP -> EMIT` tuần tự từng word — không pipeline back-to-back!), sau đó `u_log_quantizer` mất `2` chu kỳ, rồi `u_write_out` (Master 19) ghi 1 word FP4 xuống TCDM.
   - Tổng chi phí cho 1 vector trong `Readback`:
     * TCDM Read bandwidth: **4 accesses** (đọc 4 word FP16).
     * TCDM Write bandwidth: **1 access** (ghi 1 word FP4) — nhưng lại đi **SAU** khi trước đó chính hệ thống vừa mất **4 accesses TCDM Write** để ghi bản FP16 kia từ `Core` hoặc `u_sfu`!
     * **Tổng TCDM traffic nếu tách (`Compute/SFU -> TCDM FP16 -> Readback -> TCDM FP4`):** `4 (W FP16) + 4 (R FP16) + 1 (W FP4) = 9 accesses/vector` (chưa kể `4-cycle serial overhead` của `snax_stream_read_fp16` FSM làm chậm tốc độ tối đa chỉ còn **1 vector / 9–10 chu kỳ**!).
     * **Tổng TCDM traffic nếu Fused trực tiếp (`Core/SFU -> LogQ -> WO FP4`):** **`1 (W FP4)` duy nhất!** **Tiết kiệm 88.9% truy cập TCDM** và đạt throughput gần **1 vector / 1–2 chu kỳ**!
   - Chi tiết bất ngờ ở [snax_stream_read_fp16.sv#L83-L135](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_stream_read_fp16.sv#L83-L135): `u_read_fp16` hoàn toàn **không pipeline giữa `REQ` và `RSP`**: mỗi word phải đi `REQ (1+ cyc) -> RSP (1 cyc) -> REQ`, hết 4 word mới vào `EMIT (1+ cyc)` $\implies$ ngay cả khi crossbar trống hoàn toàn, `snax_stream_read_fp16` chỉ xuất được tối đa **`1` vector FP16 mỗi `9` chu kỳ** (`4 × 2 + 1 = 9 cycles/vec`)!
   - Hệ quả chí mạng: Khi `Readback` chạy bằng `u_read_fp16`, tốc độ nén bị **bóp nghẹt xuống 1/9 tốc độ thật của `log_quantizer`** (vốn nhận được 1 vector/chu kỳ ở chế độ pipelined)! Trong khi nếu fuse thẳng từ `u_core` (qua `2`-cycle drain) hoặc `u_sfu` (qua streaming), ta tránh hẳn cái cổ chai `9 cycles/vec` của `Readback`!

---

## Chương 3. Giải phẫu Sâu `Readback` (`op_mode = 2'b11`): Trùng lặp Mạch với `SFU_BYPASS` & Vì sao Phải Xóa Bỏ Khối Độc lập

### 3.1. Truy vết Từng Dây và Từng Chu kỳ của `Readback` Trong RTL Hiện tại

Để chứng minh `Readback` không phải là một "khối tính toán độc lập" mà chỉ là một mảnh cấu hình dư thừa, ta truy vết toàn bộ đường đi của nó từ CSR đến TCDM:

#### Bước 1: Kích hoạt từ CSR ([snax_flash_csr_snax.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_flash_csr_snax.sv))
* `RW_CTRL` (`idx=0`): bit `[0]` (`src_sel`) **= `0`** (chế độ Sequencer tự chủ).
* `RW_SEQ_CTRL` (`idx=31`):
  * `[1:0]` (`seq_op_mode`) `= 2'b11` (Readback).
  * `[5:4]` (`seq_out_mode`) `= 2'd0` hoặc `2'd1` (FP4 linear) hay `2'd3` (FP4 4D reorder — dùng trong `tb_snax_llm_layer_mid.sv` để sắp xếp Q/K sau RoPE).
  * `[6]` (`seq_dma_wb`) `= 0`.
  * `[26]` (`wo_wrap_en`) & `[27]` (`rd_wrap_en`) + `RW_WRAP_CFG` (`idx=52`): cấu hình vòng lặp bank `16-bank` khi đọc hoặc ghi nhiều hơn `BANK_DEPTH` word trên một bank (ví dụ ở `tb_llama3_1b_layer.sv#L925` dùng `rd_wrap_words_m1 = 511`, `wo_wrap_words_m1 = 127`).
* `RW_RD_BASE` (`27`), `RW_RD_NV` (`28`), `RW_RD_LS` (`29`): địa chỉ gốc, tổng số vector FP16 (`nvec`), bước nhảy địa chỉ theo word FP16.
* `RW_WO_BASE` (`13`): địa chỉ TCDM đích ghi gói FP4 72-bit (thông qua `cfg_rd_out_base_i = csr_wo_base` tại [snax_llm_top_clean.sv#L676](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L676)).
* Nếu dùng `seq_out_mode = 2'd3` (FP4 4D): `u_seq` mượn toàn bộ **8 thanh ghi của Attention & Base** (`RW_SEQ_ATTN_QT`, `RW_SEQ_RING`, `RW_SEQ_ATTN_DV`, `RW_SEQ_Q_BASE`, `RW_SEQ_ATTN_KV`, `RW_SEQ_OUT`, `RW_SEQ_ATTN_KB`, `RW_SEQ_DRAM_KV`) để đổ vào `wo_num_vectors_o .. wo_dim3_stride_o` ([snax_llm_sequencer.sv#L716-L726](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L716-L726)).

#### Bước 2: FSM trong `snax_llm_sequencer.sv` (`u_seq`)
1. **`ST_IDLE -> ST_PREP_DRAIN` (`L593`):** Vì `cfg_op_mode_i == 2'b11`, nhánh `ST_PREP_DRAIN` ([`#L600-L672`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L600-L672)) **bỏ trống hoàn toàn**, chỉ chuyển sang `ST_PREFETCH_INIT`.
2. **`ST_PREFETCH_INIT -> ST_RD_LAUNCH` (`L680`):** Không kích DMA nào cả (`dma_attn_pending_q = 0`, `dma_pending_q = 0`), nhảy thẳng sang `ST_RD_LAUNCH`.
3. **`ST_RD_LAUNCH` (`L709-L736`):**
   - Đợi `!rd_busy_i && !wo_busy_i`.
   - Khởi tạo `rd_done_seen_d = 1'b0`.
   - Bật `cfg_enable_slq_o = 1'b1` (thực ra ở Top-level [snax_llm_top_clean.sv#L1391](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L1391), tín hiệu này đi vào `u_core` chứ **không hề liên quan gì tới `u_log_quantizer`!** — thêm một chi tiết cho thấy `u_seq` đang điều khiển mù `cfg_enable_slq_o` lúc `u_core` ngủ!).
   - Bắn đồng thời `rd_start_o = 1'b1` và `wo_start_o = 1'b1` với `wo_base_o = cfg_rd_out_base_i`.
   - Nếu `cfg_out_mode_i == 2'd3 && cnt_kt_q == 0` chọn `wo_mode_o = 2'd3` (`MODE_FP4_4D`), ngược lại `wo_mode_o = 2'd0` (`MODE_FP4`).
   - Chuyển sang `ST_RD_RUN`.
4. **`ST_RD_RUN` (`L738-L744`):**
   - Giữ trạng thái chờ cho tới khi `rd_done_seen_d && !wo_busy_i`, rồi nhảy sang `ST_WAIT_WRITE_OUT` $\rightarrow$ `ST_DONE`.
   - Chú ý ở đây: suốt thời gian `u_read_fp16` và `u_write_out` chạy (hàng trăm đến hàng nghìn chu kỳ), `u_core` (tức toàn bộ **256 bộ nhân MAC**), `u_act_streamer`, `u_wt_streamer`, `u_sfu`, `snax_simple_dma` đều **NGỒI CHƠI HOÀN TOÀN (`100% idle`)**!

#### Bước 3: Đường truyền Dữ liệu qua 4 tầng Hardware

```text
[TCDM Banks]
     │ (Master 20: 1 word 72-bit / >= 2 cycles, 4 words = 1 FP16 vector)
     ▼
[u_read_fp16 (snax_stream_read_fp16)]
     │ FSM tuần tự: IDLE -> REQ -> RSP -> (x4 words) -> EMIT (256-bit fp16_data_o)
     │ ~9 cycles / vector ở điều kiện không conflict!
     ▼
[Mux 3-to-1 tại snax_llm_top_clean.sv#L1065-L1091]
     │ Điều kiện kích hoạt: src_rd_sel = !v_active && !sfu_en && (rd_busy || cfg_quant_src_sel==2'b10)
     │ quant_fp16_in  = rd_fp16_data
     │ quant_in_valid = rd_fp16_valid && quant_accept
     ▼
[u_log_quantizer (slq_encoder.sv)]
     │ Tầng 1 (Comb): 16x tách {sign, E[4:0], M[9:7]} -> mu[0..15] -> Max-tree 4 tầng -> mu_max
     │ Tầng 1 Reg (`pipe_advance = !out_valid || out_ready`): chốt mu_max_q1, sign_fp16_q1, mu_d1[0..15]
     │ Tầng 2 (Comb): scale_raw = mu_max_q1 - 140 -> clamp [-128, 127] -> diff = mu_max_q1 - mu_d1[i] -> 7 ngưỡng FP4
     │ Tầng 2 Reg: packed_vec <= {scale_sat, mag_sign[15:0]} (72-bit)
     ▼
[u_write_out (snax_streamer_write_out)]
     │ Push vào skid_buffer DEPTH=16 (`wo_fp4_valid && in_fifo_ready`)
     │ WRITE_FP4 hoặc WRITE_FP4_4D -> Master 19 ghi 1 word 72-bit/vec xuống TCDM
     ▼
[TCDM Banks (gói FP4 72-bit)]
```

---

### 3.2. Phân tích Chi tiết Latency & Thông lượng Từng Tầng trong `Readback`

Bảng theo dõi từng chu kỳ thực thi 1 vector FP16 (`N_vec = 1` đến `N`) của `Readback` (giả sử Crossbar không conflict):

| Chu kỳ | `u_read_fp16` FSM | Crossbar Master 20 (`b_req[b]`) | `u_log_quantizer` | `u_write_out` (`skid` + Master 19) |
| :---: | :--- | :--- | :--- | :--- |
| **`T0`** | `IDLE`: nhận `rd_start` $\rightarrow$ `REQ` | Chưa request (`0`) | Idle | `IDLE` $\rightarrow$ `WRITE_FP4` |
| **`T1`** | `REQ` (word 0): giữ `m_tcdm_req_valid_o=1` | Arbiter grant Master 20, SRAM chốt địa chỉ `W0` | Idle | Chờ `in_fifo_valid` |
| **`T2`** | `RSP` (word 0): `m_tcdm_rsp_valid_i=1` $\rightarrow$ nạp `word_buf_q[0]`, `word_cnt <= 1`, sang `REQ` | Rảnh (chuyển word tiếp) | Idle | Chờ |
| **`T3`** | `REQ` (word 1) | SRAM chốt `W1` | Idle | Chờ |
| **`T4`** | `RSP` (word 1) $\rightarrow$ nạp `word_buf_q[1]`, `word_cnt <= 2`, sang `REQ` | Rảnh | Idle | Chờ |
| **`T5`** | `REQ` (word 2) | SRAM chốt `W2` | Idle | Chờ |
| **`T6`** | `RSP` (word 2) $\rightarrow$ nạp `word_buf_q[2]`, `word_cnt <= 3`, sang `REQ` | Rảnh | Idle | Chờ |
| **`T7`** | `REQ` (word 3) | SRAM chốt `W3` | Idle | Chờ |
| **`T8`** | `RSP` (word 3) $\rightarrow$ `word_buf_q[3]` (từ `m_tcdm_rsp_rdata_i` chốt vào FF), sang `EMIT` | Rảnh | Idle | Chờ |
| **`T9`** | `EMIT`: bật `fp16_valid_o = 1`, `fp16_ready_i = 1` $\rightarrow$ quay về `REQ` (vector kế) | Rảnh (**mất thêm 1 chu kỳ không đọc SRAM!**) | Tổ hợp tính `mu_max`; cạnh `T9->T10` chốt Stage 1 | Chờ |
| **`T10`** | Bắt đầu `REQ` (word 0 của vec 1) | SRAM chốt `W0` vec 1 | Tổ hợp tính 7 ngưỡng; cạnh `T10->T11` chốt `packed_vec` | Chờ |
| **`T11`** | `RSP` (word 0 vec 1) | Rảnh | `out_valid = 1` $\rightarrow$ Push vào `u_in_fifo` của `u_write_out` | Chốt vào `u_in_fifo` |
| **`T12`** | `REQ` (word 1 vec 1) | SRAM chốt `W1` vec 1 | — | `in_fifo_valid = 1` $\rightarrow$ Master 19 ghi word FP4 xuống TCDM |

**Con số định lượng rút ra trực tiếp từ bảng timing trên:**
* Do `snax_stream_read_fp16` ([snax_stream_read_fp16.sv#L98-L132](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_stream_read_fp16.sv#L98-L132)) đợi `RSP` rồi mới gửi `REQ` tiếp theo và tốn thêm 1 chu kỳ `EMIT` tách biệt, mỗi vector FP16 mất tối thiểu **`9` chu kỳ** (`(1 REQ + 1 RSP) × 4 + 1 EMIT`).
* Với `Llama-3.2-1B` Prefill `M = 16, D = 2048` (`NUM_VEC = 16 × 128 = 2,048` vectors):
  * Mỗi lần gọi `Readback` (`run_requant_wrap` cho `Xn` và `Norm2`) tốn tối thiểu:
    $$2,048 \times 9 = \mathbf{18,432\text{ chu kỳ}}$$
  * Hai lần `Readback` trong 1 layer = **`36,864` chu kỳ** chỉ để đọc FP16 từ TCDM rồi nén lại FP4 ghi về TCDM — trong khi **bản thân `log_quantizer` có khả năng nuốt `1` vector/chu kỳ (`2,048` chu kỳ!)** nếu ăn trực tiếp từ luồng stream đang chạy!

---

### 3.3. Bóc trần Sự Trùng lặp Chồng chéo giữa `Readback` và `SFU_BYPASS` trong RTL

Đây là bằng chứng rõ ràng nhất chứng tỏ `Readback` bị "đẻ thêm" song song một cách thiếu nhất quán trong [snax_llm_top_clean.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv):

So sánh trực tiếp **2 đường mạch đang cùng tồn tại** cho **cùng 1 tác vụ `TCDM FP16 -> Log-Quantizer -> TCDM FP4`**:

#### Đường thứ 1: `Readback` qua `u_seq` (`src_rd_sel`)
Nằm tại [snax_llm_top_clean.sv#L1066](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L1066), [`#L1079-L1081`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L1079-L1081), [`#L1120`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L1120), [`#L1216`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L1216):
```systemverilog
wire src_rd_sel   = !v_active && !sfu_en && (rd_busy || (cfg_quant_src_sel == 2'b10));
...
end else if (src_rd_sel) begin
    quant_fp16_in  = rd_fp16_data;
    quant_in_valid = rd_fp16_valid && quant_accept;
end
...
assign rd_fp16_ready = (sfu_en && !sfu_in_a_src) ? sfu_in_a_ready : (src_rd_sel && quant_accept);
...
wire wo_job_start = wo_start | v_start | (rd_start && !sfu_en);
```

#### Đường thứ 2: `SFU_BYPASS` qua `u_sfu` (`sfu_en = 1, sfu_in_a_src = 0, sfu_out_dst = 0, sfu_op = 3'd0`)
Nằm tại [snax_llm_top_clean.sv#L1082-L1086](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L1082-L1086), [`#L1120`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L1120), [`#L1139`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L1139), và [snax_llm_sfu_opt.sv#L307-L316](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sfu_opt.sv#L307-L316):
```systemverilog
// Tại snax_llm_sfu_opt.sv:
if (cfg_op_i == SFU_BYPASS) begin
    active_vec_q <= 1'b0;
    chunk_q      <= '0;
    phase_q      <= 1'b0;
    if (in_a_valid_i && out_can_accept) begin
        out_valid_o <= 1'b1;
        out_fp16_o  <= in_a_fp16_i;
    end else if (out_ready_i) begin
        out_valid_o <= 1'b0;
    end
end

// Tại snax_llm_top_clean.sv:
} else if (src_core_sel) begin
    if (sfu_en && sfu_out_dst == 1'b0) begin
        quant_fp16_in  = sfu_out_data;
        quant_in_valid = sfu_out_valid && quant_accept && !wo_mode_fp16;
    end ...
```

Hãy nhìn vào **sự vô lý kỹ thuật lộ thiên** ngay trong đoạn code trên:
1. Tên gọi `src_core_sel` (`L1067`) rõ ràng ám chỉ "nguồn từ Core", nhưng thực tế khi `sfu_en == 1`, nó **bị gán ghép thêm `sfu_out_data`**! Vì vậy nếu `sfu_in_a_src == 0` và `sfu_op == SFU_BYPASS (3'd0)`, dữ liệu đi từ **`u_read_fp16` $\rightarrow$ `u_sfu (1-cycle passthrough)` $\rightarrow$ `u_log_quantizer` $\rightarrow$ `u_write_out`** — giống hệt **100%** chức năng của `Readback`!
2. Do giữ cả hai đường song song, Top-level phải thêm hack `&& !sfu_en` ở 2 chỗ cực kỳ mong manh:
   * `src_rd_sel = !v_active && !sfu_en && ...` (`L1066`): để `rd_busy` lúc chạy SFU không cướp mất quyền điều khiển `u_log_quantizer` của SFU!
   * `wo_job_start = wo_start | v_start | (rd_start && !sfu_en)` (`L1216`): vì trong testbench/mode manual, `rd_start` vốn được gán cứng để tự khởi chạy `u_write_out`, nên lúc chạy SFU pass 1 (`RMSNorm` pass 1 chỉ đọc X, không ghi ra TCDM), người thiết kế phải thòng điều kiện `&& !sfu_en` để `rd_start` không kích nhầm `u_write_out` gây deadlock!
3. Trong `snax_llm_top_clean.sv#L1081`, hãy để ý lỗi logic handshake tiềm ẩn ở `quant_in_valid = rd_fp16_valid && quant_accept;` vs `snax_logq_shared.sv`! Trong `slq_encoder.sv#L105-L106`, `in_ready = !out_valid || out_ready` — tức là `u_log_quantizer` có **combinational path** từ `out_ready` tới `in_ready` (`quant_accept`), rồi tại `snax_llm_top_clean.sv`, `quant_accept` vòng tiếp qua `wo_fp4_ready -> rd_fp16_ready` VÀ ngược vào `quant_in_valid`! Ở `snax_logq_shared.sv#L97-L140`, người ta đã viết hẳn một module bọc chuẩn với `issue_ok = (fifo_count < FIFO_DEPTH - 2)` + FIFO tagged để tách timing path đó — **nhưng `snax_llm_top_clean.sv` lại KHÔNG DÙNG `snax_logq_shared.sv`, mà tự đấu mux thủ công (`src_v_sel`, `src_rd_sel`, `src_core_sel`)!**

---

### 3.4. Phân tích 4 Trường hợp Trong Lịch sử Từng Dùng `Readback` — Khối nào Thừa, Khối nào Gộp Được?

Bảng tổng kết toàn diện 4 kịch bản thực tế có liên quan tới `Readback` (`FP16 -> FP4`) trong suốt quá trình thực thi 1 LLM layer:

| STT | Kịch bản trong LLM Layer | Cách làm Cũ (`Readback` tách riêng) | Cách làm Đúng (Fused / Unified Post-Proc) | Tiết kiệm khi bỏ `Readback` rời |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **`Q_proj / K_proj` + RoPE $\rightarrow$ FP4** | `GEMM -> TCDM FP16 -> CPU/SFU RoPE -> TCDM FP16 -> Readback (op=11) -> TCDM FP4` | **`Core -> fused_fifo -> SFU(ROPE) -> LogQ -> WO (FP4/4D)`** (1 phát ăn ngay, `tb_fused_qk_rope_compare.sv` Method B đã chứng minh!) | Xóa **100%** write FP16 + read FP16 + CPU repack + Readback job |
| **2** | **`FFN Gate + Up` + `SwiGLU` $\rightarrow$ FP4** | `Gate -> TCDM FP16`, `Up -> TCDM FP16` $\rightarrow$ `SFU(SWIGLU) -> TCDM FP16` $\rightarrow$ `Readback -> TCDM FP4` | **`SFU(SWIGLU, out_dst=0 LogQ) -> WO (FP4)`** (đã làm ở `tb_llama3_1b_layer.sv#L872, L1073`) | Xóa **100%** `SwiGLU FP16` TCDM round-trip + Readback |
| **3** | **`RMSNorm 1 / RMSNorm 2` $\rightarrow$ FP4 (`Act` cho QKV / Gate-Up)** | `u_sfu_seq(RMSNorm) -> TCDM FP16` $\rightarrow$ **gọi `Readback` (`run_requant_wrap`)** $\rightarrow$ `TCDM FP4` | **Nén thẳng ở Pass 2 của `RMSNorm` (`SFU(RMSNORM, out_dst=0 LogQ) -> WO (FP4 + wrap)`!)** Trong inference thực tế không cần lưu bản `XN_fp16` ra TCDM rồi đọc lại để quantize! | Với `M=16, D=2048`: tiết kiệm **2 × 18,432 = 36,864 cycles/layer** + 100% traffic TCDM `XN_fp16`! |
| **4** | **Chuyển đổi thuần `FP16 -> FP4` (khi CPU tạo FP16 trên TCDM hoặc `O` FP16 cần 4D repack)** | Bật `u_seq` (`op=11 Readback`) kèm mux `src_rd_sel` trùng lặp | Dùng chính **1 đường Unified SFU/Post-Proc** với `sfu_op = SFU_BYPASS (3'd0)` + `out_dst = LogQ (0)` + hỗ trợ `wrap` / `4D` / `Macro-DMA`! | Xóa bỏ nhánh `src_rd_sel` trùng lặp + xóa opcode `2'b11` khỏi Compute Sequencer! |

**Kết luận Chương 3:**
Việc tách `Readback` thành một opcode/khối riêng ngang hàng với `GEMM` và `FlashAttention` là **sai hoàn toàn cả về dataflow lẫn tổ chức module**.
* 3 trong 4 trường hợp vốn dùng `Readback` biến mất hoàn toàn nhờ Fused Epilogue (`Core -> SFU -> LogQ` và `SFU(Pass2/2D) -> LogQ`).
* Trường hợp thứ 4 còn lại (nén một tensor FP16 bất kỳ sang FP4) **đã có sẵn sẵn trong `u_sfu` thông qua `SFU_BYPASS (3'd0)`**, chỉ cần gom về **một đường Post-Process Pipeline duy nhất** (thêm hỗ trợ `wrap` / `4D` / `DMA` ở bộ điều khiển chung), thay vì duy trì hai đường đọc-nén trùng lặp (`src_rd_sel` vs `u_sfu(BYPASS)`) và hai sequencer tách biệt (`u_seq` vs `u_sfu_seq`)!

---

## Chương 4. Giải phẫu Sâu `DMA-WB` (`cfg_dma_wb_i = 1`) & Cơ chế Gearbox `snax_simple_dma.sv`

### 4.1. Động cơ DMA Vật lý Dưới Gầm (`snax_simple_dma.sv`)

Trước khi đánh giá vai trò của `DMA-WB` ở tầng Sequencer, ta phải hiểu chính xác cách phần cứng `snax_simple_dma.sv` vận hành từng bit và từng beat bus DRAM.

#### Thông số cố định từ [snax_simple_dma.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_simple_dma.sv):
* **DRAM Bus Width (`DRAM_DATA_W`):** `512 bits = 64 bytes/beat` (`DRAM_STRB_W = 64 bits`).
* **TCDM Bus (`TCDM_PORTS × TCDM_DATA_W`):** `4 ports × 72 bits = 288 bits/cycle` (`4 × 9 bytes = 36 bytes/cycle`).
  * `Master 0` = Port 0 (`dma_mem_req_*`)
  * `Masters 22..24` = Ports 1..3 (`dma_aux_mem_req_*[0..2]`)
* **Độ rộng Gearbox Shift Buffer (`BUF_W`):** `512 * 2 + 72 = 1,096 bits` (~`2.14` DRAM beats), cho phép pipeline ngầm tới nhiều request đọc DRAM mà không bị rỗng buffer.
* **Tỷ số chuyển đổi bit hoàn hảo cho FP4 Weight/KV tile:**
  * `1` tile 16-lane trong TCDM = `16 words × 72 bits = 1,152 bits = 144 bytes`.
  * `1` tile `144 bytes` bằng: $144 / 64 = 2.25$ DRAM beats $\Rightarrow$ `4 tiles = 4 × 144 = 576 bytes = 9 DRAM beats` (vừa khít nguyên bội số 64B, **0 byte lãng phí padding** trên DRAM!).
  * `1` vector FP16 (gồm 4 words 72-bit) chiếm `4 × 9 = 36 bytes` vật lý TCDM $\Rightarrow$ `16` vectors FP16 (`M=16, D=16`) = `64 words × 9B = 576 bytes = 9 DRAM beats`!

#### Bất đối xứng Cực lớn giữa 2 chiều của `snax_simple_dma.sv` (`DRAM -> TCDM` vs `TCDM -> DRAM`)

So sánh trực tiếp hai nhánh trạng thái `ST_DRAM_TO_TCDM` (`L223-L294`) và `ST_TCDM_TO_DRAM` (`L296-L337`):

| Đặc tính | Chiều `dir[0] = 0`: `DRAM -> TCDM` (`ST_DRAM_TO_TCDM`) | Chiều `dir[0] = 1`: `TCDM -> DRAM` (`ST_TCDM_TO_DRAM`) |
| :--- | :--- | :--- |
| **Số Port TCDM sử dụng** | **4 Ports (`0, 22, 23, 24`)** ghi song song qua `hold_valid_q[0..3]` | **DUY NHẤT 1 Port (`Master 0`)** đọc tuần tự! (`L139-L144`) |
| **Pipelining đọc/ghi TCDM** | Khi 4 hold regs đã trống (`next_hold_valid == '0`), refill ngay tối đa 4 words trong 1 chu kỳ | **`!tcdm_rd_inflight_q` chặn đứng pipeline!** Chỉ phát 1 TCDM read mới SAU KHI nhận xong `tcdm_rsp_valid_i[0]` của word trước $\implies$ **tối đa `1` word mỗi `2` chu kỳ** (`4.5 B/cycle`)! |
| **Thông lượng đỉnh với Layout `16-bank` (`dir[1]=0`)** | `4 words/cycle = 36 bytes/cycle` (đủ nuốt non nửa DRAM beat `64B` mỗi 1.78 cycles) | **`0.5 word/cycle = 4.5 bytes/cycle`** (chậm hơn **8×** so với chiều vào!) |
| **Thông lượng với Layout `linear` (`dir[1]=1`)** | Cả 4 port đụng chung 1 bank (`+0x200`) $\implies$ Crossbar arbitrate xuống còn **`1 word/cycle = 9 bytes/cycle`** | **`0.5 word/cycle = 4.5 bytes/cycle`** (còn chưa tính xung đột bank nếu `Act/WO/RD` chạm cùng bank!) |
| **Ảnh hưởng Crossbar Priority** | `Master 0` ở ưu tiên thấp nhất; `Masters 22..24` thua `1..21` | `Master 0` ở ưu tiên **thấp nhất** trong toàn bộ 25 masters $\implies$ bất kỳ master nào truy cập trùng bank là DMA port 0 đứng chờ ngay! |

**Phát hiện quan trọng cho người thiết kế Sequencer từ bảng trên:**
Vì chiều **`TCDM -> DRAM`** hiện chỉ chạy ở tốc độ **`1` word / `2` chu kỳ (`4.5 bytes/cycle`)**, nếu ta xả output ở chế độ rời rạc (`block hết compute` rồi mới gọi **`DMA-WB`** như `u_seq` cũ), thì cứ mỗi `1,000` words output, toàn bộ NPU Core (`256 MACs/cycle`) phải **ngồi chết tối thiểu `2,000+` chu kỳ** chỉ chờ `DMA-WB` đổ từng word 72-bit ra DRAM!
Ngược lại, khi dùng **Ping-Pong `O_buf0 / O_buf1` (`TAG_O_WB`)** tích hợp trong Sequencer hợp nhất (như đã làm ở `snax_gemm_os_sequencer.sv` với `CAP_O_TILES = 16` và cần làm y hệt cho FlashAttention & SFU): quá trình chậm `0.5 word/cycle` của chiều `TCDM -> DRAM` được **giấu hoàn toàn dưới bóng thời gian tính toán của cụm tile tiếp theo** (overlap `Compute` và `DMA Writeback`)!

---

### 4.2. Giải phẫu Lệnh `DMA-WB` Rời rạc trong `snax_llm_sequencer.sv`

Xem trực tiếp code tại [snax_llm_sequencer.sv#L943-L956](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L943-L956):

```systemverilog
            ST_DMA_WB_LAUNCH: begin
                dma_req_valid_o      = 1'b1;
                dma_req_dir_o        = cfg_out_mode_i; // [0]: 0=DRAM->TCDM, 1=TCDM->DRAM; [1]: 0=16-bank, 1=linear
                dma_req_dram_addr_o  = cfg_dram_kv_base_i;
                dma_req_tcdm_addr_o  = cfg_tcdm_out_base_i;
                dma_req_size_bytes_o = cfg_stride_tile_i;
                if (dma_req_ready_i)
                    state_d = ST_DMA_WB_WAIT;
            end

            ST_DMA_WB_WAIT: begin
                if (dma_done_i)
                    state_d = ST_DONE;
            end
```

Chỉ **14 dòng SystemVerilog**, hoàn toàn **không có vòng lặp 2D/3D, không có stride, không có Ping-Pong, không có overlap**, chỉ làm đúng việc **chuyển tiếp 4 thanh ghi CSR (`cfg_out_mode_i`, `cfg_dram_kv_base_i`, `cfg_tcdm_out_base_i`, `cfg_stride_tile_i`) thẳng xuống cổng `dma_req_*` rồi khóa chết toàn bộ Sequencer chờ `dma_done_i`!**

Và để chỉ chạy 14 dòng chuyển tiếp đó, `snax_llm_top_clean.sv` phải:
1. Dành riêng bit `RW_SEQ_CTRL[6]` (`csr_seq_dma_wb`) để chặn `gemm_os_active` (`!csr_seq_dma_wb`, [snax_llm_top_clean.sv#L351](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L351)) khiến CPU không dùng được `u_gemm_seq`, ép rơi về `u_seq`.
2. Chiếm dụng `RW_SEQ_CTRL[5:4]` (`cfg_out_mode_i`) làm `dma_req_dir_o`, `RW_SEQ_OUT` (`tcdm_addr`), `RW_SEQ_DRAM_KV` (`dram_addr`), `RW_SEQ_TILE_ST` (`size_bytes`) — gây alias lẫn lộn ngữ nghĩa thanh ghi.
3. Buộc CPU phải **polling chờ xongCompute $\rightarrow$ lập CSR mới $\rightarrow$ chạy `DMA-WB` $\rightarrow$ polling chờ xong DMA $\rightarrow$ lập lại toàn bộ bộ CSR cho phép tính kế tiếp**!

---

### 4.3. Truy vết Toàn bộ 7 Điểm Chặn `DMA-WB` Trong 1 Layer Llama-3.2-1B (`tb_llama3_1b_layer.sv`)

Kiểm chứng trực tiếp tại [tb_llama3_1b_layer.sv#L905-L1130](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_llama3_1b_layer.sv#L905-L1130) cho cấu hình chuẩn Prefill (`M_TOKENS = 16`, `D_MODEL = 2048`, `D_FF = 8192`, `H_Q = 32`, `H_KV = 8`, `D_K = 64`):

| Lần gọi | Dòng code | Gọi ở pha nào | Hướng & Kích thước (`bytes`) | Nguyên nhân thật sự phải gọi `DMA-WB` rời | Ai phải xử lý việc này trong Kiến trúc Hợp nhất? |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **#1** | `L920` (Prefill M=16) / `L1129` (M>16) | Ngay sau `[1/12] SFU RMSNorm 1` | `TCDM -> DRAM` (`8192 × 9 = 73,728 B`, `dir=01`) | Phải cất `X` (`Banks 16..31`) ra DRAM để nhường TCDM cho Q/K/V/Attention! (Lúc chạy M>16 ở `L1129` thì đẩy ra `DRAM_X` từng batch 16 tokens!) | `u_sfu` / **Host đầu layer vốn đã có `X` ở DRAM** (nếu layer bắt đầu từ DRAM); hoặc **Macro-DMA của SFU** tự đọc `X` từ DRAM và ghi output thẳng thay vì lấy TCDM làm bộ đệm vạn năng rồi xả tay! |
| **#2** | `L1007` | Sau `[5/12] Fused V + Transpose` | `TCDM -> DRAM` (`512 × 9 = 4,608 B`, `dir=01`) | `Fused GEMM V` đẩy V-transpose vào `Banks 0..15` trên TCDM (`L991`), CPU phải chép thủ công từng packet vào `DRAM_KV` và gọi `DMA-WB` xả ra `DRAM_VC`! | **GEMM 4-Loop + V-Transpose tích hợp `TAG_O_WB`** xả thẳng `V_transposed_FP4` từ `O_buf` ra DRAM KV-Cache! |
| **#3** | `L1013` | Trước `[6/12] FlashAttention` | `DRAM -> TCDM` (`1024 × 9 = 9,216 B`, `dir=10` linear) | **`u_seq` FlashAttention thiếu Macro-DMA nạp `Q`!** Buộc CPU gọi `DMA-WB` nạp sẵn `QPKT[0..15]` vào `Bank 30`. | **FlashAttention Redesigned** tự nạp từng cụm `Q_sub` từ `DRAM_Q` qua `TAG_Q` Ping-Pong (`Banks 16..31`)! |
| **#4** | `L1014` | Trước `[6/12] FlashAttention` | `DRAM -> TCDM` (`1024 × 9 = 9,216 B`, `dir=10` linear) | Như trên: nạp tiếp `QPKT[16..31]` vào `Bank 31` — và nếu `N_Q = 256` (`Q = 288 KB`), cách nạp tay trước này **CHẾT ĐỨNG hoàn toàn** vì hết bank TCDM! | Như trên (`TAG_Q` tự động kéo cho mọi `N_Q` bất kỳ, dù `16` hay `4096` tokens)! |
| **#5** | `L1030` | Trước `[7/12 & 8/12] GEMM O + Residual 1` | `DRAM -> TCDM` (`8192 × 9 = 73,728 B`, `dir=00`) | Phải kéo lại `X` từ DRAM lên `Banks 16..31 (words 512..)` để chuẩn bị cho `u_sfu_seq` tính `Residual 1` — **vì `u_sfu_seq` hoàn toàn không có DMA!** | **SFU Engine có Macro-DMA B** tự kéo từng chunk `X` từ `DRAM_X` song song trên Ping-Pong buffer khi chạy Residual 1 (thậm chí không cần chiếm 73 KB tĩnh trên TCDM)! |
| **#6** | `L1043` | Ngay sau `[8/12] Residual 1` | `TCDM -> DRAM` (`8192 × 9 = 73,728 B`, `dir=01`) | Phải cất `RES1` ra DRAM để nhường chỗ TCDM (`words 512..1023`) cho `FF1` của FFN (`Gate/Up -> SwiGLU`)! | **SFU Engine (`TAG_O_WB`)** tự xả `RES1` ra DRAM theo Ping-Pong trong khi vẫn tính vector tiếp theo — và khi `M > 16`, đây là điều **BẮT BUỘC** để `Residual 1` chạy được >288 KB! |
| **#7** | `L1086` | Ngay trước `[12/12] Residual 2` | `DRAM -> TCDM` (`8192 × 9 = 73,728 B`, `dir=00`) | Kéo lại `RES1` từ DRAM vào TCDM cho `u_sfu_seq` cộng với kết quả `GEMM Down`! | **SFU Macro-DMA** tự kéo chunk `RES1` từ DRAM vào Ping-Pong TCDM! |
| **#8** | `L1110` (cả Decode `L905` lẫn Prefill `L1110`) | Cuối `[12/12] Residual 2` | `TCDM -> DRAM` (`8192 × 9 = 73,728 B` hoặc `512 × 9 B`) | Đẩy output cuối layer từ TCDM ra `DRAM_OUT`. | **SFU Macro-DMA (`TAG_O_WB`)** tự xả `Final_Out` ra `DRAM_OUT` ngay ở cuối mỗi chunk (hoặc host lấy ngay trong DRAM sau 1 lệnh `Residual 2`)! |

**Đọc kỹ bảng 8 dòng trên:**
Tính riêng trong Prefill 1 layer `M=16`: tổng lưu lượng bưng bê thủ công bằng `DMA-WB` là:
$$73,728 + 4,608 + 9,216 + 9,216 + 73,728 + 73,728 + 73,728 + 73,728 = \mathbf{391,680\text{ bytes} \approx 382.5\text{ KB}}$$
Và quan trọng hơn lưu lượng:
1. Toàn bộ 8 lần gọi đó đều là **blocking stalls** (CPU phải dừng tính toán, setup CSR, chạy DMA, wait done, rồi mới setup CSR compute tiếp).
2. Khi tăng lên **`M_TOKENS = 256`** (`X` FP16 = `16 × 73,728 B = 1.18 MB`), **cả phần `u_sfu_seq` lẫn `FlashAttention` đều KHÔNG CHẠY ĐƯỢC TRONG 1 LỆNH**, buộc CPU phải viết vòng `for (int mb = 0; mb < M_CHUNKS; mb++)` bên ngoài (`tb_llama3_1b_layer.sv#L1118`) — vi phạm trực diện **Tiêu chí 1** của Sequencer!

---

### 4.4. So sánh `DMA-WB` Rời vs `TAG_O_WB` / `TAG_IN` Đã Chứng minh Thành công ở `snax_gemm_os_sequencer.sv`

Trong `snax_gemm_os_sequencer.sv` (mà ta đã verify pass 0 errors trên cả Verilator & QuestaSim với `tb_gemm_os_all_sizes`), ta đã thấy **điều gì xảy ra khi DMA không tách làm khối riêng mà nằm chung bên trong 1 Sequencer với 3 sub-FSM**:
1. Trong lúc Compute (`cmp_state_q`) đang tính cụm tile `(m_macro, n_tile)` vào `O_buf[0]` và `Drain FSM` (`dr_state_q`) đang xả kết quả ra TCDM, `DMA FSM` (`dma_state_q`) đồng thời làm 3 việc theo thứ tự ưu tiên:
   - **`TAG_O_WB` (Ưu tiên cao nhất):** Đẩy `O_buf[1]` (cụm vừa tính xong ở lượt trước) từ TCDM ra DRAM.
   - **`TAG_A` Urgent / Prefetch:** Nạp trước cụm `A` kế tiếp (`m_macro + 1`) vào `A_buf[!a_cmp_rd_sel_q]`.
   - **`TAG_W` Prefetch:** Nạp trước `W` tile kế tiếp vào Ring slot trống (`w_ring_credits`).
2. Kết quả: CPU chỉ bắn **ĐÚNG 1 LẦN `start_i`**, toàn bộ `M=128, N=128, K=512` (lớn hơn nhiều lần 288 KB TCDM) tự động nạp `A`, nạp `W`, tính, drain, và xả `O` thẳng ra `DRAM_OUT` — **`DMA-WB` biến mất hoàn toàn ở góc độ API lớp trên** vì nó đã tan vào bên trong cơ chế `TAG_O_WB` của Sequencer!

**Vậy câu hỏi đặt ra ngay lập tức là:** Nếu `GEMM` làm được điều đó để xóa bỏ `DMA-WB` rời, tại sao **FlashAttention** và **SFU/Post-Process (`Readback` cũ)** lại không dùng **cùng chính `DMA Engine 3-Tag` đó** ngay bên trong **1 Sequencer duy nhất**?! Câu trả lời là: **KHÔNG CÓ BẤT KỲ RÀO CẢN PHẦN CỨNG NÀO CẢ** — chỉ vì thiết kế cũ chia 3 sequencer rời nên mới giữ `DMA-WB` và `Readback` làm hai khối vá víu bên ngoài!

---

## Chương 5. Giải phẫu Hệ thống SFU (`snax_llm_sfu_opt.sv` + `snax_sfu_sequencer.sv`) & Điểm Chết ở Prefill Lớn

Mấu chốt thứ ba giải thích tại sao `Readback` và `DMA-WB` bị tách rời nằm ngay ở cách **`u_sfu` và `u_sfu_seq` đang được ghép nối tách biệt khỏi `u_seq` và `u_gemm_seq`** tại [snax_llm_top_clean.sv#L917-L976](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L917-L976) và [`#L1266-L1374`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L1266-L1374).

### 5.1. Kiến trúc Phần cứng Bên trong `snax_llm_sfu_opt.sv`

#### Tham số và Chia Chunk ALU (`ALU_LANES = 8` mặc định tại Top)
Trong [snax_llm_top_clean.sv#L46](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L46) và [`#L1352-L1355`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L1352-L1355), `u_sfu` được khởi tạo với:
* `NUM_LANES = 16` (giao tiếp ngoài luôn cố định là **vector 256-bit = 16 phần tử FP16**).
* `ALU_LANES = 8` (`SFU_ALU_LANES` — hỗ trợ `16`, `8`, hoặc `4`).
* `NUM_CHUNKS = NUM_LANES / ALU_LANES = 16 / 8 = 2` chunks (`CHUNK_W = 1`, `ALU_BITS = 128 bits`).
* Nội bộ chỉ instantiate đúng **`8` bộ nhân FP20 (`sfu_fp20_mul`)**, **`8` bộ cộng/trừ FP20 (`sfu_fp20_addsub`)**, **`8` bộ `sfu_sigmoid_pwl_fp20`**, và **`1` bộ `sfu_rsqrt_fp20`** dùng chung cho cả 5 phép toán (`SFU_BYPASS=0`, `SFU_RESIDUAL=1`, `SFU_RMSNORM=2`, `SFU_ROPE=3`, `SFU_SWIGLU=4`).

#### Bảng Bóc tách Chính xác Từng Chế độ Thực thi của `snax_llm_sfu_opt.sv` (`ALU_LANES = 8`)

| `cfg_op_i` | Tên phép toán | Định nghĩa Toán học trên từng Chunk (`8` phần tử FP20) | Độ trễ mỗi Vector (`ALU_LANES=8`) | Cổng đầu vào cần dùng | Quan hệ với `Readback` & `DMA-WB` |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **`3'd0`** | `SFU_BYPASS` | `out_fp16_o <= in_a_fp16_i` (giữ nguyên toàn bộ `256-bit`, không unpack FP20) | **`1 cycle/vec`** (pipeline thẳng `active_vec_q = 0`) | Chỉ `in_a` (`in_b` bỏ qua!) | **TRÙNG 100% chức năng đọc-đẩy của `Readback` (`op=11`)** nếu nối `in_a <- rd_fp16` và `out -> LogQ -> WO`! |
| **`3'd1`** | `SFU_RESIDUAL` | `add_res[l] = a_fp20[l] + b_fp20[l]` $\rightarrow$ pack `slice_fp16_comb` (`NUM_CHUNKS = 2`) | **`2 cycles/vec`** (`chunk 0 -> chunk 1 -> emit`) | `in_a` (`TCDM` hoặc `fused_fifo`) + `in_b` (`TCDM`) | Cần giữ `X` hoặc `RES1` ở DRAM khi `>288 KB` $\implies$ hiện phải gọi `DMA-WB` tay trước/sau mỗi chunk! |
| **`3'd2`** | `SFU_RMSNORM` | Pass 1: tích lũy $\sum x^2$ bằng cây cộng 3 tầng (`add_res[0..6] + add_res[7]`); `ST_RMS_RSQRT`: $1/\sqrt{\mu(x^2)+\epsilon}$; Pass 2 (`phase 0 -> 1`): $(x \cdot \text{inv\_rms}) \cdot \gamma$ | Pass 1: **`2 cyc/vec`**; RSQRT: **`1 cyc`**; Pass 2: **`4 cyc/vec`** (`2 chunks × 2 phases`) | Pass 1: `in_a` (X); Pass 2: `in_a` (X) + `in_b` ($\gamma$) | Hiện phải xuất `FP16` ra TCDM rồi gọi thêm `Readback` (`run_requant`) mới ra `FP4` — thừa cả pha ghi/đọc FP16! |
| **`3'd3`** | `SFU_ROPE` | Với 4 cặp `(xe, xo)` trong chunk: Phase 0 tính `xe*cos - xo*sin` (`add_sub=1`) ghi vào vị trí chẵn `out_fp16_o`; Phase 1 tính `xe*sin + xo*cos` ghi vào vị trí lẻ | **`4 cycles/vec`** (`2 chunks × 2 phases`) | `in_a` (`fused_fifo` từ `Q/K_proj` hoặc `rd_fp16`) + `in_b` (`sin/cos` từ `rd_b`) | Đã được fuse hoàn toàn với `u_core` trong `Method B` (`tb_fused_qk_rope_compare.sv`) $\implies$ xóa sạch `Readback` ở Q/K! |
| **`3'd4`** | `SFU_SWIGLU` | Phase 0: `tmp_chunk_q[l] <= a_fp20[l] * sigmoid(a_fp20[l])` (tính `SiLU(Gate)`); Phase 1: `mul_res[l] = tmp_chunk_q[l] * b_fp20[l]` (nhân `Up`) | **`4 cycles/vec`** (`2 chunks × 2 phases`) | `in_a` (`Gate`) + `in_b` (`Up`) | Có thể dùng `sfu_out_dst = 0` xuất thẳng `FP4` qua `LogQ -> WO`, hoàn toàn không cần `Readback`! |

---

### 5.2. Cơ chế Fused Datapath (`u_core -> fused_fifo -> u_sfu -> u_log_quantizer / u_write_out`)

Để hiểu vì sao `Readback` biến mất ở `Q+RoPE`, `K+RoPE`, và có thể biến mất ở hầu hết các bước khác, cần kiểm chứng chính xác mạch ghép nối giữa `u_core` và `u_sfu` tại [snax_llm_top_clean.sv#L224-L231](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L224-L231), [`#L1122-L1140`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L1122-L1140), và [`#L1322-L1351`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L1322-L1351):

#### Cấu trúc `fused_fifo` (4-entry FIFO decoupling)
```systemverilog
    logic [3:0][255:0] fused_fifo_mem;
    logic [1:0]        fused_wptr, fused_rptr;
    logic [2:0]        fused_cnt;
    wire               fused_fifo_full      = (fused_cnt == 3'd4);
    wire               fused_fifo_empty     = (fused_cnt == 3'd0);
    wire               fused_fifo_can_drain = (fused_cnt <= 3'd1);
```

Tại sao điều kiện cho phép Core phát `tile_drain_req` mới là `fused_fifo_can_drain = (fused_cnt <= 3'd1)`?
* Đường ống xả `Core` (`rmw_latch_acc_buffer` $\rightarrow$ `final_normalizer S1` $\rightarrow$ `final_normalizer S2`) có **tối đa 3 vector đang bay trong pipeline** (khi tính thêm chu kỳ cấp phát tại Drain FSM).
* Nếu tại chu kỳ cấp `tile_drain_req` có `fused_cnt <= 1`, thì dù `u_sfu` tạm dừng nhận từ `fused_fifo` ngay lúc đó (do đang bận 4 chu kỳ RoPE mỗi vector), cả `3` vector đang bay cộng `1` vector đã có sẵn trong FIFO $= 4$ entries — vừa khít chiều sâu `4` của `fused_fifo_mem` mà **KHÔNG BAO GIỜ ghi đè lên 1-entry skid `out_hold_q` trong Core**!
* So sánh sự tương đồng **100% về mặt toán học điều khiển** với hằng số `OUT_PIPE_CREDITS = 4` và `wo_fifo_headroom` (`in_fifo_level <= FIFO_DEPTH - 4`) của `snax_streamer_write_out`:
  * Ở Top: khi `(sfu_en && sfu_in_a_src)`: `core_drain_ready = fused_fifo_can_drain` (tối đa `4 - 1 = 3` slot trống).
  * Khi xuất thẳng `u_write_out`: `core_drain_ready = wo_fifo_headroom` (`16 - 4 = 12`, giữ tối thiểu `4` slot trống).
  * Và nếu `u_gemm_seq` dùng `inflight_out_cnt_q < 4` kết hợp cùng `core_fp16_valid_i && core_fp16_ready_i`, nó bảo vệ tuyệt đối cả 2 trường hợp (FIFO của `u_write_out` lẫn `fused_fifo` của `u_sfu`) bằng **cùng 1 cơ chế Credit duy nhất**!

#### Ma trận 4 Đường ra của `u_sfu` và Đường Multiplexing `u_log_quantizer` Hiện tại

Kiểm tra các bit cấu hình của `RW_SFU_CTRL` tại [snax_flash_csr_snax.sv#L294-L299](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_flash_csr_snax.sv#L294-L299) và logic routing tại [snax_llm_top_clean.sv#L1065-L1140](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L1065-L1140):

| `sfu_en` | `sfu_in_a_src` | `sfu_out_dst` | `wo_mode_fp16` | Nguồn A $\rightarrow$ Đầu ra thực tế | Ví dụ sử dụng thực tế | Có cần `Readback` sau đó? |
| :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **`0`** | `x` | `x` | `1` | `u_core` $\rightarrow$ `u_write_out` (FP16) | GEMM xuất FP16 (`GEMM Down` trước Residual 2) | Chỉ cần nếu CPU sửa FP16 rồi muốn nén FP4 |
| **`0`** | `x` | `x` | `0` | `u_core` $\rightarrow$ `u_log_quantizer` $\rightarrow$ `u_write_out` (FP4) | GEMM / FlashAttention xuất thẳng FP4 | **KHÔNG** |
| **`1`** | **`1`** | **`0`** | **`0`** | `u_core` $\rightarrow$ `fused_fifo` $\rightarrow$ `u_sfu` $\rightarrow$ `u_log_quantizer` $\rightarrow$ `u_write_out` (FP4) | **Fused Q+RoPE, Fused K+RoPE (`Method B`)** | **KHÔNG** (đã loại bỏ hoàn toàn `Readback`!) |
| **`1`** | **`1`** | **`1`** | **`1`** | `u_core` $\rightarrow$ `fused_fifo` $\rightarrow$ `u_sfu` $\rightarrow$ `u_write_out` (FP16) | **Fused `GEMM O + Residual 1`** hoặc **`GEMM Down + Residual 2`** (tính trực tiếp ở đuôi GEMM không cần chờ riêng!) | **KHÔNG** (kết quả cần giữ FP16 để vào Norm hoặc trả Layer Out!) |
| **`1`** | **`0`** | **`0`** | **`0`** | `u_read_fp16 (A)` (+ `u_read_fp16_b`) $\rightarrow$ `u_sfu` $\rightarrow$ `u_log_quantizer` $\rightarrow$ `u_write_out` (FP4) | **SwiGLU (`Gate*sigmoid*Up -> FP4`)**, **RMSNorm 1/2 xuất thẳng FP4**, VÀ **chính là `SFU_BYPASS -> LogQ` (`Readback` hợp nhất!)** | **KHÔNG** (`LogQ` chạy inline ngay ở đuôi!) |
| **`1`** | **`0`** | **`1`** | **`1`** | `u_read_fp16 (A)` (+ `u_read_fp16_b`) $\rightarrow$ `u_sfu` $\rightarrow$ `u_write_out` (FP16) | RMSNorm xuất FP16 (khi cần giữ bản FP16 để check hoặc dùng nhiều nơi), Residual từ TCDM ra FP16 | Chỉ nếu sau đó muốn tạo thêm bản FP4 mà không chạy Pass 2 |

**Nhìn vào hàng thứ 4 (`sfu_en=1, sfu_in_a_src=1, sfu_out_dst=1, wo_mode_fp16=1`) và hàng thứ 5 (`sfu_en=1, sfu_in_a_src=0, sfu_out_dst=0`):**
* Hardware **ĐÃ ĐẤU SẴN** đường để `GEMM O_proj` và `GEMM Down_proj` **cộng thẳng `Residual`** ngay trên luồng thoát của `u_core` qua `fused_fifo -> u_sfu`!
* Hardware **CŨNG ĐÃ ĐẤU SẴN** đường `u_read_fp16 -> u_sfu -> u_log_quantizer -> u_write_out` cho phép **VỪA** chạy `RMSNorm`/`SwiGLU`/`BYPASS` **VỪA** lượng tử thẳng sang `FP4`!
* **VẬY TẠI SAO TRÊN THỰC TẾ `tb_llama3_1b_layer.sv` VẪN KHÔNG KHÉP KÍN ĐƯỢC?** Câu trả lời nằm ở các giới hạn ngớ ngẩn của `snax_sfu_sequencer.sv` + sự chia cắt 3 Sequencer!

---

### 5.3. Kiểm toán Toàn bộ Giới hạn và Rủi ro của `snax_sfu_sequencer.sv` (`u_sfu_seq`)

Đọc từng dòng trong [snax_sfu_sequencer.sv#L1-L256](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_sfu_sequencer.sv#L1-L256) và cách đấu ở [snax_llm_top_clean.sv#L918-L976](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L918-L976), ta phát hiện **6 lỗi/giới hạn nghiêm trọng**:

1. **Giới hạn #1 — ZERO DMA (`Mù DRAM hoàn toàn`):**
   `snax_sfu_sequencer` không có một cổng `dma_req_*` nào. Mọi địa chỉ `cfg_base_a_i`, `cfg_base_b_i`, `cfg_base_o_i` đều phải nằm sẵn 100% trong TCDM `288 KB`.
   * **Hậu quả cho `Llama-3.2-1B` (`D_MODEL = 2048` $\implies$ `128` vectors FP16/token $\implies$ `512` words $\times$ `9B` = `4,608 B`/token):**
     * `M = 1` (Decode): `X` chỉ tốn `4.5 KB` $\rightarrow$ chứa thoải mái.
     * `M = 16` (Prefill vừa `16 banks`): mỗi token nằm trên 1 bank của `Banks 16..31`, mỗi bản sao (`X`, `XN`, `RES1`) ngốn **`512 words / 1024 words` = đúng 50% dung lượng của 16 banks cao (`72 KB`)**. Vì chỉ đủ chỗ cho **tối đa 2 bản sao FP16 (`X` và `XN`, hoặc `OUT` và `RES1`)**, nên ở `tb_llama3_1b_layer.sv`:
       - Sau `RMSNorm 1`, **BẮT BUỘC gọi `DMA-WB` đẩy `X` ra DRAM** để giải phóng chỗ!
       - Trước `Residual 1`, **BẮT BUỘC gọi `DMA-WB` kéo `X` về**, cộng xong lại **gọi `DMA-WB` đẩy `RES1` ra DRAM** nhường chỗ cho `FF1`!
       - Trước `Residual 2`, **BẮT BUỘC gọi `DMA-WB` kéo `RES1` về**!
     * **`M = 32, 64, 128, 256` (`N_Q >= 32`):**
       - `X` (chỉ riêng 1 bản tensor đầu vào!) chiếm:
         - `M = 32`: `32 × 512 × 9 B = 144 KB` (vừa khít toàn bộ `Banks 16..31`, **0 byte trống cho kết quả output `XN`!**)
         - `M = 64`: `288 KB` (chiếm **100% CẢ 32 BANKS TCDM** kể cả vùng Ring Weight!)
         - `M = 256`: `1,152 KB` (**vượt `4×` toàn bộ TCDM vật lý `288 KB`**!)
       - Do đó **`u_sfu_seq` CHẾT CỨNG (`FAIL Tiêu chí 1`)** với mọi `M >= 32` ở `D = 2048` (hoặc `M >= 16` khi `D = 4096`), buộc CPU phải chặt thành nhiều lần launch kèm `DMA-WB`!

2. **Giới hạn #2 — Chỉ chạy được khi `csr_src_sel == 1` (`A_CTRL[0] = 1`):**
   Nhìn [snax_llm_top_clean.sv#L918](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L918):
   ```systemverilog
   wire sfu_seq_active = (csr_src_sel && sfu_en && sfu_seq_en) || sfu_seq_busy;
   ```
   Tức là về mặt hình thức, `u_sfu_seq` mang danh "sequencer tự chủ", nhưng lại **ăn bám vào bit điều khiển manual `csr_src_sel = 1`**! Hệ thống không thể coi SFU là một lệnh đồng cấp trong chuỗi lệnh `seq` (`csr_src_sel = 0`) được.

3. **Giới hạn #3 — Hoàn toàn KHÔNG hỗ trợ `wo_mode` `4D` và phải mượn ké CSR của `GEMM` + `RD`:**
   * `snax_sfu_sequencer.sv` chỉ phát `rd_nv`, `wo_nv`, `wo_ls` kiểu `1D` (`32'h200` word stride cố định ở `RMSNorm` `L149, L170, L176`, hoặc gọi lặp `nv = 1` cho từng cặp `(t, g)` ở `S_2D_LAUNCH` `L202-L217`!).
   * Look closely at `S_2D_LAUNCH` (`L202-L244`)! Khi chạy `Residual` hoặc `SwiGLU` cho `M=16, N_TILES=128` (`2,048` vectors): `u_sfu_seq` phát **2,048 lần** `rd_start + rd_b_start + sfu_start + wo_start`, **mỗi lần chỉ xử lý đúng `nv = 1` vector** rồi chờ `wait_guard_q` 2 chu kỳ + chờ `!any_busy_p2` (bao gồm cả `wo_busy_i` đến tận khi `u_write_out` commit xong word cuối cùng xuống SRAM — mất tối thiểu ~15–20 chu kỳ cho 1 vector đơn lẻ)!
   * Tính thử chi phí của `S_2D_LAUNCH -> S_2D_WAIT` khi `M = 16, N = 128` (`2,048` vectors): vì `wo_busy_i` phải hạ về 0 sau MỖI 1 vector trước khi launch vector tiếp, `snax_sfu_sequencer` **phá hủy hoàn toàn cơ chế pipelining của `skid_buffer` 16-entry trong `u_write_out`**, làm tốc độ `Residual` / `SwiGLU` tụt xuống ~1/4 công suất thật của `u_sfu`!
   * Trong khi `wo_base`, `wo_nv`, `wo_nb`, `wo_bs`, `wo_dim2`, `wo_dim3` của `u_write_out` đã hỗ trợ sẵn loop 2D/4D bên trong phần cứng — chỉ cần `u_read_fp16` hỗ trợ loop hoặc `u_sfu_seq` **không đợi `wo_busy_i`** từng vector 1 (chỉ chờ khi hết chunk/job) thì tốc độ tăng gấp nhiều lần!

4. **Giới hạn #4 — Stride bị cắt ngắn còn 16-bit & nhân 16 byte cứng tại Top (`L933-L940`):**
   ```systemverilog
   .cfg_stride_a_t_i ({12'd0, csr_rd_line_stride[15:0], 4'd0}),
   .cfg_stride_a_g_i ({12'd0, csr_rd_line_stride[31:16], 4'd0}),
   ```
   Bởi vì CSR file hết chỗ sạch để cấp cho `u_sfu_seq`, Top-level đã ép 2 stride `T` và `G` vào cùng thanh ghi `csr_rd_line_stride` rồi dịch trái 4 bit (`<< 4`), nghĩa là stride tối đa bị giới hạn 16-bit word-aligned (`< 1 MB` và phải khác cách dùng 32-bit byte-stride của `Readback`).

5. **Giới hạn #5 — Chồng chéo `wo_mode_fp16_q` & `sfu_out_dst` ở Top:**
   Ở [snax_llm_top_clean.sv#L981-L986](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L981-L986), `wo_mode_fp16_q` chỉ chốt từ `wo_mode_fp16_sel = csr_src_sel ? csr_wo_mode_fp16 : (seq_wo_mode == 2'd1)`. Nếu bạn bật `u_sfu` với `sfu_out_dst = 0` (`LogQ -> FP4`) nhưng quên set `A_WO_MODE = 0` (khi `csr_src_sel = 1`), `u_log_quantizer` bị khóa bởi `!wo_mode_fp16` (`L1085`) và `u_write_out` đợi FP16 mãi mãi $\rightarrow$ **deadlock âm thầm**!

6. **Giới hạn #6 — Bug tiềm ẩn `start_i` của `u_sfu` ở `SFU_BYPASS` và `RMSNorm`:**
   Trong `u_sfu_seq` (`L146-L182`): `S_RMS_P1_LAUNCH` bắn `sfu_start_o = 1`, rồi đợi `rd_busy_i` tắt (`any_busy_p1 = rd_busy_i` ở `L91`!). Chú ý: **ở `S_RMS_P1_WAIT` KHÔNG KIỂM TRA `sfu_busy_i` hay `rms_st`**, chỉ dựa vào niềm tin rằng sau khi `rd_busy_i` tắt được 2 chu kỳ thì `u_sfu` đã chạy xong chunk cuối của vector cuối và nhảy qua `ST_RMS_RSQRT -> ST_RMS_PASS2`! Nếu `ALU_LANES = 4` (`NUM_CHUNKS = 4` $\implies$ mất `4` chu kỳ xử lý sau khi `u_read_fp16` `EMIT` beat cuối!), `wait_guard_q = 2` cộng với `1` chu kỳ RSQRT có nguy cơ **`S_RMS_P2_LAUNCH` bắn `in_a` và `in_b` sớm hơn 1 chu kỳ trong khi `u_sfu` vẫn đang xử lý chunk 3 của vector cuối Pass 1**!
   Kiểm tra lại bằng phép tính chu kỳ thực (`ALU_LANES = 4`):
   - `T`: `u_read_fp16` vào `EMIT` của vector cuối, `in_a_valid_i = 1`, `in_a_ready_o = 1` (vì `!active_vec_q`) $\rightarrow$ `u_read_fp16` chốt `done_irq_o=1 -> IDLE` (`rd_busy_i` tắt ở `T+1`), còn `u_sfu` nạp `active_vec_q=1, chunk_q=1` ở `T+1`.
   - `T+1`: `wait_guard_q` từ `2 -> 1`, `u_sfu` chạy `chunk 1`. (`rd_busy_i = 0`!)
   - `T+2`: `wait_guard_q` từ `1 -> 0`, `u_sfu` chạy `chunk 2`.
   - `T+3`: `u_sfu_seq` thấy `wait_guard_q == 0 && !rd_busy_i` $\implies$ nhảy sang `S_RMS_P2_LAUNCH`! Trong khi tại `T+3`, `u_sfu` mới đang ở `chunk 3` (`last_chunk`) của Pass 1! Phải `T+4` nó mới sang `ST_RMS_RSQRT`, và `T+5` mới vào `ST_RMS_PASS2`! Vì `S_RMS_P2_LAUNCH` lại kích `rd_start_o` mất vài chu kỳ đọc từ TCDM nên nó *ăn may* chưa đụng tới `in_a_valid_i` ở `T+4` — nhưng **nếu `u_sfu` nối với stream nhanh (FIFO 1 chu kỳ) thì `wait_guard_q = 2` LÀ BUG!** Nếu kiểm tra trực tiếp `!sfu_busy_i` hoặc đợi 1 xung `rms_pass1_done` thì mới chuẩn mực hardware!

---

## Chương 6. Giải phẫu `V-Transpose` (`snax_transpose_buffer.sv` + `snax_transpose_ctrl.sv`) & Gộp Flush KV Cache

Trong `u_seq` hiện tại có một opcode đứng riêng là `cfg_op_mode_i == 2'b10` (`ST_V_LAUNCH -> ST_V_RUN`, [snax_llm_sequencer.sv#L687-L704](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L687-L704)), đồng thời cũng có chế độ **Fused V** (`cfg_op_mode_i == 2'b00 && cfg_out_mode_i == 2'd3`) được cả `u_seq` lẫn `snax_gemm_os_sequencer.sv` gọi chung. Tại sao `V-Transpose` tồn tại và phần nào trong nó vẫn gây ra `DMA-WB` thừa?

### 6.1. Vì sao Phải Xoay Trục Ma trận `V`? (Bản chất Toán học của Latch 16 Lanes)

Đối chiếu giữa `GEMM V_proj` và `FlashAttention Mode 10 (P × V)`:
* **Đầu ra của phép nhân ma trận `V = X · W_V` (`GEMM`):**
  * Ma trận `V` kích thước `[M_tokens × D_V]` sinh ra theo từng hàng (row = token `t = 0..15`, cột = channel `c = 0..15` trên 16 lanes của Core).
  * Tức là 1 vector FP16 `core_fp16_packed` thoát ra từ `final_normalizer` chứa **16 channels của CÙNG 1 token `t`**: $V[t, 0..15]$.
  * Nếu đem `16` phần tử cùng 1 token đi nén thành 1 gói FP4 72-bit ngay lập tức, ta sẽ nén chung 1 scale `Log8` cho `16` **channels khác nhau của cùng 1 token**.
* **Đầu vào mà `FlashAttention Mode 10 (P × V)` cần ([snax_fused_llm_core.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv)):**
  * Trong pha `P × V`, 16 lanes phần cứng của `log8_fp4_mac_array` chạy song song theo **16 channels `d_v` (`c = 0..15`)**, còn chiều reduction bên trong mỗi gói FP4 72-bit phải là **16 tokens KV liên tiếp (`kj = 0..15`) của CÙNG 1 channel `c`**!
  * Cụ thể:
    - Lane `c` nhận 1 word 72-bit chứa `[V[kj=0..15, c]]` dùng chung `1` scale `Log8` cho 16 tokens dọc theo chiều `kj` của channel `c`.
    - `u_wt_streamer` đọc cả 16 lanes `[c=0..15]` song song từ 16 banks `0..15` trong 1 chu kỳ (`Master 3..18`)!

Vì **không thể** đổi chiều nén 16-to-1 của gói FP4 sau khi đã mất thông tin FP16 từng phần tử, phép xoay trục `16 tokens × 16 channels` **bắt buộc phải thực hiện ở dạng `FP16` TRƯỚC khi qua `u_log_quantizer`**!

### 6.2. Cơ chế Bắt tay `snax_transpose_buffer` + `snax_transpose_ctrl`

Xem [snax_transpose_buffer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_transpose_buffer.sv) và [snax_transpose_ctrl.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_transpose_ctrl.sv):

```text
  u_core (core_fp16_packed: 16 ch của row t)
         │ (row_valid_i = core_fp16_valid && core_fp16_ready)
         ▼
  [GATHER: 16 rows t=0..15] ---> mem[0..15][0..15] (256 x 16b FF = 4,096 bits)
         │ (khi row_q == 15)
         ▼
  [ISSUE -> WAIT] x 16 cột c=0..15:
         │ rd_data[t] = mem[t][rd_col] (16 tokens của channel c)
         │ col_valid_o = 1 (1 cycle pulse khi quant_accept == 1)
         │ WAIT 1 chu kỳ cho log_quantizer pipe ổn định (pipe_idle_i = quant_accept)
         ▼
  [u_log_quantizer] ---> [u_write_out (MODE_V: 16 channel packets, stride=+0x10 bank, batch_stride=+0x200)]
```

**Chú ý chi tiết đắt giá ở [tb_llama3_1b_layer.sv#L988-L1009](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_llama3_1b_layer.sv#L988-L1009):**
Trong phép `[5/12] Fused V + Transpose`:
1. `run_gemm_v_dma_ex` kích hoạt `GEMM` với `cfg_out_mode = 2'd3` (`Fused V`): `u_gemm_seq` (hoặc `u_seq`) tự bật `v_start_o = 1` ở nhịp Launch đầu tiên (`ST_GEMM_LAUNCH` / `CMP_KT_LAUNCH`), đồng thời `u_write_out` chạy `MODE_V` ghi `16` gói FP4 xoay trục dàn đều qua 16 banks `0..15` (`addr = base + batch*0x200 + ch*0x10`).
2. Nhưng ở `tb_llama3_1b_layer.sv#L991-L1008`, sau khi mỗi tile `(hkv, dvi)` xong, **CPU testbench phải dùng vòng `for` đọc từng word trong TCDM `cpu_mem_read(tcdm_addr(128+..., l), vp_pkt)` nhét thủ công vào `DRAM_KV` rồi gọi thêm `run_dma_writeback(2'b01, ...)` để lưu `DRAM_VC`!**
3. **Tại sao lại phải cực khổ vậy?**
   - Trong `snax_gemm_os_sequencer.sv` mới (4-Loop với `CAP_O_TILES` + `TAG_O_WB`), khi `cfg_gemm_macro_dma_i = 1` và `cfg_out_mode_i = 2'd3` (`Fused V`), mỗi batch `V` đã xoay trục xong được `u_write_out (MODE_V)` ghi thành **đúng `1` tile 16-bank (`16 words × 9B = 144 bytes`)** — định dạng **y hệt** layout 16-bank mà `snax_simple_dma` (`dir = 2'b01` -> TCDM->DRAM 16-bank) đọc ra!
   - Nghĩa là chỉ cần gắn thêm tính toán offset DRAM KV-cache ở pha `TAG_O_WB` (hoặc khi chạy ở chế độ Fused V Macro-DMA), `O_buf` chứa ngay `16` channel packets của tile `V` và **`TAG_O_WB` tự động đẩy thẳng ra `DRAM_KV_Cache` ngay trong lúc Compute đang chạy tile `V` tiếp theo** $\implies$ **XÓA ĐƯỢC THÊM 1 LỆNH `DMA-WB` (`DMA WB V-cache`) + XÓA ĐƯỢC `cfg_op_mode_i = 2'b10` (`ST_V_LAUNCH` standalone vô dụng khi đã Fuse với GEMM)!**

Kiểm tra nhanh trong toàn bộ `tb/`: có testbench nào còn gọi standalone `op_mode = 2'b10` (`ST_V_LAUNCH`) tách khỏi GEMM không? Gần như toàn bộ pipeline mới đều dùng `cfg_out_mode = 2'd3` (`Fused V` trong `GEMM`), tức `u_transpose_ctrl` không cần phải là 1 opcode sequencer đứng riêng nữa, mà chỉ là **1 chế độ output adapter (`out_mode == V_TRANSPOSE`) của động cơ GEMM**!

---

## Chương 7. Truy vết End-to-End 1 Layer LLM (`Llama-3.2-1B` & `Qwen2.5-1.5B`) — Hiện trạng Rời rạc vs Kiến trúc Hợp nhất

Để chấm dứt hoàn toàn lối tư duy cục bộ "mạnh khối nào khối nấy chạy", chương này truy vết **từng bước một từ `[1/12]` đến `[12/12]`** trong toàn bộ vòng đời thực thi 1 Transformer Layer tại [tb_llama3_1b_layer.sv#L784-L1292](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_llama3_1b_layer.sv#L784-L1292), đối chiếu trực tiếp qua 4 chế độ kích thước thực tế:
1. **Decode (`M_TOKENS = 1, N_KV = 512`)** trên `Llama-3.2-1B`
2. **Prefill nhỏ (`M_TOKENS = 16, N_KV = 16`)** vừa 1 chunk 16-bank của `Banks 16..31`
3. **Prefill vượt TCDM (`M_TOKENS = 64..256, N_KV = 256`)** — nơi toàn bộ kiến trúc rời rạc sụp đổ thành vòng lặp CPU + `DMA-WB` + `Readback`
4. **Đối chiếu mở rộng sang `Qwen2.5-1.5B`** (`D_MODEL = 1536`, `D_FF = 8960`, `H_Q = 12`, `H_KV = 2`, `GQA = 6:1`, `d_k = d_v = 128`)

### 7.1. Bảng Quy đổi Dung lượng Vật lý TCDM (`288 KB`) cho Tất cả Tensor Trong 1 Layer `Llama-3.2-1B`

Thông số mô hình chuẩn lấy từ [tb_llama3_1b_layer.sv#L29-L66](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_llama3_1b_layer.sv#L29-L66):
* `D_MODEL = 2048` (`ROW_WORDS = 512` words FP16/token; `K_PACKETS_QKV = 128` words FP4/token).
* `D_FF = 8192` (`K_PACKETS_DOWN = 512` words FP4/token; `N_TILES_FF = 512`).
* `H_Q = 32`, `H_KV = 8` (`GQA 4:1`, `gqa_group_m1 = 3`), `D_K = D_V = 64` (`K_PACKETS_QK = 4`, `DV_TILES = 4`).
* Tỉ lệ nén vật lý trên TCDM (`1 word = 72 bit = 9 bytes`):
  * 1 token **FP16** (`2048` phần tử): $2048 / 4 = \mathbf{512\text{ words} \times 9\text{ B} = 4,608\text{ B}}$ (chiếm 50% chiều sâu `1024 words` của `1` bank nếu dàn `1 token/bank`).
  * 1 token **FP4 + Log8** (`2048` phần tử): $2048 / 16 = \mathbf{128\text{ words} \times 9\text{ B} = 1,152\text{ B}}$ (chiếm 12.5% chiều sâu của `1` bank).
  * 1 token **FFN trung gian FP4** (`8192` phần tử): $8192 / 16 = \mathbf{512\text{ words} \times 9\text{ B} = 4,608\text{ B}}$.

| Tensor trong Layer | Định dạng | `M = 1` (Decode) | `M = 16` (1 chunk) | `M = 64` (`4` chunks) | `M = 256` (`16` chunks) | Chứa lọt `Banks 16..31` (`144 KB`)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`X` (Input hidden)** | FP16 | `4.5 KB` | `72 KB` (`512w × 16b`) | `288 KB` (`2×` Banks 16..31) | **`1,152 KB` (`8×` nửa trên)** | Chỉ khi `M <= 32` (với `M=32` đã kín `100%` không còn chỗ output!) |
| **`XN` (Sau RMSNorm1)** | FP16 | `4.5 KB` | `72 KB` (`512w × 16b`) | `288 KB` | **`1,152 KB`** | Chỉ khi `M <= 16` nếu cần chứa song song cùng `X` |
| **`Xn_fp4` (Input QKV)** | FP4 | `1.125 KB` | `18 KB` (`128w × 16b`) | `72 KB` | **`288 KB` (`2×` Banks 16..31)** | Tối đa `M <= 128` một mình; `<= 64` nếu giữ kèm buffer khác |
| **`Q` (Toàn bộ 32 heads)** | FP4 | `1.125 KB` | `18 KB` (`2,048 w`) | `72 KB` (`8,192 w`) | **`288 KB` (`32,768 w`)** | **Tràn toàn TCDM** ngay từ `M >= 128` khi cộng chung `O`! |
| **`K_new` (8 KV heads)** | FP4 | `0.281 KB` | `4.5 KB` | `18 KB` | `72 KB` | Vừa nếu đứng riêng, nhưng cộng KV cũ `N_KV >= 1024` $\implies$ tràn |
| **`V_new` (8 KV heads)** | FP4 | `0.281 KB` | `4.5 KB` | `18 KB` | `72 KB` | Như trên |
| **`K + V` Cache (`N_KV = 512`)** | FP4 | `36 KB` | `36 KB` | `36 KB` | `36 KB` | `Banks 0..15` (`144 KB`) giữ lọt `1` KV-head (thậm chí cả `8` KV-heads nếu cộng 18KB*8? Chờ: 1 KV-head `d=64`: $512 \times 4 \times 9 = 18\text{ KB}$ cho K + $18\text{ KB}$ cho V = **`36 KB`/head**; cả `8` KV-heads = **`288 KB`**!) |
| **`K + V` Cache (`N_KV = 2048`)** | FP4 | `144 KB`/head | `144 KB`/head | `144 KB`/head | `144 KB`/head | Vừa đúng `1` KV-head nằm gọn trong `Banks 0..15` (`144 KB`)! |
| **`Attn_O` (Toàn bộ 32 heads)** | FP4 / FP16 | `1.125 KB` / `4.5 KB` | `18 KB` / `72 KB` | `72 KB` / `288 KB` | **`288 KB` / `1,152 KB`** | Nếu xuất FP4: `M=16` tốn `18 KB`; `M=256` tốn `288 KB` (vượt nửa trên!) |
| **`RES1` (`O_proj + X`)** | FP16 | `4.5 KB` | `72 KB` | `288 KB` | **`1,152 KB`** | Phải đẩy DRAM khi `M >= 32`, hoặc ngay `M=16` nếu muốn nhường chỗ cho FFN! |
| **`Gate_fp16` / `Up_fp16`** | FP16 | `18 KB` × 2 | `288 KB` × 2 | `1,152 KB` × 2 | **`4,608 KB` × 2** | **Không bao giờ chứa nổi toàn bộ `D_FF=8192` ở FP16 ngay cả khi `M=16`!** Phải cắt theo `N`-chunk (`32` N-tiles/lượt) |
| **`FF1` (`SwiGLU -> FP4`)** | FP4 | `4.5 KB` | `72 KB` (`512w × 16b`) | `288 KB` | **`1,152 KB`** | `M=16` vừa đúng 50% `Banks 16..31`; `M>=32` vượt `144 KB` $\implies$ bắt buộc Macro-DMA! |

Nhìn bảng kích thước này ta rút ra ngay một nhận định xương sống:
> **Ở quy mô LLM thực (`D = 2048`, `D_FF = 8192`, `M >= 32`), `288 KB` TCDM CHỈ LÀ CACHE/SCRATCHPAD L1, KHÔNG PHẢI BỘ NHỚ CHỨA TOÀN TENSOR.**
> Do đó bất kỳ khối xử lý nào (dù là `GEMM`, `FlashAttention`, hay `SFU + LogQ`) **KHÔNG CÓ MACRO-DMA TỰ CHỦ** sẽ ngay lập tức biến thành điểm gãy của toàn layer, ép CPU quay lại vòng lặp cắt mảnh thủ công kèm `DMA-WB` và `Readback`!

---

### 7.2. Truy vết Toàn cảnh 12 Bước Trong Hiện trạng Rời rạc (`tb_llama3_1b_layer.sv`)

Hãy xem chính xác điều gì đang xảy ra từng bước tại [tb_llama3_1b_layer.sv#L908-L1291](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_llama3_1b_layer.sv#L908-L1291) khi `M_TOKENS = 64..256` (`M_CHUNKS = M_TOKENS / 16 > 1`):

#### Pha I — Tiền Attention (Lặp `M_CHUNKS` lần bởi CPU Host! `L1118-L1194`)

1. **Chuẩn bị thủ công bởi CPU (`L1119-1123`):**
   - CPU ghi trực tiếp `X[mb]` (`16 × 512` words FP16) vào `Banks 16..31 (words 0..511)` và `gamma1` (`512` words) vào `Bank 0 (words 0..511)` trên TCDM.
   - *(Đáng lẽ việc nạp `X` nếu xuất phát từ DRAM phải do DMA tự nạp, nhưng vì `u_sfu_seq` không có DMA nên TB không thể tự stream từ `DRAM_X` vào!)*

2. **Bước `[1/12]` `RMSNorm 1` (`L1125-1129`):**
   - CPU lật **`csr_src_sel = 1`** (`A_CTRL = 1`), cấu hình `u_sfu_seq` (`RW_SEQ_GEMM_MT = 16`, `RW_RD_LS`, `RW_WO_LS`), chạy 2-pass RMSNorm đọc `Banks 16..31 (words 0..511)` + `Bank 0 (gamma1)` $\rightarrow$ ghi `XN_fp16` vào `Banks 16..31 (words 512..1023)`.
   - **Ngay lập tức gọi `DMA-WB #1` (`L1129`):** CPU lật **`csr_src_sel = 0`**, set `seq_dma_wb = 1` (`A_SEQ_CTRL = 32'h50`), đẩy `X` từ `Banks 16..31 (words 0..511)` ra `DRAM_X + mb * 73,728` (`8,192 words × 9B = 73,728 B`) — chỉ để giải phóng `words 0..511`!

3. **Bước `[2/12]` Nén `XN -> ACT_FP4` bằng `Readback` (`L1131`):**
   - CPU gọi `run_requant_wrap`: đặt `A_SEQ_CTRL` dùng **`op_mode = 2'b11` (Readback)** trên `u_seq` (`wo_wrap_en=1, rd_wrap_en=1`).
   - `u_read_fp16` đọc lại chính `2,048` vectors `XN_fp16` ở `words 512..1023` (vừa mới ghi ở bước `[1/12]` xong!), chạy chậm `~9 cyc/vec` $\rightarrow$ qua nhánh `src_rd_sel` $\rightarrow$ `u_log_quantizer` $\rightarrow$ `u_write_out` ghi `18 KB` `ACT_FP4` đè lại vào `words 0..127` của `Banks 16..31`!
   - *Phát hiện vô lý #1:* Toàn bộ `73,728 B` ghi `XN_fp16` ở `[1/12]` và `73,728 B` đọc lại ở `[2/12]` (`~18,432` chu kỳ) là **hoàn toàn dư thừa** nếu `u_sfu_seq` xuất thẳng `sfu_out_dst = 0 (LogQ -> FP4)` ngay ở Pass 2 của RMSNorm 1! (Nhất là khi bản `X` nguyên gốc đã vừa được cất hoặc đã nằm ở DRAM rồi, chẳng có phép toán nào phía sau dùng tới `XN_fp16` cả!).
   - Sau đó CPU lại ghi tay `cos/sin` (`256` words) vào `Bank 29 (words 128..383)` (`L1132-1133`).

4. **Bước `[3/12]` `Fused Q + RoPE -> LogQ` (`L1135-1152`):**
   - CPU gọi **`u_gemm_seq`** (`op_mode = 2'b00, out_mode = 2'b01, enable_dma = 1`) kết hợp bật thủ công `u_sfu` + `u_read_fp16_b` (`A_START = 32'h183` kích 3 khối đồng thời!).
   - `Core` tính `128 N-tiles × 16 tokens` (của `mb`) $\rightarrow$ `fused_fifo` $\rightarrow$ `u_sfu (RoPE)` $\rightarrow$ `u_log_quantizer` $\rightarrow$ `u_write_out (wrap 255)` ghi `QPKT` vào `Banks 16..31 (words 128..255)`.
   - **Điểm đứt gãy (`L1137-1152`):** Vì `u_gemm_seq` ở test này chạy `out_mode = 2'd1` ghi vào **TCDM** (không bật Macro-DMA `TAG_O_WB` để đẩy thẳng ra DRAM và không tự xếp 4D cho từng head!), **CPU phải chạy vòng lặp `for (h, t, kp)` đọc hết `2,048` words TCDM và tự copy sang `dram_mem` (`DRAM_QPKT`)**!

5. **Bước `[4/12]` `Fused K + RoPE -> LogQ` (`L1154-1174`):**
   - Tương tự bước `[3/12]` cho `32 N-tiles` (`H_KV = 8, D_K = 64`): chạy xong trên TCDM, **CPU lại tiếp tục copy thủ công từng gói `K` từ TCDM sang `DRAM_KV` VÀ NHÂN BẢN LẶP LẠI `DV_TILES = 4` LẦN (`L1163-1173`)** chỉ vì `u_seq` Attention cũ ép layout DRAM của `K` và `V` phải gộp chung `768 B` cho mỗi `(hkv, dvi, kj)`!
   - **Thảm họa nhân bản dữ liệu:** Riêng với `d_v = 64` (`DV_TILES = 4`), `K` trong DRAM KV-Cache bị phình gấp **`4×`** dung lượng thực (nếu `d_v = 128` là **`8×`**)!

6. **Bước `[5/12]` `Fused V + Transpose` (`L1176-1194`):**
   - CPU gọi `u_gemm_seq` (`cfg_out_mode = 2'd3`, `Fused V`) **tới `32` lần riêng lẻ** (`hkv = 0..7`, `dvi = 0..3`, mỗi lần `N_tiles = 1`), ghi `16` gói FP4 vào `Banks 0..15` trên TCDM (`word = 128 + hkv*4 + dvi`).
   - Sau **mỗi lần** xong 1 tile `V` (trong tổng số 32 lần/chunk $\times$ `M_CHUNKS`), CPU lại đọc 16 packets từ TCDM chép thủ công vào `DRAM_KV`! *(Ở bản `M=16` dòng `1007`, sau vòng for còn gọi thêm **`DMA-WB #2` (`DMA WB V-cache`)** để đẩy `512 × 9 B` ra `DRAM_VC`!)*

#### Pha II — Attention (Hoạt động Đứt gãy theo Nhóm Head! `L1197-1219`)

7. **Bước `[6/12]` `HW Causal FlashAttention` (`L1200-1218`):**
   - Nhìn đoạn code [tb_llama3_1b_layer.sv#L1201-L1217](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_llama3_1b_layer.sv#L1201-L1217) để thấy giới hạn chí mạng của `u_seq` (`snax_llm_sequencer.sv`):
     * `h_grp = 1024 / (M_TOKENS * K_PACKETS_QK);` (ví dụ `M_TOKENS = 64` $\implies$ `1024 / 256 = 4` heads mỗi nhóm; `M_TOKENS = 256` $\implies$ **`h_grp = 1`** head mỗi nhóm — tức là khi `M_TOKENS > 256`, **`1` head Q thôi cũng vượt quá `1024` word chiều sâu của `Bank 30`, KHÔNG THỂ CHẠY ĐƯỢC NỮA!**).
     * CPU phải chạy vòng lặp ngoài `for (int hg = 0; hg < n_hgrp; hg++)`:
       - Gọi **`DMA-WB #3` (`DMA Load QPKT`)** (`dir = 2'b10` linear $\implies$ chỉ dùng 1 port qua crossbar vì đụng chung `Bank 30`!) kéo `QPKT` từ DRAM vào `Bank 30`.
       - Chạy `u_seq` (`op_mode = 2'b01`, FlashAttention). Bên trong `u_seq`, DMA Ring cứ **kéo lặp lại từng tile `(K+V)` từ `DRAM_KV` cho mọi `qi` và mọi `h`** (không hề pin KV vào `Banks 0..15` dù `N_KV = 256` chỉ nặng `18 KB`/head!).
       - Chạy xong `u_seq` ghi `O` vào `Bank 28` (tối đa `1024` words!). Nếu `M_TOKENS = 256, H_Q = 32`: kết quả `O` của 1 nhóm còn không đủ chỗ `1024` word của 1 bank (`256 × 4 = 1024` chỉ vừa đúng 1 head nếu FP4, và **TRÀN CHẮC CHẮN** nếu xuất `FP16` đòi `4096` words!).
       - CPU dùng `cpu_mem_read` bưng kết quả từ `Bank 28` cất vào mảng phần mềm `o_act_buf`!

#### Pha III — Hậu Attention & FFN (Lại Lặp `M_CHUNKS` lần bởi CPU! `L1221-1288`)

8. **Bước `[7/12]` & `[8/12]` `GEMM O_proj + Residual 1 + RMSNorm 2` (`L1224-1248`):**
   - CPU chép tay `o_act_buf[mb]` vào `Banks 16..31 (words 0..127)`.
   - **Gọi `DMA-WB #4` (`DMA Load X`, `L1227`)**: kéo lại `73,728 B` của `X[mb]` từ `DRAM_X` lên `words 512..1023` (vì `u_sfu_seq` không tự đọc từ DRAM).
   - Chạy `2` lần `GEMM O` (`64` N-tiles/lần) ghi FP16 vào `words 128..383`, rồi `2` lần `u_sfu_seq (Residual 1)` cộng vào `words 512..1023`. *(Tại sao không bật Macro-DMA hoặc fuse thẳng Residual 1 vào đuôi `GEMM O` qua `fused_fifo`? Vì `GEMM O` xuất FP16 vào TCDM xong CPU mới chạy `u_sfu_seq`!)*
   - **Gọi `DMA-WB #5` (`DMA Save RES1`, `L1240`)**: đẩy `73,728 B` `RES1` từ TCDM ra `DRAM_RES1` để lấy chỗ cho FFN!
   - CPU nạp tay `gamma2` vào `Bank 0`, chạy `u_sfu_seq (RMSNorm 2)` ghi `N2_fp16` vào `words 0..511`.

9. **Bước `[9/12]` `Requant Norm2 -> ACT` bằng `Readback` (`L1249`):**
   - Lại phải đổi từ `u_sfu_seq` (`csr_src_sel=1`) về `u_seq` (`csr_src_sel=0, op_mode=2'b11`), gọi `run_requant_wrap` đọc lại `N2_fp16` (`73,728 B`) nén thành `ACT_FP4` (`18 KB`) ở `words 0..127`!

10. **Bước `[10/12]` `FFN Gate & Up + SwiGLU` (`L1250-1263`):**
    - Vì `D_FF = 8192` (`512` N-tiles), `Gate_fp16` tốn `288 KB` (chứa không nổi!) nên TB lặp **`16` chunks × (`GEMM Gate 32 NT` + `GEMM Up 32 NT` + `u_sfu_seq SwiGLU out_fp4=1`)** = **`48` lần launch CSR khác nhau cho 1 chunk `16` tokens**!
    - Kết quả `FF1_FP4` (`72 KB`) dồn vào `words 512..1023` của `Banks 16..31`. *(Lưu ý: nếu dùng Fused / `O_buf`, đáng lẽ có thể tính hết hoặc với `M > 16` thì phải đẩy thẳng ra DRAM chứ không bị kẹt 16 tokens/lần!)*

11. **Bước `[11/12]` `GEMM Down (2048 × 8192, K-Tiling 2×4096)` (`L1264-1266`):**
    - Chạy `u_gemm_seq` (`enable_dma=1, k_tiles_m1=1, K_TILE_DOWN=256`) xuất `OUT_fp16` (`73,728 B`) vào `words 0..511`.

12. **Bước `[12/12]` `Residual 2 -> Final Output` (`L1267-1287`):**
    - **Gọi `DMA-WB #6` (`DMA Load RES1`, `L1267`)**: kéo `73,728 B` `RES1` từ `DRAM_RES1` lên `words 512..1023`.
    - Lật `csr_src_sel=1`, chạy `u_sfu_seq (Residual 2)` cộng `words 0..511` + `words 512..1023` ghi vào `words 0..511`.
    - Lật lại `csr_src_sel=0`, gọi **`DMA-WB #7` (`DMA WB Final Out`, `L1286`)**: đẩy `73,728 B` ra `DRAM_OUT`!

---

### 7.3. Bảng Định lượng Thảm họa Chi phí Điều khiển Trong Hiện trạng (`M_TOKENS = 256` trên `Llama-3.2-1B`)

Hãy đếm con số thực tế khi chạy Prefill `M_TOKENS = 256` (`M_CHUNKS = 16`, `N_KV = 256`) bằng hiện trạng 3 Sequencer + `Readback` + `DMA-WB` tại [tb_llama3_1b_layer.sv#L1113-L1291](file:///c:/Users/nguyen/Desktop/NPU_LOG/tb/tb_llama3_1b_layer.sv#L1113-L1291):

| Chỉ số đo lường | Số lượng trên Hiện trạng (`M = 256`) | Nguyên nhân gốc rễ | Khi Hợp nhất `1 Unified Sequencer` (chỉ giữ Macro-DMA + Inline Epilogue) |
| :--- | :---: | :--- | :---: |
| **Số lần gọi `Readback` (`op=11`)** | **`32` lần** (`16 × Requant Xn` + `16 × Requant Norm2`) | `u_sfu_seq` đứng tách rời và xuất FP16 rồi mới Readback | **`0` lần!** `RMSNorm` Pass 2 bắn thẳng `out_dst = LogQ (FP4)`! *(Nếu có lệnh Requant rời thì đi qua chính `OP_SFU_POST` với `SFU_BYPASS`!)* |
| **Số lần gọi `DMA-WB` rời (`cfg_dma_wb=1`)** | **`72+` lần** (`16×Save X` + `8×Load Q` + `16×Load X` + `16×Save RES1` + `16×Load RES1` + `16×WB Out` = `88` lần!) | `u_seq` Attention thiếu Macro-DMA `Q/O`; `u_sfu_seq` 0 có DMA | **`0` lần!** Mọi Engine (`GEMM`, `ATTN`, `SFU`) đều tự kéo từ DRAM và flush `O_buf` Ping-Pong ra DRAM! |
| **Số lần CPU phải đọc/chép TCDM bằng tay (`cpu_mem_read/write`)** | **`> 200,000` words** (X, Q, K nhân 4×, V 32 lần, O_act) | Thiếu layout thống nhất + thiếu Macro-DMA `Q/O/KV` + `V` chưa flush trực tiếp | **`0` lần!** Toàn bộ tensor lớn nằm hoàn toàn trên DRAM, chỉ stream Ping-Pong qua TCDM! |
| **Số lần CPU launch `V_proj`** | `16 × 32 =` **`512` jobs** | Vòng lặp CPU ngoài theo `(mb, hkv, dvi)` | **`1` job duy nhất** (`GEMM V` với `out_mode = V_TRANSPOSE` + `Macro-DMA` cho toàn bộ `M = 256, N = 512`)! |
| **Số lần CPU launch `Gate/Up + SwiGLU`** | `16 × (16 × 3) =` **`768` jobs** | Giới hạn `TCDM` không chứa nổi `D_FF` FP16 + thiếu Macro-DMA | **`48` jobs** (hoặc **`3` jobs `M=256`**: `GEMM Gate` Chunked Macro-DMA $\rightarrow$ `GEMM Up` $\rightarrow$ `SFU SwiGLU` Macro-DMA; hoặc `16` N-chunks $\times$ `3` jobs nhưng mỗi job chạy trọn **`M = 256` không cắt `mb`!**) |
| **Số lần `FlashAttention` đọc lại `K` & `V` từ DRAM** (`N_KV = 256`, `K+V/head = 18 KB`) | `K` đọc `32 × 32 × 4 =` **`4,096×`** (lại còn phình 4× thành `K+V` gộp!); `V` đọc **`1,024×`** | `u_seq` không có GQA grouping `h_kv` ngoài `h_grp` và không có `KV_Pin_En` | **`1×` duy nhất cho mỗi KV-head** (`KV_Pin_En = 1`, `18 KB <= 144 KB`) — giảm tới **`4,096×` traffic đọc `K`**! |
| **Tổng số lần CPU can thiệp launch job cả 1 layer** | **`~1,460+` lần** đổi mode CSR & chờ `wait_job` | 3 Sequencer rời + không có Macro-DMA cho Attn & SFU | **`<= 12` lần** (nếu giữ 1 job cho mỗi phép toán chuẩn, không phân mảnh theo `mb`!) — hoặc tối đa ~58 lần cho 16 N-chunks của Gate/Up/SwiGLU! |

---

### 7.4. Truy vết Kiến trúc Mới: 1 Layer `Llama-3.2-1B` (`M = 256` hoặc `M = 1024`) Chạy Như Thế Nào Trên `1 Unified Sequencer`?

Hãy xem từng bước `[1..12]` khi hợp nhất về **`1 Sequencer Duy Nhất`** (`snax_llm_unified_sequencer`) — **ZERO `Readback`**, **ZERO `DMA-WB` rời**, **ZERO vòng lặp cắt `M_CHUNKS` ở CPU**:

1. **Job #1 — `OP_SFU_POST (RMSNorm 1 -> FP4)`:**
   - **Input A (`DRAM_X`, FP16, `1.15 MB` cho `M=256`)**: `DMA TAG_IN_A` stream Ping-Pong từng cụm `M_sub = 16` tokens (`72 KB`) vào `Banks 16..31` (`Buf0` words `0..511` / `Buf1` words `512..1023`).
   - **Input B (`gamma1`, FP16, `4.5 KB`)**: giữ cố định trong `Bank 0` (`words 0..511`) hoặc nạp 1 lần đầu job.
   - **Thực thi trên `u_sfu` (`RMSNORM`):**
     * Pass 1 đọc từ `Buf_active` (`Master 20`), tính $\text{inv\_rms}$.
     * Pass 2 nhân $\gamma_1$, đặt **`out_dst = LogQ (FP4)`**! Mỗi token xuất thẳng `128` gói FP4 (`1,152 B`), gom đủ `M_sub = 16` tokens = `18 KB` (`O_buf`) $\rightarrow$ `TAG_O_WB` tự động xả ra `DRAM_XN_FP4` song song lúc DMA nạp cụm `M_sub` tiếp theo!
   - **Kết quả:** **`0` Readback**, **`0` DMA-WB rời**, `X` nguyên thủy vẫn còn nguyên trên `DRAM_X` cho Residual 1, không cần `DMA Save X`!

2. **Job #2 — `OP_GEMM (Q_proj + Fused RoPE -> FP4, 4-Loop Macro-DMA)`:**
   - **Input A:** `DRAM_XN_FP4` (`288 KB` cho `M=256`), **Weights:** `DRAM_WQ` (`2 MB`), **SFU Input B (`sin/cos`):** giữ cố định trên TCDM (chỉ vài KB cho mỗi `M_sub`, hoặc stream cùng `TAG_IN_B`).
   - `Core` chạy 4-Loop OS GEMM (`M_macro -> N_16 -> M_sub -> K_TILES`) $\rightarrow$ `fused_fifo` $\rightarrow$ `u_sfu (ROPE)` $\rightarrow$ `u_log_quantizer` $\rightarrow$ `O_buf` (FP4) $\rightarrow$ `TAG_O_WB` đẩy thẳng ra `DRAM_Q` theo đúng layout head-major cho Attention!
   - **Kết quả:** 1 lệnh duy nhất cho toàn bộ `M = 256` (`256 × 2048 × 2048`), **`0` Readback**, **`0` copy CPU**.

3. **Job #3 — `OP_GEMM (K_proj + Fused RoPE -> FP4, 4-Loop Macro-DMA)`:**
   - Y hệt Job #2 cho `W_K` (`512 × 2048`), đẩy thẳng `K_new` FP4 vào vùng `K-Cache` trên `DRAM_K` (**không cần nhân 4× sang `DV_TILES`!**).

4. **Job #4 — `OP_GEMM (V_proj + Fused V-Transpose -> FP4, 4-Loop Macro-DMA)`:**
   - Chạy `GEMM` với `out_mode = V_TRANSPOSE`: `Core` $\rightarrow$ `u_transpose_buffer/ctrl` $\rightarrow$ `u_log_quantizer` $\rightarrow$ `u_write_out (MODE_V)` vào `O_buf` $\rightarrow$ `TAG_O_WB` đẩy thẳng từng lát `[16 tokens × 16 channels]` (`144 B`) ra vùng `V-Cache` trên `DRAM_V`!
   - **Kết quả:** Rút **`512` lần launch ở CPU** xuống **`1` lệnh**, **xóa `DMA WB V-cache`**!

5. **Job #5 — `OP_ATTN (FlashAttention Redesigned, Macro-DMA Q/O + GQA KV_Pin_En)`:**
   - Với `N_KV = 256` (`K+V` của `1` KV-head = `18 KB <= 144 KB` `Banks 0..15`): tự động bật **`KV_Pin_En = 1`**.
   - Vòng ngoài `h_kv = 0..7`: nạp `K[h_kv]` + `V[h_kv]` **ĐÚNG 1 LẦN** (`18 KB`), chạy toàn bộ `G = 4` Q-heads × `32` Q-tiles (`M=256`) × `4` d_v slices bằng dữ liệu `KV` ghim sẵn trên TCDM!
   - Vùng `Banks 16..31` dùng Ping-Pong `Q_buf` (`Q_sub` từ `DRAM_Q`) và `O_buf` (`CAP_O_ATTN`, `TAG_O_WB` đẩy ra `DRAM_ATTN_O` ở định dạng **FP4** hoặc **FP16**).
   - **Kết quả:** **1 lệnh duy nhất cho toàn bộ `32 heads × 256 tokens`**, **xóa `DMA Load QPKT0/1`**, **giảm `4,096×` lượng đọc DRAM của `K`**!

6. **Job #6 — `OP_GEMM (O_proj + Fused Residual 1 -> FP16, 4-Loop Macro-DMA)`:**
   - *(Khả năng tối ưu cực mạnh sẵn có trên phần cứng của bạn!)*
   - Nhờ `u_core -> fused_fifo -> u_sfu (RESIDUAL) -> u_write_out (FP16)` (`sfu_en=1, sfu_in_a_src=1, sfu_out_dst=1, wo_mode=FP16`), nếu ta nạp đồng thời `X[m_macro]` từ `DRAM_X` vào Ping-Pong cho `u_read_fp16_b`, **`O_proj` vừa tính ra từng vector FP16 là CỘNG THẲNG VỚI `X` ngay trên đường drain**, rồi `TAG_O_WB` đẩy `RES1_fp16` ra `DRAM_RES1` (hoặc giữ trên TCDM nếu `M <= 16`)!
   - *(Hoặc nếu tách thành 2 job chuẩn trên 1 Sequencer: `OP_GEMM (O_proj -> DRAM_WO_OUT)` xong `OP_SFU_POST (RESIDUAL, DRAM_WO_OUT + DRAM_X -> DRAM_RES1)` — cả 2 đều chạy full `M = 256` trong 1 lệnh mà **0 cần `DMA-WB` rời**, **0 cần `mb` loop ở CPU**!)*

7. **Job #7 — `OP_SFU_POST (RMSNorm 2 -> FP4)`:**
   - Y hệt Job #1: đọc `DRAM_RES1` qua Ping-Pong DMA `TAG_IN_A`, đọc `gamma2`, **bắn thẳng `out_dst = LogQ -> FP4`**, đẩy ra `DRAM_N2_FP4` (vì `RES1` đã nằm sẵn an toàn trên `DRAM_RES1` cho Residual 2!).
   - **Kết quả:** Xóa sạch **`Readback` (`Requant Norm2`)** + **`DMA Save RES1`**!

8. **Job #8..10 (hoặc vòng N-chunk theo `D_FF`) — `OP_GEMM (Gate)` + `OP_GEMM (Up)` + `OP_SFU_POST (SwiGLU -> FP4)`:**
   - Ở `M = 256`: kích thước `Gate_fp16` (1 chunk `32 N-tiles` $\implies 256 \times 32 \times 16 \times 2\text{B} = 256\text{ KB}$ hay `16 N-tiles` = `128 KB`). Với Macro-DMA `TAG_O_WB`, `OP_GEMM` tự đẩy `Gate` & `Up` ra DRAM theo mọi `M`, sau đó `OP_SFU_POST (SWIGLU, out_dst=LogQ)` đọc 2 luồng `DRAM_GATE` + `DRAM_UP` qua Ping-Pong `TAG_IN_A / TAG_IN_B` và xả `FF1_FP4` ra `DRAM_FF1` (vẫn là **1 lệnh cho toàn bộ `M=256`, không phải chặt `16` chunks `mb` ở CPU**)!

9. **Job #11 — `OP_GEMM (Down_proj, 4-Loop OS K-Tiling 2×4096)` + Job #12 — `OP_SFU_POST (Residual 2 -> DRAM_OUT)` (hoặc Fused):**
   - `OP_GEMM` đọc `DRAM_FF1` (`M=256, K=8192`) qua 2 K-tiles (`K=4096` mỗi tile) giữ Output-Stationary `100%` trên Latch $\rightarrow$ xả `DOWN_fp16` ra `DRAM_DOWN` (hoặc Fused thẳng với `RES1`).
   - Cuối cùng `OP_SFU_POST (RESIDUAL)` đọc `DRAM_DOWN` + `DRAM_RES1` và ghi `Final_Out` thẳng ra `DRAM_OUT` — **xóa hoàn toàn `DMA Load RES1` và `DMA WB Final Out` rời**!

---

### 7.5. Kiểm chứng Tổng quát trên Mô hình thứ 2: `Qwen2.5-1.5B` (`d_k = 128`, `GQA = 6:1`, `D_MODEL = 1536`, `D_FF = 8960`)

Để chứng minh kiến trúc hợp nhất **không bị thiết kế khớp cứng (overfit) vào mỗi `Llama-3.2-1B`**, hãy kiểm tra qua `Qwen2.5-1.5B` (`D_MODEL = 1536`, `D_FF = 8960`, `H_Q = 12`, `H_KV = 2`, `d_k = d_v = 128`):

* **Đặc điểm hình học khác với `Llama-3.2-1B`:**
  * `d_k = d_v = 128` $\implies$ `K_PACKETS_QK = 8`, `DV_TILES = 8` (gấp **2×** chiều sâu head so với Llama!).
  * `GQA = 12 / 2 = 6:1` (`gqa_group_m1 = 5`, chia hết cho `2` nhưng không phải lũy thừa của `2`!).
  * `D_MODEL = 1536` $\implies$ `96` tiles (`K_PACKETS = 96`, **không phải lũy thừa của `2`**; `384` words FP16/token).
  * `D_FF = 8960` $\implies$ `560` tiles (`560 / 256` $\implies$ `2 × 4096 + 1 × 768` = **`3` K-tiles** với tail K-tile chỉ còn `48` packets, **không bằng nhau**!).
* **Kiểm tra khả năng chịu tải của 2 kiến trúc:**
  * **Kiến trúc Rời rạc Hiện tại:**
    - `K` ở `DRAM_KV` bị nhân **`DV_TILES = 8×`** thay vì `4×` (lãng phí gấp 8 lần băng thông DRAM + dung lượng KV!).
    - Nếu dùng `u_gemm_seq` kết hợp `u_seq`: `K_TILES` đuôi lẻ của `8960` bắt buộc phải chỉnh đúng `cfg_gemm_k_blocks_i` hoặc kiểm tra tail K-tile (lưu ý: `snax_gemm_os_sequencer.sv` hiện dùng chung `cfg_gemm_k_blocks_i` cho mọi K-tile nên với `8960 = 35 × 256` = **chẵn 35 lần `256`**, ta có thể chọn `K_TILE = 256`? `560 = 256 + 256 + 48`, không chia hết cho `256`, nhưng chia hết cho **`112` (`5 × 112 = 560` packets $\implies 5$ K-tiles $\times 1,792$ K-dim)**! Đây là ưu điểm có sẵn của `K_TILES` cấu hình được!).
    - Với Attention `N_KV = 1024`, `1` KV-head của Qwen2.5-1.5B (`d = 128`) tốn: $1024 \times 8 \times 9 \times 2 = \mathbf{144\text{ KB}}$ — vẫn **vừa khít `100%` vùng `Banks 0..15` (`144 KB`)** nếu bật `KV_Pin_En`! Còn với `u_seq` cũ, `K` bị đọc lặp tới **`6 × Q_T × 8 = 48 × Q_T` lần** cho mỗi KV-head!
    - Trong `OP_SFU_POST (RMSNorm)` cho `D = 1536` (`96` vectors/token): `16` tokens vừa khít `16 × 384 = 6,144` words (`54 KB`), chỉ cần 1 lệnh hợp nhất `OP_SFU_POST (out_dst=LogQ)` với Macro-DMA là chạy hết mọi `M` từ `1` tới `4096+` mà không đụng tới `Readback` hay `DMA-WB`!
