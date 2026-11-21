# Phân tích Toàn diện Thuật toán FlashAttention & Thiết kế Sequencer trên Phần cứng SNAX-FP4 (`TCDM 288 KB` + `16×16` Latch)

Tài liệu này phân tích từ gốc **toán học – phần cứng – băng thông DRAM – FSM chu kỳ** cho phép toán **FlashAttention (Online Exp-Free Softmax)** trên kiến trúc [snax_fused_llm_core.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv) và [snax_llm_top_clean.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv), chỉ rõ:
1. Thuật toán Online Softmax không `exp()` và cơ chế tính toán trong từng module datapath.
2. Tại sao cấu trúc 4 vòng (`Head -> Q_tile -> DV_slice -> KV_tile`) **đúng về toán học Core** nhưng **vẫn FAIL 2 tiêu chí Sequencer** nếu thiếu Macro-DMA `Q/O` và tối ưu GQA/`KV`.
3. Phân tích băng thông DRAM định lượng cho tất cả các hoán vị vòng lặp khả dĩ trên TCDM `288 KB`.
4. Chẩn đoán từng giới hạn/rủi ro cụ thể trong [snax_llm_sequencer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv) hiện tại.
5. Đặc tả kiến trúc FlashAttention Sequencer mới để hợp nhất sạch cùng GEMM 4-Loop thành **1 Sequencer duy nhất**.

---

## MỤC LỤC
* [Phần 1. Thông số Phần cứng Cố định & Bản đồ Datapath Attention](#phần-1-thông-số-phần-cứng-cố-định--bản-đồ-datapath-attention)
* [Phần 2. Chứng minh Toán học Exp-Free Online Softmax & Rescaling trên RTL](#phần-2-chứng-minh-toán-học-exp-free-online-softmax--rescaling-trên-rtl)
* [Phần 3. Phân tích Timing Từng Chu kỳ (Pipeline, Bypass, Overlap & Hazard)](#phần-3-phân-tích-timing-từng-chu-kỳ-pipeline-bypass-overlap--hazard)
* [Phần 4. Đánh giá Thuật toán 4 Vòng (`H -> Q_tile -> d_v -> KV_tile`) theo Đúng 2 Tiêu chí của Sequencer](#phần-4-đánh-giá-thuật-toán-4-vòng-h---q_tile---d_v---kv_tile-theo-đúng-2-tiêu-chí-của-sequencer)
* [Phần 5. Phân tích Băng thông DRAM & Dung lượng TCDM 288 KB (Chứng minh Phương án Tối ưu)](#phần-5-phân-tích-băng-thông-dram--dung-lượng-tcdm-288-kb-chứng-minh-phương-án-tối-ưu)
* [Phần 6. Rà soát Toàn bộ Lỗi Tiềm ẩn & Giới hạn trong `snax_llm_sequencer.sv` Hiện tại](#phần-6-rà-soát-toàn-bộ-lỗi-tiềm-ẩn--giới-hạn-trong-snax_llm_sequencersv-hiện-tại)
* [Phần 7. Đặc tả Thiết kế Lại FlashAttention Sequencer (Đạt Cả 2 Tiêu chí & Hợp nhất 1 Sequencer Duy nhất)](#phần-7-đặc-tả-thiết-kế-lại-flashattention-sequencer-đạt-cả-2-tiêu-chí--hợp-nhất-1-sequencer-duy-nhất)
* [Phần 8. Checklist Kiểm chứng Đầy đủ (Testbench Verification Matrix)](#phần-8-checklist-kiểm-chứng-đầy-đủ-testbench-verification-matrix)

---

## Phần 1. Thông số Phần cứng Cố định & Bản đồ Datapath Attention

### 1.1. Các tham số phần cứng tại Top-Level ([snax_llm_top_clean.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv))
* **TCDM Scratchpad (`u_tcdm`):** `32 banks × 1024 words × 72 bits = 294,912 bytes = 288 KB` (`32,768` bank-words; mỗi word 72-bit = `9` bytes vật lý chứa 16 phần tử FP4 + 8-bit scale Log8).
  * Ánh xạ địa chỉ: `addr = (word << 9) | (bank << 4)` (`0x200` byte-address space mỗi bước `word` xuyên 32 banks; `0x10` mỗi bước `bank`).
* **Bus DRAM (`snax_simple_dma`):** Rộng `512 bits = 64 bytes/beat` DRAM $\leftrightarrow$ `4 ports × 72 bits = 288 bits/cycle` vào TCDM.
* **Compute Core (`u_core` — [snax_fused_llm_core.sv#L1377-L1385](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv#L1377-L1385)):**
  * `NUM_LANES_P = 16`: Mảng MAC (`log8_fp4_mac_array`) tính 16 output channels / 16 keys song song mỗi chu kỳ (`16 × 16 = 256 MACs/cycle`).
  * `ACC_DEPTH_P = 16`: Latch (`rmw_latch_acc_buffer`) giữ 16 vector 32-bit (`Q16.16`), mỗi vector gồm 16 lanes.
  * `NUM_QUERIES = 8`: Số query tối đa cho một tile FlashAttention $\implies$ **`M_Q <= 8`**.
  * `P_FIFO_DEPTH = 16`: Bộ đệm `p_fifo` nội bộ Core chứa tối đa `16` gói xác suất $P$ (mỗi gói = `1` query × `16` keys kèm `15` bit sideband ở Top khi `QW=3`).
  * `USE_OUTPUT_READY = 1'b1`: Cổng ra FP16 tuân thủ đầy đủ backpressure `out_vec_fp16_ready_i`.

### 1.2. Phân hoạch Vùng Latch 16 Slots cho FlashAttention
Khác với GEMM dùng cả `16` slots (`0..15`) làm `O_tile`, FlashAttention phải **chẻ đôi cứng** `rmw_latch_acc_buffer`:

| Vùng Latch | Địa chỉ Slot (`4-bit`) | Kích thước vật lý | Chứa đại lượng nào | Ghi bởi | Đọc / Xả bởi |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Vùng O** | **`4'b0xxx` (`Slots 0..7`)** | `8 × 16 × 32b = 512 B` | Kết quả tích lũy $O_{\text{unnorm}}[0..M_Q-1, 0..15]$ | **Pha PV (`cfg_mode = 2'b10`)** qua `u_attn_rescaler` | **Background Drain (`nrm` / cuối `kj`)** $\rightarrow$ `u_final_normalizer` |
| **Vùng S** | **`4'b1xxx` (`Slots 8..15`)** | `8 × 16 × 32b = 512 B` | Điểm thô $S = Q \times K^T[0..M_Q-1, 0..15]$ | **Pha QK (`cfg_mode = 2'b01`)** | **SM Drain** $\rightarrow$ `LOG2E_Q16` $\rightarrow$ `u_softmax_requantizer` |

Cờ phân biệt vùng trong [snax_fused_llm_core.sv#L288-L295](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv#L288-L295) hoạt động bằng bit MSB của `tile_drain_idx_i` (`slot[3]`):
* `tile_drain_idx_i[3] == 1` (`Slots 8..15`) $\Rightarrow$ `drain_is_o_q = 0` $\Rightarrow$ rẽ vào **Softmax Requantizer** (khi `cfg_mode == 2'b01`).
* `tile_drain_idx_i[3] == 0` (`Slots 0..7`) $\Rightarrow$ `drain_is_o_q = 1` $\Rightarrow$ rẽ vào **Final Normalizer** (nhân với `1/l`), **bất kể `cfg_mode_i` lúc đó là `2'b01` hay `2'b10`**. Đây chính là lý do phần cứng cho phép xả `O` của nhóm trước ngay trong khi `QK` của nhóm sau đang chạy!

### 1.3. Sơ đồ Luồng Dữ liệu Khép kín Trên Chip (On-Chip Attention Loop)

```text
  TCDM (Q: Bank port 2)      TCDM Ring (K & V: 16 banks port 3..18)
          |                               |
          |  [QK: K_BLOCKS = D_K/16]      | [QK: K, K_BLOCKS]   [PV: V, 1 blk (16 KV)]
          v                               v
   +-------------+                 +-------------+
   | act_stream  |                 | wt_stream   |
   +------+------+                 +------+------+
          |                               |
          | cfg_mode=01: chọn ext_act     | (luôn vào Port wt_in_i)
          +--------► [Mux Act] ◄----------+-------- (cfg_mode=10: chọn p_fifo_packet)
                         |                                        ^
                         v                                        |
              +---------------------+                             |
              |  log8_fp4_mac_array | (Stage 1A -> 1B -> Stage 2) |
              +----------+----------+                             |
                         | ^                                      |
             mac_wdata   | | effective_old_res                    |
                         v |                                      |
              +---------------------+     [cfg_mode=10]     +-----+--------+
              | rmw_latch_acc_buffer| ◄── attn_o_rescaler ◄-+   p_fifo     |
              | Slots 0..7 : O      |     2^(m_old-m_new)   | (16 entries) |
              | Slots 8..15: S      |                       +-----+--------+
              +----------+----------+                             ^
                         | Port B (1 cyc lat)                     | (lat 2)
                         +--------------------+                   |
                         |                    |                   |
       [slot[3]==1]      |                    | [slot[3]==0]      |
       (Scores S)        v                    v (Output O)        |
               +-------------------+    +-------------------+     |
               | * LOG2E_Q16 (CSD) |    | final_normalizer  |     |
               +---------+---------+    |  O * (1/l) -> FP16|     |
                         |              +---------+---------+     |
                         v                        ^               |
               +-------------------+              | (1/l LUT)     |
               |softmax_requantizer|--------------+---------------+
               | m_q[8],resc,FP4 P |--- l_contrib_out ---> [l_accumulator]
               +-------------------+                        l_q[8] + LZC + LUT
```

---

## Phần 2. Chứng minh Toán học Exp-Free Online Softmax & Rescaling trên RTL

Định nghĩa chính xác tại sao phần cứng của bạn tính Softmax **mà không cần bất kỳ hàm `exp()` hay bộ chia dấu phẩy động nào**, đối chiếu 1-1 với [softmax_requantizer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/softmax_requantizer.sv), [l_accumulator.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/l_accumulator.sv), [attn_o_rescaler_bank.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/attn_o_rescaler_bank.sv), và [final_normalizer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/final_normalizer.sv).

### 2.1. Chuyển miền từ cơ số $e$ sang cơ số $2$ không dùng bộ nhân (`LOG2E_Q16`)
Với 1 lát key $j \in \{0..15\}$ của hàng query $q$, tích vô hướng $QK^T$ tích lũy ở định dạng **Q16.16 signed two's complement** (`FRAC_BITS = 16`):
$$s_j = \frac{1}{\sqrt{d_k}} Q_q \cdot K_j^T \in \text{Q16.16}$$
*(Hệ số $1/\sqrt{d_k}$ có thể gấp sẵn vào scale của `Q` hoặc `K` trước Attention.)*

Vì $\exp(s) = 2^{s \cdot \log_2 e}$, tại [snax_fused_llm_core.sv#L89](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv#L89) và [`#L311-L318`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv#L311-L318), lõi nhân với $\log_2(e) \approx \frac{94548}{65536} = 1.442688$ bằng **7 phép dịch-cộng CSD** (0 DSP/multiplier):
$$94548 = 2^{16} + 2^{15} - 2^{12} + 2^8 + 2^6 + 2^4 + 2^2$$
Kết quả đưa vào `u_softmax_requantizer` là:
$$s'_j = s_j \cdot \log_2 e \quad (\text{Q16.16})$$

### 2.2. Vì sao triệt tiêu được `exp()` khi lượng tử $P$ sang FP4 E2M1?
Xét trong một block `16` keys ($j = 0..15$):
1. Đặt $s'_{\text{bmax}} = \max_{j \in \text{visible}} s'_j$ là max cục bộ của block (qua cây so sánh 4 tầng 8-4-2-1 ở [softmax_requantizer.sv#L182-L190](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/softmax_requantizer.sv#L182-L190); key bị causal mask `key_mask[j] == 0` bị ép thành `S_MIN = 0x8000_0000`).
2. Cập nhật running max $m_q$ cho hàng $q$:
   * Nếu `row_start == 1` (`kj == 0`): $m_{\text{next}} = s'_{\text{bmax}}$ (`resc_req = 0`).
   * Nếu `row_start == 0`: $m_{\text{next}} = \max(m_{\text{curr}}, s'_{\text{bmax}})$, và `resc_req = (s'_bmax > m_curr)`.
   * Nếu block bị mask hoàn toàn (`any_valid == 0`): giữ nguyên $m_{\text{next}} = m_{\text{curr}}$, `resc_req = 0`, `block_empty = 1`.
3. Đặt $E_j = 2^{s'_j - m_{\text{next}}}$, khi đó với định nghĩa Softmax chuẩn $p_j = \frac{E_j}{l}$, giả sử ta chọn scale của block FP4 E2M1 (mức cực đại của E2M1 là $6.0$) sao cho khớp với giá trị lớn nhất chưa chuẩn hóa $E_{\text{bmax}} = 2^{s'_{\text{bmax}} - m_{\text{next}}}$:
   $$S_{\text{blk}} = \frac{E_{\text{bmax}}}{6 \cdot l}$$
   Khi lượng tử hóa $p_j$ bằng cách chia $p_j / S_{\text{blk}}$, **cả $l$ lẫn $m_{\text{next}}$ TRIỆT TIÊU TUYỆT ĐỐI**:
   $$\frac{p_j}{S_{\text{blk}}} = \frac{E_j / l}{E_{\text{bmax}} / (6 l)} = 6 \cdot \frac{2^{s'_j - m_{\text{next}}}}{2^{s'_{\text{bmax}} - m_{\text{next}}}} = 6 \cdot 2^{-(s'_{\text{bmax}} - s'_j)} = 6 \cdot 2^{-d_j}$$
   với $d_j = s'_{\text{bmax}} - s'_j \ge 0$ chỉ là **1 phép trừ số nguyên Q16.16**!
4. Việc làm tròn $6 \cdot 2^{-d_j}$ về 8 mức không dấu của E2M1 ($\{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\}$) trở thành **so sánh trực tiếp $d_j$ với 7 hằng số tĩnh** (`TH_6P0..TH_0P5`, dòng `100-106`), hoàn toàn **không có bất kỳ mạch `exp2` hay `exp` nào**:
   * Ví dụ ranh giới giữa $6.0$ và $4.0$ là điểm giữa $5.0 \implies d = \log_2(6/5) \approx 0.2630 \implies \text{TH\_6P0} = 17238$ (trong Q16).

### 2.3. Chuẩn hóa Log8 Scale của `P` & Hủy hằng số `P_OFFSET = -21`
Vì chưa biết tổng mẫu số $l$ khi chưa chạy hết chuỗi KV, `softmax_requantizer` xuất gói $P$ ở scale **chưa chia $l$**:
$$\hat{S}_{\text{blk}} = l \cdot S_{\text{blk}} = \frac{2^{s'_{\text{bmax}} - m_{\text{next}}}}{6} = 2^{(s'_{\text{bmax}} - m_{\text{next}}) - \log_2 6}$$
Chuyển sang mã **Log8 S4.3** (đơn vị $1/8$ octave, tức $p = 8 \log_2 \hat{S}_{\text{blk}}$):
$$p_{\text{blk}} = \text{round}\!\left(\frac{s'_{\text{bmax}} - m_{\text{next}}}{2^{13}}\right) - \underbrace{21}_{\approx 8\log_2(6) = 20.68}$$
Chú ý 2 bất biến cực đẹp của mạch ([softmax_requantizer.sv#L52](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/softmax_requantizer.sv#L52), [`#L284`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/softmax_requantizer.sv#L284)):
1. Vì $s'_{\text{bmax}} \le m_{\text{next}}$ luôn đúng $\implies p_{\text{blk}} \le -21$ luôn âm/bảo đảm không overflow `signed [7:0]`.
2. Cả tử số $\hat{O} = \sum \hat{P}_{\text{quant}} V$ (tính qua MAC array) lẫn mẫu số $\hat{l} = \sum \hat{S}_{\text{blk}} \cdot \text{sum}(\text{code\_level})$ (tính trong `l_accumulator`) **đều cùng dùng chính $\hat{S}_{\text{blk}}$ mang hằng số dịch $-21$ và nhân $2$ (`M_int = 2*|level|`)**. Do đó khi lấy thương $\frac{\hat{O}}{\hat{l}}$ ở `final_normalizer`, cả yếu tố $2^{-21/8}$ lẫn thừa số $2$ **triệt tiêu chính xác ở mẫu số & tử số**.

### 2.4. Online Rescaling qua CSD & Bit-Shift cho cả `O` và `l`
Khi phát hiện block mới có $s'_{\text{bmax}} > m_{\text{curr}}$, mọi tích lũy cũ $O_{\text{old}}$ và $l_{\text{old}}$ phải nhân hệ số thu nhỏ:
$$c = 2^{m_{\text{curr}} - m_{\text{next}}} = 2^{\Delta m} \quad (\Delta m \le 0)$$
Tách $\Delta m$ (dạng Q16.16) thành:
* Phần nguyên octave: $q = \lfloor \Delta m / 2^{16} \rfloor$ (`resc_q_d`, saturating trong $[-128, 127]$)
* Phần lẻ 3 bit: $r = \Delta m[15:13] \in \{0..7\}$ (`resc_r_d`)
Sử dụng `vsq_rescaler_32b` (xấp xỉ 3-bit CSD $2^{r/8} \approx \text{CSD}[r]/32$ kèm dịch số học `q`) tại **cùng lúc 2 nơi**:
1. **`u_l_rescaler` trong `l_accumulator`:** Rescale $l_{\text{curr}}$ ngay khi nhận `l_contrib_in` (độ trễ đồng bộ với `s1 -> lat 2` của softmax).
2. **`u_attn_rescaler` (`attn_o_rescaler_bank`, 16 lanes):** KHÔNG rescale `O` tại lúc chạy Softmax! Mà **cất `{resc_valid, resc_q, resc_r}` đi kèm gói $P$ vào trong `p_fifo`**. Đến lúc `P×V` tiêu thụ gói $P$ (`cfg_mode == 2'b10`), sideband đi qua 2 thanh ghi pipeline (`s1a -> q1`) cho **đúng nhịp Stage 2 RMW** của `rmw_latch_acc_buffer`, rescale đúng vector $O_{\text{old}}$ đang đọc lên từ Latch trước khi cộng dồn tích $P \times V$ mới!

---

## Phần 3. Phân tích Timing Từng Chu kỳ (Pipeline, Bypass, Overlap & Hazard)

Đây là phần cực kỳ dễ sai nếu không đếm từng chu kỳ khi thiết kế FSM FlashAttention.

### 3.1. Chu kỳ pha QK (`ST_ATTN_QK_LAUNCH -> ST_ATTN_QK_RUN`)
* `u_act_streamer` đọc `M_Q` tokens của `Q[h, qi]` qua `D_K / 16` (`k_blocks`) nhịp; thứ tự beat: **Token trong (`0..M_Q-1`), K-block ngoài (`0..k_blocks-1`)**.
* `u_wt_streamer` đọc `1` hàng key-block của `K[h_kv, kj]` và `replay` đủ `M_Q` lần rồi mới tăng K-block; thứ tự beat **khớp 1-1** với Activation Streamer.
* Mỗi beat tính, `lane_acc_idx_o = {1'b1, cnt_m_q[2:0]}` tích lũy trực tiếp vào **Slots `8..15`**.
* Khi cả `!act_busy_i && !wt_busy_i` tắt ở cuối `ST_ATTN_QK_RUN`: vẫn phải **đợi `!nrm_busy`** ([snax_llm_sequencer.sv#L795](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L795)) trước khi sang `ST_ATTN_SM_DRAIN` — lý do bắt buộc là `nrm_req` và `SM_DRAIN` **dùng chung duy nhất Port B của `rmw_latch_acc_buffer`**!

### 3.2. Chu kỳ pha SM Drain & Pipe Stage Alignment (`ST_ATTN_SM_DRAIN`)
Theo dõi chính xác từng chu kỳ từ lúc phát `tile_drain_req_o = 1` ở slot $8+m$:
* **Cycle `T`:** Sequencer đặt `tile_drain_req_o = 1`, `tile_drain_idx_o = {1'b1, m[2:0]}`, `tile_clear_on_drain_o = 1`, `cfg_row_start_o = kv_first_q`.
  * Chú ý **same-slot Stage-2 bypass** trong [rmw_latch_acc_buffer.sv#L86-L88](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/rmw_latch_acc_buffer.sv#L86-L88): Nếu beat cuối cùng của `QK` đang ở Stage 2 ghi vào chính slot $8+m$ đúng vào chu kỳ `T` mà `SM_DRAIN` đọc slot đó, `drain_word_comb` lấy thẳng `acc_wdata_i` thay vì `rdata2` nên **KHÔNG cần chờ bubble 1 chu kỳ** cho `M_Q > 1`! *(Trường hợp đặc biệt: nếu `M_Q == 1` như Decode, kiểm tra cẩn thận độ trễ Stage 1A -> Stage 1B -> Stage 2 của `log8_fp4_mac_array` để slot 8 đã kịp hoàn thành Stage 2 trước khi drain!)*
* **Cycle `T+1`:** `raw_drained_valid = 1` xuất `score_array` $\rightarrow$ qua tổ hợp `LOG2E_Q16` + max-tree $\rightarrow$ bắt đầu `sm_in_valid_q` ở cuối chu kỳ `T+1`?
  * Kiểm tra kỹ trong [snax_fused_llm_core.sv#L340-L363](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv#L340-L363):
    * `T`: `tile_drain_req_i = 1` $\implies$ latch `drain_is_o_q`, `drain_is_row_start_q`, `drain_query_idx`.
    * `T+1`: `raw_drained_valid = 1`, `sm_in_valid = 1` $\implies$ `score_for_softmax_q <= score_for_softmax`, `sm_in_valid_q <= 1`, `cfg_key_mask_q <= cfg_key_mask_i`, `row_start_d1 <= drain_is_row_start_q`, `sm_query_idx_d1 <= drain_query_idx`.
    * *(Đây chính là lý do ở [snax_llm_sequencer.sv#L455-L468](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L455-L468) phải có `causal_q_idx_q` trễ 1 chu kỳ so với `cnt_dr_q` — để `cfg_key_mask_i` khớp đúng chu kỳ `T+1` khi `sm_in_valid_q` chốt!)*
    * `T+2`: `u_softmax_requantizer` chốt Stage 1 (`m_next`, `resc_q`, `resc_r`).
    * `T+3`: `u_softmax_requantizer` xuất Latency 2 (`out_valid = 1`, `code_out`, `p_blk_out`, `l_contrib_out`), đồng thời `sm_resc_*_d1` khớp cùng nhịp $\rightarrow$ push thẳng vào `u_p_fifo` và cập nhật `u_l_accumulator`!
* **Điểm tinh tế ở `ST_ATTN_PV_LAUNCH` ([snax_llm_sequencer.sv#L820-L835](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L820-L835)):**
  * Ngay chu kỳ cuối của `ST_ATTN_SM_DRAIN` (`T_end`), xung cuối được phát ra; ở `T_end + 1` (`ST_ATTN_PV_LAUNCH`), `raw_drained_valid` mới đưa vào `sm_in_valid_q`, nên **bắt buộc phải giữ `cfg_mode_o = 2'b01` thêm đúng chu kỳ `ST_ATTN_PV_LAUNCH`** thì gói điểm cuối cùng mới được `softmax_requantizer` nhận! Chu kỳ sau đó (`ST_ATTN_PV_RUN`), `p_fifo` lần lượt pop ra (khi `!p_fifo_empty_i`) cho `M_Q` beats của `P × V`.

### 3.3. Hazard cần kiểm soát ở Background `O` Drain (`nrm`)
Nhìn vào [snax_llm_sequencer.sv#L292-L346](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L292-L346):
* `nrm_beat = core_fp16_valid_i && core_out_fifo_ready_i;` (L305) — **ĐÂY LÀ ĐIỂM YẾU KỸ THUẬT** nếu so với chuẩn GEMM mới:
  1. `core_fp16_valid_i && core_out_fifo_ready_i` khác với handshake thật của cổng ra Core (`core_fp16_valid_i && core_fp16_ready_i` — trong [snax_llm_top_clean.sv#L1125-L1135](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv#L1125-L1135), `core_fp16_ready` còn phụ thuộc `wo_fp16_ready`, còn `core_drain_ready` dùng `wo_fifo_headroom`).
  2. `nrm` phát `tile_drain_req_o` liên tiếp miễn `core_out_fifo_ready_i == 1`, **không có bộ đếm `inflight_out_cnt_q < OUT_PIPE_CREDITS` (`4`)** bảo vệ đường pipe 3 chu kỳ (`rmw_latch_acc_buffer` $\rightarrow$ `final_normalizer S1` $\rightarrow$ `S2`), nên chỉ an toàn tình cờ khi `M_Q <= 8` và `u_write_out` FIFO (`DEPTH=16`) còn trống; nếu `u_log_quantizer` hoặc `u_write_out` bị backpressure bất chợt giữa lúc 3 vector đang nằm trên pipe, 1-entry skid `out_hold_q` của Core có nguy cơ bị ghi đè!

---

## Phần 4. Đánh giá Thuật toán 4 Vòng (`H -> Q_tile -> d_v -> KV_tile`) theo Đúng 2 Tiêu chí của Sequencer

Xét trực tiếp cấu trúc 4 vòng đang dùng chung cả ở hình thiết kế FlashAttention của bạn lẫn bên trong [snax_llm_sequencer.sv#L1083-L1113](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L1083-L1113):

```text
Vòng 1 (Ngoài cùng): h   = 0 .. H_Q - 1         (Heads)
  Vòng 2:            qi  = 0 .. Q_TILES - 1     (M_Q <= 8 queries mỗi tile)
    Vòng 3:          dvi = 0 .. DV_TILES - 1    (Mỗi lát 16 channels của d_v)
      Vòng 4 (Trong):kj  = 0 .. KV_TILES - 1    (Mỗi lát 16 tokens KV; bật cfg_row_start ở kj=0)
```

### 4.1. Thẩm định Công bằng theo Từng Tầng
* **Tầng 1 — Tính toán Core (`Latch 8 S-slots + 8 O-slots`):**
  * **ĐÚNG 100%.** Do Core chỉ giữ được **1 lát `8 × 16` của $O$** trên Latch tại một thời điểm (vì 8 slots kia đang bận giữ $S$), bất kỳ thuật toán nào muốn giữ **0 byte partial-sum $O$ và $(m, l)$ trên TCDM** đều bắt buộc phải đặt `kj` (Vòng 4) nằm trong cùng của `dvi` (Vòng 3), và xả $O$ một lần duy nhất sau khi xong toàn chuỗi `kj = 0..KV_TILES-1`.
* **Tầng 2 — Điều phối toàn Hệ thống (`DRAM <-> 288 KB TCDM <-> Core`):**
  Áp dụng **y hệt 2 tiêu chí đã chốt cho GEMM**:
  1. **Tiêu chí 1 — Chạy được MỌI kích thước ma trận trên TCDM 288 KB?**
     * Trong RTL `snax_llm_sequencer.sv` hiện tại: **FAIL (SAI).** Vì chỉ mới có DMA cho `K, V` (Vòng 4), **hoàn toàn chưa có Macro-DMA nạp `Q` và xả `O`**, nên mọi ma trận có `Q + O > 144 KB` (`N_Q` dài ở Prefill hoặc `H_Q` lớn) sẽ **tràn TCDM**.
     * Tuy nhiên **về mặt thuật toán 4 vòng đó**, nếu bổ sung **DMA nạp từng cụm `Q` và DMA xả từng cụm `O`** ở Vòng 1–2, thì tại bất kỳ thời điểm nào TCDM chỉ cần giữ ít nhất `1` cụm `Q_sub`, vài slot `K, V`, và Ping-Pong `O_buf` $\implies$ **chạy được mọi kích thước `(H, N_Q, N_KV, d_k, d_v)` không giới hạn!**
  2. **Tiêu chí 2 — Đã TỐI ƯU NHẤT về băng thông DRAM & Tái sử dụng dữ liệu chưa?**
     * **CHƯA TỐI ƯU.** Ngay cả khi đã thêm DMA `Q/O`, nếu chạy thẳng thứ tự `h -> qi -> dvi -> kj` và DMA cứ kéo `(K+V)` ở Vòng 4 như code hiện tại thì mắc **3 lãng phí băng thông DRAM nặng** (phân tích định lượng ở Phần 5 ngay dưới đây).

---

## Phần 5. Phân tích Băng thông DRAM & Dung lượng TCDM 288 KB (Chứng minh Phương án Tối ưu)

### 5.1. Số liệu kích thước thực tế trên TCDM `288 KB` (`144 KB` cho `Banks 0..15` + `144 KB` cho `Banks 16..31`)
Xét cấu hình Edge LLM chuẩn (`Llama-3.2-1B` & `Qwen2.5-1.5B` với `d_k = d_v = 64` hoặc `128`):
* `1` tile **Query** `Q_tile` (`8 tokens × 64 dims` FP4): $8 \times (64/16) \times 9\text{ B} = \mathbf{288\text{ B}}$ (`d_k = 128`: `576 B`).
* `1` token **K** (hoặc **V**) của `1` KV-head (`d = 64` FP4): $4 \times 9\text{ B} = \mathbf{36\text{ B}}$ (`d = 128`: `72 B`).
* `1` lát `16` tokens của `1` KV-head (`K` hoặc `V`, `d = 64` FP4): $16 \times 36\text{ B} = \mathbf{576\text{ B}}$ (`K + V` = `1,152 B`).
* **Tổng dung lượng KV của CẢ 1 KV-head theo độ dài context $N_{KV}$ (`d_k = d_v = 64`):**

| Độ dài context $N_{KV}$ | `K` (1 head) | `V` (1 head) | **`K + V` (1 head)** | Nằm gọn trong `Banks 0..15` (`144 KB`)? |
| :--- | :---: | :---: | :---: | :---: |
| `N_KV = 512` | `18 KB` | `18 KB` | **`36 KB`** | **Vừa dư (25% ngân hàng KV)** |
| `N_KV = 1,024` | `36 KB` | `36 KB` | **`72 KB`** | **Vừa gọn (50%, chạy được cả Ping-Pong `KV0/KV1`!)** |
| `N_KV = 2,048` | `72 KB` | `72 KB` | **`144 KB`** | **Vừa khít 100% `Banks 0..15`** |
| `N_KV = 4,096..32,768` | `144 KB..1.15 MB` | `144 KB..1.15 MB` | `288 KB..2.3 MB` | **Vượt `144 KB` $\rightarrow$ bắt buộc phải stream Ring** |

* **Tổng dung lượng `Q` + `O` toàn layer `Llama-3.2-1B` (`H_Q = 32, d = 64`):**
  * Decode (`N_Q = 1`): `Q = 1.125 KB` (FP4), `O = 4 KB` (FP16) $\rightarrow$ rất nhỏ (`< 6 KB`).
  * Prefill (`N_Q = 256`): `Q = 288 KB` (FP4), `O = 1,024 KB` (`1 MB` FP16) hoặc `288 KB` (FP4) $\rightarrow$ **vượt xa `144 KB` (`Banks 16..31`)**!

### 5.2. Định lượng Lượng Đọc DRAM của các Chiến lược Vòng lặp

Ký hiệu: $H_{kv}$ số KV-head, $G = H_Q / H_{kv}$ tỷ lệ GQA (`4` trên Llama-3.2-1B, `6` trên Qwen2.5-1.5B), $Q_T = \lceil N_Q / 8 \rceil$, $DV_T = d_v / 16$ (`4` với `d_v = 64`), $\text{Size}(Q_1)$, $\text{Size}(K_1)$, $\text{Size}(V_1)$ là kích thước của **1 head**.

#### Phương án A — Theo `snax_llm_sequencer.sv` hiện tại (`h -> qi -> dvi -> kj`, mỗi bước DMA kéo `K + V`)
* Không có DMA `Q/O` (không chạy được $N_Q$ lớn trong 1 lệnh).
* Lượng đọc **`K`** từ DRAM cho 1 layer:
  $$\text{DRAM\_Read}(K) = H_{kv} \times G \times Q_T \times DV_T \times \text{Size}(K_1)$$
  Ví dụ Prefill `Llama-3.2-1B` (`N_Q = 256 => Q_T = 32`, `G = 4`, `DV_T = 4`, `N_KV = 1024`):
  Cùng 1 dữ liệu $K$ bị kéo lại từ DRAM tới **$4 \times 32 \times 4 = 512\text{ lần}$**!
* Lượng đọc **`V`** từ DRAM cho 1 layer (cũng bị lặp theo $G \times Q_T$):
  $$\text{DRAM\_Read}(V) = H_{kv} \times G \times Q_T \times \text{Size}(V_1) \quad (\text{lặp } 128\times)$$

#### Phương án B — Sắp lại GQA (`h_kv -> h_grp -> qi -> dvi -> kj`) + Phân 2 chế độ tự động theo sức chứa `Banks 0..15` (`144 KB`)

**Trường hợp B1 — `Size(K_1 + V_1) <= 144 KB` (hoặc `<= 72 KB` khi Ping-Pong, phủ tới `N_KV <= 1,024..2,048` với `d = 64`):**
* **Tại Vòng `h_kv` (ngoài cùng):** Nạp trọn **`K[h_kv]` + `V[h_kv]` ĐÚNG 1 LẦN** từ DRAM lên `Banks 0..15` (`144 KB`)!
  *(Nếu `<= 72 KB`, dùng Ping-Pong `KV0/KV1` để nạp ngầm `h_kv + 1` trong lúc tính `h_kv`!)*
* Bên trong chạy hết mọi Q-head `h_grp = 0..G-1`, mọi `qi = 0..Q_T-1`, mọi `dvi = 0..DV_T-1` trực tiếp từ TCDM — **0 byte đọc lại `K` và `V` từ DRAM!**
* **Lượng đọc DRAM cho `K` và `V`:**
  $$\text{DRAM\_Read}(K+V) = \mathbf{1\times} \quad (\text{giảm } \mathbf{512\times} \text{ cho } K \text{ và } \mathbf{128\times} \text{ cho } V \text{ ở ví dụ trên!})$$

**Trường hợp B2 — Long-Context `Size(K_1 + V_1) > 144 KB` (ví dụ `N_KV = 8,192..32,768`):**
Không thể nhét hết `1` KV-head vào `144 KB`, nhưng vẫn có 2 phép tối ưu lớn so với Phương án A ngay trên cùng Latch `8 S + 8 O`:
1. **Tách địa chỉ & tái sử dụng từng khối `K_macro` (`B_KV` tokens, vừa `144 KB`) trong TCDM hay cắt đôi?**
   * *Chú ý giới hạn Latch `8 O-slots`:* Nếu đưa `KV_macro` (`kj_outer`) ra ngoài `Q_T × G`, vì chưa kết thúc toàn chuỗi `KV`, kết quả $O$ và $(m, l)$ lúc đó còn dở dang $\rightarrow$ muốn thoát ra sang `Q_tile` khác thì phải **xả partial $O, m, l$ 32-bit ra TCDM và nạp lại** (cần thêm đường `acc_preload` vào `rmw_latch_acc_buffer` + `u_l_accumulator` + `u_softmax_requantizer`, hiện phần cứng Core **CHƯA** có cổng preload lại $O, m, l$ từ TCDM!).
   * Vì phần cứng Core **chỉ hỗ trợ giữ nguyên $O, m, l$ cho đến hết `kj` (không có preload $O$ 32-bit)**, nên **khi `N_KV` vượt `144 KB`, `kj` BẮT BUỘC phải hoàn thành trọn `0..KV_TILES-1` bên trong `(h_grp, qi, dvi)`**, và streaming qua Ring Buffer `4 slots` ở `Banks 0..15` (**tuyệt đối đúng quy luật vật lý giống hệt lý do `K_TILES` nằm trong cùng bên GEMM!**).
   * **Nhưng vẫn tối ưu được DRAM ngay ở Ring Streaming:**
     - **Ở pha Decode (`N_Q = 1, Q_T = 1`):** nếu toàn bộ `d_k` và `d_v` nhỏ, trong GQA `4:1` tại cùng `(h_kv, dvi, kj)` có thể chỉ chạy `4` Q-heads (`4 <= 8` slots? Không: S và O chỉ 8 slots nhưng chạy từng head 1 -> dùng chung Ring slot `K+V` qua hết `h_grp = 0..G-1` hay không? *Kiểm tra:* nếu đưa `h_grp` vào trong `kj` thì `8` O-slots không giữ cùng lúc 4 head nếu `M_Q=8`, NHƯNG **ở Decode mỗi head chỉ có `M_Q = 1` token!** `4` Q-heads của GQA `4:1` cộng lại chỉ chiếm **đúng `4` slots `<= 8`!** Còn `d_v` vẫn giữ nguyên `dvi` ngoài `kj`!).

#### Bảng Tổng hợp Chứng minh Đánh đổi Vòng lặp FlashAttention trên Chip này

| Cấp phần cứng / Workload | Vòng lặp Tối ưu | Số lần đọc `Q` từ DRAM | Số lần đọc `K, V` từ DRAM | Ghi partial `O/m/l` ra TCDM? |
| :--- | :--- | :---: | :---: | :---: |
| **`KV_head <= 144 KB`** (`N_KV <= 1K..2K`) | `h_kv -> h_grp -> qi -> dvi -> kj` (Pin KV ở `Banks 0..15`, Ping-Pong `Q/O` ở `Banks 16..31`) | **`1×`** | **`1×` duy nhất!** | **Không (`0 B`)** |
| **`KV_head > 144 KB`**, Decode (`M_Q=1`) | `h_kv -> (pack <=8 GQA heads?)` hoặc `h_kv -> h_grp -> dvi -> kj` (Ring `K, V`) | **`1×`** | **`DV_T×` (`K`), `1×..G×`** | **Không (`0 B`)** |
| **`KV_head > 144 KB`**, Prefill (`N_Q > 8`) | `h_kv -> h_grp -> qi -> dvi -> kj` (Macro-DMA Ping-Pong `Q` & `O`, Ring `K, V`) | **`1×`** | **`G × Q_T × DV_T` (`K`), `G × Q_T` (`V`)** | **Không (`0 B`)** |

*Ghi chú phản biện quan trọng cho bản thiết kế Paper / Kiến trúc:* Nếu muốn cắt luôn hệ số `DV_T` và `Q_T` khi `N_KV > 144 KB`, phần cứng Core tương lai chỉ cần đúng 2 nâng cấp rõ ràng: (1) mở rộng Latch `O` thành `8 × DV_T = 32` slots để rút `dvi` vào trong `kj` (bỏ lặp `QK^T` `4×`), và (2) thêm đường nạp lại `(O_32b, m_32b, l_32b)` vào `u_acc_buffer / softmax / l_accum` để đặt `KV_macro` ra ngoài `Q_sub`. **Còn trên phần cứng hiện tại (`snax_fused_llm_core.sv` không có cổng preload `O/m/l`), giữ `kj` hoàn tất bên trong `(qi, dvi)` + Pin `KV` khi vừa `144 KB` + Ping-Pong `Q/O` là nghiệm tối ưu toán học duy nhất.**

---

## Phần 6. Rà soát Toàn bộ Lỗi Tiềm ẩn & Giới hạn trong `snax_llm_sequencer.sv` Hiện tại

Kiểm tra trực tiếp từng dòng trong [snax_llm_sequencer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv) cho Mode `op_mode = 2'b01`:

1. **Không có DMA cho `Q` và `O` ([`#L388-L390`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L388-L390), [`#L650-L671`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L650-L671)):**
   * `q_addr_calc` giả định cả tensor `Q[H × Q_TILES × M_Q × D_K]` nằm sẵn trong TCDM.
   * `u_write_out` ghi cả tensor `O[H × Q_TILES × M_Q × D_V]` xuống TCDM, không có Ping-Pong flush `O -> DRAM`.
2. **DMA Prefetch chỉ hoạt động kiểu Ring cho từng bộ `(dma_hkv, dma_dvi, dma_kj)`, lặp `dma_h_q` & `dma_qi_q` ([`#L531-L543`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L531-L543), [`#L1140-L1170`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L1140-L1170)):**
   * Không có chế độ **Pin `KV` trong `Banks 0..15`** khi `KV <= 144 KB`.
   * Vẫn kéo lặp `K, V` theo từng head GQA (`dma_h_q`) và từng `dma_qi_q`, `dma_dvi_q`.
   * Buộc cứng layout DRAM của `K` và `V` phải gộp liền `(k_blocks + 1) × 144 B` cho **mỗi `dvi`**, khiến `K` bị đóng gói lặp lại trong bộ nhớ nếu `DV_TILES > 1` hoặc đọc thừa.
3. **Nằm chung vòng `nrm` Drain thiếu Credit Pipeline (`inflight_out_cnt`) ([`#L305-L346`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L305-L346)):**
   * `nrm_beat` dùng `core_fp16_valid_i && core_out_fifo_ready_i` thay vì kiểm soát chính xác `inflight_out_cnt < 4` qua `core_fp16_valid_i && core_fp16_ready_i` (có nguy cơ tràn 1-slot skid `out_hold_q` của Core nếu downstream `log_quantizer` hoặc `u_write_out` stall bất chợt khi pipe 3 tầng đang chứa sẵn 3 vector).
4. **Hạn chế khi `N_Q` không chia hết `M_Q` (hoặc `M_Q == 1` Decode ở `QK_RUN -> SM_DRAIN`):**
   * Khi `N_Q` không phải bội của `M_Q`, cần kiểm tra masking hoặc tail `cur_m_q = min(M_Q, N_Q - qi*M_Q)` để không viết rác ở tile `Q` cuối và tính đúng Causal Mask (`auto_causal_mask` dùng `cfg_attn_m_q_i` cố định ở [`#L483`](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sequencer.sv#L483)).
   * Khi `N_KV` không chia hết cho `16`: `auto_causal_mask` vẫn dùng key index tuyệt đối, nhưng nếu không mask thêm biên `valid_kv_len` thì những token key pad ở lát `KV_TILES - 1` (khi `!cfg_is_causal_i` hoặc tại hàng cuối lớn hơn `N_KV_actual`) sẽ lọt vào Softmax nếu ngoài phần mềm không pad `-inf`.
5. **Chồng chéo CSR Aliasing ([snax_flash_csr_snax.sv#L326-L347](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_flash_csr_snax.sv#L326-L347)):**
   * Dùng chung 6 thanh ghi CSR `32..37` cho cả Attention, Readback 4D, và GEMM Macro-DMA nên rất khó đọc cấu hình cùng lúc nếu nâng cấp Attention thêm Macro-DMA `Q/O`.

---

## Phần 7. Đặc tả Thiết kế Lại FlashAttention Sequencer (Đạt Cả 2 Tiêu chí & Hợp nhất 1 Sequencer Duy nhất)

Mục tiêu kiến trúc: Khi gom về **1 Sequencer duy nhất** trong [snax_llm_top_clean.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv) (gỡ bỏ hoàn toàn việc instantiate song song 2–3 sequencer), phần FlashAttention Engine phải nâng cấp theo **đúng chuẩn công nghiệp như GEMM 4-Loop**:

### 7.1. Nguyên tắc Phân bổ TCDM 288 KB cho FlashAttention Tổng quát
Giữ nguyên quy tắc phân đôi Zero-Conflict của hệ thống:
* **`Banks 0..15` (`144 KB`) — Dành riêng cho `K, V` (16-lane parallel):**
  * **Mode A (`KV_Pin_En = 1`, khi `1` KV-head `<= 72 KB` Ping-Pong hoặc `<= 144 KB`):** Nạp `K[h_kv] + V[h_kv]` **1 lần duy nhất** trên TCDM cho toàn bộ `G` heads trong nhóm GQA, toàn bộ `Q_tiles` và toàn bộ `dvi`.
  * **Mode B (`KV_Pin_En = 0`, Long Context `KV_head > 144 KB`):** Chia `Banks 0..15` làm `4` Ring Slots (`36 KB`/slot), streaming từng cặp `(K[kj], V[dvi, kj])`.
* **`Banks 16..31` (`144 KB`) — Dành cho `Q` và `O` (Ping-Pong Macro-DMA):**
  * **`Q_buf0 / Q_buf1`:** Chứa `Q_sub` tiles (`M_Q_sub × M_Q` tokens × `D_K`) cho 1 hoặc nhiều Q-head (chỉ tốn vài KB — ví dụ `Q_sub = 8` tiles = `64 tokens` × `64 dims` chỉ nặng `2.25 KB`!).
  * **`O_buf0 / O_buf1`:** Chứa từng cụm `O` đã chuẩn hóa xong (ngưỡng xả **`CAP_O_ATTN`** tương tự GEMM, tròn `64B/beat` DRAM) $\rightarrow$ DMA `TAG_O` đẩy dần ra DRAM song song với lúc Compute đang chạy sang buffer còn lại!

### 7.2. Cấu trúc Vòng lặp Chuẩn hóa cho FlashAttention
```text
// Tham số: H_KV, GQA_RATIO (H_Q = H_KV * GQA_RATIO), Q_TILES, DV_TILES, KV_TILES, M_Q (<=8),
//          Q_SUB (số Q-tiles mỗi lô DMA Q), KV_PIN_EN (1 nếu 1 KV-head vừa Banks 0..15)

for (h_kv = 0; h_kv < H_KV; h_kv++):                   // Nhóm theo KV head để chia sẻ GQA!
  // Nếu KV_PIN_EN: DMA_TAG_KV nạp trọn K[h_kv] + V[h_kv] 1 LẦN (Ping-Pong 2 KV-head nếu <=72KB)

  for (h_grp = 0; h_grp < GQA_RATIO; h_grp++):         // Các Q-head dùng chung h_kv
    for (q_base = 0; q_base < Q_TILES; q_base += Q_SUB):
      // Nếu Attn Macro-DMA bật: DMA_TAG_Q nạp Q[h, q_base..q_base+cur_q-1] vào Q_buf[q_wr]

      for (qi = 0; qi < cur_q_sub; qi++):              // Từng Q_tile (M_Q <= 8 queries)
        for (dvi = 0; dvi < DV_TILES; dvi++):          // Từng lát 16 channels d_v (8 O-slots Latch)
          kv_first = 1;
          for (kj = 0; kj < KV_TILES; kj++):           // Duyệt hết chuỗi KV (0 partial O ra TCDM)
            // [1] QK_LAUNCH -> QK_RUN: Q[q_base+qi] x K[h_kv, kj]^T -> S-slots 8..15
            //     (Chờ !nrm_busy trước khi vào SM_DRAIN)
            // [2] SM_DRAIN: Drain S-slots 8..15 (cfg_row_start = kv_first, causal_mask) -> p_fifo
            // [3] PV_LAUNCH -> PV_RUN: Pop p_fifo x V[h_kv, dvi, kj] -> Rescale & cộng O-slots 0..7
            //     (Nếu !KV_PIN_EN: nhả credit Ring KV ở đây)

          // Xong toàn bộ kj: kích Background Drain Engine (nrm) dùng chung Drain Guard (inflight < 4)
          // Xả O-slots 0..7 -> final_normalizer -> u_write_out -> O_buf[o_cmp]

        // Cuối mỗi Q_tile (hoặc khi o_cnt >= CAP_O_ATTN):
        // Nếu o_cnt đạt ngưỡng CAP_O_ATTN hoặc hết (q_base + cur_q_sub):
        //   Chờ nrm + wo xong cụm hiện tại -> DMA_TAG_O xả O_buf[o_cmp] ra DRAM, đảo O_buf!
```

### 7.3. Chia sẻ Tài nguyên với GEMM 4-Loop trong 1 Sequencer Hợp nhất (Không đẻ rác / Không copy code)
Khi gộp **GEMM 4-Loop** + **FlashAttention** + **Readback** + **DMA-WB** vào **ĐÚNG 1 Sequencer duy nhất**:
1. **Dùng CHUNG 1 DMA Descriptor & Credit Arbiter:**
   * Cả GEMM lẫn Attention dùng chung **1 bộ máy DMA 3 tag** (`TAG_O_WB > Urgent TAG_IN_A/Q > Prefetch TAG_IN_A/Q / TAG_W/KV`) đẩy qua **1 cổng `dma_req_*` duy nhất**.
   * `ST_DMA_WB` (`cfg_dma_wb_i = 1`) chỉ là **2 state** gọi thẳng vào cùng cổng DMA đó (không tạo arbiter riêng).
2. **Dùng CHUNG 1 Output Drain Credit Counter (`inflight_out_cnt_q < 4`):**
   * Dùng cho **cả `GEMM Drain` lẫn `Attention nrm Drain`** (chỉ khác nhau dải slot xả: `0..15` với GEMM Prefill, `0` với GEMM Decode, `0..M_Q-1` với Attention), xử lý triệt để rủi ro backpressure pipe 3 chu kỳ ở cả 2 mode!
3. **Tách rõ thanh ghi điều khiển nội bộ — Chỉ chia 2 Execution Engines sạch sẽ:**
   * **Dispatcher + Shared DMA/Drain Wrapper** (~120 dòng): nhận lệnh tại `IDLE` (`OP_GEMM`, `OP_ATTN`, `OP_READBACK` 2 state, `DMA_WB` 2 state), bỏ `op = 2'b10` thừa.
   * **GEMM 4-Loop Engine** (đã verify pass QuestaSim + Verilator).
   * **FlashAttention Engine** (`QK -> SM -> PV -> nrm` + Causal Mask + GQA + KV Pin / Ring + Macro `Q/O`).

---

## Phần 8. Checklist Kiểm chứng Đầy đủ (Testbench Verification Matrix)

Để khẳng định 100% FlashAttention đạt **cả 2 tiêu chí** trước khi chốt hệ thống hợp nhất:

1. **Tương thích ngược các test hiện có (Regression 0 lỗi):**
   * `tb_ktiling_gqa`: `K-Tiling non-DMA`, `K-Tiling + 4-port DMA`, `GQA 4:1`, `Multi-bank Act K=8192`.
   * `tb_llama3_1b_layer`: Decode (`M=1, N_KV=512..2048`) và Prefill (`M=16, GQA 4:1, D=64`) khớp SQNR / CosSim (`>= 0.99` vs Golden).
2. **Kiểm chứng Tiêu chí 1 (`Q + O > 288 KB` trong 1 lệnh Sequencer duy nhất):**
   * Chạy **1 lệnh đơn lẻ** Prefill Attention kích thước vượt TCDM, ví dụ `H_Q = 32, H_KV = 8, N_Q = 128..256, N_KV = 256, D_K = D_V = 64` (`O` FP16 = `512 KB..1 MB > 288 KB TCDM`), tự động Macro-DMA `Q` vào Ping-Pong và xả `O` ra DRAM khớp 100% bit-exact từng head & từng token.
3. **Kiểm chứng Tiêu chí 2 (`KV_Pin_En` + GQA reuse):**
   * Đo trực tiếp tổng `dram_rd_beats` khi `N_KV = 512..1024` (`K+V <= 72..144 KB`) với `H_Q = 8, H_KV = 2 (GQA 4:1), N_Q = 32`: chứng minh `K` và `V` chỉ đọc từ DRAM **đúng `1×` cho mỗi KV-head** thay vì lặp `4 × 4 × 4 = 64×`.
4. **Kiểm chứng Long-Context Fallback (`N_KV > 2048` với `K+V > 144 KB`):**
   * Ví dụ `N_KV = 4096` (KV vượt `144 KB` `Banks 0..15`), tự động chạy qua Ring 4 slots, kèm Causal Mask cho cả lát lẻ và `N_Q` không chia hết `M_Q = 8`.
5. **Drain Backpressure Stress cho Attention:**
   * Chèn LFSR stall lên `wo_m_req_ready` / `log_quantizer` ngay khi `nrm` xả slots `0..7` gối đầu vào `ST_ATTN_QK_RUN` kế tiếp $\rightarrow$ xác nhận `inflight_out_cnt_q <= 4`, `0` vector rơi, `0` sai khác so với bản không stall.
