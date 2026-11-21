# Spec Thiết kế GEMM Output-Stationary FSM (Chuẩn 4-Loop) cho `snax_gemm_os_sequencer.sv`

Tài liệu này là **đặc tả chuẩn duy nhất (Single Source of Truth)** thống nhất từ **toán học tiling**, **phân bổ tài nguyên TCDM 288 KB**, **thuật toán 4 vòng lặp**, **ngưỡng gom & xả Output ra DRAM**, đến **đặc tả 3 Sub-FSM ở mức chu kỳ** — dùng làm cơ sở để đối chiếu và triển khai RTL mà không có bất kỳ bước hiểu ngầm hay mập mờ nào.

---

## 1. Các lỗi ở `snax_llm_sequencer.sv` gốc cần loại bỏ

1. **Chưa hỗ trợ DMA Macro-Tiling cho Activation (`A`) & Output (`O`):** `snax_llm_sequencer.sv` gốc chỉ prefetch Weight (`W`). Khi `A` (`M×K`) hoặc `O` (`M×N`) vượt phần trống của TCDM 288 KB, 1 lệnh sequencer không chạy trọn được ma trận lớn.
2. **Deadlock 100% ở `cfg_out_mode_i == 2'd3` (Fused V-Transpose) khi `K_tiles > 1` hoặc `M_tiles × N_tiles > 1`:** `ST_GEMM_LAUNCH` phát `v_start_o = 1` ở `kt=0` làm `v_busy_i = 1`, nhưng lại kiểm tra `!v_busy_i` ngay sau đó $\rightarrow$ treo cứng.
3. **Sai địa chỉ `gemm_act_addr` khi `cfg_act_bank_layout_i == 1` và `M_tiles > 1`:** Offset theo `cnt_gm_q` bị ép bằng `0`, khiến mọi nhóm 16 token phía sau đọc lặp lại 16 token đầu.
4. **Thiếu `cfg_gemm_m_tiles_i` ở `cfg_out_mode_i == 2'd2`:** `wo_num_batches_o` chỉ bằng `cfg_gemm_n_tiles_i`, làm `u_write_out` dừng sau 16 token đầu khi `M_tiles > 1`.
5. **Bubble mỗi K-tile & thiếu Drain Pipeline Credit:** Đường xả `Port B -> final_normalizer S1 -> S2` dài 3 chu kỳ dễ rớt vector khi `u_v_quant_path` hoặc `u_write_out` backpressure.

---

## 2. Định nghĩa Chuẩn Phần cứng & Ký hiệu (Không trùng lặp, không ẩn ý)

Xét phép nhân ma trận tổng quát:
$$O_{[M \times N]} = A_{[M \times K]} \times W_{[N \times K]}^T$$

### 2.1. Thông số phần cứng cố định
* **Mảng tính toán Core (`snax_fused_llm_core`):**
  * **`NUM_LANES = 16`** (`log8_fp4_mac_array`): 16 cột Output Channel tính song song mỗi chu kỳ.
  * **`ACC_DEPTH = 16`** (`rmw_latch_acc_buffer`): 16 slot tích lũy Output-Stationary (`1` hàng dùng ở Decode `M = 1`, `16` hàng dùng ở Prefill `M = 16×`).
  * $\implies$ **1 Output Tile chuẩn = `16 × 16`** (`16` tokens × `16` output channels; Decode là `1 × 16`).
  * Core **KHÔNG có Accumulator SRAM 32-bit trên TCDM**: Mọi phép cộng dồn `psum` **chỉ diễn ra bên trong 16×16 Latch Array**; xả ra ngoài qua `final_normalizer` thành FP16 hoặc FP4 ngay khi cộng xong toàn bộ chiều `K`.
* **Độ hạt Packet (`BLOCK_SIZE = 16`):** 1 packet 72-bit = `64b FP4 + 8b Log8 scale` = **9 bytes vật lý** (1 bank-word TCDM), chứa **16 phần tử liên tiếp theo chiều `K`**.
* **TCDM Scratchpad (`multi_bank_tcdm_72b`):** `32 banks × 1024 words × 9 bytes = 294,912 bytes = 288 KB` (`32,768 packets`).
  * Quy ước địa chỉ TCDM: `addr = (word << 9) | (bank << 4)` (1 word xuyên 32 banks ứng với `0x200` trên bus địa chỉ; 1 bank bước `0x10`).
* **Bus DRAM của `snax_simple_dma`:** Rộng **512-bit = 64 bytes/beat** (dùng chung 1 bộ trọng tài DMA cho cả 3 luồng: nạp `A`, nạp `W`, xả `O`).

### 2.2. Bảng ký hiệu kích thước ma trận (Dùng thống nhất 1 bộ tên)

| Ký hiệu | Tên thanh ghi / Ý nghĩa chính xác | Đơn vị | Ghi chú bắt buộc |
| :--- | :--- | :--- | :--- |
| **`M_tiles`** | Tổng số tile 16-token của toàn ma trận (`M / 16`; Decode `= 1`) | Tile (16 tokens) | Với chiều `M` **CHỈ dùng 2 số: `M_tiles` và `M_sub`** |
| **`M_sub`** | Số tile 16-token nạp **mỗi đợt** của `A` lên TCDM (`1 <= M_sub <= M_tiles`) | Tile (16 tokens) | Tổng token của 1 đợt = `M_sub × 16`. Không dùng thêm tên `B_M` hay biến thứ 3 |
| **`N_tiles`** | Tổng số tile 16-channel của `W` và `O` (`N / 16`) | Tile (16 channels) | Mỗi tile của `W` luôn chỉ rộng `16` channels |
| **`K_BLOCKS`** | Số packet (16 phần tử `K`) trong **1 slot Ring Buffer** (`1` lần launch streamer) | Packet (9B) | `1` tile 16-channel dài `K_BLOCKS` packets tốn `K_BLOCKS × 16 × 9B = K_BLOCKS × 144 B` |
| **`K_TILES`** | Số lát cắt dọc chiều `K`: `K / (K_BLOCKS × 16)` | Lát `K` | Nếu toàn bộ `K` của 1 tile `W[16×K]` vừa 1 slot Ring $\Rightarrow$ `K_TILES = 1` |

### 2.3. Định nghĩa Chỉ số 2 chiều `(hàng, cột)` & Bản chất `psum` vs `O` (Tránh nhầm `m` với `k`)

Phải viết đủ **2 chỉ số** cho mọi mảnh ma trận:
* **`A[m, k]`**: Lát `16 × (K_BLOCKS×16)` tại **nhóm token `m`** (`m = 0 .. M_tiles-1`) và **lát `k`** dọc chiều `K` (`k = 0 .. K_TILES-1`).
* **`W[n, k]`**: Lát `16 × (K_BLOCKS×16)` tại **nhóm channel `n`** (`n = 0 .. N_tiles-1`) và **lát `k`** dọc chiều `K` (`k = 0 .. K_TILES-1`).

Công thức toán học **DUY NHẤT** cho một ô Output `O[m, n]` (`16 × 16`):
$$O[m, n] = \sum_{k=0}^{K_{\text{TILES}}-1} A[m, k] \times W[n, k]^T = A[m, 0]\times W[n, 0]^T + A[m, 1]\times W[n, 1]^T + \dots$$

**Hai nguyên tắc bất di bất dịch:**
1. **`psum` CHỈ cộng dồn dọc theo chỉ số `k` (vòng `K_TILES`) bên trong `16×16 Latch Array`**, **KHÔNG BAO GIỜ ghi `psum` dở dang ra TCDM**.
2. **`A[m=0, :] × W[n, :]` (`O[0, n]`) và `A[m=1, :] × W[n, :]` (`O[1, n]`) là 2 hàng token khác nhau của `O` — KHÔNG BAO GIỜ cộng dồn vào nhau.**
   * Chỉ sau khi tính **xong lát `k = K_TILES - 1`** của `O[0, n]`, giá trị trong Latch **đã là kết quả cuối cùng `O[0, n]`** $\rightarrow$ `Drain FSM` mới rút ra ghi vào `O_buf` trên TCDM và xóa sạch Latch (`clear_on_drain = 1`) để nhường Latch cho `O[1, n]`.

---

## 3. Phân vùng Tài nguyên TCDM 288 KB (Zero Bank-Conflict)

Toàn bộ TCDM 288 KB chia thành **2 nửa bank tách biệt hoàn toàn** (`Banks 0..15` cho `W`, `Banks 16..31` dùng chung Ping-Pong cho `A` và `O`):

```text
======================================================================================
           MULTI-BANK TCDM SCRATCHPAD (32 Banks x 1024 Words x 9B = 288 KB)
======================================================================================
        NỬA TRÁI: BANKS 0..15 (144 KB)          |      NỬA PHẢI: BANKS 16..31 (144 KB)
         (Dành riêng cho WEIGHT Ring)           |    (Dùng chung PING-PONG cho A và O)
------------------------------------------------+-------------------------------------
Word 0..255   : [ W Ring Slot 0 ]  (36 KB)      |  Word 0..255   : [ A_buf0 ] (36 KB)
              :  (16 lanes = 16 banks song song)|                :  (Chứa M_sub x K)
--------------+---------------------------------+----------------+--------------------
Word 256..511 : [ W Ring Slot 1 ]  (36 KB)      |  Word 256..511 : [ A_buf1 ] (36 KB)
              :                                 |                :  (DMA nạp mẻ sau)
--------------+---------------------------------+----------------+--------------------
Word 512..767 : [ W Ring Slot 2 ]  (36 KB)      |  Word 512..767 : [ O_buf0 ] (36 KB)
              :                                 |                :  (Core ghi kết quả)
--------------+---------------------------------+----------------+--------------------
Word 768..1023: [ W Ring Slot 3 ]  (36 KB)      |  Word 768..1023: [ O_buf1 ] (36 KB)
              :                                 |                :  (DMA xả ra DRAM)
======================================================================================
        |                                            ^                    |
        | 16-ch Weight Streamer                      | Act Streamer       | Write-Out /
        | (16 banks/cycle, 0 conflict)               | (Port 1)           | DMA WB
        v                                            |                    v
+------------------------------------------------------------------------------------+
|               snax_fused_llm_core  (16 x 16 Output-Stationary Latch)               |
|  - Port A: Cộng dồn A[gm, kt] x W[gn, kt]^T qua toàn bộ K_TILES (KHÔNG xả psum)    |
|  - Port B: Xong last_kt -> rút kết quả CUỐI O[gm, gn] qua Normalizer -> O_buf      |
+------------------------------------------------------------------------------------+
```

### Quy tắc chia ngân sách chính xác trên phần cứng:
1. **Nửa trái — `Banks 0..15` (`144 KB`, cố định cho Weight Ring):**
   * `NUM_RING_SLOTS = 4` slots $\times$ `256` words/bank $\times$ `16` banks $\times$ `9B` = `36 KB` vật lý/slot (ứng với bước địa chỉ `SLOT_SIZE_BYTES = 128 KB`).
   * Mỗi slot `W` nạp tối đa `16 channels × 256 packets = 16 × 4096` phần tử FP4. Nếu `K > 4096` (ví dụ `K = 8192`), bắt buộc chia làm `K_TILES >= 2` lát sao cho `K_BLOCKS <= 256` (thường dùng `K_BLOCKS = 128` $\rightarrow$ `18 KB`/slot).
   * `snax_streamer_wt_16ch` đọc 16 channels từ 16 banks `0..15` trong **cùng 1 chu kỳ**, **0 bank conflict**.
2. **Nửa phải — `Banks 16..31` (`144 KB`, chia động giữa `A_buf0/1` và `O_buf0/1`):**
   * Ranh giới giữa `A` và `O` cấu hình qua CSR (`cfg_macro_a_buf_stride_i`, `cfg_macro_o_buf_stride_i`, `cfg_tcdm_out_base_i`) sao cho luôn thỏa mãn ràng buộc cứng:
     $$2 \times \text{Size}(A\_\text{buf}) + 2 \times \text{Size}(O\_\text{buf}) \le 144\text{ KB}$$
   * **Kích thước 1 đợt `A_buf` (`M_sub × K`):**
     $$\text{Size}(A\_\text{buf}) = M_{\text{sub}} \times \left(\frac{K}{16}\right) \times 144\text{ B} \quad (\text{hoặc } \times 9\text{ B ở Decode } M=1)$$
     * Ví dụ `K = 2048` (`18 KB` cho mỗi tile 16-token): chọn `M_sub = 2` (`36 KB`/buffer $\rightarrow$ `A0+A1 = 72 KB`, còn `72 KB` cho `O0+O1`), hoặc `M_sub = 3` (`54 KB`/buffer $\rightarrow$ `A0+A1 = 108 KB`, còn `36 KB` cho `O0+O1`).
     * Ví dụ `K = 8192` (`72 KB` cho mỗi tile 16-token): chọn `M_sub = 1` (`A0+A1 = 144 KB` nếu giữ được, hoặc chỉ giữ `A_buf` đơn + `O` nhỏ vài KB).
   * **Kích thước 1 tile Output `O` ghi vào `O_buf`:**
     * Decode (`M = 1`): `36 B/tile` (FP16) hoặc `9 B/tile` (FP4).
     * Prefill (`M = 16×`): `576 B/tile` (`64 words × 9B`, `out_mode = 0` FP16) hoặc `144 B/tile` (`16 words × 9B`, `out_mode = 1, 2, 3` FP4).

---

## 4. Thuật toán Chính thức: 4-Loop Output-Stationary (`M_macro -> N_16 -> M_sub -> K_TILES`)

### 4.1. Tại sao dùng ĐÚNG 4 vòng này? (So sánh định lượng trên Edge LLM $M \ll N$)

Với Edge LLM (`Llama-3.2-1B`: Chunked Prefill $M = 128..256$, `Gate/Up` $K=2048, N=8192$, `Down` $K=8192, N=2048$):

| Thuật toán | Thứ tự vòng | Đọc `A` từ DRAM | Đọc `W` từ DRAM | Tràn 288 KB TCDM? | Ghi chú kỹ thuật |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **3 vòng (N ngoài M)** | `N_16 -> M_16 -> K_TILES` | **`N/16 = 512×`** (`~144 MB`) | `1×` | Không | **Phế bỏ hoàn toàn** vì ở $M=128..256$, `A` không vừa hết TCDM $\rightarrow$ đọc lại `A` 512 lần |
| **3 vòng (M ngoài N)** | `M_16 -> N_16 -> K_TILES` | `1×` | **`M/16 = 16×`** (`~138 MB`) | Không | Chạy đúng mọi size, nhưng chỉ giữ 16 token `A` nên đọc lại toàn bộ `W` tới `M/16` lần |
| **4 vòng chuẩn (Spec này)** | **`M_macro -> N_16 -> M_sub -> K_TILES`** | **`1×`** | **Giảm `M_sub×`** (`(M_tiles / M_sub)×` khi `K_TILES=1`) | **Không** | **Vừa đọc `A` đúng 1 lần, vừa tái sử dụng được `W` qua `M_sub` tiles, vừa 0 partial-sum** |

* Lưu ý khi `M_tiles <= M_sub` (ví dụ `Decode M = 1` hoặc ma trận nhỏ vừa `A_buf`): vòng `M_macro` tự chạy đúng **1 lần**, không phát sinh overhead.

### 4.2. Mã giả tường minh 100% của thuật toán 4 vòng

```text
// Tham số đầu vào: M_tiles, N_tiles, K_BLOCKS, K_TILES, M_sub (<= M_tiles), CAP_O (xem Mục 5)

for (m_base = 0; m_base < M_tiles; m_base += M_sub):        // [VÒNG 1: M_macro] Đợt M_sub tiles của A
  // -> DMA_TAG_A nạp A[m_base .. m_base+cur_m-1, 0..K-1] 1 LẦN vào A_buf[a_wr]
  //    (Double-buffer: trong lúc Core tính A_buf hiện tại, DMA nạp ngầm đợt kế vào A_buf còn lại)
  o_tile_cnt = 0;
  o_dram_col_base = m_base;

  for (gn = 0; gn < N_tiles; gn++):                         // [VÒNG 2: N_16] Từng lát 16 channel W
    // -> DMA_TAG_W nạp W[gn, kt] (16 x K_blocks*16) vào Ring Slot r

    for (gm = 0; gm < cur_m_sub; gm++):                     // [VÒNG 3: M_sub] Quét qua các nhóm 16 token trong A_buf
      for (kt = 0; kt < K_TILES; kt++):                     // [VÒNG 4: K_TILES] Cộng dồn K TRONG LATCH
        // Core lấy A[m_base + gm, kt] (từ A_buf) x W[gn, kt] (từ Ring Slot)
        // Cộng dồn psum ngay trong 16x16 Latch Array (KHÔNG ghi psum ra TCDM)

        if (kt == K_TILES - 1):
          // Đã có KẾT QUẢ CUỐI O[m_base + gm, gn]!
          // Drain FSM rút 16 hàng từ Port B -> Normalizer -> u_write_out ghi vào O_buf[o_cmp]
          // Xóa sạch Latch (clear_on_drain = 1) trước khi sang gm+1

        // Cơ chế nhả Credit W Ring Slot (RẤT QUAN TRỌNG, xem Mục 4.3):
        if (K_TILES == 1):
          // Chỉ nhả credit W[gn] khi ĐÃ XONG gm cuối cùng (gm == cur_m_sub - 1)!
          // -> Mọi gm = 0..cur_m_sub-1 dùng chung ĐÚNG 1 slot W[gn] trong TCDM!
        else:
          // K_TILES > 1: Xem mục 4.3 bên dưới

    // Kết thúc 1 cột gn (đã ghi thêm cur_m_sub tiles kết quả vào O_buf[o_cmp]):
    o_tile_cnt += cur_m_sub;

    // Kiểm tra ngưỡng "gom đủ" để xả O_buf ra DRAM (KHÔNG cần vòng N_macro!):
    if (macro_dma_en && (o_tile_cnt >= CAP_O || gn == N_tiles - 1)):
      // -> Chờ u_write_out đẩy xong cột hiện tại vào O_buf[o_cmp]
      // -> Gắn DMA_TAG_O đẩy O_buf[o_cmp] (o_tile_cnt tiles) ra DRAM (TCDM -> DRAM)
      // -> Đảo O_buf[o_cmp] <-> O_buf[~o_cmp] (Ping-Pong), reset o_tile_cnt = 0
```

### 4.3. Điều kiện tái sử dụng Weight `W` qua `M_sub` (Làm rõ `K_TILES == 1` vs `K_TILES > 1`)
Không được hiểu ngầm rằng *"cứ có `M_sub` là auto reuse `W` với mọi `K`"*:
* **Khi `K_TILES == 1`** *(toàn bộ `16 × K` của 1 `N_16` nằm gọn trong 1 slot Ring Buffer — ví dụ `K <= 4096` với slot 36 KB)*:
  * Slot `W[gn, 0]` nạp 1 lần vào Ring Buffer.
  * Trong vòng `gm = 0 .. cur_m_sub - 1`, `w_release_credit` **CHỈ phát ở `gm == cur_m_sub - 1`** (`last_gm`).
  * Nhờ vậy, **`W[gn, 0]` được dùng chung đủ `cur_m_sub` lần $\implies$ giảm `cur_m_sub×` băng thông đọc `W` từ DRAM!**
* **Khi `K_TILES > 1`** *(ví dụ `K = 8192`, phải cắt thành `K_TILES = 4` lát)*:
  * Vì `K_TILES` nằm trong `gm`, với mỗi `gm` Core lần lượt cần `W[gn, 0], W[gn, 1], ..., W[gn, K_TILES-1]`.
  * **Chế độ mặc định (an toàn cho mọi `K_TILES`):** Nếu không giữ đủ cả `K_TILES` slot cùng lúc trong Ring, sau mỗi `(gm, kt)` sequencer nhả slot ngay và DMA stream lại tuần tự (`M_sub` hiệu dụng về stream thường, **không bao giờ deadlock** nếu `K_TILES > NUM_RING_SLOTS`).
  * *(Mở rộng tương lai nếu `2 <= K_TILES <= NUM_RING_SLOTS`, ví dụ `K = 8192` với `K_TILES = 4` slots trong 144 KB: có thể nạp cả `K_TILES` lát của `W[gn]` vào 4 slots và giữ nguyên cả `K_TILES` slots qua hết `gm = 0..cur_m_sub-1`.)*

---

## 5. Cơ chế Gom & Xả Output `O` ra DRAM (Thay thế hoàn toàn ý niệm vòng thứ 5 `N_macro`)

### 5.1. Tại sao KHÔNG xả từng tile `144 B` thẳng ra DRAM ngay khi Drain xong?
1. Bus DRAM rộng **512-bit = 64 bytes/beat**: 1 tile `16×16` FP4 chỉ nặng **`144 B = 2.25 beats`** (bị lẻ beat, phải RMW/byte-strobe phí băng thông).
2. `snax_simple_dma` là **1 cổng DMA dùng chung cho cả 3 luồng (`A`, `W`, `O`)**: cứ 16 chu kỳ tính lại ngắt ngang xin DMA đẩy `144 B` sẽ tranh chấp liên tục với luồng prefetch `W`, gây đói Core.

### 5.2. Vậy "Gom đến khi nào là ĐỦ" để DMA xả `O_buf` ra DRAM?
Không cần đẻ thêm biến vòng lặp thứ 5 `N_macro` hay cấu hình rườm rà — phần cứng đếm ngay sau mỗi cột `gn` (`N_16`) dựa trên **ngưỡng `CAP_O`** (tính bằng số tile) thỏa **cùng lúc 3 điều kiện vật lý**:

1. **Không tràn nửa `O_buf` trên TCDM:**
   $$\text{CAP\_O} \times \text{tile\_bytes} \le \text{Size}(O\_\text{buf0})$$
   *(Ví dụ khi `A0+A1 = 72 KB`, mỗi `O_buf` có tối đa `36 KB` $\Rightarrow$ chứa được tới `256` tile FP4 hoặc `64` tile FP16.)*
2. **Tròn bội `64 B/beat` trên bus DRAM (không bị lẻ beat):**
   * **Prefill FP4 (`144 B/tile`):** Vì $\gcd(144, 64) = 16$ và $4 \times 144\text{ B} = 576\text{ B} = 9 \times 64\text{ B}$, chỉ cần số tile xả **là bội của 4 tiles** là tổng byte **chia hết tuyệt đối cho 64B**!
   * **Prefill FP16 (`576 B/tile`):** Mỗi tile vốn đã bằng đúng `9 × 64 B` (tròn 64B với mọi bội $\ge 1$).
3. **Đủ lớn để giấu độ trễ DMA writeback & giữ layout DRAM gọn:**
   * Chọn `CAP_O` sao cho mỗi lần xả gửi từ **`8–32` tiles** (`~1.15–4.6 KB` = `18–72 beats` ở FP4; `~4.5–18 KB` = `72–288 beats` ở FP16) và xả **tại biên hết một cột `gn`** (`o_tile_cnt` tăng bước `cur_m_sub` sau mỗi `N_16`).
   * **Điều kiện kích xả chính xác trong RTL (không vòng lặp phụ):**
     $$\text{Flush\_Trigger} = (\text{kt} == K_{\text{TILES}}-1) \land (\text{gm} == \text{cur\_m\_sub}-1) \land \big[(\text{o\_tile\_cnt} + \text{cur\_m\_sub} \ge \text{CAP\_O}) \lor (\text{gn} == N_{\text{tiles}}-1)\big]$$
   * *(Ghi chú về layout DRAM: Vì trong mỗi cụm xả gồm `cur_m_sub` hàng × `cols_in_chunk` cột `N_16`, `u_write_out` ghi liên tiếp vào `O_buf0` theo thứ tự cột `gn -> gm`, nên `DMA_TAG_O` đẩy cụm đó ra DRAM tại `dram_out_base + offset(m_base, col_start)` thành một khối chữ nhật `(cur_m_sub × cols_in_chunk)` hoàn toàn xác định — hoặc viết thẳng theo row-stride nếu cần.)*

---

## 6. Đặc tả Chi tiết 3 Sub-FSM cho Implementation RTL 4-Loop

```text
 +--------------------------------------------------------------------+
 | SUB-FSM 3: DMA Descriptor & Credit Controller (dma_state_q)        |
 |   Trọng tài DMA 3 tags: DMA_TAG_O (xả O) -> DMA_TAG_A -> DMA_TAG_W |
 +----------+---------------------------------+--------------------+--+
            | [DMA -> CMP]                    | [CMP -> DMA]       ^
            | a_buf_ready, w_buf_ready,       | w_rel_credit,      |
            | o_buf_wb_busy_q, dma_busy_q     | o_flush_commit     |
            v                                 v                    |
 +-----------------------------------------------------------------+--+
 | SUB-FSM 1: Compute FSM (cmp_state_q)                               |
 |   Đúng 4 vòng: M_macro (m_base) -> N_16 (gn) -> M_sub (gm) -> K (kt)|
 |   Điều phối act_streamer, wt_streamer, wo/v_ctrl & MAC Port A      |
 +----------+------------------------------------------------------+--+
            | [CMP -> DR] drain_start_pulse   ^ [DR -> CMP]
            v                                 | dr_busy, inflight_out_cnt_q,
 +--------------------------------------------+ gemm_beat_ok (WAR guard)
 | SUB-FSM 2: Drain & Output Guard FSM (dr_state_q)                   |
 |   Port B Drain (0..15 / 0) + Credit 3-stage pipe + WAR Guard       |
 +--------------------------------------------------------------------+
```

### 6.1. Sub-FSM 1: Compute FSM (`cmp_state_q`)
* **`CMP_IDLE`**: Chờ `start_i`, reset mọi counter (`cmp_m_base_q`, `cnt_gn_q`, `cnt_gm_q`, `cnt_kt_q`, `cnt_gk_q`, `cnt_gbeat_q`, `o_tile_cnt_q`).
* **`CMP_O_INIT`** *(Chạy lúc đầu mẻ `O_buf` mới — `1` chu kỳ nếu `O_buf[o_cmp_sel_q]` rảnh)*:
  * Đợi `!wo_busy_i && !v_busy_i && !dr_busy && (inflight_out_cnt_q == 0) && !o_buf_wb_busy_q[o_cmp_sel_q]`.
  * Tính trước số tile của cụm `O_buf` hiện tại: `cur_o_cols = min(CAP_O_COLS, N_tiles - o_col_base_q)` $\rightarrow$ `cur_o_batches = cur_m_sub × cur_o_cols`.
  * Nếu `out_mode == 2'd3` (Fused V): phát `v_start_o = 1` **1 lần duy nhất** với `v_num_batches_o = cur_o_batches`.
  * Ngược lại: phát `wo_start_o = 1` cho `cur_o_batches` tiles vào `O_buf[o_cmp_sel_q]`.
  * Chuyển sang **`CMP_KT_LAUNCH`**.
* **`CMP_KT_LAUNCH`**:
  * Điều kiện vào: `a_buf_ready && w_buf_ready && !act_busy_i && !wt_busy_i` (**tuyệt đối KHÔNG kiểm tra `!v_busy_i`**).
  * Phát `act_start_o = 1`, `wt_start_o = 1` cho `(m_base + gm, gn, kt)` với `k_blocks = K_BLOCKS` $\rightarrow$ **`CMP_KT_RUN`**.
  * *Công thức địa chỉ bắt buộc phải đúng cho cả 2 layout `act_bank_layout` (khắc phục dứt điểm lỗi ở RTL cũ):*
    * Nếu `cfg_act_bank_layout_i == 1` (16-bank A): `act_tile_offset = gm * total_k_blocks + kt * K_BLOCKS`.
    * Nếu `cfg_act_bank_layout_i == 0` (linear A): `act_tile_offset = (gm * 16) * total_k_blocks + kt * K_BLOCKS`.
* **`CMP_KT_RUN`**:
  * Khi `gemm_compute_fire = act_valid_i && (&wt_valid_i) && gemm_beat_ok`: xuất `lane_valid_o = 16'hFFFF`, `lane_acc_idx_o = cnt_gbeat_q`.
  * Thứ tự tăng bộ đếm **đúng chuẩn 4 vòng**: `cnt_gbeat_q` (`0` với Decode, `0..15` với Prefill) $\rightarrow$ `cnt_gk_q` (`0..K_BLOCKS-1`) $\rightarrow$ `cnt_kt_q` (`0..K_TILES-1`) $\rightarrow$ `cnt_gm_q` (`0..cur_m_sub-1`) $\rightarrow$ `cnt_gn_q` (`0..N_tiles-1`) $\rightarrow$ `cmp_m_base_q` (`+= M_sub`).
  * Tại beat cuối của `kt` (`last_gbeat && last_gk`):
    * Nếu `last_kt`: bật `drain_start_pulse = 1` sang Drain FSM.
    * Nhả credit Ring `W`: `w_release_credit = cfg_enable_dma_i && ((cfg_gemm_k_tiles_m1 == 0) ? last_gm : 1'b1)`.
    * Chuyển trạng thái:
      * Nếu chạm ngưỡng xả `O_buf` (`last_kt && last_gm && (o_chunk_done || last_gn)`) và `macro_dma_en`: sang **`CMP_O_COMMIT`**.
      * Nếu `!macro_dma_en` và xong toàn job (`last_kt && last_gm && last_gn`): sang **`CMP_WAIT_DONE`**.
      * Ngược lại: quay về **`CMP_KT_LAUNCH`**.
* **`CMP_O_COMMIT`**:
  * Chờ `!wo_busy_i && !v_busy_i && !dr_busy && (inflight_out_cnt_q == 4'd0) && !act_busy_i && !wt_busy_i`.
  * Phát xung **`o_flush_commit_pulse = 1`** bàn giao `O_buf[o_cmp_sel_q]` cho `DMA_TAG_O`, đảo `o_cmp_sel_q <= ~o_cmp_sel_q`.
  * Nếu vừa xong cả cột `last_gn` (hết cả đợt `M_sub` của `A_buf`): phát `macro_a_rel_pulse = 1` giải phóng `A_buf[a_cmp_rd_sel_q]`, đảo `a_cmp_rd_sel_q <= ~a_cmp_rd_sel_q`, tăng `cmp_m_base_q += eff_m_sub`.
  * Nếu xong toàn ma trận (`last_gn && last_macro_m`) $\rightarrow$ **`CMP_WAIT_DONE`**; ngược lại $\rightarrow$ **`CMP_O_INIT`**.
* **`CMP_WAIT_DONE`**: Chờ `!dr_busy && (inflight_out_cnt_q == 0) && !wo_busy_i && !v_busy_i && (!macro_dma_en || (!o_buf_wb_busy_q[0] && !o_buf_wb_busy_q[1] && !dma_busy_q))` $\rightarrow$ **`CMP_DONE`**.

### 6.2. Sub-FSM 2: Drain & Output Guard FSM (`dr_state_q`)
* **Pipeline Credit (`inflight_out_cnt_q`):**
  * `+1` mỗi chu kỳ phát `tile_drain_req_o`, `-1` mỗi chu kỳ `core_fp16_valid_i && core_fp16_ready_i`.
  * Chỉ phát drain khi `core_out_fifo_ready_i && (inflight_out_cnt_q < OUT_PIPE_CREDITS)` (`OUT_PIPE_CREDITS = 4`, che đủ 3 tầng `acc_buffer -> final_normalizer S1 -> S2`).
* **Trạng thái:**
  * **`DR_IDLE`**: Nhận `drain_start_pulse` $\rightarrow$ Prefill (`M=16`) sang **`DR_RUN`** (`dr_slot = 0`); Decode (`M=1`) sang **`DR_WAIT_D1`**.
  * **`DR_WAIT_D1`**: Chờ 1 chu kỳ (slot 0 đi hết Stage 1A $\rightarrow$ Stage 2) $\rightarrow$ **`DR_RUN`** (`dr_slot = 0`).
  * **`DR_RUN`**: Xả lần lượt `dr_slot_q = 0 .. (decode ? 0 : 15)` với `tile_clear_on_drain_o = 1`, xong về **`DR_IDLE`**.
* **Bảo vệ ghi đè WAR Hazard sang Compute:**
  $$\text{gemm\_beat\_ok} = (\text{dr\_state\_q} == \text{DR\_IDLE}) \;\lor\; (\text{dr\_state\_q} == \text{DR\_RUN} \land \text{cnt\_gbeat\_q} < \text{dr\_slot\_q})$$

### 6.3. Sub-FSM 3: DMA Descriptor & Credit Controller (`dma_state_q`)
Điều phối tuần tự 1 descriptor mỗi lần tới [snax_simple_dma.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_simple_dma.sv) (`dma_state_q` $\in$ `{DMA_IDLE, DMA_INFLIGHT}`):
1. **Ưu tiên 1 — Flush `O` (`DMA_TAG_O`, `TCDM -> DRAM`):** Khi `o_buf_wb_busy_q[o_dma_sel_q] == 1` (đẩy ngay để giải phóng vùng `O_buf` cho Ping-Pong tiếp theo).
2. **Ưu tiên 2 — Urgent `A` (`DMA_TAG_A`, `DRAM -> TCDM`):** Khi `A_buf[a_cmp_rd_sel_q]` chưa sẵn sàng mà Compute đang chờ (`dma_a_urg_req`).
3. **Ưu tiên 3 — Prefetch `A` hoặc `W` (`DMA_TAG_A` / `DMA_TAG_W`):** Nạp trước `A` đợt kế (`a_dma_wr_sel_q`) nếu trống và không chặn urgent `W`; ngược lại nạp `W[gn, kt]` vào `Ring[ring_wr_ptr_q]` khi `w_free_credits_q > 0`.
* **Lưu ý căn địa chỉ `snax_simple_dma.sv`:** Các cụm `A`, `W`, `O` phải hỗ trợ địa chỉ DRAM không chia hết 64B (qua `rd_skip_bytes` / `wr_skip_bytes` & exact `dram_req_strb_o`) để xử lý đúng mọi kích thước phần dư khi `M_tiles % M_sub != 0` hoặc `N_tiles % CAP_O_COLS != 0`.

---

## 7. Checklist Kiểm chứng & Tình trạng Đồng bộ Hiện tại

### 7.1. Ma trận kiểm chứng bắt buộc cho Spec 4-Loop
1. **Decode (`M = 1`):** `K_TILES = 1` và `K_TILES > 1` (`K = 4096`), kiểm tra `DR_WAIT_D1` trên cả FP16 (`out_mode = 0`) và FP4 (`out_mode = 1, 2`).
2. **Prefill (`M = 32`):** Tất cả 4 giá trị `cfg_out_mode_i` (`0, 1, 2, 3`), cả `cfg_act_bank_layout_i = 0/1`, và Fused-V (`out_mode = 3`) với `M_tiles × N_tiles > 1`, `K_TILES > 1` (0 deadlock).
3. **Weight Ring Reuse (`K_TILES == 1`, `M_sub >= 4`):** Đo số word `W` đọc từ DRAM phải giảm đúng `M_sub×` (ví dụ `M_tiles = 4, M_sub = 4` $\rightarrow$ chỉ fetch `1×` `W` từ DRAM).
4. **Ngưỡng xả `O_buf` (`CAP_O`) & Phần dư không chia hết:**
   * Kiểm tra `M_tiles % M_sub != 0` và `N_tiles % CAP_O_COLS != 0` (với `K_TILES >= 2`) đảm bảo vét sạch tile dư ở `O` không lệch địa chỉ DRAM.
   * Kiểm tra quy mô **vượt 288 KB TCDM** (ví dụ `M = 128, N = 64..8192, K = 4096`) với `A0/A1 + O0/O1 <= 144 KB` ở Banks `16..31`.
5. **Drain Backpressure Stress:** Bơm stall ngẫu nhiên trên `out_ready` / `u_v_quant_path` trong lúc Drain chạy $\rightarrow$ xác nhận `inflight_out_cnt_q <= 4` và 0 vector bị rơi.

### 7.2. Ghi chú trạng thái đồng bộ với RTL hiện tại
* Tài liệu này đã chốt dứt điểm kiến trúc **4-Loop (`M_macro -> N_16 -> M_sub -> K_TILES`) + Ngưỡng xả `CAP_O`** (bỏ `N_macro` và `B_M`).
* Hiện tại file [snax_gemm_os_sequencer.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_gemm_os_sequencer.sv) vẫn đang cài đặt cơ chế đếm cố định `cfg_gemm_n_sub_tiles_i` (`N_macro`, 5 bộ đếm). Khi chuẩn hóa code khớp Spec này, chỉ cần thay `eff_n_sub` bằng bộ đếm ngưỡng cột `CAP_O` ngay trong vòng `N_16` là RTL khớp 100% với đặc tả trên.
