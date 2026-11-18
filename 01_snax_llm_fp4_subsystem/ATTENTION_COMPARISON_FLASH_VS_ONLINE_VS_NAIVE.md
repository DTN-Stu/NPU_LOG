# Phân tích Kiến trúc Attention trên Edge NPU: Standard Attention vs. Online Softmax vs. FlashAttention (`snax_fused_llm_core`)

Tài liệu này phân tích chi tiết cách hệ thống phần cứng (`u_core` + `u_sfu` + `TCDM 288 KB` + `Simple Gearbox DMA 512b<->4×72b`) phải vận hành và chi phí bộ nhớ/băng thông DRAM trong 3 kịch bản thiết kế Attention:
1. **Không có FlashAttention và không có Online Softmax** (*Standard / Naive Attention*).
2. **Chỉ có Online Softmax** (*không có Tile-level `QK^T -> PV` Fusion & `O`-Rescaling*).
3. **FlashAttention hoàn chỉnh** (*kiến trúc hiện tại trên `snax_fused_llm_core`*).

---

## 1. Bối cảnh & Hệ quy chiếu (Reference Workload)

Để dễ định lượng, xét 1 layer của mô hình **`NousResearch/Llama-3.2-1B`** ở chế độ Prefill với độ dài chuỗi $M = N_{\text{kv}} = 1,024$ tokens:
- **Cấu hình mô hình**: $H_Q = 32$ heads, $H_{KV} = 8$ heads (GQA 4:1), $d_k = d_v = 64$.
- **Dung lượng SRAM on-chip (TCDM)**: $32\text{ banks} \times 1,024\text{ words} \times 72\text{-bit} \approx \mathbf{288\text{ KB}}$.
- **Băng thông DMA**: `512-bit DRAM` $\leftrightarrow$ `4 × 72-bit TCDM` (`288 bits/cycle = 36 B/cycle`, tức `18–36 GB/s` ở `500 MHz – 1 GHz`).

Ma trận điểm Attention đầy đủ cho 1 layer có kích thước:
$$H_Q \times M \times N_{\text{kv}} = 32 \times 1,024 \times 1,024 = \mathbf{33,554,432\text{ phần tử}}$$

---

## 2. Kịch bản 1: Không có FlashAttention và không có Online Softmax (Standard Attention)

Công thức chuẩn:
$$S = Q K^T, \qquad P_{ij} = \frac{\exp(S_{ij} - \max_k S_{ik})}{\sum_k \exp(S_{ik} - \max_k S_{ik})}, \qquad O = P V$$

Vì hàm $\text{softmax}$ chuẩn đòi hỏi phải biết **giá trị lớn nhất $m_i = \max_k S_{ik}$** và **tổng chuẩn hóa $l_i = \sum_k e^{S_{ik} - m_i}$ trên toàn bộ hàng $N_{\text{kv}}$** trước khi tính được bất kỳ phần tử xác suất $P_{ij}$ nào, phần cứng **bắt buộc phải tách Attention thành 3 pha tuần tự** và vật chất hóa (materialize) toàn bộ các ma trận trung gian kích thước $O(N^2)$ ra bộ nhớ ngoài.

### 2.1. Luồng thực thi 3 pha trên phần cứng
1. **Pha 1 — Tính toàn bộ $S = Q \cdot K^T$:**
   - `u_core` chạy ở chế độ GEMM nhân từng tile của $Q$ với $K^T$.
   - Kết quả từng tile của $S$ (định dạng FP16) được ghi ra TCDM (`OUT_BANK`), sau đó **DMA Write (`TCDM -> DRAM`)** phải xả toàn bộ ma trận $S$ ra DRAM vì TCDM chỉ có `288 KB`.
2. **Pha 2 — Tính Softmax 2–3 pass trên toàn bộ hàng $S$:**
   - Hệ thống **DMA Read (`DRAM -> TCDM`)** nạp lại từng hàng của $S$ cho `u_sfu` / CPU thực hiện:
     - **Pass 1 (Max-reduction):** Duyệt cả $N_{\text{kv}}$ phần tử để tìm $m_i = \max_j S_{ij}$.
     - **Pass 2 (Exp & Sum-reduction):** Đọc lại $S_{ij}$, tính $e^{S_{ij} - m_i}$ và cộng dồn mẫu số $l_i = \sum_j e^{S_{ij} - m_i}$.
     - **Pass 3 (Normalize & Requant):** Chia cho $l_i$ (hoặc nhân $1/l_i$) để ra $P_{ij}$, đưa qua `u_log_quantizer` nén sang packet 72-bit (16×FP4 + Log8 scale), rồi **DMA Write (`TCDM -> DRAM`)** ghi toàn bộ ma trận $P$ ra lại DRAM.
3. **Pha 3 — Tính $O = P \cdot V$:**
   - Hệ thống **DMA Read** nạp lại cả ma trận $P$ (vừa ghi ở Pha 2) lẫn ma trận $V$ từ DRAM vào TCDM để `u_core` chạy một phép GEMM tách biệt thứ hai tạo ra $O$.

### 2.2. Chi phí bộ nhớ & lưu lượng DRAM ($M = N_{\text{kv}} = 1,024$)
- **Kích thước ma trận $S$ (FP16, 2B/phần tử):** $32 \times 1024 \times 1024 \times 2\text{ B} = \mathbf{64\text{ MB}}$ *(gấp ~227× tổng TCDM)*.
- **Kích thước ma trận $P$ (FP4+Log8, 9B/16 phần tử):** $32 \times 1024 \times 64 \times 9\text{ B} = \mathbf{18\text{ MB}}$ *(gấp ~64× tổng TCDM)*.
- **Lưu lượng DMA $\leftrightarrow$ DRAM phát sinh thêm (chỉ tính phần trung gian $S, P$):**
  - Ghi $S$ (`64 MB`) + Đọc $S$ để Softmax (`64–128 MB` tùy số pass) + Ghi $P$ (`18 MB`) + Đọc $P$ để nhân $V$ (`18 MB`) $\approx \mathbf{164\text{ – }228\text{ MB}}$ traffic DRAM thừa cho mỗi layer!
- **Hệ quả:** Attention hoàn toàn bị **memory-bound**, latency và năng lượng DRAM tăng vọt (do năng lượng truy cập DRAM đắt hơn truy cập SRAM/register on-chip khoảng ~100×).

---

## 3. Kịch bản 2: Chỉ có Online Softmax (Không có FlashAttention Tiling & `PV` Fusion)

**Online Softmax** (Milakov & Gimelshein) cho phép cập nhật đồng thời giá trị lớn nhất $m$ và tổng chuẩn hóa $l$ chỉ trong **1 pass** khi quét tuần tự qua từng block điểm $S_j$:
$$m_{\text{new}} = \max(m_{\text{old}}, \max(S_j))$$
$$l_{\text{new}} = l_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}}) + \sum \exp(S_j - m_{\text{new}})$$

Tuy nhiên, nếu hệ thống **chỉ có Online Softmax** mà **không có cơ chế FlashAttention** (tức là không có `p_fifo` on-chip và không có `attn_o_rescaler_bank` để nhân cuốn chiếu $P_j V_j$ và rescale $O$ cũ theo hệ số $\alpha = \exp(m_{\text{old}} - m_{\text{new}})$), thì vẫn gặp 1 trong 2 giới hạn lớn tùy cách tổ chức:

### 3.1. Trường hợp 2A: Online Softmax tách pha khỏi `PV` (Chưa fuse `QK^T -> PV`)
- **Cách chạy:**
  - `u_core` vẫn tính $S = QK^T$.
  - Nhờ Online Softmax, bộ phận Softmax chỉ cần duyệt $S$ **1 lần** để có được $m_i$ và $l_i$ cuối cùng (giảm từ 2–3 pass xuống 1 pass).
  - Tuy nhiên, vì $l_i$ cuối cùng chỉ biết được **sau khi đã quét hết toàn bộ $N_{\text{kv}}$**, phần cứng vẫn chưa thể tính xác suất chuẩn hóa $P_{ij} = \frac{e^{S_{ij} - m_i}}{l_i}$ cho các token đầu tiên trong lúc đang quét!
- **Hệ quả:**
  - Vẫn phải lưu lại toàn bộ $S$ (`64 MB`) hoặc tính lại $S$, sau đó tính xong toàn bộ $P$ (`18 MB`) và **DMA Write toàn bộ ma trận $P$ ($O(N^2)$) ra DRAM**, rồi lại **DMA Read $P$** từ DRAM về để tính $O = P \cdot V$.
  - **Kết luận:** Giảm được số lần đọc lại $S$ ở bước Softmax, nhưng **không xóa bỏ được bài toán bộ nhớ & traffic $O(N^2)$** của $S$ và $P$.

### 3.2. Trường hợp 2B: Online Softmax có ghép `PV` nhưng không dùng 2D Tile Fusion (`M_Q × KV_tile`)
- Nếu cố tính nối tiếp `QK^T -> Softmax -> PV` trên từng token hoặc từng vector nhỏ mà không tổ chức theo 2D tile (`M_Q` queries $\times$ `KV_tile` keys/values trong SRAM):
  - Mỗi query sẽ phải đọc lại `K` và `V` từ DRAM riêng lẻ thay vì dùng chung một `KV_tile` cho cả nhóm `M_Q` queries.
  - Lưu lượng đọc `K, V` từ DRAM tăng lên tới **$M_Q$ lần**, đồng thời `u_core` không thể tái sử dụng dữ liệu trên 16 làn MAC như tile GEMM (tụt mạnh hiệu suất tính toán).

---

## 4. Kịch bản 3: FlashAttention hoàn chỉnh trên `snax_fused_llm_core`

Kiến trúc hiện tại của `snax_fused_llm_core` kết hợp đầy đủ **3 mảnh ghép**:
1. **Online Softmax (base-2 `LOG2E_Q16 = 94548`)** trong `softmax_requantizer` và `l_accumulator`.
2. **Tile-level `QK^T -> P -> PV` Fusion** thông qua hàng đợi on-chip **`p_fifo`** (giữ các tile xác suất cục bộ $\tilde{P}_j$ đã lượng tử hóa FP4 ngay bên trong core, không bao giờ đẩy ra TCDM/DRAM).
3. **Online Output Rescaling** thông qua **`attn_o_rescaler_bank`** (`O_new = alpha * O_old + P_j * V_j` theo từng `KV_tile`) và **`final_normalizer`** (chia cho tổng $l_i$ đúng 1 lần duy nhất ở cuối hàng sau `KV_tile` cuối cùng, rồi `4D Write-Out` thẳng ra layout token-major cho `GEMM O`).

Nhờ vậy:
- **Dung lượng $S$ và $P$ ghi ra TCDM / DRAM = `0 bytes` ($O(1)$ on-chip).**
- Mỗi `KV_tile` (`16` tokens) đọc qua 4-slot Ring Buffer từ DRAM được tái sử dụng cho cả nhóm `M_Q` queries, giữ hiệu suất MAC cao và giảm tối đa lưu lượng DMA.

---

## 5. Bảng so sánh tổng hợp (1 Layer `Llama-3.2-1B`, $M = N_{\text{kv}} = 1,024$)

| Tiêu chí | Standard Attention | Chỉ có Online Softmax | FlashAttention (`snax_fused_llm_core`) |
| :--- | :---: | :---: | :---: |
| **Lưu ma trận $S$ ($O(N^2)$)** | Có (`~64 MB` FP16) | Có *(hoặc tính lại)* | **Không (`0 B`)** |
| **Lưu ma trận $P$ ($O(N^2)$)** | Có (`~18 MB` FP4) | **Vẫn có** (`~18 MB` FP4) | **Không (`0 B`, trong `p_fifo`)** |
| **Số pass đọc $S$ cho Softmax** | 2–3 pass | 1 pass | **1 pass fused** |
| **Traffic DRAM trung gian ($S, P$)** | `~164 – 228 MB` | `~100 – 164 MB` | **`0 MB`** |
| **Tải lên DMA & TCDM** | Rất nặng (memory-bound) | Nặng (vẫn ghi/đọc $P$) | **Tối ưu** (chỉ stream $Q, K, V$ & ghi $O$) |
| **Áp dụng trên `TCDM 288 KB`** | Phải offload $S, P$ ra DRAM | Phải offload $P$ ra DRAM | **Chạy trọn vẹn từng tile trên chip** |
