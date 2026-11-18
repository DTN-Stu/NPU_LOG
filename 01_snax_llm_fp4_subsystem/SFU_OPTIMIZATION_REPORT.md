# Báo cáo Tối ưu Diện tích & Cập nhật RTL cho SFU (`snax_llm_sfu_opt.sv`)

Tài liệu này ghi lại các nâng cấp tối ưu phần cứng từ bản baseline (`snax_llm_sfu.sv`) sang bản tối ưu diện tích tối đa (`snax_llm_sfu_opt.sv`), cùng kết quả kiểm chứng trên QuestaSim 2024.1.

---

## 1. Bảng So sánh Tài nguyên Trước & Sau Tối ưu

| Thành phần phần cứng | Baseline (`snax_llm_sfu.sv`) | Tối ưu (`snax_llm_sfu_opt.sv`) | Mức giảm |
| :--- | :---: | :---: | :---: |
| **Độ chính xác nội bộ** | IEEE FP32 (`8b exp + 23b mant`) | **Custom FP20** (`8b exp + 11b mant`) | Giảm `~4×` diện tích mỗi bộ nhân (`12×12` vs `24×24`) |
| **Bộ đệm RMSNorm (`rms_buf`)** | `256 × 256b = 65,536` Flip-Flops | **0 Flip-Flops** (2-pass streaming) | **-100%** bộ nhớ đệm |
| **Số lượng Multiplier** | **132 × FP32 (`24×24`)** | **16 × FP20 (`12×12`)** trong mảng chia sẻ + **3** trong `sfu_rsqrt_fp20` = **19** | **-85.6%** (mảng ALU); toàn SFU 132→19 |
| **Số lượng Adder/Sub** | **98 × FP32** | **16 × FP20** trong mảng + **1** trong rsqrt + **48** trong PWL sigmoid = **65** | **-33.7%** (toàn SFU); mảng ALU 16 |
| **Khối `Sigmoid` (SwiGLU)** | 16 × 3 FP32 Add = **48 Add** | **16 × `sfu_sigmoid_pwl_fp20`** (dịch exponent) = **48 Add, 0 Mul** | **0 Multiplier** cho Sigmoid |
| **Khối `1/sqrt` (RMSNorm)** | `sfu_rsqrt_fp32` | `sfu_rsqrt_fp20` có **3 Mul + 1 Add riêng** | Không tái sử dụng `mul[0]/add[0]`; vẫn gọn hơn FP32 |

---

## 2. Chi tiết 4 Kỹ thuật Tối ưu đã Triển khai

1. **Định dạng nội bộ Custom FP20 ([snax_sfu_fp20_units.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_sfu_fp20_units.sv)):**
   * Vì giao tiếp vào/ra của SFU là **FP16 (10-bit mantissa)**, việc dùng FP32 (23-bit mantissa) gây lãng phí lớn ở bộ nhân và LZC.
   * **FP20** giữ nguyên **8-bit exponent (bias 127)** của FP32 để chống tràn khi bình phương $x^2$ trong RMSNorm, nhưng thu gọn mantissa xuống **11-bit** (`10b FP16 + 1b guard`).
2. **Zero-Buffer 2-Pass Streaming RMSNorm:**
   * Xóa bỏ hoàn toàn mảng `rms_buf` 64 Kbit Flip-Flop.
   * **Pass 1 (`ST_RMS_PASS1`):** Stream `x` từ TCDM qua `in_a` để tính $\sum x^2$ và `inv_rms`.
   * **Pass 2 (`ST_RMS_PASS2`):** Stream lại `x` (`in_a`) và `gamma` (`in_b`) từ TCDM để tính `(x * inv_rms) * gamma`.
3. **Chia sẻ chung đúng `16 × Mul + 16 × Add` cho mọi chế độ ([snax_llm_sfu_opt.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_sfu_opt.sv)):**
   * Tại một thời điểm SFU chỉ chạy 1 phép toán, nên toàn bộ `Residual`, `RMSNorm`, `RoPE`, và `SwiGLU` dùng chung 1 mảng ALU 16-lane qua Mux:
     * `RMS_PASS1`: dùng `16 Mul` ($x^2$) + `15 Add` làm cây cộng `16→1` + `1 Add` tích lũy tổng.
     * `RMS_RSQRT`: mượn `mul[0]` và `add[0]` để tính `sum * (1/D) + eps`.
     * `ROPE` (2 chu kỳ/vector): Chu kỳ 0 tính 8 phần tử chẵn (`xe*cos - xo*sin`), Chu kỳ 1 tính 8 phần tử lẻ (`xe*sin + xo*cos`).
     * `SWIGLU` (2 chu kỳ/vector): Chu kỳ 0 tính `silu = a * sig(a)`, Chu kỳ 1 tính `silu * b`.
4. **Multiplier-less PWL Sigmoid (`sfu_sigmoid_pwl_fp20`):**
   * Dùng 4 đoạn tuyến tính đối xứng với hệ số góc lũy thừa 2 (`0.25`, `0.125`, `0.03125`) — thực hiện bằng phép trừ exponent đơn giản, **không tốn bất kỳ bộ nhân nào**.

---

## 3. Giao thức & Thông lượng theo Chế độ (`cfg_op_i`)

| `cfg_op_i` | Chế độ | Thông lượng | Lưu ý cho Agent tích hợp |
| :---: | :--- | :---: | :--- |
| `3'd0` | `SFU_BYPASS` | `1 cyc/vec` | Chỉ dùng `in_a`. |
| `3'd1` | `SFU_RESIDUAL` | `1 cyc/vec` | `in_a` và `in_b` bắt tay đồng thời. |
| `3'd2` | `SFU_RMSNORM` | Pass 1: `1 cyc/vec`<br>Pass 2: `2 cyc/vec` | Kích `start_i` 1 chu kỳ; Pass 1 chỉ cấp `in_a`; chờ 1 chu kỳ `ST_RMS_RSQRT`; Pass 2 cấp `in_a` ở pha 0 và giữ `in_b` (`gamma`) đến khi `in_b_ready_o=1` (pha 1). |
| `3'd3` | `SFU_ROPE` | `2 cyc/vec` | Cấp `in_a` và `in_b` (`cos/sin`) ở pha 0 (`in_a_ready_o && in_b_ready_o`); kết quả 16-lane hợp lệ sau 2 chu kỳ. |
| `3'd4` | `SFU_SWIGLU` | `2 cyc/vec` | Cấp `in_a` (`gate`) và `in_b` (`up`) ở pha 0; kết quả hợp lệ sau 2 chu kỳ. |

---

## 4. Kết quả Kiểm chứng QuestaSim 2024.1 (`ALU_LANES = 16, 8, 4`)

* **Filelist & Testbench:** `sim/filelist_sfu_opt.f`, `01_snax_llm_fp4_subsystem/tb_snax_llm_sfu_opt.sv`
* **Lệnh chạy (thay `ALU_LANES` bằng `16`, `8`, hoặc `4`):**
  ```bat
  c:\Users\nguyen\Desktop\NPU_LOG\sim\run_questa.bat -Filelist filelist_sfu_opt.f -Top tb_snax_llm_sfu_opt -Define ALU_LANES=8
  ```
* **Kết quả kiểm chứng cả 3 cấu hình (`Errors: 0, Warnings: 0`):**
  * `ALU_LANES = 16`: **`ALL SFU_OPT (ALU_LANES=16) TESTS PASSED (0 errors)`** (`226 ns`)
  * `ALU_LANES = 8`: **`ALL SFU_OPT (ALU_LANES=8) TESTS PASSED (0 errors)`** (`396 ns`)
  * `ALU_LANES = 4`: **`ALL SFU_OPT (ALU_LANES=4) TESTS PASSED (0 errors)`** (`736 ns`)

---

## 5. Kiến trúc Kết nối Hệ thống (`snax_llm_top_clean.sv`)

Thay vì nối trực tiếp SFU lên Crossbar TCDM (gây phình crossbar và trùng lặp logic với streamer), `snax_llm_sfu_opt` được nối kiểu **In-Line Streaming** giữa các module 256-bit sẵn có:

```text
                       ┌──────────────────────────┐
                       │  u_csr / u_seq (Control) │
                       └────────────┬─────────────┘
                                    │ start_i, cfg_op_i[2:0], cfg_num_vecs_i,
                                    │ cfg_inv_n_fp32_i, cfg_eps_fp32_i
                                    ▼
u_core (core_fp16_packed) ──► [Mux A] ──► in_a (256b) ──┐
                                │                       ▼
u_read_fp16 (rd_fp16_data) ─────┴──► [Mux B] ──► in_b ──► [ snax_llm_sfu_opt ]
u_read_fp16_b (hoặc TCDM) ──────────►                   └─────────┬──────────┘
                                                                  │ out_fp16_o (256b)
                                    ┌─────────────────────────────┼─────────────────────────────┐
                                    ▼                             ▼                             ▼
                          u_logq_shared (Mux)           u_write_out (FP16)           u_transpose_buffer
                        (Nén 72b FP4 -> TCDM)          (Ghi 256b FP16 -> TCDM)        (Bypass cho V_proj)
```

### 5.1. Bảng Chi tiết Kết nối

| Nhóm cổng của `u_sfu` | Kết nối với module trong `snax_llm_top_clean.sv` | Mục đích |
| :--- | :--- | :--- |
| **Control & Status** | `u_csr` (`snax_flash_csr_snax`) & `u_seq` (`snax_llm_sequencer`) | Nhận `start_i`, `cfg_op_i`, `cfg_num_vecs_i`, `cfg_inv_n_fp32_i`, `cfg_eps_fp32_i`; trả `busy_o`, `done_o`. |
| **`in_a` (256b)** | **Mux 2:1** chọn giữa `u_core` (`core_fp16_packed`) và `u_read_fp16` (`rd_fp16_data`) | Lấy trực tiếp từ Core (`RoPE`, `SwiGLU`, `Residual`) hoặc từ TCDM (`RMSNorm` Pass 1 & 2). |
| **`in_b` (256b)** | `u_read_fp16` (khi `in_a` từ Core) hoặc `u_read_fp16_b` (khi `RMSNorm Pass 2`) | Cấp $\sin/\cos$ (`RoPE`), `up_proj` (`SwiGLU`), `residual_y` (`Residual`), hoặc $\gamma$ (`RMSNorm`). |
| **`out` (256b)** | `u_logq_shared` (Mux 0/2) và `u_write_out` (`core_out_fp16_data_i`) | Nén thẳng xuống 72b FP4 (`RMSNorm`, `RoPE`, `SwiGLU`) hoặc ghi 256b FP16 xuống TCDM. |

### 5.2. Vai trò của Nhánh Write FP16 (`u_write_out`)
Vẫn **giữ nguyên nhánh ghi 256-bit FP16** xuống TCDM trên `u_write_out` (Master 19) để phục vụ 3 trường hợp bắt buộc giữ độ chính xác FP16:
1. **Nhánh Residual (`H_mid`, `H_out`):** Kết quả sau `Residual Add` cần lưu dạng FP16 làm đầu vào cho `RMSNorm` 2-pass và phép cộng Residual tiếp theo.
2. **Lưu tạm `Up_proj` cho `SwiGLU`:** Core tính `Up_proj` ghi tạm FP16 xuống TCDM, sau đó tính `Gate_proj` đẩy thẳng vào `in_a` của SFU đồng thời `u_read_fp16` đọc `Up_proj` vào `in_b`.
3. **Fallback cho RV32I / LM Head:** Ghi logits hoặc các tác vụ đặc biệt ngoài 4 phép của SFU.

---

## 6. Hướng dẫn Tích hợp Chi tiết Trường hợp `ALU_LANES = 8` vào `snax_llm_top_clean.sv`

Khi tích hợp bản **`ALU_LANES = 8`** vào [snax_llm_top_clean.sv](file:///c:/Users/nguyen/Desktop/NPU_LOG/01_snax_llm_fp4_subsystem/snax_llm_top_clean.sv), bạn không cần sửa `u_core`, `u_transpose_buffer`, hay `u_log_quantizer`. Chỉ cần thực hiện **4 bước nối dây** sau:

### 6.1. Khai báo & Instantiate `u_sfu` (`ALU_LANES = 8`)
```systemverilog
logic        sfu_en;          // 1 khi csr_sfu_op != 3'd0 (khác BYPASS)
logic        sfu_in_a_sel_rd; // 1: lấy in_a từ rd_fp16_data (RMSNorm); 0: từ core_fp16_packed
logic [255:0] sfu_in_a_data, sfu_in_b_data, sfu_out_data;
logic         sfu_in_a_valid, sfu_in_a_ready;
logic         sfu_in_b_valid, sfu_in_b_ready;
logic         sfu_out_valid,  sfu_out_ready;
logic         sfu_busy,       sfu_done;

assign sfu_en          = (csr_sfu_op != 3'd0);
assign sfu_in_a_sel_rd = (csr_sfu_op == 3'd2); // SFU_RMSNORM đọc x từ TCDM

assign sfu_in_a_data   = sfu_in_a_sel_rd ? rd_fp16_data  : core_fp16_packed;
assign sfu_in_a_valid  = sfu_in_a_sel_rd ? rd_fp16_valid : (core_fp16_valid && !v_active);

// Khi in_a lấy từ Core, tận dụng luôn u_read_fp16 (Master 20) cấp in_b!
// (Nếu cần RMSNorm Pass 2 đọc cả x và gamma từ TCDM cùng lúc, dùng thêm u_read_fp16_b cho in_b)
assign sfu_in_b_data   = rd_fp16_data;
assign sfu_in_b_valid  = rd_fp16_valid;

snax_llm_sfu_opt #(
    .NUM_LANES (16),
    .ALU_LANES (8)
) u_sfu (
    .clk              (clk),
    .rst_n            (rst_n),
    .start_i          (csr_sfu_start),
    .cfg_op_i         (csr_sfu_op),
    .cfg_num_vecs_i   (csr_sfu_num_vecs),
    .cfg_inv_n_fp32_i (csr_sfu_inv_n_fp32),
    .cfg_eps_fp32_i   (csr_sfu_eps_fp32),
    .busy_o           (sfu_busy),
    .done_o           (sfu_done),
    .in_a_fp16_i      (sfu_in_a_data),
    .in_a_valid_i     (sfu_in_a_valid),
    .in_a_ready_o     (sfu_in_a_ready),
    .in_b_fp16_i      (sfu_in_b_data),
    .in_b_valid_i     (sfu_in_b_valid),
    .in_b_ready_o     (sfu_in_b_ready),
    .out_fp16_o       (sfu_out_data),
    .out_valid_o      (sfu_out_valid),
    .out_ready_i      (sfu_out_ready)
);
```

### 6.2. Sửa Mux đầu vào `u_log_quantizer` & `u_write_out`
Thay nguồn `core_fp16_packed` / `core_fp16_valid` bằng bộ mux chọn `sfu_en`:
* **Vào `u_log_quantizer` (dòng 712–715 của `snax_llm_top_clean.sv`):**
  ```systemverilog
  end else if (src_core_sel) begin
      quant_fp16_in  = sfu_en ? sfu_out_data : core_fp16_packed;
      quant_in_valid = (sfu_en ? sfu_out_valid : core_fp16_valid) && quant_accept && !wo_mode_fp16;
  end
  ```
* **Vào `u_write_out` nhánh FP16 (dòng 845–846 của `snax_llm_top_clean.sv`):**
  ```systemverilog
  .core_out_fp16_data_i  (sfu_en ? sfu_out_data  : core_fp16_packed),
  .core_out_fp16_valid_i ((sfu_en ? sfu_out_valid : core_fp16_valid) && wo_mode_fp16 && !v_active),
  ```

### 6.3. Cập nhật tín hiệu Handshake (`ready`)
```systemverilog
assign sfu_out_ready = wo_mode_fp16 ? wo_fp16_ready : (src_core_sel ? quant_accept : 1'b0);

// Cập nhật dòng 741 & 746:
assign rd_fp16_ready = sfu_en
    ? (sfu_in_a_sel_rd ? sfu_in_a_ready : sfu_in_b_ready)
    : (src_rd_sel && quant_accept);

assign core_fp16_ready = v_active ? v_row_ready
    : (sfu_en && !sfu_in_a_sel_rd) ? sfu_in_a_ready
    : wo_mode_fp16 ? wo_fp16_ready
    : src_core_sel ? quant_accept : 1'b0;
```

### 6.4. Mở rộng CSR (`snax_flash_csr_snax.sv`) & Đồng bộ nhịp 8-Lane
1. **Thêm 4 thanh ghi CSR** trong `snax_flash_csr_snax.sv`:
   * `CSR_SFU_CTRL`: `{sfu_start (pulse), sfu_op[2:0], sfu_num_vecs[15:0]}`
   * `CSR_SFU_INV_N`: `sfu_inv_n_fp32[31:0]` ($1/D$ dạng IEEE FP32)
   * `CSR_SFU_EPS`: `sfu_eps_fp32[31:0]` (mặc định `32'h3727C5AC` = $10^{-5}$)
   * `CSR_SFU_STATUS`: đọc `{sfu_done, sfu_busy}` (gộp `sfu_busy` vào `subsystem_busy_o`).
2. **Nhịp chạy thực tế khi dùng `ALU_LANES = 8`:**
   * Với `ROPE` / `SWIGLU` / `RMS_PASS2`: SFU mất **4 chu kỳ/vector** (`2 chunks × 2 phases`) — **khớp chính xác 1-1** với tốc độ đọc/ghi 256-bit FP16 của `u_read_fp16` & `u_write_out` qua port TCDM 64-bit (`4 chu kỳ/vector`), nên hoàn toàn không gây nghẽn thêm cho toàn hệ thống.
   * Với `RESIDUAL` / `RMS_PASS1`: SFU chỉ mất **2 chu kỳ/vector**, luôn nhanh hơn tốc độ cấp dữ liệu từ TCDM.

---

## 7. As-Built Integration (đã triển khai trong RTL)

Các snippet ở mục 5–6 là mô tả khái niệm; phần dưới là cấu hình **thực tế đã compile và chạy** trong `snax_llm_top_clean.sv`. Khác biệt quan trọng: cả hai toán hạng SFU đều đọc từ TCDM qua **hai read streamer**, không dùng `u_core` làm `in_a`.

### 7.1. Master / Streamer
* `NUM_MASTERS` tăng **21 → 22**.
* Master 20: `u_read_fp16` (A) — tái sử dụng streamer đọc cũ.
* Master 21: `u_read_fp16_b` (B) — streamer mới cho gamma / cos-sin / up / x.
* `u_sfu`: `snax_llm_sfu_opt #(.NUM_LANES(16), .ALU_LANES(8))`.
* SFU input A/B nối trực tiếp `rd_fp16_data` / `rd_b_fp16_data`; không có mux từ core.

### 7.2. CSR map as-built
`snax_flash_csr_snax.sv` dùng `NUM_RW=54`, `NUM_RO=10`; `snax_csr_manager` coi **RW[53] là START**.

| Địa chỉ | Tên | Ý nghĩa |
| :---: | :--- | :--- |
| 46 | `RW_SFU_CTRL` | `[0]` en, `[1]` out_dst (0=logq FP4, 1=write-out FP16), `[4:2]` op, `[20:5]` num_vecs |
| 47 | `RW_SFU_INV_N` | FP32 `1/D` |
| 48 | `RW_SFU_EPS` | FP32 epsilon |
| 49 | `RW_SFU_B_BASE` | địa chỉ TCDM cho operand B |
| 50 | `RW_SFU_B_NV` | số vector B |
| 51 | `RW_SFU_B_LS` | line stride B |
| 53 | `RW_START` | bit0 commit, bit4 wo, bit6 read A, bit7 SFU, bit8 read B |
| 54 | `RO[0]` | seq busy/done (RO base mới) |
| 61 | `RO[7]` | ID `0x534E4146` |

Các TB cũ đã được cập nhật `A_START 47→53`, `A_RO_ID 55→61`, `A_RO_SEQ 48→54`, `A_RO_WO 51→57`.

### 7.3. Ba lỗi handshake đã sửa khi tích hợp
1. **Không thể lấy cả A và B từ `rd_fp16_data`**: RMSNorm pass 2 cần `x` và `gamma` đồng thời → thêm `u_read_fp16_b`/Master 21.
2. **`rd_start` vô tình start `u_write_out`**: `wo_job_start` cũ là `wo_start | v_start | rd_start`; RMS pass 1 dùng `rd_start` nên write-out bật và giữ `wo_busy`, chặn CSR pass 2. Đã sửa:
   ```systemverilog
   wire wo_job_start = wo_start | v_start | (rd_start && !sfu_en);
   ```
3. **Pass 2 cần ghi CSR trong lúc SFU busy**: bỏ `sfu_busy` khỏi `csr_cfg_ready` (chỉ cho phép start read A/B), và read A phải được restart ở pass 2. Đây là giải pháp cho TB/CPU điều khiển; nếu cần autonomous, nên thêm một SFU controller phần cứng quản lý pass 2.

### 7.4. Mux / ready as-built
* `src_rd_sel` bị ép 0 khi `sfu_en` → logq không tranh read A với SFU.
* `rd_fp16_ready = sfu_en ? sfu_in_a_ready : (src_rd_sel && quant_accept)`.
* `rd_b_fp16_ready = sfu_in_b_ready`.
* `core_fp16_ready = 0` khi `sfu_en`.
* `sfu_out_ready` = `wo_fp16_ready` nếu `out_dst=1`, ngược lại `src_core_sel && quant_accept`.
* logq lấy `sfu_out_data` khi `sfu_en && out_dst==0`; write-out lấy `sfu_out_data` khi `sfu_en && out_dst==1`.
* `subsystem_busy_o` và `subsystem_done_irq_o` đã gộp `sfu_busy`/`sfu_done`/`rd_b_busy`.

### 7.5. Filelist
Đã thêm vào các filelist dùng top:
```text
../01_snax_llm_fp4_subsystem/snax_sfu_fp20_units.sv
../01_snax_llm_fp4_subsystem/snax_llm_sfu_opt.sv
```

### 7.6. Kết quả simulation
* `tb_snax_sfu_system` (TCDM/CSR/read A+B/write-out):
  * `RESIDUAL`, `ROPE`, `SWIGLU`, `RMSNORM` đều **PASS**.
* `tb_snax_llm_layer_mid` với toàn bộ 6 stage CPU thay bằng SFU:
  * Simulation chạy hết layer (39,141 job cycles; 187,148 tổng cycles).
  * `SFU RMS1`: max_abs `0.01367`, fail 0/4096.
  * `SFU RoPE Q/K`: sai khác chủ yếu **1 LSB FP16** (max `0.545`/`2.0` tại giá trị lớn), do FP20→FP16 rounding.
  * `SFU residual 1` / final: sai số bị khuếch đại qua attention/GEMM; final max_abs `180.7` (14/4096 exact). **Không phải bit-exact**; đây là giới hạn độ chính xác của SFU xấp xỉ, không phải lỗi treo/handshake.
  * Muốn bit-exact layer: dùng FP32/FP16 exact cho RoPE/SwiGLU hoặc reload golden sau từng stage (không còn là đường SFU thuần).

### 7.7. Lệnh chạy
```bat
run_questa.bat -Filelist filelist_sfu_system.f -Top tb_snax_sfu_system -Wave none
run_questa.bat -Filelist filelist_layer_mid_tb.f -Top tb_snax_llm_layer_mid -Wave none
```


