# End-to-End LLM-Layer Verification Guide (`snax_llm_top_clean`)

Tài liệu này mô tả kiến trúc testbench, luồng dữ liệu 19 bước của một Transformer/LLaMA prefill layer đầy đủ, kiểm thử đường nối trực tiếp `Core -> SFU` (`sfu_in_a_src = 1`), lệnh chạy QuestaSim và kết quả mô phỏng đã xác minh.

---

## 1. Tổng quan Kiến trúc Testbench

Hệ thống kiểm thử end-to-end cho `snax_llm_top_clean` gồm 2 testbench bổ trợ lẫn nhau:

1. **[tb_snax_llm_layer_mid.sv](file:///d:/LVThs/NPU_PRJ/NPU_LOG/tb/tb_snax_llm_layer_mid.sv)** (End-to-End Full Prefill Layer TB):
   - Chạy trọn vẹn 1 LLaMA layer kích thước trung bình: `M_TOKENS=32`, `D_MODEL=128`, `D_FF=256`, `N_HEADS=4` (`d_k=32`, `d_v=32`, `Q_TILES=2`, `KV_TILES=2`, `DV_TILES=2`).
   - Kết hợp đầy đủ các khối phần cứng trong `snax_llm_top_clean`:
     - **DMA / CPU host** nạp trọng số & activation vào 32-bank TCDM qua `interleaved_interconnect_72b`.
     - **`u_sfu` (`snax_llm_sfu_opt`)**: thực thi RMSNorm, RoPE, Residual Add và SwiGLU (`SiLU(gate) * up`).
     - **`u_log_quantizer` (`log_quantizer`)**: lượng tử hóa online FP16 -> gói 72-bit Log8+FP4.
     - **`u_core` (`snax_fused_llm_core`)**: thực thi Dense GEMM (`Q, K, V, O, Gate, Up, Down`), Fused `V + Transpose` (qua `u_transpose_buffer`), và Causal FlashAttention (`QK^T -> Softmax/Online-Rescale -> PV`).

2. **[tb_snax_sfu_system.sv](file:///d:/LVThs/NPU_PRJ/NPU_LOG/tb/tb_snax_sfu_system.sv)** (System-Level SFU & Fused `Core -> SFU` TB):
   - Xác minh các phép toán SFU qua đường TCDM (`RESIDUAL`, `ROPE`, `SWIGLU`, `RMSNORM` với `sfu_in_a_src = 0`).
   - Xác minh đường nối trực tiếp **`u_core -> 4-entry fused_fifo -> u_sfu -> u_write_out`** (`sfu_in_a_src = 1`, bit 21 của `RW_SFU_CTRL`): chạy `16x16` GEMM (`A=1.0, W=1.0 -> 16.0`) cộng trực tiếp với vector Residual `B` đọc từ `u_read_fp16_b` rồi ghi kết quả FP16 ra TCDM.

---

## 2. Luồng Dữ liệu 19 Bước trong `tb_snax_llm_layer_mid.sv`

| Bước | Giai đoạn | Khối thực thi | Mô tả ngắn |
| :--- | :--- | :--- | :--- |
| `[1]` | DMA Preload | `dma_write` -> TCDM | Nạp sẵn toàn bộ trọng số FP4 (`W_Q, W_K, W_V, W_O, W_GU, W_D`), input `X`, `gamma1/2` từ `.hex` vào TCDM; kiểm tra `A_RO_ID == 0x534E4146` (`SNAF`). |
| `[2]` | RMSNorm 1 | `u_sfu` (`RMSNORM`) | Chuẩn hóa `X` theo `gamma1` -> `XN_BANK`. |
| `[3]` | Online Quant `Xn` | `u_log_quantizer` | Nén FP16 `XN_BANK` -> gói 72-bit FP4+Log8 tại `ACT_BANK`. |
| `[4]` | GEMM `Q` | `u_core` (`GEMM`) | Tính `Q = Xn * W_Q` (`M=2, N=8, K=8` tiles). |
| `[5]` | RoPE `Q` | `u_sfu` (`ROPE`) | Áp dụng Rotary Position Embedding lên `Q`. |
| `[6]` | Quant `Q` | `u_log_quantizer` | Nén `Q` sau RoPE -> gói FP4 tại `QPKT_BANK`. |
| `[7]` | GEMM `K` | `u_core` (`GEMM`) | Tính `K = Xn * W_K`. |
| `[8]` | RoPE `K` | `u_sfu` (`ROPE`) | Áp dụng RoPE lên `K`. |
| `[9]` | Quant `K` | `u_log_quantizer` | Nén `K` sau RoPE -> gói FP4 tại `KPKT_BANK`. |
| `[10]` | Fused `V` + Transpose | `u_core` + `u_transpose_buffer` | Tính `V` và chuyển vị trực tiếp sang layout `V` cho FlashAttention (`out_mode=3`). |
| `[11]` | Reorder `Q/K` | CPU helper | Sắp xếp gói `Q/K` sang layout tile của FlashAttention. |
| `[12]` | Causal FlashAttention | `u_core` (`ATTN`) | Thực thi FlashAttention `H=4, KV=2, DV=2` có causal mask + online softmax/rescale. |
| `[13]` | GEMM `O` | `u_core` (`GEMM`) | Chiếu đầu ra Attention `O_proj = Attn * W_O`. |
| `[14]` | Residual 1 + RMSNorm 2 | `u_sfu` | `Res1 = O_proj + X`, sau đó `Norm2 = RMSNorm(Res1, gamma2)`. |
| `[15]` | Quant `Norm2` | `u_log_quantizer` | Nén `Norm2` -> gói FP4 cho FFN. |
| `[16]` | GEMM Gate/Up + SwiGLU | `u_core` + `u_sfu` (`SWIGLU`) | Tính 2 chunk `Gate` và `Up`, sau đó `u_sfu` tính `SiLU(Gate) * Up`. |
| `[17]` | Quant FFN | `u_log_quantizer` | Nén 2×256 vector đầu ra SwiGLU -> gói FP4. |
| `[18]` | GEMM Down | `u_core` (`GEMM`) | Tính `Down = FFN_act * W_D` (`M=2, N=8, K=16`). |
| `[19]` | Residual 2 & Final Check | `u_sfu` (`RESIDUAL`) | `Final = Down + Res1`, đối chiếu kết quả cuối với golden `layer_final_fp16.hex`. |

---

## 3. Hướng dẫn Chạy Simulation (QuestaSim)

Thực hiện từ thư mục `d:\LVThs\NPU_PRJ\NPU_LOG\sim`:

### A. Sinh dữ liệu `.hex` và chạy `tb_snax_llm_layer_mid`
```powershell
cd d:\LVThs\NPU_PRJ\NPU_LOG\sim
python ..\tb\generate_snax_llm_layer_mid_data.py
vlib work_mid
vlog -work work_mid -sv -f filelist_layer_mid_tb.f
vsim -c work_mid.tb_snax_llm_layer_mid -do "run -all; quit"
```

### B. Chạy `tb_snax_sfu_system` (bao gồm đường Fused `Core -> SFU`)
```powershell
cd d:\LVThs\NPU_PRJ\NPU_LOG\sim
vlib work_sfu
vlog -work work_sfu -sv -f filelist_sfu_system.f
vsim -c work_sfu.tb_snax_sfu_system -do "run -all; quit"
```

> Lưu ý: Xóa các thư mục tạm `work_mid` / `work_sfu` sau khi chạy xong nếu không cần giữ build cache.

---

## 4. Kết quả Đã Xác minh (QuestaSim 2024.1)

- **`tb_snax_llm_layer_mid`**:
  - Compile & Simulation: `Errors: 0`
  - Structural & stage checks: `Checks: 80 passed, 0 failed`
  - Latency: `Total job cycles: 39141`
  - Cycle mix: `total=197388 seq_busy=39048 core_compute=22016 core_fp16=2560 cpu=48481 dma=11584`
  - Kết thúc với thông điệp: `>>> SFU MID LAYER COMPLETED: end-to-end SFU path exercised; final values are approximate <<<`

- **`tb_snax_sfu_system`**:
  - Compile & Simulation: `Errors: 0, Warnings: 17`
  - Tất cả 5 test case đều đạt `[PASS]`:
    - `[PASS] System RESIDUAL`
    - `[PASS] System ROPE`
    - `[PASS] System SWIGLU`
    - `[PASS] System RMSNORM`
    - `[PASS] System FUSED Core->SFU RESIDUAL (16x16 GEMM + Residual, sfu_in_a_src=1)`
  - Tổng kết: `=== ALL SFU SYSTEM TESTS PASSED (0 errors) ===` (`Time: 8295 ns`)
