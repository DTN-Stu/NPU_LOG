# SNAX Cluster & Fused FP4 LLM Acceleration Subsystem

Thư mục này chứa toàn bộ mã nguồn RTL SystemVerilog hoàn chỉnh cho hệ thống tăng tốc suy luận LLM ở định dạng FP4 (Microscaling / MXFP4) tích hợp vào kiến trúc SNAX Cluster.

## Cấu trúc thư mục

### 1. Phân hệ tăng tốc LLM FP4 (`01_snax_llm_fp4_subsystem/`)
- `pe_defines.svh`: Khai báo hằng số vi kiến trúc (Block Size = 16, FP4 E2M1, Packet Width = 72-bit, 16 Lanes, Accumulator Depth = 64/128).
- `snax_llm_top.sv`: **Top-level Subsystem** tích hợp Multi-Bank SRAM, Interconnect, Streamers và Fused LLM Core.
- `snax_llm_shell.sv`: Khung đóng gói chuẩn **SNAX Shell Wrapper** tích hợp CSR Manager và bộ chốt **Activation Freeze & Broadcast** cho Phase Decode.
- `snax_fused_llm_core.sv`: Core tính toán tích hợp nối chu trình khép kín giữa MatMul, Softmax, P_FIFO, Normalizer và Log Quantizer.
- `multi_bank_tcdm_72b.sv`: Khối bộ nhớ nội cụm TCDM gồm **32 Banks Single-Port (SP) SRAM (DW=72-bit, Strb=9-bit)**, tổng dung lượng 288 KB.
- `interleaved_interconnect_72b.sv`: Mạng chuyển mạch ma trận xen kẽ **18 Masters x 32 Banks** với phân xử Round-Robin độc lập cho từng bank và theo dõi phản hồi đọc 1 chu kỳ.
- `vsq_matmul_engine.sv`: Lõi nhân ma trận 16-lane FP4 Dot-Product với bộ cộng log-scale, CSD scaler và mảng chốt Latch Array 1W2R.
- `softmax_requantizer.sv`: Bộ sinh xác suất Softmax không cần tính hàm số mũ (Exp-Free), so sánh 7 ngưỡng để tái lượng tử hóa trực tiếp sang FP4.
- `p_fifo.sv`: Bộ đệm vòng Transient Ring Buffer 72-bit giữ ma trận xác suất P hoàn toàn trên chip.
- `l_accumulator.sv`: Bộ tích lũy mẫu số $l = \sum P$ và mạch sinh số nghịch đảo $1/l$.
- `final_normalizer.sv`: Nhân vector $O_{\text{unnorm}}$ với $1/l$ và chuẩn hóa ra vector 256-bit FP16.
- `slq_encoder.sv`: Bộ nén Log Quantizer chuyển đổi 16 x FP16 thành gói 72-bit FP4.
- `vsq_rescaler_32b.sv`: Mạch dịch và co giãn CSD 32-bit phục vụ Online FlashAttention Rescaling.
- `fp4_mult_lut.sv`: Bảng tra cứu nhân 2 số FP4 E2M1.
- `latch_array.sv`: Mảng chốt 1W2R D-Latch 2 pha tốc độ cao.

## Hướng dẫn tổng hợp & mô phỏng
Biên dịch tất cả các file SystemVerilog cùng cờ `+incdir+.` trên Verilator, Synopsys VCS hoặc Siemens QuestaSim.
Top module là `snax_llm_top`.
