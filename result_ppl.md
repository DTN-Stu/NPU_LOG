30. Kết Quả Đo Kiểm Thực Nghiệm Thực Tế Trên Mô Hình LLaMA-3.2-1B (Real-World GPU Experimental Validation)
30.1. Môi Trường Đo Kiểm & Thiết Lập Thực Nghiệm
Mô hình mục tiêu: NousResearch/Llama-3.2-1B (1.23 tỷ tham số).
Bộ dữ liệu đánh giá: Wikitext-2 (Test Split) với độ dài ngữ cảnh mở rộng 40.096 tokens (Sequence Length = 512, Batch Size = 1).
Cấu hình lượng tử hóa: Chế độ Full W4A4 + Attention Quantization (toàn bộ trọng số Linear, activations đầu vào, cùng các ma trận Query, Key, Value, Probability trong khối Self-Attention đều được lượng tử hóa xuống FP4 E2M1 với kích thước khối $B = 16$).
Cấu hình nén DLCSE: Super-Block $SB = 8$, độ rộng Delta $\Delta = 4\text{ bits}$, hệ số bước nhảy ngoại lai $M = 3$ (khớp hoàn toàn với mạch phần cứng RTL dlcse_decoder).
30.2. Bảng Kết Quả Thực Nghiệm Đo Được Từ GPU
30.3. Ba Giá Trị Khoa Học Vàng Dành Cho Paper IEEE TVLSI

Cấu Hình Đánh Giá (Configuration)	Kiểu Lượng Tử Hóa	Định Dạng Scale	Perplexity (PPL)	Δ PPL vs FP16	Metadata Scale Overhead	Đánh Giá & Ý Nghĩa Kỹ Thuật
Float Baseline	FP16 Unquantized	N/A	13.3025	+0.0000	0.0% (0 b/elem)	Chuẩn đối chứng FP16 gốc của LLaMA-3.2-1B.
OCP MXFP4 (e8m0_mxfp4)	Block FP4 (E2M1)	E8M0 ($2^E$)	37.4757	+24.1732	12.5% (0.50 b/elem)	Chuẩn công nghiệp OCP v1.0. Bước nhảy thô 2x gây clipping/underflow nặng, làm suy giảm PPL trầm trọng.
NVIDIA NVFP4 (e4m3_nvfp4)	Block FP4 (E2M1)	E4M3 (FP8)	20.1864	+6.8839	12.5% (0.50 b/elem)	Chuẩn NVIDIA Blackwell. Độ chính xác cao nhờ 3 mantissa bits nhưng yêu cầu bộ nhân FP8 phức tạp.
Proposed Log8 Baseline	Log-Scale FP4 (E2M1)	Log8 ($2^{e/8}$)	20.1631	+6.8606	12.5% (0.50 b/elem)	Đề xuất Log-Scale chưa nén. Đạt PPL tốt hơn cả NVFP4 và vượt trội hoàn toàn so với OCP MXFP4 (giảm 46.2% PPL).
Proposed DLCSE (dlcse_sb8_d4)	Log-Scale FP4 (E2M1)	DLCSE Compressed	20.3222	+7.0197	7.81% (0.3125 b/elem)	Giảm 37.5% băng thông & dung lượng metadata scale, trong khi PPL chỉ tăng nhẹ +0.1591 so với Log8 chưa nén (&lt; 0.78% relative). Vượt trội OCP MXFP4 (giảm 45.8% PPL) và tiết kiệm 37.5% bộ nhớ so với NVFP4.