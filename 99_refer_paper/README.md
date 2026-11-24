# Reference Papers (`99_refer_paper`)

Thư mục này tập hợp các bản PDF open-access chính thức từ `arXiv` của các bài báo liên quan trực tiếp đến đề tài **Edge LLM Accelerator / Microscaling 4-bit / Exp-Free Softmax & FlashAttention / SNAX Co-Design** để đối chiếu đánh giá cho Section V (Evaluation) trên IEEE TCAD / IEEE TVLSI.

> **Nguyên tắc truy cập:** Toàn bộ 16 PDF bên dưới được tải hợp lệ từ bản preprint open-access chính thức trên `arxiv.org` (không vượt paywall của IEEE Xplore hay các nhà xuất bản thu phí). Các bài **IEEE TVLSI** và **IEEE** chỉ có trên IEEE Xplore được ghi nhận riêng ở Mục 2 để bạn tải bằng tài khoản trường/tổ chức khi cần đọc toàn văn.

---

## 1. Đã tải về trong `99_refer_paper/` (16 Open-Access PDFs, đã kiểm tra `%PDF-`)

### Nhóm A — LLM Quantization & Co-Design Accelerators (ISCA / MICRO / HPCA / FPGA)
| Short Name | Venue | arXiv ID | Title | File Size | Local File |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **BitMoD** | HPCA'25 | `2411.11745` | BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration | 1,137.5 KB | [BitMoD_2411.11745.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/BitMoD_2411.11745.pdf) |
| **MicroScopiQ** | ISCA'25 | `2411.05282` | MicroScopiQ: Accelerating Foundational Models through Outlier-Aware Microscaling Quantization | 9,056.5 KB | [MicroScopiQ_2411.05282.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/MicroScopiQ_2411.05282.pdf) |
| **FuseMax** | MICRO'24 | `2406.10491` | FuseMax: Leveraging Extended Einsums to Optimize Attention Accelerator Design | 771.2 KB | [FuseMax_2406.10491.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/FuseMax_2406.10491.pdf) |
| **Tender** | ISCA'24 | `2406.12930` | Tender: Accelerating Large Language Models via Tensor Decomposition and Runtime Requantization | 3,258.9 KB | [Tender_2406.12930.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/Tender_2406.12930.pdf) |
| **Cambricon-LLM** | ISCA'24 | `2409.15654` | Cambricon-LLM: A Chiplet-Based Hybrid Architecture for On-Device Inference of 70B LLM | 1,836.9 KB | [Cambricon-LLM_2409.15654.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/Cambricon-LLM_2409.15654.pdf) |
| **FlightLLM** | FPGA'24 | `2401.03868` | FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs | 2,119.8 KB | [FlightLLM_2401.03868.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/FlightLLM_2401.03868.pdf) |
| **M-ANT** | HPCA'25 | `2502.18755` | M-ANT: Efficient Low-bit Group Quantization for LLMs via Mathematically Adaptive Numerical Type | 992.4 KB | [M-ANT_2502.18755.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/M-ANT_2502.18755.pdf) |

### Nhóm B — Hardware Softmax, Non-Linear & FlashAttention VLSI Accelerators (DATE / DAC / ISLPED)
| Short Name | Venue | arXiv ID | Title | File Size | Local File |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **FLASH-D** | ISLPED'25 | `2505.14201` | FLASH-D: FlashAttention with Hidden Softmax Division (28 nm) | 960.7 KB | [FLASH-D_2505.14201.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/FLASH-D_2505.14201.pdf) |
| **SoftEx** | DATE'25 | `2412.06321` | A Flexible Template for Edge Generative AI with High-Accuracy Accelerated Softmax & GELU (12 nm) | 2,942.6 KB | [SoftEx_2412.06321.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/SoftEx_2412.06321.pdf) |
| **ConSmax** | arXiv'24 | `2402.10930` | ConSmax: Hardware-Friendly Alternative Softmax with Learnable Parameters (16 nm) | 1,803.5 KB | [ConSmax_2402.10930.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/ConSmax_2402.10930.pdf) |
| **ITA** | ISLPED'23 | `2307.03493` | ITA: An Energy-Efficient Attention and Softmax Accelerator for Quantized Transformers (22 nm) | 776.3 KB | [ITA_2307.03493.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/ITA_2307.03493.pdf) |
| **Softermax** | DAC'21 | `2103.09301` | Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers (16 nm) | 1,051.3 KB | [Softermax_2103.09301.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/Softermax_2103.09301.pdf) |

### Nhóm C — IEEE TCAS-I Regular Papers (2024–2026)
| Short Name | Venue | arXiv ID | Title | File Size | Local File |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **HiKV** | IEEE TCAS-I'26 | `2607.22389` | HiKV: Hierarchical Importance-Aware KV Cache with Hardware Acceleration for LLM Decoding | 1,670.6 KB | [TCASI_HiKV_2607.22389.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/TCASI_HiKV_2607.22389.pdf) |
| **LowPowerViT** | IEEE TCAS-I'26 *(Vol. 73, No. 1)* | `2510.14393` | Low Power Vision Transformer Accelerator with Hardware-Aware Pruning and Optimized Dataflow | 4,128.2 KB | [TCASI_LowPowerViT_2510.14393.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/TCASI_LowPowerViT_2510.14393.pdf) |
| **DiP** | IEEE TCAS-I'25 | `2412.09709` | DiP: A Scalable, Energy-Efficient Systolic Array for Matrix Multiplication Acceleration | 19,397.8 KB | [TCASI_DiP_2412.09709.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/TCASI_DiP_2412.09709.pdf) |
| **TinyFormer** | IEEE TCAS-I'24 | `2311.01759` | TinyFormer: Efficient Transformer Design and Deployment on Tiny Devices | 3,427.5 KB | [TCASI_TinyFormer_2311.01759.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/99_refer_paper/TCASI_TinyFormer_2311.01759.pdf) |

---

## 2. Các bài IEEE TVLSI / IEEE không có bản Open-Access trên arXiv (Cần IEEE Xplore)

Những công trình sau đây là paywalled trên **IEEE Xplore / IEEE Computer Society** và chưa có preprint công khai trên `arxiv.org` (cần tải trực tiếp qua tài khoản thư viện / IEEE Xplore nếu muốn đọc toàn văn):

- **EdgeQ-GEMM** (*IEEE TVLSI'26*): *"EdgeQ-GEMM: A Single-Engine Bit-Partitioned GEMM Accelerator With Integrated On-the-Fly Quantization for Edge AI Processors"* (C. Jung et al.) — Require IEEE Xplore access.
- **DayPQ** (*IEEE TVLSI'26*, Vol. 34, Issue 6): *"DayPQ: Dynamic Layerwise Pruning and Quantization for LLM Inference Acceleration"* (Y. Xu et al.) — Require IEEE Xplore access.
- **BooLUT-CIM** (*IEEE TVLSI'25/26*): Digital Compute-in-Memory Transformer accelerator — Require IEEE Xplore access.
- **28-nm Sparse Transformer Accelerator (RRLC)** (*IEEE TVLSI'24/25*): Row-Wise Run-Length Compression + non-linear layer co-design — Require IEEE Xplore access.
- **FIGNA** (*HPCA'24*): *"FIGNA: Integer Unit-Based Accelerator Design for FP-INT GEMM Preserving Numerical Accuracy"* — Require IEEE Xplore / IEEE Computer Society access.

---

## 3. Các bài báo tham khảo đã có sẵn ở thư mục gốc Workspace (`NPU_LOG/`)

- **VSQ** (*IEEE JSSC'23*): [A_95.6-TOPS_W_Deep_Learning_Inference_Accelerator_With_Per-Vector_Scaled_4-bit_Quantization_in_5_nm.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/A_95.6-TOPS_W_Deep_Learning_Inference_Accelerator_With_Per-Vector_Scaled_4-bit_Quantization_in_5_nm.pdf)
- **MiX** (*MICRO'26*): [MiX Micro-Inverted-Scaling for End-to-End Low-Bit Vision-Language Model Acceleration.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/MiX%20Micro-Inverted-Scaling%20for%20End-to-End%20Low-Bit%20Vision-Language%20Model%20Acceleration.pdf)
- **Voltra** (*arXiv:2602.11357*): [2602.11357v1.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/2602.11357v1.pdf)
- **DataMaestro** (*arXiv:2504.14091*): [2504.14091.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/2504.14091.pdf)
- **Open-Source Multi-Accelerator SNAX Framework** (*arXiv:2508.14582*): [2508.14582.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/2508.14582.pdf)
- **M-ANT** (*arXiv:2502.18755v1*): [2502.18755v1.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/2502.18755v1.pdf)
- **OpenGeMM / ACM Reference** (`3658617.3697652`): [3658617.3697652.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/3658617.3697652.pdf)
- **Tiny RISC-V FPU** (Bertaccini): [Bertaccini-A_Tiny_RISC-V_FPU.pdf](file:///c:/Users/nguyen/Desktop/NPU_LOG/Bertaccini-A_Tiny_RISC-V_FPU.pdf)
