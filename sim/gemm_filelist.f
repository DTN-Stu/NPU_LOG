# ---------------------------------------------------------------------------
# TB for snax_fused_llm_core mode 00 (W*A)
# Paths are relative to the project root (the folder holding this file's parent)
# ---------------------------------------------------------------------------

+incdir+../01_snax_llm_fp4_subsystem
+incdir+data

// ------- RTL: fused core subsystem -------
../01_snax_llm_fp4_subsystem/fp4_mult_lut.sv
../rtl/compressor_4to2.sv
../rtl/dot_product_16.sv
../01_snax_llm_fp4_subsystem/latch_array.sv
../01_snax_llm_fp4_subsystem/vsq_rescaler_32b.sv
../01_snax_llm_fp4_subsystem/vsq_matmul_engine.sv
../01_snax_llm_fp4_subsystem/p_fifo.sv
../01_snax_llm_fp4_subsystem/softmax_requantizer.sv
../01_snax_llm_fp4_subsystem/l_accumulator.sv
../01_snax_llm_fp4_subsystem/final_normalizer.sv
../01_snax_llm_fp4_subsystem/slq_encoder.sv
../01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv

// ------- Testbench -------
../tb/tb_snax_gemm_prefill.sv
