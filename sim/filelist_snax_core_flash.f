# SNAX fused LLM core - full pipeline flash tiling TB (softmax + P FIFO)
# Paths are relative to sim/
+incdir+../01_snax_llm_fp4_subsystem

# ------- RTL -------
../01_snax_llm_fp4_subsystem/fp4_mult_lut.sv
../rtl/compressor_4to2.sv
../01_snax_llm_fp4_subsystem/dot_product_16.sv
../01_snax_llm_fp4_subsystem/latch_array.sv
../01_snax_llm_fp4_subsystem/vsq_rescaler_32b.sv
../01_snax_llm_fp4_subsystem/log8_fp4_arith_units.sv
../01_snax_llm_fp4_subsystem/vsq_matmul_engine.sv
../01_snax_llm_fp4_subsystem/softmax_requantizer.sv
../01_snax_llm_fp4_subsystem/p_fifo.sv
../01_snax_llm_fp4_subsystem/l_accumulator.sv
../01_snax_llm_fp4_subsystem/final_normalizer.sv
../01_snax_llm_fp4_subsystem/slq_encoder.sv
../01_snax_llm_fp4_subsystem/log8_fp4_mac_array.sv
../01_snax_llm_fp4_subsystem/rmw_latch_acc_buffer.sv
../01_snax_llm_fp4_subsystem/attn_o_rescaler_bank.sv
../01_snax_llm_fp4_subsystem/snax_fused_llm_core.sv

# ------- Testbench -------
../tb/tb_snax_core_flash_tiling.sv
