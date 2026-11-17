# SNAX FP4 vsq_matmul_engine - standalone flash tiling / online rescale TB
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

# ------- Testbench -------
../tb/tb_vsq_matmul_flash_tiling.sv
