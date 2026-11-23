# =============================================================================
# filelist_gls.f - Gate-Level Simulation Filelist (QuestaSim 2024.1)
# Uses Multi-Vt ASAP7 Verilog models (RVT/LVT/SLVT), SRAM macro,
# snax_simple_dma (TB DMA model), and synthesized Top netlist.
# =============================================================================

# ASAP7 RVT
./lib/asap7_rvt/asap7sc7p5t_AO_RVT_TT_201020.v
./lib/asap7_rvt/asap7sc7p5t_INVBUF_RVT_TT_201020.v
./lib/asap7_rvt/asap7sc7p5t_OA_RVT_TT_201020.v
./lib/asap7_rvt/asap7sc7p5t_SEQ_RVT_TT_220101.v
./lib/asap7_rvt/asap7sc7p5t_SIMPLE_RVT_TT_201020.v
./lib/asap7_rvt/asap7sc7p5t_TBUF_RVT_TT_260831.v

# ASAP7 LVT
./lib/asap7_lvt/asap7sc7p5t_AO_LVT_TT_201020.v
./lib/asap7_lvt/asap7sc7p5t_INVBUF_LVT_TT_201020.v
./lib/asap7_lvt/asap7sc7p5t_OA_LVT_TT_201020.v
./lib/asap7_lvt/asap7sc7p5t_SEQ_LVT_TT_220101.v
./lib/asap7_lvt/asap7sc7p5t_SIMPLE_LVT_TT_201020.v
./lib/asap7_lvt/asap7sc7p5t_TBUF_LVT_TT_260831.v

# ASAP7 SLVT
./lib/asap7_slvt/asap7sc7p5t_AO_SLVT_TT_201020.v
./lib/asap7_slvt/asap7sc7p5t_INVBUF_SLVT_TT_201020.v
./lib/asap7_slvt/asap7sc7p5t_OA_SLVT_TT_201020.v
./lib/asap7_slvt/asap7sc7p5t_SEQ_SLVT_TT_220101.v
./lib/asap7_slvt/asap7sc7p5t_SIMPLE_SLVT_TT_201020.v
./lib/asap7_slvt/asap7sc7p5t_TBUF_SLVT_TT_260831.v

# SRAM Macro
./lib/srambank_256x4x72_6t122.v

# Top-Level Synthesized Gate Netlist (600 MHz ASAP7 Multi-Vt)
../05_system_ppa/eda_backup/output_netlist_bu/snax_llm_top_clean.v

# Testbench + DMA Bridge
../01_snax_llm_fp4_subsystem/snax_simple_dma.sv
./tb_gemm_os_gls.sv
