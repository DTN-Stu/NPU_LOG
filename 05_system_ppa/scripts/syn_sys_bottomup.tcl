# =============================================================================
# syn_sys_bottomup.tcl - Top-Level Integration + Incremental Compile @ 600 MHz
# Reads pre-compiled .ddc for 6 heavy blocks, compiles glue/TCDM/interconnect,
# and runs compile_ultra -incremental -no_autoungroup.
# =============================================================================
set DESIGN_NAME "snax_llm_top_clean"
set PERIOD_PS   1666.7
set_host_options -max_cores 16

set BASE_DIR   [file normalize [file dirname [info script]]]
set RTL_DIR    [file join $BASE_DIR rtl]
set SRAM_DIR   [file join $BASE_DIR sram_lib]
set BU_DDC_DIR [file join $BASE_DIR ddc_bu]
set REPORT_DIR [file join $BASE_DIR reports]
set OUTPUT_DIR [file join $BASE_DIR output]
file mkdir $REPORT_DIR
file mkdir $OUTPUT_DIR

set DB_ASAP7 "/home/ICer/vsq_syn/syn/libs/db/asap7"
set DB_MVT   "/home/ICer/asap7_multivt/db"
set CELL_DBS [list \
    [file join $DB_ASAP7 AO_RVT_TT.db] \
    [file join $DB_ASAP7 OA_RVT_TT.db] \
    [file join $DB_ASAP7 SIMPLE_RVT_TT.db] \
    [file join $DB_ASAP7 INVBUF_RVT_TT.db] \
    [file join $DB_ASAP7 SEQ_RVT_TT.db] \
    [file join $DB_MVT   AO_LVT_TT.db] \
    [file join $DB_MVT   OA_LVT_TT.db] \
    [file join $DB_MVT   SIMPLE_LVT_TT.db] \
    [file join $DB_MVT   INVBUF_LVT_TT.db] \
    [file join $DB_MVT   SEQ_LVT_TT.db] \
    [file join $DB_MVT   AO_SLVT_TT.db] \
    [file join $DB_MVT   OA_SLVT_TT.db] \
    [file join $DB_MVT   SIMPLE_SLVT_TT.db] \
    [file join $DB_MVT   INVBUF_SLVT_TT.db] \
    [file join $DB_MVT   SEQ_SLVT_TT.db]]
set SRAM_DB [file join $SRAM_DIR srambank_256x4x72_6t122.db]

set_app_var search_path        [concat [list $RTL_DIR $DB_ASAP7 $DB_MVT $SRAM_DIR] [get_app_var search_path]]
set_app_var synthetic_library  [list dw_foundation.sldb]
set_app_var target_library     $CELL_DBS
set_app_var link_library       [concat [list "*"] $CELL_DBS [list $SRAM_DB]]
set_app_var hdlin_auto_save_templates true
set_app_var verilogout_no_tri         true
set_app_var bus_naming_style          {%s[%d]}

# 1. Analyze & elaborate Top first, then bind pre-compiled sub-block DDCs
set files [list \
    [file join $RTL_DIR fp4_mult_lut.sv] \
    [file join $RTL_DIR compressor_4to2.sv] \
    [file join $RTL_DIR dot_product_16.sv] \
    [file join $RTL_DIR log8_fp4_arith_units.sv] \
    [file join $RTL_DIR latch_array.sv] \
    [file join $RTL_DIR log8_fp4_mac_array.sv] \
    [file join $RTL_DIR rmw_latch_acc_buffer.sv] \
    [file join $RTL_DIR vsq_rescaler_32b.sv] \
    [file join $RTL_DIR attn_o_rescaler_bank.sv] \
    [file join $RTL_DIR softmax_requantizer.sv] \
    [file join $RTL_DIR p_fifo.sv] \
    [file join $RTL_DIR l_accumulator.sv] \
    [file join $RTL_DIR final_normalizer.sv] \
    [file join $RTL_DIR snax_fused_llm_core.sv] \
    [file join $RTL_DIR snax_sfu_fp20_units.sv] \
    [file join $RTL_DIR snax_llm_sfu_opt_pipe.sv] \
    [file join $RTL_DIR slq_encoder.sv] \
    [file join $RTL_DIR snax_transpose_buffer.sv] \
    [file join $RTL_DIR snax_transpose_ctrl.sv] \
    [file join $RTL_DIR skid_buffer.sv] \
    [file join $RTL_DIR snax_streamer_act_bcast.sv] \
    [file join $RTL_DIR snax_streamer_wt_16ch.sv] \
    [file join $RTL_DIR snax_streamer_write_out.sv] \
    [file join $RTL_DIR snax_stream_read_fp16.sv] \
    [file join $RTL_DIR fixed_priority_arb_n.sv] \
    [file join $RTL_DIR interleaved_interconnect_72b.sv] \
    [file join $RTL_DIR multi_bank_tcdm_72b.sv] \
    [file join $RTL_DIR snax_csr_manager.sv] \
    [file join $RTL_DIR snax_flash_csr_snax.sv] \
    [file join $RTL_DIR snax_llm_sequencer.sv] \
    [file join $RTL_DIR snax_sfu_sequencer.sv] \
    [file join $RTL_DIR snax_llm_top_clean.sv]]

analyze -format sverilog -vcs [list "+incdir+$RTL_DIR" "+define+IGC_CELL" "+define+LATCH_CELL" "+define+USE_ASAP7_SRAM"] $files
elaborate $DESIGN_NAME
current_design $DESIGN_NAME

# Point WORK to an empty directory so DC cannot re-elaborate removed sub-blocks
set WORK_EMPTY [file join $BASE_DIR work_empty]
file delete -force $WORK_EMPTY
file mkdir $WORK_EMPTY
define_design_lib WORK -path $WORK_EMPTY

# Remove unmapped sub-designs for the 6 pre-compiled blocks so read_ddc is not shadowed (avoids UIO-92)
set bu_inst_names {u_core u_sfu u_seq u_write_out u_wt_streamer u_csr}
foreach inst $bu_inst_names {
    set c [get_cells $inst]
    set r [get_attribute $c ref_name]
    puts "INFO: Removing unmapped design $r (for $inst) before loading pre-compiled DDC"
    catch { remove_design -hierarchy $r }
}

# Read pre-compiled sub-block DDCs
read_ddc [file join $BU_DDC_DIR u_core.ddc]
read_ddc [file join $BU_DDC_DIR u_sfu.ddc]
read_ddc [file join $BU_DDC_DIR u_seq.ddc]
read_ddc [file join $BU_DDC_DIR u_write_out.ddc]
read_ddc [file join $BU_DDC_DIR u_wt_streamer.ddc]
read_ddc [file join $BU_DDC_DIR u_csr.ddc]

current_design $DESIGN_NAME
change_link u_core        snax_fused_llm_core_NUM_LANES_P16_ACC_DEPTH_P16_PKT_W72_FRAC_BITS16_P_FIFO_DEPTH16_NUM_QUERIES8_USE_OUTPUT_READY1
change_link u_sfu         snax_llm_sfu_opt_pipe_NUM_LANES16_ALU_LANES8
change_link u_seq         snax_llm_sequencer_ADDR_WIDTH32_NUM_LANES16_NUM_RING_SLOTS4_SLOT_SIZE_BYTES131072_OUT_PIPE_CREDITS4_CAP_O_TILES16
change_link u_write_out   snax_streamer_write_out_DATA_WIDTH72_ADDR_WIDTH32_NUM_LANES16_STRB_WIDTH9
change_link u_wt_streamer snax_streamer_wt_16ch_DATA_WIDTH72_ADDR_WIDTH32_NUM_LANES16_STRB_WIDTH9
change_link u_csr         snax_flash_csr_snax_ENABLE_LEGACY_DIRECT_CSR0_DOUBLE_BUFFER0
if {[link] == 0} {
    puts "ERROR: Top-level link failed with unresolved references! Aborting."
    exit 1
}

# Verify all 6 sub-blocks are linked to mapped designs
foreach inst $bu_inst_names {
    set c [get_cells $inst]
    set r [get_attribute $c ref_name]
    set unmapped_cnt [sizeof_collection [get_cells -quiet -hier -filter "full_name =~ ${inst}/* && is_hierarchical == false && is_unmapped == true"]]
    set total_cnt    [sizeof_collection [get_cells -quiet -hier -filter "full_name =~ ${inst}/* && is_hierarchical == false"]]
    puts "INFO: Verified $inst ($r): total_leaf=$total_cnt, unmapped_leaf=$unmapped_cnt"
    if {$total_cnt == 0 || $unmapped_cnt > 0} {
        puts "ERROR: Sub-block $inst ($r) did not bind to a mapped DDC! Aborting."
        exit 1
    }
}

# Protect SRAM macros
set_dont_touch [get_cells -hier -filter "ref_name == srambank_256x4x72_6t122"] true

source [file join $BASE_DIR constraints_sys_600mhz.sdc]
set_max_area 0

redirect [file join $REPORT_DIR ${DESIGN_NAME}_check_design.rpt] { check_design }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_check_timing.rpt] { check_timing }

# Step A: Map unmapped glue, interconnect, TCDM, and small blocks while keeping 6 DDCs intact
puts "INFO: Bottom-Up Step A — Compiling unmapped Top/TCDM/Interconnect..."
set bu_cells [get_cells {u_core u_sfu u_seq u_write_out u_wt_streamer u_csr}]
set_dont_touch $bu_cells true
compile_ultra -no_autoungroup

# Step B: Unprotect 6 sub-blocks and run incremental compile to optimize boundary paths
puts "INFO: Bottom-Up Step B — Incremental boundary optimization @ 600 MHz..."
set_dont_touch $bu_cells false
compile_ultra -incremental -no_autoungroup

redirect [file join $REPORT_DIR ${DESIGN_NAME}_area.rpt]          { report_area -hierarchy }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_power_default.rpt] { report_power -hierarchy }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_timing.rpt]        { report_timing -max_paths 50 -nets -transition_time }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_qor.rpt]           { report_qor }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_clock_gating.rpt]  { report_clock_gating -verbose }

write -format verilog -hierarchy -output [file join $OUTPUT_DIR ${DESIGN_NAME}_netlist.v]
write -format ddc     -hierarchy -output [file join $OUTPUT_DIR ${DESIGN_NAME}.ddc]
write_sdc -nosplit                       [file join $OUTPUT_DIR ${DESIGN_NAME}.sdc]

puts "INFO: Bottom-Up Top Synthesis complete."
exit
