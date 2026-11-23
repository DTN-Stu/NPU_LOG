# =============================================================================
# syn_sys_top.tcl - Full Subsystem Synthesis (snax_llm_top_clean) on ASAP7 7nm
# Usage: dc_shell -f syn_sys_top.tcl
# =============================================================================
set DESIGN_NAME "snax_llm_top_clean"
set PERIOD_PS   1666.7

set_host_options -max_cores 16

set BASE_DIR   [file normalize [file dirname [info script]]]
set RTL_DIR    [file join $BASE_DIR rtl]
set SRAM_DIR   [file join $BASE_DIR sram_lib]
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
set_app_var link_library       [concat [list "*"] $CELL_DBS [list $SRAM_DB dw_foundation.sldb]]

set_app_var hdlin_auto_save_templates true
set_app_var verilogout_no_tri         true
set_app_var bus_naming_style          {%s[%d]}

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
link

# Protect SRAM macros from optimization
set_dont_touch [get_cells -hier -filter "ref_name == srambank_256x4x72_6t122"] true

source [file join $BASE_DIR constraints_sys_600mhz.sdc]
set_max_area 0

redirect [file join $REPORT_DIR ${DESIGN_NAME}_check_design.rpt] { check_design }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_check_timing.rpt] { check_timing }

puts "INFO: Synthesizing $DESIGN_NAME @ period=${PERIOD_PS}ps (600 MHz)"
compile_ultra -no_autoungroup

redirect [file join $REPORT_DIR ${DESIGN_NAME}_area.rpt]          { report_area -hierarchy }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_power_default.rpt] { report_power -hierarchy }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_timing.rpt]        { report_timing -max_paths 20 -nets -transition_time }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_qor.rpt]           { report_qor }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_clock_gating.rpt]  { report_clock_gating -verbose }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_reference.rpt]     { report_reference -hierarchy }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_vth.rpt]           { report_threshold_voltage_group }

write -format verilog -hierarchy -output [file join $OUTPUT_DIR ${DESIGN_NAME}_netlist.v]
write -format ddc     -hierarchy -output [file join $OUTPUT_DIR ${DESIGN_NAME}.ddc]
write_sdc -nosplit                       [file join $OUTPUT_DIR ${DESIGN_NAME}.sdc]

puts "INFO: $DESIGN_NAME synthesis complete."
exit
