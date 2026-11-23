# =============================================================================
# syn_subblock.tcl - Bottom-Up Sub-Block Synthesis @ 600 MHz (ASAP7 Multi-Vt)
# Usage: dc_shell -f syn_subblock.tcl -x "set BLK_TAG u_sfu; set TOP_MOD snax_llm_sfu_opt_pipe; set PARAMS {NUM_LANES=16,ALU_LANES=8}"
# =============================================================================
if {![info exists BLK_TAG]} { set BLK_TAG "u_sfu" }
if {![info exists TOP_MOD]} { set TOP_MOD "snax_llm_sfu_opt_pipe" }
if {![info exists PARAMS]}  { set PARAMS  "" }
set PERIOD_PS 1666.7

set_host_options -max_cores 4

set BASE_DIR   [file normalize [file dirname [info script]]]
set RTL_DIR    [file join $BASE_DIR rtl]
set REPORT_DIR [file join $BASE_DIR reports_bu]
set OUTPUT_DIR [file join $BASE_DIR ddc_bu]
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

set_app_var search_path        [concat [list $RTL_DIR $DB_ASAP7 $DB_MVT] [get_app_var search_path]]
set_app_var synthetic_library  [list dw_foundation.sldb]
set_app_var target_library     $CELL_DBS
set_app_var link_library       [concat [list "*"] $CELL_DBS]
set_app_var hdlin_auto_save_templates true
set_app_var verilogout_no_tri         true
set_app_var bus_naming_style          {%s[%d]}

# Isolate work library per block so parallel DC runs never collide!
set WORK_LIB [file join $BASE_DIR "work_${BLK_TAG}"]
file mkdir $WORK_LIB
define_design_lib WORK -path $WORK_LIB

if {$BLK_TAG eq "u_core"} {
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
        [file join $RTL_DIR snax_fused_llm_core.sv]]
} elseif {$BLK_TAG eq "u_sfu"} {
    set files [list \
        [file join $RTL_DIR snax_sfu_fp20_units.sv] \
        [file join $RTL_DIR snax_llm_sfu_opt_pipe.sv]]
} elseif {$BLK_TAG eq "u_seq"} {
    set files [list [file join $RTL_DIR snax_llm_sequencer.sv]]
} elseif {$BLK_TAG eq "u_write_out"} {
    set files [list \
        [file join $RTL_DIR skid_buffer.sv] \
        [file join $RTL_DIR snax_streamer_write_out.sv]]
} elseif {$BLK_TAG eq "u_wt_streamer"} {
    set files [list \
        [file join $RTL_DIR skid_buffer.sv] \
        [file join $RTL_DIR snax_streamer_wt_16ch.sv]]
} elseif {$BLK_TAG eq "u_csr"} {
    set files [list \
        [file join $RTL_DIR snax_csr_manager.sv] \
        [file join $RTL_DIR snax_flash_csr_snax.sv]]
}

analyze -format sverilog -vcs [list "+incdir+$RTL_DIR" "+define+IGC_CELL" "+define+LATCH_CELL"] $files
if {$PARAMS ne ""} {
    elaborate $TOP_MOD -parameters $PARAMS
} else {
    elaborate $TOP_MOD
}
set ELAB_DESIGN [get_object_name [current_design]]
link

# Block-level constraints @ 600 MHz (1666.7 ps)
create_clock -name clk -period $PERIOD_PS -waveform [list 0.0 [expr {$PERIOD_PS * 0.5}]] [get_ports clk]
set_clock_uncertainty -setup 83.3 [get_clocks clk]
set_clock_uncertainty -hold  20.0 [get_clocks clk]
set_clock_transition  25.0 [get_clocks clk]

set rst_port [get_ports -quiet {rst_n reset_n}]
set in_ports [remove_from_collection [all_inputs] [add_to_collection [get_ports clk] $rst_port]]
set_driving_cell -lib_cell BUFx4_ASAP7_75t_R -pin Y $in_ports
set_load -pin_load 5.0 [all_outputs]
set_input_delay  -clock clk -max [expr {$PERIOD_PS * 0.20}] $in_ports
set_input_delay  -clock clk -min 15.0 $in_ports
set_output_delay -clock clk -max [expr {$PERIOD_PS * 0.20}] [all_outputs]
set_output_delay -clock clk -min -10.0 [all_outputs]
if {[sizeof_collection $rst_port] > 0} {
    set_false_path -from $rst_port
}
set_max_transition 100.0 [current_design]
set_max_fanout 48 [current_design]
set_clock_gating_check -setup 25.0 -hold 10.0 [get_clocks clk]
set_max_time_borrow [expr {$PERIOD_PS * 0.25}] [all_clocks]
set_max_area 0

compile_ultra -no_autoungroup

redirect [file join $REPORT_DIR ${BLK_TAG}_area.rpt]   { report_area -hierarchy }
redirect [file join $REPORT_DIR ${BLK_TAG}_timing.rpt] { report_timing -max_paths 10 -nets -transition_time }
redirect [file join $REPORT_DIR ${BLK_TAG}_qor.rpt]    { report_qor }

write -format ddc -hierarchy -output [file join $OUTPUT_DIR ${BLK_TAG}.ddc]
puts "INFO: Sub-block $BLK_TAG ($ELAB_DESIGN) complete."
exit
