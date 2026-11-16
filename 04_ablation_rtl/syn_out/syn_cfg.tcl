# =============================================================================
# syn_cfg.tcl - Parameterized ASAP7 7nm DC synthesis (500 MHz)
# Usage: dc_shell -f syn_cfg.tcl -x "set DESIGN_NAME cfg2_nvfp4_fp32_ff"
# =============================================================================
if {![info exists DESIGN_NAME]} { set DESIGN_NAME "cfg2_nvfp4_fp32_ff" }

set BASE_DIR   [file normalize [file dirname [info script]]]
set RTL_DIR    [file join $BASE_DIR rtl]
set REPORT_DIR [file join $BASE_DIR reports]
set OUTPUT_DIR [file join $BASE_DIR output]
file mkdir $REPORT_DIR
file mkdir $OUTPUT_DIR

set DB_ASAP7 "/home/ICer/vsq_syn/syn/libs/db/asap7"
set CELL_DBS [list \
    [file join $DB_ASAP7 AO_RVT_TT.db] \
    [file join $DB_ASAP7 OA_RVT_TT.db] \
    [file join $DB_ASAP7 SIMPLE_RVT_TT.db] \
    [file join $DB_ASAP7 INVBUF_RVT_TT.db] \
    [file join $DB_ASAP7 SEQ_RVT_TT.db]]

set_app_var search_path [concat [list $RTL_DIR $DB_ASAP7] [get_app_var search_path]]
set_app_var synthetic_library [list dw_foundation.sldb]
set_app_var target_library $CELL_DBS
set_app_var link_library   [concat [list "*"] $CELL_DBS]

set_app_var hdlin_auto_save_templates true
set_app_var verilogout_no_tri         true
set_app_var bus_naming_style          {%s[%d]}

set PERIOD_NS 2.0
set TSCALE    1000.0 ;# ASAP7 time_unit = 1 ps

set files [list \
    [file join $RTL_DIR fp4_mult_lut.sv] \
    [file join $RTL_DIR compressor_4to2.sv] \
    [file join $RTL_DIR dot_product_16.sv] \
    [file join $RTL_DIR latch_array.sv] \
    [file join $RTL_DIR vsq_rescaler_32b.sv] \
    [file join $RTL_DIR vsq_matmul_engine.sv] \
    [file join $RTL_DIR nvfp4_scale_mac_lane.sv] \
    [file join $RTL_DIR MXFP4_g16_MUL.sv] \
    [file join $RTL_DIR NVFP4_MUL.sv] \
    [file join $RTL_DIR fp32_lane_backend.sv] \
    [file join $RTL_DIR cfg1_mxfp4_g16_fp32_ff.sv] \
    [file join $RTL_DIR cfg2_nvfp4_fp32_ff.sv] \
    [file join $RTL_DIR cfg3_nvfp4_int32_lazy_ff.sv] \
    [file join $RTL_DIR cfg4_log8_int32_lazy_ff.sv] \
    [file join $RTL_DIR cfg5_log8_int32_lazy_latch.sv] \
    [file join $RTL_DIR cfg6_log8_fp32_ff.sv]]

analyze -format sverilog -vcs [list "+incdir+$RTL_DIR"] $files
elaborate $DESIGN_NAME
current_design $DESIGN_NAME
link

set CLK [expr {$PERIOD_NS * $TSCALE}]
create_clock -name clk -period $CLK [get_ports clk]
set_clock_uncertainty [expr {0.05 * $TSCALE}] [get_clocks clk]

set in_ports [remove_from_collection [all_inputs] [get_ports clk]]
set_input_transition [expr {0.05 * $TSCALE}] $in_ports
set_driving_cell -lib_cell NAND2x1_ASAP7_75t_R -pin Y $in_ports
set_input_delay  -clock clk -max [expr {0.2 * $CLK}] $in_ports
set_input_delay  -clock clk -min 0                 $in_ports
set_output_delay -clock clk -max [expr {0.2 * $CLK}] [all_outputs]
set_output_delay -clock clk -min 0                 [all_outputs]
set_load 0.02 [all_outputs]
set_false_path -from [get_ports reset_n]
set_max_area 0

puts "INFO: PE=$DESIGN_NAME period(design units)=[get_attribute [get_clocks clk] period]"
compile_ultra -no_autoungroup

redirect [file join $REPORT_DIR ${DESIGN_NAME}_area.rpt]   { report_area -hierarchy }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_power.rpt]  { report_power -hierarchy }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_timing.rpt] { report_timing -max_paths 5 }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_qor.rpt]    { report_qor }

write -format verilog -hierarchy -output [file join $OUTPUT_DIR ${DESIGN_NAME}_netlist.v]
write -format ddc     -hierarchy -output [file join $OUTPUT_DIR ${DESIGN_NAME}.ddc]
write_sdc -nosplit                 [file join $OUTPUT_DIR ${DESIGN_NAME}.sdc]

puts "INFO: $DESIGN_NAME synthesis complete."
exit
