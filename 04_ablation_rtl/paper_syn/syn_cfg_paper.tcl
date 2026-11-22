# =============================================================================
# syn_cfg_paper.tcl - Unified ASAP7 7nm DC Synthesis for 5-Config Paper Ablation
# Usage: dc_shell -f syn_cfg_paper.tcl -x "set DESIGN_NAME cfg1_mxfp4_g16_fp32_ff"
# =============================================================================
if {![info exists DESIGN_NAME]} { set DESIGN_NAME "cfg5_log8_int32_lazy_latch" }
if {![info exists PERIOD_PS]}   { set PERIOD_PS 2000.0 }

set_host_options -max_cores 8

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

set_app_var search_path        [concat [list $RTL_DIR $DB_ASAP7] [get_app_var search_path]]
set_app_var synthetic_library  [list dw_foundation.sldb]
set_app_var target_library     $CELL_DBS
set_app_var link_library       [concat [list "*"] $CELL_DBS]

set_app_var hdlin_auto_save_templates true
set_app_var verilogout_no_tri         true
set_app_var bus_naming_style          {%s[%d]}

set files [list \
    [file join $RTL_DIR latch_array.sv] \
    [file join $RTL_DIR log8_fp4_arith_units.sv] \
    [file join $RTL_DIR MXFP4_g16_MUL.sv] \
    [file join $RTL_DIR NVFP4_MUL.sv] \
    [file join $RTL_DIR LOG8_MUL.sv] \
    [file join $RTL_DIR fp32_lane_backend.sv] \
    [file join $RTL_DIR log8_int32_lazy_core.sv] \
    [file join $RTL_DIR cfg1_mxfp4_g16_fp32_ff.sv] \
    [file join $RTL_DIR cfg2_nvfp4_fp32_ff.sv] \
    [file join $RTL_DIR cfg6_log8_fp32_ff.sv] \
    [file join $RTL_DIR cfg4_log8_int32_lazy_ff.sv] \
    [file join $RTL_DIR cfg5_log8_int32_lazy_latch.sv]]

# Always map icg_cell -> ICGx4_ASAP7_75t_R and dlatch -> DHLx2_ASAP7_75t_R
analyze -format sverilog -vcs [list "+incdir+$RTL_DIR" "+define+IGC_CELL" "+define+LATCH_CELL"] $files
elaborate $DESIGN_NAME
current_design $DESIGN_NAME
link

# Apply unified SDC constraints
source [file join $BASE_DIR constraints_500mhz.sdc]
set_max_area 0

# Pre-compile sanity checks
redirect [file join $REPORT_DIR ${DESIGN_NAME}_check_design.rpt] { check_design }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_check_timing.rpt] { check_timing }

puts "INFO: Synthesizing $DESIGN_NAME @ period=${PERIOD_PS}ps with unified compile_ultra -no_autoungroup"
compile_ultra -no_autoungroup

# Post-compile reports
redirect [file join $REPORT_DIR ${DESIGN_NAME}_area.rpt]         { report_area -hierarchy }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_power_default.rpt] { report_power -hierarchy }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_timing.rpt]       { report_timing -max_paths 10 -nets -transition_time }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_qor.rpt]          { report_qor }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_clock_gating.rpt] { report_clock_gating -verbose }

write -format verilog -hierarchy -output [file join $OUTPUT_DIR ${DESIGN_NAME}_netlist.v]
write -format ddc     -hierarchy -output [file join $OUTPUT_DIR ${DESIGN_NAME}.ddc]
write_sdc -nosplit                       [file join $OUTPUT_DIR ${DESIGN_NAME}.sdc]

puts "INFO: $DESIGN_NAME synthesis complete."
exit
