# =============================================================================
# read_saif_paper.tcl - Read gate-level SAIF into DC and report power
# Usage: dc_shell -f read_saif_paper.tcl -x "set DESIGN_NAME cfg5_log8_int32_lazy_latch; set STIM_TAG real"
# =============================================================================
if {![info exists DESIGN_NAME]} { set DESIGN_NAME "cfg5_log8_int32_lazy_latch" }
if {![info exists STIM_TAG]}    { set STIM_TAG "real" }

set BASE_DIR   [file normalize [file dirname [info script]]]
set REPORT_DIR [file join $BASE_DIR reports]
set OUTPUT_DIR [file join $BASE_DIR output]
set SIM_DIR    [file join $BASE_DIR sim]

set DB_ASAP7 "/home/ICer/vsq_syn/syn/libs/db/asap7"
set CELL_DBS [list \
    [file join $DB_ASAP7 AO_RVT_TT.db] \
    [file join $DB_ASAP7 OA_RVT_TT.db] \
    [file join $DB_ASAP7 SIMPLE_RVT_TT.db] \
    [file join $DB_ASAP7 INVBUF_RVT_TT.db] \
    [file join $DB_ASAP7 SEQ_RVT_TT.db]]

set_app_var search_path    [concat [list $DB_ASAP7] [get_app_var search_path]]
set_app_var target_library $CELL_DBS
set_app_var link_library   [concat [list "*"] $CELL_DBS]

read_ddc [file join $OUTPUT_DIR ${DESIGN_NAME}.ddc]
current_design $DESIGN_NAME
link

read_saif -input [file join $SIM_DIR ${DESIGN_NAME}_${STIM_TAG}.saif] -instance_name "tb_ablation_saif/dut"
redirect [file join $REPORT_DIR ${DESIGN_NAME}_power_saif_${STIM_TAG}.rpt] { report_power -hierarchy }
puts "INFO: $DESIGN_NAME SAIF ($STIM_TAG) power report written."
exit
