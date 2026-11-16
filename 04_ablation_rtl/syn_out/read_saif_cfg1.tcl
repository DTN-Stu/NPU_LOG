# =============================================================================
# read_saif_cfg1.tcl - Read gate-level SAIF into DC and report accurate power
# =============================================================================
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

set_app_var search_path [concat [list $DB_ASAP7] [get_app_var search_path]]
set_app_var target_library $CELL_DBS
set_app_var link_library   [concat [list "*"] $CELL_DBS]

set DESIGN_NAME "cfg1_mxfp4_g16_fp32_ff"
read_ddc [file join $OUTPUT_DIR ${DESIGN_NAME}.ddc]
current_design $DESIGN_NAME
link

read_saif -input [file join $SIM_DIR ablation_dut.saif] -instance_name "tb_ablation_saif/dut"
redirect [file join $REPORT_DIR ${DESIGN_NAME}_power_saif.rpt] { report_power -hierarchy }
puts "INFO: SAIF power report written."
exit
