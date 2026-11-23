# =============================================================================
# syn_sys_netlist_bu.tcl - Top-Level Synthesis with Pre-Compiled Gate Netlists
# Reads 6 gate-level netlists (netlist_bu/*.v), elaborates Top + uncompiled RTL
# (snax_llm_top_clean_bu.sv), runs Stage A (dont_touch netlists, compile RTL)
# and Stage B (unprotect netlists, compile_ultra -incremental -no_autoungroup).
# =============================================================================
set DESIGN_NAME "snax_llm_top_clean"
set PERIOD_PS   1666.7
set_host_options -max_cores 16

set BASE_DIR   [file normalize [file dirname [info script]]]
set RTL_DIR    [file join $BASE_DIR rtl]
set SRAM_DIR   [file join $BASE_DIR sram_lib]
set NET_DIR    [file join $BASE_DIR netlist_bu]
set REPORT_DIR [file join $BASE_DIR reports_netlist_bu]
set OUTPUT_DIR [file join $BASE_DIR output_netlist_bu]
file mkdir $REPORT_DIR
file mkdir $OUTPUT_DIR

set WORK_LIB [file join $BASE_DIR work_netlist_bu]
file delete -force $WORK_LIB
file mkdir $WORK_LIB
define_design_lib WORK -path $WORK_LIB

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

set_app_var search_path        [concat [list $RTL_DIR $NET_DIR $DB_ASAP7 $DB_MVT $SRAM_DIR] [get_app_var search_path]]
set_app_var synthetic_library  [list dw_foundation.sldb]
set_app_var target_library     $CELL_DBS
set_app_var link_library       [concat [list "*"] $CELL_DBS [list $SRAM_DB]]
set_app_var hdlin_auto_save_templates true
set_app_var verilogout_no_tri         true
set_app_var bus_naming_style          {%s[%d]}

# 1. Read the 6 pre-compiled gate-level netlists
foreach blk {u_core u_sfu u_seq u_write_out u_wt_streamer u_csr} {
    puts "INFO: Reading gate-level netlist ${blk}.v"
    read_verilog [file join $NET_DIR ${blk}.v]
}

# 2. Analyze & elaborate ONLY the remaining uncompiled RTL + snax_llm_top_clean_bu.sv
set rtl_files [list \
    [file join $RTL_DIR slq_encoder.sv] \
    [file join $RTL_DIR snax_transpose_buffer.sv] \
    [file join $RTL_DIR snax_transpose_ctrl.sv] \
    [file join $RTL_DIR snax_streamer_act_bcast.sv] \
    [file join $RTL_DIR snax_stream_read_fp16.sv] \
    [file join $RTL_DIR fixed_priority_arb_n.sv] \
    [file join $RTL_DIR interleaved_interconnect_72b.sv] \
    [file join $RTL_DIR multi_bank_tcdm_72b.sv] \
    [file join $RTL_DIR snax_sfu_sequencer.sv] \
    [file join $RTL_DIR snax_llm_top_clean_bu.sv]]

analyze -format sverilog -vcs [list "+incdir+$RTL_DIR" "+define+IGC_CELL" "+define+LATCH_CELL" "+define+USE_ASAP7_SRAM"] $rtl_files
elaborate $DESIGN_NAME
current_design $DESIGN_NAME

if {[link] == 0} {
    puts "ERROR: Top-level link failed! Aborting."
    exit 1
}

# Verify all 6 sub-blocks are mapped
set bu_inst_names {u_core u_sfu u_seq u_write_out u_wt_streamer u_csr}
foreach inst $bu_inst_names {
    set c [get_cells $inst]
    set r [get_attribute $c ref_name]
    set unmapped_cnt [sizeof_collection [get_cells -quiet -hier -filter "full_name =~ ${inst}/* && is_hierarchical == false && is_unmapped == true"]]
    set total_cnt    [sizeof_collection [get_cells -quiet -hier -filter "full_name =~ ${inst}/* && is_hierarchical == false"]]
    puts "INFO: Verified $inst ($r): total_leaf=$total_cnt, unmapped_leaf=$unmapped_cnt"
    if {$total_cnt == 0 || $unmapped_cnt > 0} {
        puts "ERROR: Sub-block $inst ($r) is not cleanly mapped! Aborting."
        exit 1
    }
}

# Protect SRAM macros
set_dont_touch [get_cells -hier -filter "ref_name == srambank_256x4x72_6t122"] true

source [file join $BASE_DIR constraints_sys_600mhz.sdc]
set_max_area 0

redirect [file join $REPORT_DIR ${DESIGN_NAME}_check_design.rpt] { check_design }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_check_timing.rpt] { check_timing }

# Stage A: Map uncompiled Top/TCDM/Interconnect/glue RTL while keeping 6 netlists dont_touch
puts "INFO: Netlist Bottom-Up Stage A — Compiling remaining RTL @ 600 MHz..."
set bu_cells [get_cells $bu_inst_names]
set_dont_touch $bu_cells true
compile_ultra -no_autoungroup

# Stage B: Unprotect 6 netlists and run incremental compile across boundaries
puts "INFO: Netlist Bottom-Up Stage B — Incremental boundary optimization @ 600 MHz..."
set_dont_touch $bu_cells false
compile_ultra -incremental -no_autoungroup

redirect [file join $REPORT_DIR ${DESIGN_NAME}_area.rpt]          { report_area -hierarchy }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_power_default.rpt] { report_power -hierarchy }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_timing.rpt]        { report_timing -max_paths 50 -nets -transition_time }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_qor.rpt]           { report_qor }
redirect [file join $REPORT_DIR ${DESIGN_NAME}_clock_gating.rpt]  { report_clock_gating -verbose }

change_names -rules verilog -hierarchy
write -format verilog -hierarchy -output [file join $OUTPUT_DIR ${DESIGN_NAME}.v]
write -format ddc     -hierarchy -output [file join $OUTPUT_DIR ${DESIGN_NAME}.ddc]
write_sdc -nosplit                       [file join $OUTPUT_DIR ${DESIGN_NAME}.sdc]
puts "INFO: Netlist Bottom-Up Top Synthesis complete."
exit
