set DB_ASAP7 "/home/ICer/vsq_syn/syn/libs/db/asap7"
set DB_MVT   "/home/ICer/asap7_multivt/db"
set SRAM_DB  "/home/ICer/sys_syn/sram_lib/srambank_256x4x72_6t122.db"
set CELL_DBS [list \
    $DB_ASAP7/AO_RVT_TT.db $DB_ASAP7/OA_RVT_TT.db $DB_ASAP7/SIMPLE_RVT_TT.db $DB_ASAP7/INVBUF_RVT_TT.db $DB_ASAP7/SEQ_RVT_TT.db \
    $DB_MVT/AO_LVT_TT.db   $DB_MVT/OA_LVT_TT.db   $DB_MVT/SIMPLE_LVT_TT.db   $DB_MVT/INVBUF_LVT_TT.db   $DB_MVT/SEQ_LVT_TT.db \
    $DB_MVT/AO_SLVT_TT.db  $DB_MVT/OA_SLVT_TT.db  $DB_MVT/SIMPLE_SLVT_TT.db  $DB_MVT/INVBUF_SLVT_TT.db  $DB_MVT/SEQ_SLVT_TT.db]
set_app_var target_library $CELL_DBS
set_app_var link_library   [concat [list "*"] $CELL_DBS [list $SRAM_DB]]

read_ddc output/snax_llm_top_clean.ddc
current_design snax_llm_top_clean

set dl_d_pins   [get_pins -hier *u_dhl/D]
set dl_q_pins   [get_pins -hier *u_dhl/Q]
set icg_en_pins [get_pins -hier *u_icg/ENA]

redirect reports/timing_latch_icg.rpt {
    puts "=== 1. TO D-LATCH (SETUP / TIME BORROW) ==="
    report_timing -to $dl_d_pins -max_paths 3 -nosplit -transition_time
    puts "=== 2. FROM D-LATCH (SETUP / READ PATHS) ==="
    report_timing -from $dl_q_pins -max_paths 5 -nosplit -transition_time
    puts "=== 3. TO ICG ENA (CLOCK GATING SETUP) ==="
    report_timing -to $icg_en_pins -max_paths 3 -nosplit -transition_time
    puts "=== 4. HOLD CHECK (MIN DELAY) TO D-LATCH & ICG ==="
    report_timing -delay_type min -to $dl_d_pins -max_paths 3 -nosplit
    report_timing -delay_type min -to $icg_en_pins -max_paths 3 -nosplit
}
exit
