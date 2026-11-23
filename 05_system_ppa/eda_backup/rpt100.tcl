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

redirect reports/timing_top100.rpt { report_timing -max_paths 100 -nosplit -transition_time }
redirect reports/timing_top20_nonsfu.rpt { report_timing -max_paths 20 -path_type end -to [get_cells -hier -filter "is_sequential==true && full_name !~ *u_sfu*"] -nosplit }
exit
