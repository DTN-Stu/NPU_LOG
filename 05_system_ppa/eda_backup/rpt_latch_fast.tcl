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

set pfx "u_core/u_acc_buffer/u_latch_array/gen_latch_storage"
set dl_q_sample  [get_pins "${pfx}.col_gen[0].icg_gen[0].row_gen[0].bit_gen.u_dlatch/u_dhl/Q ${pfx}.col_gen[0].icg_gen[8].row_gen[0].bit_gen.u_dlatch/u_dhl/Q ${pfx}.col_gen[0].icg_gen[15].row_gen[0].bit_gen.u_dlatch/u_dhl/Q"]
set dl_d_sample  [get_pins "${pfx}.col_gen[0].icg_gen[0].row_gen[0].bit_gen.u_dlatch/u_dhl/D ${pfx}.col_gen[0].icg_gen[8].row_gen[0].bit_gen.u_dlatch/u_dhl/D ${pfx}.col_gen[0].icg_gen[15].row_gen[0].bit_gen.u_dlatch/u_dhl/D"]
set icg_sample   [get_pins "${pfx}.col_gen[0].icg_gen[0].u_icg_cell/u_icg/ENA ${pfx}.col_gen[0].icg_gen[8].u_icg_cell/u_icg/ENA"]

puts "=== 2. FROM D-LATCH (SETUP / READ PATHS) ==="
report_timing -from $dl_q_sample -max_paths 5 -path_type end -nosplit
report_timing -from $dl_q_sample -max_paths 1 -nosplit

puts "=== 3. TO ICG ENA (SETUP) ==="
report_timing -to $icg_sample -max_paths 3 -path_type end -nosplit

puts "=== 4. HOLD (MIN DELAY) TO D-LATCH & ICG ==="
report_timing -delay_type min -to $dl_d_sample -max_paths 3 -path_type end -nosplit
report_timing -delay_type min -to $icg_sample -max_paths 3 -path_type end -nosplit
exit
