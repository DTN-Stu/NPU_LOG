set BASE_DIR [pwd]
set BU_DDC_DIR [file join $BASE_DIR ddc_bu]
set NET_DIR [file join $BASE_DIR netlist_bu]
file mkdir $NET_DIR

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
set_app_var target_library $CELL_DBS
set_app_var link_library [concat [list "*"] $CELL_DBS]
set_app_var verilogout_no_tri true
set_app_var bus_naming_style {%s[%d]}

foreach blk {u_core u_sfu u_seq u_write_out u_wt_streamer u_csr} {
    remove_design -all
    read_ddc [file join $BU_DDC_DIR ${blk}.ddc]
    link
    change_names -rules verilog -hierarchy
    write -format verilog -hierarchy -output [file join $NET_DIR ${blk}.v]
    puts "EXPORTED $blk -> [get_object_name [current_design]]"
}
exit
