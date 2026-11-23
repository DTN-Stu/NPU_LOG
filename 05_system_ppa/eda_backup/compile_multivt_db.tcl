# =============================================================================
# compile_multivt_db.tcl - Compile ASAP7 LVT & SLVT TT .lib -> .db
# =============================================================================
set BASE_DIR [file normalize [file dirname [info script]]]
set LIB_DIR  "/home/ICer/asap7_multivt/lib"
set DB_DIR   "/home/ICer/asap7_multivt/db"
file mkdir $DB_DIR

set items [list \
    [list asap7sc7p5t_AO_LVT_TT_nldm_211120.lib     asap7sc7p5t_AO_LVT_TT_nldm_211120     AO_LVT_TT.db] \
    [list asap7sc7p5t_INVBUF_LVT_TT_nldm_220122.lib asap7sc7p5t_INVBUF_LVT_TT_nldm_211120 INVBUF_LVT_TT.db] \
    [list asap7sc7p5t_OA_LVT_TT_nldm_211120.lib     asap7sc7p5t_OA_LVT_TT_nldm_211120     OA_LVT_TT.db] \
    [list asap7sc7p5t_SEQ_LVT_TT_nldm_220123.lib    asap7sc7p5t_SEQ_LVT_TT_nldm_220123    SEQ_LVT_TT.db] \
    [list asap7sc7p5t_SIMPLE_LVT_TT_nldm_211120.lib asap7sc7p5t_SIMPLE_LVT_TT_nldm_211120 SIMPLE_LVT_TT.db] \
    [list asap7sc7p5t_AO_SLVT_TT_nldm_211120.lib    asap7sc7p5t_AO_SLVT_TT_nldm_211120    AO_SLVT_TT.db] \
    [list asap7sc7p5t_INVBUF_SLVT_TT_nldm_220122.lib asap7sc7p5t_INVBUF_SLVT_TT_nldm_211120 INVBUF_SLVT_TT.db] \
    [list asap7sc7p5t_OA_SLVT_TT_nldm_211120.lib    asap7sc7p5t_OA_SLVT_TT_nldm_211120    OA_SLVT_TT.db] \
    [list asap7sc7p5t_SEQ_SLVT_TT_nldm_220123.lib   asap7sc7p5t_SEQ_SLVT_TT_nldm_220123   SEQ_SLVT_TT.db] \
    [list asap7sc7p5t_SIMPLE_SLVT_TT_nldm_211120.lib asap7sc7p5t_SIMPLE_SLVT_TT_nldm_211120 SIMPLE_SLVT_TT.db] \
]

foreach it $items {
    set f   [lindex $it 0]
    set lib [lindex $it 1]
    set out [lindex $it 2]
    puts "INFO: Compiling $f ($lib) -> $out"
    read_lib [file join $LIB_DIR $f]
    write_lib $lib -format db -output [file join $DB_DIR $out]
    remove_lib -all
}
puts "INFO: Multi-Vt DB compilation complete."
exit
