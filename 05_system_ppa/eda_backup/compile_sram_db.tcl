# =============================================================================
# compile_sram_db.tcl - Compile ASAP7 srambank_256x4x72_6t122.lib -> .db
# Run: lc_shell -f compile_sram_db.tcl
# =============================================================================
set BASE_DIR [file normalize [file dirname [info script]]]
set LIB_DIR  [file join $BASE_DIR sram_lib]
file mkdir $LIB_DIR

read_lib [file join $LIB_DIR srambank_256x4x72_6t122.lib]
write_lib srambank_256x4x72_6t122 -format db -output [file join $LIB_DIR srambank_256x4x72_6t122.db]
exit
