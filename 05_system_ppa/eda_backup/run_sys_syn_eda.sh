#!/bin/bash
set -e

LC=/EDA/Synopsys/libcompiler_v201806/bin/lc_shell
DC=/EDA/Synopsys/syn_v201806/bin/dc_shell

mkdir -p logs reports output

if [ ! -f sram_lib/srambank_256x4x72_6t122.db ]; then
    echo "[1/2] Compiling SRAM Liberty -> DB via lc_shell..."
    $LC -f compile_sram_db.tcl > logs/lc_sram.log 2>&1
fi

echo "[2/2] Running Full-Subsystem Synthesis (snax_llm_top_clean)..."
$DC -f syn_sys_top.tcl > logs/syn_sys_top.log 2>&1
echo "SYSTEM_SYN_DONE"
