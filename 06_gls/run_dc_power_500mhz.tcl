# =============================================================================
# run_dc_power_500mhz.tcl
# Evaluate 500 MHz active-window SAIF power on the 600 MHz-synthesized
# snax_llm_top_clean gate-level netlist (ASAP7 7nm TT 0.7V 25C).
# Strictly follows 08_system_models/SYSTEM_ENERGY_EXPERIMENT_PLAN.md (WS2A).
# =============================================================================

set DB_ASAP7 "/home/ICer/vsq_syn/syn/libs/db/asap7"
set DB_MVT   "/home/ICer/asap7_multivt/db"
set SRAM_DB  "/home/ICer/sys_syn/sram_lib/srambank_256x4x72_6t122.db"
set CELL_DBS [list \
    $DB_ASAP7/AO_RVT_TT.db $DB_ASAP7/OA_RVT_TT.db $DB_ASAP7/SIMPLE_RVT_TT.db $DB_ASAP7/INVBUF_RVT_TT.db $DB_ASAP7/SEQ_RVT_TT.db \
    $DB_MVT/AO_LVT_TT.db   $DB_MVT/OA_LVT_TT.db   $DB_MVT/SIMPLE_LVT_TT.db   $DB_MVT/INVBUF_LVT_TT.db   $DB_MVT/SEQ_LVT_TT.db \
    $DB_MVT/AO_SLVT_TT.db  $DB_MVT/OA_SLVT_TT.db  $DB_MVT/SIMPLE_SLVT_TT.db  $DB_MVT/INVBUF_SLVT_TT.db  $DB_MVT/SEQ_SLVT_TT.db]

set_app_var target_library $CELL_DBS
set_app_var link_library   [concat [list "*"] $CELL_DBS [list $SRAM_DB]]

set NETLIST "/home/ICer/sys_syn/output_netlist_bu/snax_llm_top_clean.v"
set SDC_500 "/home/ICer/sys_gls_500m/constraints_sys_500mhz.sdc"
set OUT_DIR "/home/ICer/sys_gls_500m/reports_500mhz"
file mkdir $OUT_DIR

read_verilog $NETLIST
current_design snax_llm_top_clean
link

# Apply 500 MHz (2000 ps) evaluation SDC so clk matches SAIF (0.001000 toggles/ps, zero PWR-12)
source $SDC_500

proc eval_saif_workload {name saif_file inst_path out_dir} {
    if {![file exists $saif_file]} {
        puts "WARN: SAIF not found for $name: $saif_file (skipping)"
        return
    }
    puts "============================================================"
    puts "Evaluating 500 MHz SAIF Power: $name"
    puts "  SAIF: $saif_file  Instance: $inst_path"
    puts "============================================================"
    reset_switching_activity
    read_saif -input $saif_file -instance_name $inst_path -verbose

    redirect -file "${out_dir}/power_500m_${name}_summary.rpt" {
        puts "=== Workload: $name (500 MHz, 2000 ps) ==="
        puts "--- Clock Switching Activity Verification ---"
        catch { puts "clk toggle_rate = [get_attribute [get_nets clk] toggle_rate]  static_prob = [get_attribute [get_nets clk] static_probability]" }
        catch { report_saif }
        puts "--- Total Power Summary ---"
        report_power -analysis_effort medium -nosplit
    }
    redirect -file "${out_dir}/power_500m_${name}_hier.rpt" {
        puts "=== Workload: $name Hierarchical Power (500 MHz, 2000 ps) ==="
        report_power -hier -analysis_effort medium -nosplit
    }
    puts "Done $name -> ${out_dir}/power_500m_${name}_hier.rpt"
}

eval_saif_workload "gemm" "/home/ICer/sys_gls_500m/gls_500m_gemm.saif" "tb_gemm_os_gls/dut"    $OUT_DIR
eval_saif_workload "attn" "/home/ICer/sys_gls_500m/gls_500m_attn.saif" "tb_gemm_os_gls/dut"    $OUT_DIR
eval_saif_workload "sfu"  "/home/ICer/sys_gls_500m/gls_500m_sfu.saif"  "tb_sfu_gls_500mhz/dut" $OUT_DIR

exit
