# =============================================================================
# Optimal Synthesis / PnR SDC for snax_llm_top_clean (ASAP7 7.5T, 1ps / 1fF)
# Target: 600 MHz (1666.7 ps) — margin for 500 MHz PnR sign-off
# =============================================================================
if {![info exists synopsys_program_name]} {
    set sdc_version 1.7
    set_units -time 1.0ps -capacitance 1.0fF
}
current_design snax_llm_top_clean

# -----------------------------------------------------------------------------
# 1. Target Clock & Derived Budgets (ps / fF)
#    500 MHz = 2000.0 | 600 MHz = 1666.7 | 700 MHz = 1428.6 | 900 MHz = 1111.1
# -----------------------------------------------------------------------------
set CLK_PERIOD      1666.7
set CLK_HALF        [expr {$CLK_PERIOD * 0.5}]

# Setup uncertainty ~5% chu kỳ (min 50ps) cho skew + jitter + OCV pre-CTS
# Hold uncertainty giữ 20ps để tránh chèn thừa delay buffer ở synthesis
set SETUP_UNCERT    [expr {max(50.0, $CLK_PERIOD * 0.05)}]
set HOLD_UNCERT     20.0
set CLK_TRANS       25.0

# Max transition tự co theo tần số (ASAP7 NLDM index_1 = 5..320 ps)
set MAX_TRAN        [expr {min(100.0, max(60.0, $CLK_PERIOD * 0.07))}]
set MAX_FANOUT      48

create_clock -name clk -period $CLK_PERIOD -waveform [list 0.0 $CLK_HALF] [get_ports clk]
set_clock_uncertainty -setup $SETUP_UNCERT [get_clocks clk]
set_clock_uncertainty -hold  $HOLD_UNCERT  [get_clocks clk]
set_clock_transition  $CLK_TRANS           [get_clocks clk]

set_dont_touch_network [get_ports {clk rst_n}]
set_ideal_network      [get_ports rst_n]

# -----------------------------------------------------------------------------
# 2. Port Grouping (khớp 100% port list của snax_llm_top_clean.sv)
# -----------------------------------------------------------------------------
set clk_rst_ports [get_ports {clk rst_n}]
set all_data_in   [remove_from_collection [all_inputs] $clk_rst_ports]

set in_tcdm [get_ports { \
    cpu_mem_req_valid_i cpu_mem_req_addr_i* cpu_mem_req_write_i cpu_mem_req_wdata_i* cpu_mem_req_strb_i* \
    dma_mem_req_valid_i dma_mem_req_addr_i* dma_mem_req_write_i dma_mem_req_wdata_i* dma_mem_req_strb_i* \
    dma_aux_mem_req_valid_i* dma_aux_mem_req_addr_i* dma_aux_mem_req_write_i* dma_aux_mem_req_wdata_i* dma_aux_mem_req_strb_i* \
}]
set out_tcdm_rdy [get_ports { \
    cpu_mem_req_ready_o dma_mem_req_ready_o dma_aux_mem_req_ready_o* \
}]
set out_tcdm_rsp [get_ports { \
    cpu_mem_rsp_valid_o cpu_mem_rsp_rdata_o* \
    dma_mem_rsp_valid_o dma_mem_rsp_rdata_o* \
    dma_aux_mem_rsp_valid_o* dma_aux_mem_rsp_rdata_o* \
}]

set in_csr [get_ports { \
    cpu_csr_req_valid_i cpu_csr_req_write_i cpu_csr_req_addr_i* cpu_csr_req_data_i* cpu_csr_rsp_ready_i \
}]
set out_csr [get_ports { \
    cpu_csr_req_ready_o cpu_csr_rsp_valid_o cpu_csr_rsp_data_o* \
}]

set in_dma_ctrl [get_ports {dma_desc_ready_i dma_desc_done_i}]
set out_dma_desc [get_ports { \
    dma_desc_valid_o dma_desc_dir_o* dma_desc_dram_addr_o* dma_desc_tcdm_addr_o* dma_desc_size_o* \
}]

set out_status [get_ports {subsystem_busy_o subsystem_done_irq_o}]

# -----------------------------------------------------------------------------
# 3. I/O Timing Budgets
# -----------------------------------------------------------------------------
set D_TCDM_IN   [expr {$CLK_PERIOD * 0.15}]
set D_TCDM_RDY  [expr {$CLK_PERIOD * 0.15}]
set D_TCDM_RSP  [expr {$CLK_PERIOD * 0.15}]
set D_CTRL_IN   [expr {$CLK_PERIOD * 0.20}]
set D_CTRL_OUT  [expr {$CLK_PERIOD * 0.20}]
set D_STAT_OUT  [expr {$CLK_PERIOD * 0.15}]

set_input_delay  -clock clk -max $D_TCDM_IN  $in_tcdm
set_input_delay  -clock clk -min 15.0        $in_tcdm
set_input_delay  -clock clk -max $D_CTRL_IN  $in_csr
set_input_delay  -clock clk -min 15.0        $in_csr
set_input_delay  -clock clk -max $D_CTRL_IN  $in_dma_ctrl
set_input_delay  -clock clk -min 15.0        $in_dma_ctrl

set_output_delay -clock clk -max $D_TCDM_RDY $out_tcdm_rdy
set_output_delay -clock clk -min -10.0       $out_tcdm_rdy
set_output_delay -clock clk -max $D_TCDM_RSP $out_tcdm_rsp
set_output_delay -clock clk -min -10.0       $out_tcdm_rsp
set_output_delay -clock clk -max $D_CTRL_OUT $out_csr
set_output_delay -clock clk -min -10.0       $out_csr
set_output_delay -clock clk -max $D_CTRL_OUT $out_dma_desc
set_output_delay -clock clk -min -10.0       $out_dma_desc
set_output_delay -clock clk -max $D_STAT_OUT $out_status
set_output_delay -clock clk -min -10.0       $out_status

# -----------------------------------------------------------------------------
# 4. Driving Cell, Output Load & DRC Limits (ASAP7 RVT baseline)
# -----------------------------------------------------------------------------
set_driving_cell -lib_cell BUFx4_ASAP7_75t_R -pin Y $all_data_in
set_load -pin_load 5.0 [all_outputs]

set_max_transition  $MAX_TRAN   [current_design]
set_max_fanout      $MAX_FANOUT [current_design]

set_clock_gating_check -setup 25.0 -hold 10.0 [get_clocks clk]
set_max_time_borrow [expr {$CLK_PERIOD * 0.25}] [all_clocks]

# -----------------------------------------------------------------------------
# 5. Silicon-Safe Timing Exceptions
# -----------------------------------------------------------------------------
set_false_path -from [get_ports rst_n]

set csr_cfg_regs [add_to_collection \
    [get_cells -quiet -hierarchical *sb_cfg_q*reg*] \
    [get_cells -quiet -hierarchical *act_cfg_q*reg*]]

if {[sizeof_collection $csr_cfg_regs] > 0} {
    set_multicycle_path -setup 2 -from $csr_cfg_regs
    set_multicycle_path -hold  1 -from $csr_cfg_regs
}
puts "INFO: SDC loaded (CLK_PERIOD=$CLK_PERIOD ps), CSR_CFG_REGS_COUNT=[sizeof_collection $csr_cfg_regs]"
