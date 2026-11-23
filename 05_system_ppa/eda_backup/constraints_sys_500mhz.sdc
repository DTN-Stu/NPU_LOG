# =============================================================================
# constraints_sys_500mhz.sdc - Full-Subsystem SDC (500 MHz, ASAP7 1ps units)
# Target: snax_llm_top_clean (rst_n active-low)
# =============================================================================
set PERIOD_PS 2000.0

create_clock -name clk -period $PERIOD_PS [get_ports clk]
set_clock_uncertainty 50.0 [get_clocks clk]

# Clock-gating guard for ICGx4_ASAP7_75t_R inside rmw_latch_acc_buffer
set_clock_gating_check -setup 50.0 -hold 50.0 [current_design]

# Cap D-latch time borrowing in u_core/u_acc_buffer to 25% of period (500 ps)
set_max_time_borrow 500.0 [all_clocks]

set in_ports [remove_from_collection [all_inputs] [get_ports clk]]
set_input_transition 50.0 $in_ports
set_driving_cell -lib_cell NAND2x1_ASAP7_75t_R -pin Y $in_ports

set_input_delay  -clock clk -max 400.0 $in_ports
set_input_delay  -clock clk -min 0.0   $in_ports
set_output_delay -clock clk -max 400.0 [all_outputs]
set_output_delay -clock clk -min 0.0   [all_outputs]
set_load 0.02 [all_outputs]

set_false_path -from [get_ports rst_n]
