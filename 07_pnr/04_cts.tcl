# =============================================================================
# Stage 4: Clock Tree Synthesis (CCOpt) & Post-CTS Optimization (500 MHz)
# Restores out/03_place.enc.dat -> ccopt_design -> optDesign -postCTS (setup/hold)
# =============================================================================
setMultiCpuUsage -localCpu 8
setLibraryUnit -time 1ps -cap 1ff

restoreDesign out/03_place.enc.dat snax_llm_top_clean

setMaxRouteLayer 6
setOptMode -usefulSkew false \
    -allEndPoints true \
    -fixHoldAllowSetupTnsDegrade false

# ASAP7 CTS NanoRoute & CCOpt settings
setNanoRouteMode -drouteMinimizeLithoEffectOnLayer {f t t t t t t t t t} \
    -routeTopRoutingLayer 5 \
    -routeBottomRoutingLayer 2 \
    -routeWithViaInPin true

set_ccopt_property target_skew 5ps
set_ccopt_property target_max_trans 30ps
create_route_type -name ccopt_route_group -bottom_preferred_layer 4 -top_preferred_layer 5
create_ccopt_clock_tree_spec
ccopt_design -cts

optDesign -postCTS
optDesign -postCTS -hold

timeDesign -postCTS -outDir reports/04_postCTS_setup
timeDesign -postCTS -hold -outDir reports/04_postCTS_hold
report_ccopt_clock_trees -file reports/04_cts_trees.rpt
report_ccopt_skew_groups -file reports/04_cts_skew.rpt

saveDesign out/04_cts.enc
puts "INFO: Stage 4 (04_cts.tcl) COMPLETE!"
exit
