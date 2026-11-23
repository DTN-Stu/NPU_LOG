# =============================================================================
# Stage 3: Placement & Pre-CTS Optimization (500 MHz target)
# Restores out/02_power.enc.dat -> place_opt_design -> timeDesign -preCTS
# =============================================================================
setMultiCpuUsage -localCpu 8
setLibraryUnit -time 1ps -cap 1ff
restoreDesign out/02_power.enc.dat snax_llm_top_clean

setMaxRouteLayer 6
setOptMode -usefulSkew false \
    -allEndPoints true \
    -fixHoldAllowSetupTnsDegrade false

# ASAP7 pre-CTS NanoRoute litho settings
setNanoRouteMode -drouteMinimizeLithoEffectOnLayer {f t t t t t t t t t} \
    -routeTopRoutingLayer 5 \
    -routeBottomRoutingLayer 2 \
    -routeWithViaInPin true

place_opt_design

checkPlace reports/03_place_check.rpt
timeDesign -preCTS -outDir reports/03_preCTS_timing
report_area > reports/03_place_area.rpt

saveDesign out/03_place.enc
puts "INFO: Stage 3 (03_place.tcl) COMPLETE!"
exit
