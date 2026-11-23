# =============================================================================
# Stage 5: Detailed Routing, Post-Route Opt, Decap/Filler & Signoff Exports
# Restores out/04_cts.enc.dat -> routeDesign -> optDesign -> Filler -> RC/DRC
# =============================================================================
setMultiCpuUsage -localCpu 8
setLibraryUnit -time 1ps -cap 1ff

restoreDesign out/04_cts.enc.dat snax_llm_top_clean

setMaxRouteLayer 6
setOptMode -usefulSkew false \
    -allEndPoints true \
    -fixHoldAllowSetupTnsDegrade false

# ASAP7 post-CTS NanoRoute settings
setNanoRouteMode -drouteMinimizeLithoEffectOnLayer {t t t t t t t t t t}
setNanoRouteMode -routeWithViaInPin true \
    -routeDesignFixClockNets true \
    -routeTopRoutingLayer 6 \
    -routeBottomRoutingLayer 2

routeDesign
saveDesign out/05_routed.enc

setDelayCalMode -SIAware false
catch {setOptMode -siAware false}
optDesign -postRoute
optDesign -postRoute -hold

# ASAP7 Decap, Tap & Filler sequence
addFiller -cell {DECAPx10_ASAP7_75t_R} -prefix FILLER_DECAP_
addFiller -cell {DECAPx6_ASAP7_75t_R}  -prefix FILLER_DECAP_
addFiller -cell {DECAPx4_ASAP7_75t_R}  -prefix FILLER_DECAP_
addFiller -cell {TAPCELL_ASAP7_75t_R}  -prefix FILLER_TAP_
addFiller -cell {FILLER_ASAP7_75t_R FILLERxp5_ASAP7_75t_R} -prefix FILLER_

# Post-route RC extraction & Signoff timing/power/DRC
setExtractRCMode -engine postRoute -effortLevel medium
extractRC
rcOut -spef out/snax_llm_top_clean_500m.spef

timeDesign -postRoute -outDir reports/05_postRoute_setup
timeDesign -postRoute -hold -outDir reports/05_postRoute_hold
report_area > reports/05_final_area.rpt
report_power > reports/05_final_power.rpt

verifyConnectivity -type all -noAntenna -error 1000 -report reports/05_final_conn.rpt
verify_drc -limit 1000 -report reports/05_final_drc.rpt

defOut -floorplan -netlist -routing out/snax_llm_top_clean_500m.def.gz
saveNetlist out/snax_llm_top_clean_pnr.v
saveDesign out/05_final.enc
puts "INFO: Stage 5 (05_route.tcl) COMPLETE!"
exit
