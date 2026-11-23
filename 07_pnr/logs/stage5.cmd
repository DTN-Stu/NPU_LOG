#######################################################
#                                                     
#  Innovus Command Logging File                     
#  Created on Mon Nov 23 07:34:41 2026                
#                                                     
#######################################################

#@(#)CDS: Innovus v15.20-p005_1 (64bit) 11/11/2015 11:16 (Linux 2.6.18-194.el5)
#@(#)CDS: NanoRoute 15.20-p005_1 NR151028-1715/15_20-UB (database version 2.30, 298.6.1) {superthreading v1.26}
#@(#)CDS: AAE 15.20-p002 (64bit) 11/11/2015 (Linux 2.6.18-194.el5)
#@(#)CDS: CTE 15.20-p001_1 () Oct 29 2015 01:50:39 ( )
#@(#)CDS: SYNTECH 15.20-b002_1 () Oct 20 2015 02:35:29 ( )
#@(#)CDS: CPE v15.20-p002
#@(#)CDS: IQRC/TQRC 15.1.2-s269 (64bit) Mon Aug 24 18:22:18 PDT 2015 (Linux 2.6.18-194.el5)

set_global _enable_mmmc_by_default_flow      $CTE::mmmc_default
suppressMessage ENCEXT-2799
getVersion
getVersion
setMultiCpuUsage -localCpu 8
setLibraryUnit -time 1ps -cap 1ff
restoreDesign out/04_cts.enc.dat snax_llm_top_clean
setMaxRouteLayer 6
setOptMode -usefulSkew false -allEndPoints true -fixHoldAllowSetupTnsDegrade false
setNanoRouteMode -drouteMinimizeLithoEffectOnLayer {t t t t t t t t t t}
setNanoRouteMode -routeWithViaInPin true -routeDesignFixClockNets true -routeTopRoutingLayer 6 -routeBottomRoutingLayer 2
routeDesign
optDesign -postRoute
saveDesign out/05_routed.enc
setDelayCalMode -SIAware false
setOptMode -siAware false
optDesign -postRoute
optDesign -postRoute -hold
addFiller -cell DECAPx10_ASAP7_75t_R -prefix FILLER_DECAP_
addFiller -cell DECAPx6_ASAP7_75t_R -prefix FILLER_DECAP_
addFiller -cell DECAPx4_ASAP7_75t_R -prefix FILLER_DECAP_
addFiller -cell TAPCELL_ASAP7_75t_R -prefix FILLER_TAP_
addFiller -cell {FILLER_ASAP7_75t_R FILLERxp5_ASAP7_75t_R} -prefix FILLER_
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
