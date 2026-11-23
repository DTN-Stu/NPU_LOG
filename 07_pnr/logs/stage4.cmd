#######################################################
#                                                     
#  Innovus Command Logging File                     
#  Created on Mon Nov 23 06:52:40 2026                
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
restoreDesign out/03_place.enc.dat snax_llm_top_clean
setMaxRouteLayer 6
setOptMode -usefulSkew false -allEndPoints true -fixHoldAllowSetupTnsDegrade false
setNanoRouteMode -drouteMinimizeLithoEffectOnLayer {f t t t t t t t t t} -routeTopRoutingLayer 5 -routeBottomRoutingLayer 2 -routeWithViaInPin true
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
