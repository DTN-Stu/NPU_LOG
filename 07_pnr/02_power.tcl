# =============================================================================
# Stage 2: ASAP7 Power Grid (sroute + m2followRail + M3..M6 addStripe) + Verify
# =============================================================================
setMultiCpuUsage -localCpu 8

# Route standard-cell followpins & macro VDD/VSS pins
sroute -connect { blockPin padPin padRing corePin floatingStripe } \
    -layerChangeRange { M1 Pad } \
    -blockPinTarget { nearestTarget } \
    -padPinPortConnect { allPort oneGeom } \
    -padPinTarget { nearestTarget } \
    -corePinTarget { firstAfterRowEnd } \
    -floatingStripeTarget { blockring padring ring stripe ringpin blockpin followpin } \
    -allowJogging 1 \
    -crossoverViaLayerRange { M1 Pad } \
    -nets { VDD VSS } \
    -allowLayerChange 1 \
    -blockPin useLef \
    -targetViaLayerRange { M1 }

# ASAP7 M1/M2 follow-rail overlay (unset scalar 'var' before & array 'var' after for Innovus 15.2 saveDesign)
catch {unset var}
source "/mnt/hgfs/NPU_LOG/lib/ASAP7/scripts/m2followRail.tcl"
catch {unset var}

setViaGenMode -viarule_preference { M6_M5widePWR1p152 M5_M4widePWR0p864 M4_M3widePWR0p864 }

# M3 vertical stripes
set m3pwrwidth [expr 0.072 * (5 + (4 * 2))]
set m3pwrset2setdist [expr 220 * 0.144]
set m3pwrspacing [expr 0.072 * 21]
set m3pwrxoffset [expr (0.072 * 26) + 0.036]

addStripe -extend_to design_boundary \
    -skip_via_on_wire_shape Noshape \
    -max_same_layer_jog_length 0.32 \
    -set_to_set_distance $m3pwrset2setdist \
    -skip_via_on_pin Standardcell \
    -stacked_via_top_layer Pad \
    -spacing $m3pwrspacing \
    -xleft_offset $m3pwrxoffset \
    -merge_stripes_value 0.16 \
    -layer M3 \
    -width $m3pwrwidth \
    -nets {VDD VSS} \
    -stacked_via_bottom_layer M2

# M4 horizontal stripes
set m4pwrwidth [expr 0.096 * (5 + (4 * 1))]
set m4pwrset2setdist [expr 320 * 0.192]
set m4pwrspacing [expr 0.096 * 21]
set m4pwrxoffset [expr (0.096 * 26) - 0.036]

setAddStripeMode \
    -max_via_size { Stripe 100 100 100 } \
    -via_using_exact_crossover_size true \
    -stacked_via_bottom_layer M3 \
    -stacked_via_top_layer M4

addStripe -extend_to design_boundary \
    -direction horizontal \
    -skip_via_on_wire_shape Noshape \
    -max_same_layer_jog_length 0.32 \
    -set_to_set_distance $m4pwrset2setdist \
    -skip_via_on_pin Standardcell \
    -spacing $m4pwrspacing \
    -start_offset $m4pwrxoffset \
    -merge_stripes_value 0.16 \
    -layer M4 \
    -width $m4pwrwidth \
    -nets {VDD VSS}

# M5 vertical stripes
set m5pwrwidth [expr 0.096 * (5 + (4 * 1))]
set m5pwrset2setdist [expr 320 * 0.192]
set m5pwrspacing [expr 0.096 * 21]
set m5pwrxoffset [expr (0.096 * 70) + 0.024]

setAddStripeMode -stacked_via_bottom_layer M4 \
    -stacked_via_top_layer M5

addStripe -extend_to design_boundary \
    -direction vertical \
    -skip_via_on_wire_shape Noshape \
    -max_same_layer_jog_length 0.32 \
    -set_to_set_distance $m5pwrset2setdist \
    -skip_via_on_pin Standardcell \
    -spacing $m5pwrspacing \
    -start_offset $m5pwrxoffset \
    -merge_stripes_value 0.16 \
    -layer M5 \
    -width $m5pwrwidth \
    -nets {VDD VSS}

# M6 horizontal stripes
set m6pwrwidth [expr 0.128 * (5 + (4 * 1))]
set m6pwrset2setdist [expr 220 * 0.256]
set m6pwrspacing [expr 0.128 * 21]
set m6pwrxoffset [expr (0.128 * 50) + 0.008]

setAddStripeMode -stacked_via_bottom_layer M5 \
    -stacked_via_top_layer M6

addStripe -extend_to design_boundary \
    -direction horizontal \
    -skip_via_on_wire_shape Noshape \
    -max_same_layer_jog_length 0.32 \
    -set_to_set_distance $m6pwrset2setdist \
    -skip_via_on_pin Standardcell \
    -spacing $m6pwrspacing \
    -start_offset $m6pwrxoffset \
    -merge_stripes_value 0.16 \
    -layer M6 \
    -width $m6pwrwidth \
    -nets {VDD VSS}

verifyConnectivity -type special -noAntenna -error 1000 -report reports/02_pwr_conn.rpt
verify_drc -limit 1000 -report reports/02_pwr_drc.rpt
saveDesign out/02_power.enc
puts "INFO: Stage 2 (02_power.tcl) COMPLETE!"
