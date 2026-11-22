#!/bin/bash
set -e

DC=/EDA/Synopsys/syn_v201806/bin/dc_shell
VCS=/EDA/Synopsys/vcsmx_v201809/bin/vcs

mkdir -p logs sim reports output

if [ ! -e asap7_v ]; then
    ln -s /home/ICer/ablation_test/sim/asap7_v asap7_v
fi

# Format: TOP_NAME|TB_DEF|FMT_HEX
CONFIGS=(
    "cfg1_mxfp4_g16_fp32_ff|DUT_CFG1|mxfp4"
    "cfg2_nvfp4_fp32_ff|DUT_CFG2|nvfp4"
    "cfg6_log8_fp32_ff|DUT_CFG6|log8"
    "cfg4_log8_int32_lazy_ff|DUT_CFG4|log8"
    "cfg5_log8_int32_lazy_latch|DUT_CFG5|log8"
)

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r TOP DEF FMT <<< "$entry"
    echo "=========================================="
    echo "[1/3] Synthesis: $TOP"
    echo "=========================================="
    if [ ! -f "output/${TOP}.ddc" ]; then
        $DC -f syn_cfg_paper.tcl -x "set DESIGN_NAME $TOP" > "logs/syn_${TOP}.log" 2>&1
    else
        echo "  -> Skipping synthesis (output/${TOP}.ddc already exists)"
    fi

    echo "[2/3] GLS + SAIF (real + uniform): $TOP"
    if [ ! -f "sim/${TOP}_real.saif" ]; then
        cp "stim/stim_${FMT}_act.hex" stim_act.hex
        cp "stim/stim_${FMT}_wt.hex"  stim_wt.hex
        $VCS -sverilog -full64 +v2k -timescale=1ns/10ps +neg_tchk \
            asap7_v/asap7sc7p5t_AO_RVT_TT_201020.v \
            asap7_v/asap7sc7p5t_INVBUF_RVT_TT_201020.v \
            asap7_v/asap7sc7p5t_OA_RVT_TT_201020.v \
            asap7_v/asap7sc7p5t_SEQ_RVT_TT_220101.v \
            asap7_v/asap7sc7p5t_SIMPLE_RVT_TT_201020.v \
            "output/${TOP}_netlist.v" \
            tb_ablation_saif.sv \
            +define+${DEF} +define+DUMP_SAIF +define+STIM_LLM_HEX \
            -o "simv_${TOP}_real" > "logs/vcs_${TOP}_real.log" 2>&1
        "./simv_${TOP}_real" > "logs/sim_${TOP}_real.log" 2>&1
        mv ablation_dut.saif "sim/${TOP}_real.saif"
    fi

    if [ ! -f "sim/${TOP}_uniform.saif" ]; then
        $VCS -sverilog -full64 +v2k -timescale=1ns/10ps +neg_tchk \
            asap7_v/asap7sc7p5t_AO_RVT_TT_201020.v \
            asap7_v/asap7sc7p5t_INVBUF_RVT_TT_201020.v \
            asap7_v/asap7sc7p5t_OA_RVT_TT_201020.v \
            asap7_v/asap7sc7p5t_SEQ_RVT_TT_220101.v \
            asap7_v/asap7sc7p5t_SIMPLE_RVT_TT_201020.v \
            "output/${TOP}_netlist.v" \
            tb_ablation_saif.sv \
            +define+${DEF} +define+DUMP_SAIF +define+STIM_UNIFORM \
            -o "simv_${TOP}_uniform" > "logs/vcs_${TOP}_uniform.log" 2>&1
        "./simv_${TOP}_uniform" > "logs/sim_${TOP}_uniform.log" 2>&1
        mv ablation_dut.saif "sim/${TOP}_uniform.saif"
    fi

    echo "[3/3] SAIF Power Analysis: $TOP"
    $DC -f read_saif_paper.tcl -x "set DESIGN_NAME $TOP; set STIM_TAG real"    > "logs/pwr_${TOP}_real.log" 2>&1
    $DC -f read_saif_paper.tcl -x "set DESIGN_NAME $TOP; set STIM_TAG uniform" > "logs/pwr_${TOP}_uniform.log" 2>&1
done

echo "ALL_DONE"
