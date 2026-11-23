#!/bin/bash
set -e
cd ~/sys_syn
DC=/EDA/Synopsys/syn_v201806/bin/dc_shell
mkdir -p logs_bu reports_bu ddc_bu logs reports output

echo "=== [STAGE 1A] Sub-Block Synthesis Batch 1 (u_core, u_seq, u_sfu) ==="
$DC -f syn_subblock.tcl -x 'set BLK_TAG u_sfu;  set TOP_MOD snax_llm_sfu_opt_pipe; set PARAMS "NUM_LANES=16,ALU_LANES=8"' > logs_bu/u_sfu.log 2>&1 &
$DC -f syn_subblock.tcl -x 'set BLK_TAG u_seq;  set TOP_MOD snax_llm_sequencer;    set PARAMS "ADDR_WIDTH=32,NUM_LANES=16,NUM_RING_SLOTS=4,SLOT_SIZE_BYTES=131072,OUT_PIPE_CREDITS=4,CAP_O_TILES=16"' > logs_bu/u_seq.log 2>&1 &
$DC -f syn_subblock.tcl -x 'set BLK_TAG u_core; set TOP_MOD snax_fused_llm_core;   set PARAMS "NUM_LANES_P=16,ACC_DEPTH_P=16,NUM_QUERIES=8,PKT_W=72,FRAC_BITS=16,P_FIFO_DEPTH=16,USE_OUTPUT_READY=1"' > logs_bu/u_core.log 2>&1 &
wait

echo "=== [STAGE 1B] Sub-Block Synthesis Batch 2 (u_write_out, u_wt_streamer, u_csr) ==="
$DC -f syn_subblock.tcl -x 'set BLK_TAG u_write_out;   set TOP_MOD snax_streamer_write_out; set PARAMS "DATA_WIDTH=72,ADDR_WIDTH=32,NUM_LANES=16,STRB_WIDTH=9"' > logs_bu/u_write_out.log 2>&1 &
$DC -f syn_subblock.tcl -x 'set BLK_TAG u_wt_streamer; set TOP_MOD snax_streamer_wt_16ch;   set PARAMS "DATA_WIDTH=72,ADDR_WIDTH=32,NUM_LANES=16,STRB_WIDTH=9"' > logs_bu/u_wt_streamer.log 2>&1 &
$DC -f syn_subblock.tcl -x 'set BLK_TAG u_csr;         set TOP_MOD snax_flash_csr_snax;     set PARAMS "ENABLE_LEGACY_DIRECT_CSR=0,DOUBLE_BUFFER=0"' > logs_bu/u_csr.log 2>&1 &
wait

echo "=== [STAGE 1 SUMMARY] ==="
for r in reports_bu/*_qor.rpt; do
    echo ">>> $r"
    grep -E "Critical Path Slack|Critical Path Length|Levels of Logic|Total Negative Slack|Cell Area:" "$r"
done

echo "=== [STAGE 2] Top-Level Incremental Integration ==="
$DC -f syn_sys_bottomup.tcl > logs/syn_sys_bottomup.log 2>&1
echo "BOTTOMUP_ALL_DONE"
