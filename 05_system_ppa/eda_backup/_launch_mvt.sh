#!/bin/bash
set -e
cd ~/sys_syn

echo "=== [1] Kill existing RVT dc_shell ==="
for p in $(pgrep -f "common_shell_exec.*syn_sys_top" || true); do kill -9 "$p" || true; done
for p in $(pgrep -f "run_sys_syn_eda.sh" || true); do kill -9 "$p" || true; done
sleep 2

echo "=== [2] Compile LVT & SLVT .lib -> .db ==="
/EDA/Synopsys/libcompiler_v201806/bin/lc_shell -f compile_multivt_db.tcl > logs/lc_multivt.log 2>&1
ls -lh /home/ICer/asap7_multivt/db/

echo "=== [3] Relaunch Multi-Vt System Synthesis ==="
mkdir -p logs/prev
[ -f logs/syn_sys_top.log ] && mv logs/syn_sys_top.log logs/prev/syn_sys_top_RVT_killed.log
rm -f reports/snax_llm_top_clean_*
nohup setsid bash run_sys_syn_eda.sh > logs/run_sys_syn.out 2>&1 < /dev/null &
sleep 12
ps -eo pid,etime,cmd | grep -E "common_shell|run_sys_syn" | grep -v grep
grep -E "Loading target library|asap7_multivt" logs/syn_sys_top.log | head -n 20
