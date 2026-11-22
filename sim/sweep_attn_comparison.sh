#!/usr/bin/env bash
# ==============================================================================
# sweep_attn_comparison.sh
#
# Thermal-safe sequential runner for Attention Comparison Sweep.
# Runs ONE Verilator simulation at a time with nice -n 19 and saves CSV rows.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CSV_FILE="${CSV_FILE:-attn_comparison_results.csv}"
LOG_DIR="${LOG_DIR:-/tmp/attn_cmp_logs}"
STAGE=""
REBUILD_FIRST=0
INIT_CSV=0
RESUME=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)    STAGE="$2"; shift 2 ;;
        --rebuild)  REBUILD_FIRST=1; shift ;;
        --init-csv) INIT_CSV=1; shift ;;
        --no-resume) RESUME=0; shift ;;
        *)          echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

HEADER="mode,M,N_KV,DRAM_BW,cyc_hw,p2_sfu_est,cyc_total_w_sfu,cyc_mac,cyc_dma,dram_stalls,rd_bytes,wr_bytes,total_dram_bytes,sp_rd_bytes,sp_wr_bytes,mac_util_pct,sqnr_fq,cos_fq,sqnr_fp32,cos_fp32,max_abs_fp32,cpu_mem_cyc,max_inflight,checks_fail"

if [[ $INIT_CSV -eq 1 || ! -f "$CSV_FILE" ]]; then
    echo "$HEADER" > "$CSV_FILE"
fi

run_one() {
    local mode="$1"
    local m="$2"
    local n_kv="$3"
    local bw="$4"

    if [[ $RESUME -eq 1 ]] && grep -q "^${mode},${m},${n_kv},${bw}," "$CSV_FILE"; then
        echo "[SKIP] ${mode} M=${m} KV=${n_kv} BW=${bw} (already in ${CSV_FILE})"
        return 0
    fi

    local extra_args=("--mode" "$mode" "--m-tokens" "$m" "--n-kv" "$n_kv" "--dram-bw" "$bw" "--skip-gen")
    if [[ $REBUILD_FIRST -eq 1 ]]; then
        extra_args+=("--rebuild")
        REBUILD_FIRST=0
    fi

    local log_path="${LOG_DIR}/${mode}_m${m}_kv${n_kv}_bw${bw}.log"
    local t0 t1 dt
    t0="$(date +%s)"
    echo "------------------------------------------------------------------------"
    echo "[RUN ] ${mode} | M=${m}, N_KV=${n_kv}, BW=${bw} B/cyc"
    nice -n 19 ./run_attn_comparison.sh "${extra_args[@]}" | tee "$log_path"
    t1="$(date +%s)"
    dt=$((t1 - t0))

    local csv_line
    csv_line="$(grep -m1 '^CSV_ROW,' "$log_path" | sed 's/^CSV_ROW,//')"
    if [[ -z "$csv_line" ]]; then
        echo "[ERROR] Missing CSV_ROW in ${log_path}" >&2
        exit 1
    fi
    echo "$csv_line" >> "$CSV_FILE"
    echo "[DONE] ${mode} M=${m} KV=${n_kv} BW=${bw} (${dt}s) -> ${csv_line}"
}

ALL_MODES=(FLASH_KVPIN FLASH_RING ONLINE_2B ONLINE_2A NAIVE)
FAST4_MODES=(FLASH_KVPIN FLASH_RING ONLINE_2A NAIVE)

case "$STAGE" in
    smoke)
        for mode in "${ALL_MODES[@]}"; do
            run_one "$mode" 64 64 64
        done
        ;;
    A)
        for pair in "64 64" "128 128"; do
            read -r m n_kv <<< "$pair"
            for bw in 64 32 16; do
                for mode in "${ALL_MODES[@]}"; do
                    run_one "$mode" "$m" "$n_kv" "$bw"
                done
            done
        done
        ;;
    B)
        for bw in 64 32 16; do
            for mode in "${ALL_MODES[@]}"; do
                run_one "$mode" 256 256 "$bw"
            done
        done
        for n_kv in 256 512; do
            for mode in "${ALL_MODES[@]}"; do
                run_one "$mode" 64 "$n_kv" 16
            done
        done
        ;;
    C)
        for mode in "${ALL_MODES[@]}"; do
            run_one "$mode" 512 512 64
        done
        for mode in "${FAST4_MODES[@]}"; do
            run_one "$mode" 512 512 16
        done
        ;;
    D)
        for bw in 64 16; do
            for mode in "${FAST4_MODES[@]}"; do
                run_one "$mode" 1024 1024 "$bw"
            done
        done
        ;;
    *)
        echo "Usage: $0 --stage {smoke|A|B|C|D} [--rebuild] [--init-csv] [--no-resume]" >&2
        exit 1
        ;;
esac
