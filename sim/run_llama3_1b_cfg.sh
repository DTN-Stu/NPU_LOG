#!/usr/bin/env bash
# ==============================================================================
# run_llama3_1b_cfg.sh
#
# Generate real NousResearch/Llama-3.2-1B layer stimulus via ~/qenv/bin/python3.12
# and run Verilator simulation with arbitrary runtime M_TOKENS and N_KV.
#
# Usage:
#   ./run_llama3_1b_cfg.sh --mode prefill --m-tokens 32 --n-kv 32
#   ./run_llama3_1b_cfg.sh --mode prefill --m-tokens 32 --n-kv 128
#   ./run_llama3_1b_cfg.sh --mode decode  --n-kv 256
#   ./run_llama3_1b_cfg.sh --rebuild ...
# ==============================================================================
set -euo pipefail

MODE="prefill"
M_TOKENS=""
N_KV=""
LAYER=1
Q_SUB=0
REBUILD=0
SKIP_GEN=0
DUMP=0
PY_BIN="${PY_BIN:-/home/nguyen/qenv/bin/python3.12}"
BUILD_DIR="obj_dir_llama3_cfg"
EXE="Vtb_llama3_1b_cfg"
WEIGHTS_DIR="sim_llama3_1b_weights"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)      MODE="$2";     shift 2 ;;
        --m-tokens)  M_TOKENS="$2"; shift 2 ;;
        --n-kv)      N_KV="$2";     shift 2 ;;
        --layer)     LAYER="$2";    shift 2 ;;
        --q-sub)     Q_SUB="$2";    shift 2 ;;
        --rebuild)   REBUILD=1;     shift ;;
        --skip-gen)  SKIP_GEN=1;    shift ;;
        --dump)      DUMP=1;        shift ;;
        *)           echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$MODE" == "decode" ]]; then
    M_TOKENS=1
    N_KV="${N_KV:-64}"
    DECODE_FLAG=1
else
    M_TOKENS="${M_TOKENS:-16}"
    N_KV="${N_KV:-$M_TOKENS}"
    DECODE_FLAG=0
fi

DATA_DIR="sim_llama3_1b_${MODE}_m${M_TOKENS}_kv${N_KV}"
WEIGHTS_HEX="${WEIGHTS_DIR}/dram_weights_l${LAYER}.hex"

if [[ $SKIP_GEN -eq 0 ]]; then
    "$PY_BIN" ../tb/gen_llama3_1b_configurable.py \
        --mode "$MODE" \
        --m-tokens "$M_TOKENS" \
        --n-kv "$N_KV" \
        --layer "$LAYER" \
        --weights-dir "$WEIGHTS_DIR" \
        --out-dir "$DATA_DIR"
fi

if [[ $REBUILD -eq 1 || ! -x "${BUILD_DIR}/${EXE}" ]]; then
    echo "==> Building Verilator simulator (-CFLAGS -O3) in ${BUILD_DIR} ..."
    rm -rf "$BUILD_DIR"
    verilator --binary --timing -j 8 \
        -Wno-fatal -CFLAGS "-O3" \
        -f filelist_llama3_1b_cfg.f \
        --top tb_llama3_1b_configurable \
        --Mdir "$BUILD_DIR" \
        -o "$EXE"
fi

PLUSARGS=(
    "+DECODE=${DECODE_FLAG}"
    "+M_TOKENS=${M_TOKENS}"
    "+N_KV=${N_KV}"
    "+DATA_DIR=${DATA_DIR}"
    "+WEIGHTS_HEX=${WEIGHTS_HEX}"
)
[[ $Q_SUB -gt 0 ]] && PLUSARGS+=("+Q_SUB=${Q_SUB}")
[[ $DUMP -eq 1 ]] && PLUSARGS+=("+DUMP")

echo "==> Running ${BUILD_DIR}/${EXE} ${PLUSARGS[*]}"
"${BUILD_DIR}/${EXE}" "${PLUSARGS[@]}"
