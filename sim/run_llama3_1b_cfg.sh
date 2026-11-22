#!/usr/bin/env bash
# ==============================================================================
# run_llama3_1b_cfg.sh
#
# Generate real NousResearch/Llama-3.2-1B layer stimulus via ~/qenv/bin/python3.12
# and run Verilator simulation with arbitrary runtime M_TOKENS and N_KV
# (100% 1D-DMA, ZERO cpu_mem, OOM-safe /tmp 2-stage build: -O0 TB / -O2 RTL).
#
# Usage:
#   ./run_llama3_1b_cfg.sh --mode prefill --m-tokens 32 --n-kv 32
#   ./run_llama3_1b_cfg.sh --mode prefill --m-tokens 32 --n-kv 128
#   ./run_llama3_1b_cfg.sh --mode decode  --n-kv 256 --dram-bp 1
#   ./run_llama3_1b_cfg.sh --skip-gen --rebuild ...
# ==============================================================================
set -euo pipefail

MODE="prefill"
M_TOKENS=""
N_KV=""
LAYER=1
Q_SUB=0
DRAM_BP=0
REBUILD=0
SKIP_GEN=0
DUMP=0
PY_BIN="${PY_BIN:-/home/nguyen/qenv/bin/python3.12}"
EXE="Vtb_llama3_1b_cfg"
WEIGHTS_DIR="sim_llama3_1b_weights"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)      MODE="$2";     shift 2 ;;
        --m-tokens)  M_TOKENS="$2"; shift 2 ;;
        --n-kv)      N_KV="$2";     shift 2 ;;
        --layer)     LAYER="$2";    shift 2 ;;
        --q-sub)     Q_SUB="$2";    shift 2 ;;
        --dram-bp)   DRAM_BP="$2";  shift 2 ;;
        --rebuild)   REBUILD=1;     shift ;;
        --skip-gen)  SKIP_GEN=1;    shift ;;
        --dump)      DUMP=1;        shift ;;
        *)           echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $DUMP -eq 1 ]]; then
    BUILD_DIR="/tmp/obj_llama3_cfg_fst"
    TRACE_FLAGS="--trace-fst -DFST_TRACE"
else
    BUILD_DIR="/tmp/obj_llama3_cfg"
    TRACE_FLAGS=""
fi

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

BIN="${BUILD_DIR}/${EXE}"

needs_build() {
    [[ $REBUILD -eq 1 ]] && return 0
    [[ ! -x "$BIN" ]] && return 0
    [[ "../tb/tb_llama3_1b_configurable.sv" -nt "$BIN" ]] && return 0
    [[ "filelist_llama3_1b_cfg.f" -nt "$BIN" ]] && return 0
    local newer
    newer="$(find "../01_snax_llm_fp4_subsystem" -type f -newer "$BIN" -print -quit 2>/dev/null || true)"
    [[ -n "$newer" ]] && return 0
    return 1
}

if needs_build; then
    echo "==> [Stage 1/3] Generating Verilator C++ in ${BUILD_DIR} ..."
    rm -rf "$BUILD_DIR"
    verilator --cc --main --exe --timing ${TRACE_FLAGS} \
        -Wno-fatal -f filelist_llama3_1b_cfg.f \
        --top tb_llama3_1b_configurable \
        --prefix "$EXE" \
        --Mdir "$BUILD_DIR" \
        -o "$EXE"

    echo "==> [Stage 2/3] Prebuilding PCH (-O2) & compiling TB coroutine object(s) at -O0 ..."
    make -C "$BUILD_DIR" -f "${EXE}.mk" OPT_FAST="-O2" OPT_SLOW="" "${EXE}__pch.h.fast.gch" "${EXE}__pch.h.slow.gch"
    TB_OBJS=()
    while IFS= read -r cpp; do
        TB_OBJS+=("$(basename "${cpp%.cpp}.o")")
    done < <(grep -rl '_eval_initial__TOP__Vtiming' "${BUILD_DIR}"/*.cpp | sort -u || true)
    if [[ ${#TB_OBJS[@]} -gt 0 ]]; then
        make -C "$BUILD_DIR" -f "${EXE}.mk" -j 4 OPT_FAST="-O0" OPT_SLOW="-O0" VK_PCH_I_FAST="" VK_PCH_I_SLOW="" "${TB_OBJS[@]}"
    fi

    echo "==> [Stage 3/3] Compiling RTL (OPT_FAST=-O2, -j 4) & linking ${EXE} ..."
    make -C "$BUILD_DIR" -f "${EXE}.mk" -j 4 OPT_FAST="-O2" OPT_SLOW=""
else
    echo "==> Using cached ${BIN} (skipping recompile)"
fi

PLUSARGS=(
    "+DECODE=${DECODE_FLAG}"
    "+M_TOKENS=${M_TOKENS}"
    "+N_KV=${N_KV}"
    "+DRAM_BP=${DRAM_BP}"
    "+DATA_DIR=${DATA_DIR}"
    "+WEIGHTS_HEX=${WEIGHTS_HEX}"
)
[[ $Q_SUB -gt 0 ]] && PLUSARGS+=("+Q_SUB=${Q_SUB}")
[[ $DUMP -eq 1 ]] && PLUSARGS+=("+DUMP")

echo "==> Running ${BIN} ${PLUSARGS[*]}"
"${BIN}" "${PLUSARGS[@]}"
