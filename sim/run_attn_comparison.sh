#!/usr/bin/env bash
# ==============================================================================
# run_attn_comparison.sh
#
# Build ONCE in /tmp/obj_attn_cmp (-O2 PCH, -O0 TB coroutine, -O2 RTL) and run
# Attention Architecture Comparison (NAIVE vs ONLINE_2A vs ONLINE_2B vs
# FLASH_RING vs FLASH_KVPIN) on real Llama-3.2-1B Layer 1 tensors via 100% 1D-DMA.
# ==============================================================================
set -euo pipefail

MODE="FLASH_KVPIN"
M_TOKENS=64
N_KV=""
DRAM_BW=64
LAYER=1
REBUILD=0
SKIP_GEN=0
DUMP=0
PY_BIN="${PY_BIN:-/home/nguyen/qenv/bin/python3.12}"
EXE="Vtb_attn_comparison"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)      MODE="$2";     shift 2 ;;
        --m-tokens)  M_TOKENS="$2"; shift 2 ;;
        --n-kv)      N_KV="$2";     shift 2 ;;
        --dram-bw)   DRAM_BW="$2";  shift 2 ;;
        --dram-bp)   DRAM_BW="$2";  shift 2 ;;
        --layer)     LAYER="$2";    shift 2 ;;
        --rebuild)   REBUILD=1;     shift ;;
        --skip-gen)  SKIP_GEN=1;    shift ;;
        --dump)      DUMP=1;        shift ;;
        *)           echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

N_KV="${N_KV:-$M_TOKENS}"
DATA_DIR="sim_attn_cmp_m${M_TOKENS}_kv${N_KV}"

if [[ $DUMP -eq 1 ]]; then
    BUILD_DIR="/tmp/obj_attn_cmp_fst"
    TRACE_FLAGS="--trace-fst -DFST_TRACE"
else
    BUILD_DIR="/tmp/obj_attn_cmp"
    TRACE_FLAGS=""
fi

if [[ $SKIP_GEN -eq 0 ]]; then
    "$PY_BIN" ../tb/gen_attn_comparison.py \
        --m-tokens "$M_TOKENS" \
        --n-kv "$N_KV" \
        --layer "$LAYER" \
        --out-dir "$DATA_DIR"
fi

BIN="${BUILD_DIR}/${EXE}"

needs_build() {
    [[ $REBUILD -eq 1 ]] && return 0
    [[ ! -x "$BIN" ]] && return 0
    [[ "../tb/tb_attn_comparison.sv" -nt "$BIN" ]] && return 0
    [[ "filelist_attn_comparison.f" -nt "$BIN" ]] && return 0
    local newer
    newer="$(find "../01_snax_llm_fp4_subsystem" -type f -newer "$BIN" -print -quit 2>/dev/null || true)"
    [[ -n "$newer" ]] && return 0
    return 1
}

if needs_build; then
    echo "==> [Stage 1/3] Generating Verilator C++ in ${BUILD_DIR} ..."
    rm -rf "$BUILD_DIR"
    verilator --cc --main --exe --timing ${TRACE_FLAGS} \
        -Wno-fatal -f filelist_attn_comparison.f \
        --top tb_attn_comparison \
        --prefix "$EXE" \
        --Mdir "$BUILD_DIR" \
        -o "$EXE"

    echo "==> [Stage 2/3] Prebuilding PCH (-O2) & compiling TB coroutine(s) at -O0 ..."
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
    "+MODE=${MODE}"
    "+M_TOKENS=${M_TOKENS}"
    "+N_KV=${N_KV}"
    "+DRAM_BW=${DRAM_BW}"
    "+DATA_DIR=${DATA_DIR}"
)
[[ $DUMP -eq 1 ]] && PLUSARGS+=("+DUMP")

echo "==> Running ${BIN} ${PLUSARGS[*]}"
"${BIN}" "${PLUSARGS[@]}"
