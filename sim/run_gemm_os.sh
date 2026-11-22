#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MDIR="/tmp/obj_gemm_os_all_sizes"
BIN="${MDIR}/Vtb_gemm_os_all_sizes"

TEST_SEL="ALL"
TEST_SET=0
REBUILD=0
EXTRA_ARGS=()

print_list() {
    cat <<'EOF'
Usage: ./run_gemm_os.sh [TEST] [OPTIONS]

Valid TEST values (default: ALL):
  ALL         Run all 16 subtests -> tb_gemm_os_all_sizes.fst
  1           Test 1:  Decode M=1, N=32, K=128 (K_TILES=4) -> tb_gemm_os_test_1.fst
  2           Test 2:  Prefill M=32, N=32, K=64 (2A..2E)   -> tb_gemm_os_test_2.fst
    2A        Test 2A: out_mode=0 (FP16), act linear layout
    2B        Test 2B: out_mode=0 (FP16), act 16-bank layout
    2C        Test 2C: out_mode=1 (FP4 linear)
    2D        Test 2D: out_mode=2 (FP4 16-bank scatter)
    2E        Test 2E: out_mode=3 (Fused V-Transpose + SLQ)
  3           Test 3:  Weight Ring 4x Reuse (M=64, N=64, K=128) -> tb_gemm_os_test_3.fst
  4           Test 4:  Macro-DMA Ping-Pong (4A + 4B)            -> tb_gemm_os_test_4.fst
    4A        Test 4A: Remainders M=48, N=48, K=1024
    4B        Test 4B: >288KB TCDM Scale M=128, N=64, K=4096 (32/32 tiles)
  5           Test 5:  Genuine DRAM + TCDM Backpressure (5A..5C)  -> tb_gemm_os_test_5.fst
    5A        Test 5A: Backpressure out_mode=0 (FP16)
    5B        Test 5B: Backpressure out_mode=2 (FP4 scatter)
    5C        Test 5C: Backpressure out_mode=3 (Fused V-Transpose)
  6           Test 6:  Unified FlashAttn + Decode V-Trans (6A..6C)-> tb_gemm_os_test_6.fst
    6A        Test 6A: FlashAttn KV-Pin + GQA (6A-1 + 6A-2)
    6A-1      Test 6A-1: Non-Causal FlashAttn KV-Pin + GQA
    6A-2      Test 6A-2: Chunked Causal FlashAttn (kv_past=16)
    6B        Test 6B: FlashAttn Macro-DMA Q/O + KV-Pin + Causal + 4D FP4
    6C        Test 6C: Hardware Decode V-Transpose (both layouts)

Options:
  --rebuild       Force Verilator recompile before running
  --no-wave       Pass +NO_WAVE (skip FST dump)
  --deep          Pass +WAVE_DEEP (dump full TB + DUT hierarchy)
  --wave FILE     Override output waveform path (+WAVE_FILE=FILE)
  --list, -h      Show this test list and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list|-h|--help)
            print_list
            exit 0
            ;;
        --rebuild)
            REBUILD=1
            shift
            ;;
        --no-wave)
            EXTRA_ARGS+=("+NO_WAVE")
            shift
            ;;
        --deep)
            EXTRA_ARGS+=("+WAVE_DEEP")
            shift
            ;;
        --wave)
            if [[ $# -lt 2 ]]; then
                echo "Error: --wave requires a filename argument" >&2
                exit 2
            fi
            EXTRA_ARGS+=("+WAVE_FILE=$2")
            shift 2
            ;;
        +*)
            EXTRA_ARGS+=("$1")
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            print_list >&2
            exit 2
            ;;
        *)
            if [[ "${TEST_SET}" -eq 1 ]]; then
                echo "Error: multiple TEST arguments given ('${TEST_SEL}' and '$1')" >&2
                exit 2
            fi
            TEST_SEL="$1"
            TEST_SET=1
            shift
            ;;
    esac
done

needs_build() {
    [[ "${REBUILD}" -eq 1 ]] && return 0
    [[ ! -x "${BIN}" ]] && return 0
    [[ "../tb/tb_gemm_os_all_sizes.sv" -nt "${BIN}" ]] && return 0
    [[ "filelist_gemm_os_all_sizes.f" -nt "${BIN}" ]] && return 0
    local newer
    newer="$(find "../01_snax_llm_fp4_subsystem" -type f -newer "${BIN}" -print -quit 2>/dev/null || true)"
    [[ -n "${newer}" ]] && return 0
    return 1
}

if needs_build; then
    echo "[BUILD] Compiling tb_gemm_os_all_sizes into ${MDIR} ..."
    verilator --binary --timing --trace-fst -DFST_TRACE -j 4 -Wno-fatal \
        -f filelist_gemm_os_all_sizes.f --top tb_gemm_os_all_sizes \
        --Mdir "${MDIR}" -o Vtb_gemm_os_all_sizes
else
    echo "[RUN] Using cached ${BIN} (skipping recompile)"
fi

exec "${BIN}" "+TEST=${TEST_SEL}" "${EXTRA_ARGS[@]}"
