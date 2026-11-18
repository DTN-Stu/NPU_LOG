#!/usr/bin/env bash
# ============================================================
#  run.sh - Chay mo phong Verilator tu mot filelist (.f)
#
#  Cach dung co ban:
#      ./run.sh                        # dung filelist.f
#      ./run.sh -f rtl/filelist.f
#      ./run.sh -f filelist.f -t tb    # chi dinh top module
#      ./run.sh --wave                 # chay xong mo Surfer luon
#
#  Xem day du:  ./run.sh -h
# ============================================================
set -uo pipefail

FILELIST="filelist.f"
TOP=""
EXE="sim"
BUILD_DIR="obj_dir"
JOBS="0"
TRACE=1
FST=0
OPEN_WAVE=0
CLEAN=0
VERBOSE=0
EXTRA=()

if [[ -t 1 ]]; then
    R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; C=$'\033[36m'; B=$'\033[1m'; N=$'\033[0m'
else
    R=""; G=""; Y=""; C=""; B=""; N=""
fi
info() { printf '%s==>%s %s\n' "$C" "$N" "$*"; }
ok()   { printf '%s[ok]%s %s\n' "$G" "$N" "$*"; }
warn() { printf '%s[!]%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '%s[ERROR]%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

usage() {
cat <<'EOF'
run.sh - Chay mo phong Verilator tu filelist

  -f, --filelist <file>   File filelist (mac dinh: filelist.f)
  -t, --top <module>      Top module (tu dong doan neu bo trong)
  -o, --exe <name>        Ten file thuc thi (mac dinh: sim)
  -j, --jobs <n>          So job build song song (mac dinh: 0 = auto)
      --no-trace          Tat xuat waveform (build nhanh hon)
      --fst               Dung dinh dang FST thay vi VCD (nho va nhanh hon)
      --wave              Mo Surfer voi waveform sau khi chay
      --clean             Xoa build dir truoc khi build
  -v, --verbose           In them thong tin
  -h, --help              Hien tro giup

Moi tham so khac duoc chuyen thang cho verilator, vi du:
  ./run.sh -f filelist.f -- -Wno-WIDTH --trace-depth 50 -CFLAGS -O3

Vi du:
  ./run.sh
  ./run.sh -f rtl.f -t uart_tb --wave
  ./run.sh --clean -j 8
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--filelist) FILELIST="${2:-}"; shift 2 || die "Thieu gia tri cho $1" ;;
        -t|--top)      TOP="${2:-}";      shift 2 || die "Thieu gia tri cho $1" ;;
        -o|--exe)      EXE="${2:-}";      shift 2 || die "Thieu gia tri cho $1" ;;
        -j|--jobs)     JOBS="${2:-}";     shift 2 || die "Thieu gia tri cho $1" ;;
        --no-trace)    TRACE=0;           shift ;;
        --fst)         TRACE=1; FST=1;    shift ;;
        --wave)        OPEN_WAVE=1;       shift ;;
        --clean)       CLEAN=1;           shift ;;
        -v|--verbose)  VERBOSE=1;         shift ;;
        -h|--help)     usage; exit 0 ;;
        --)            shift; EXTRA+=("$@"); break ;;
        *)             EXTRA+=("$1");     shift ;;
    esac
done

command -v verilator >/dev/null 2>&1 || die "Khong tim thay verilator trong PATH"

[[ -f $FILELIST ]] || die "Khong tim thay filelist: $FILELIST"

TMPD="$(mktemp -d)" || die "Khong tao duoc thu muc tam"
trap 'rm -rf "$TMPD"' EXIT
NORM="$TMPD/filelist.f"

# Chuan hoa: bo CRLF, doi "C:\a\b" -> "/mnt/c/a/b", "\" -> "/"
sed -e 's/\r$//' "$FILELIST" \
  | sed -E 's#([A-Za-z]):[\\/]#/mnt/\L\1/#g; s#\\#/#g' > "$NORM"

info "Filelist: $FILELIST"

# ---------- Liet ke file nguon (phuc vu doan top module) ----------
mapfile -t SRCS < <(
    awk '
        /^[[:space:]]*(\/\/|#)/ { next }
        {
            for (i = 1; i <= NF; i++) {
                t = $i
                if (t ~ /^[+-]/) continue
                if (t ~ /\.(v|sv|vh|svh)$/) print t
            }
        }' "$NORM" | sort -u
)

if [[ $VERBOSE -eq 1 ]]; then
    info "Nguon (${#SRCS[@]} file):"
    printf '      %s\n' "${SRCS[@]}"
fi

# ---------- Doan top module ----------
detect_top() {
    local -A decl=()
    local f m
    for f in "${SRCS[@]}"; do
        [[ -f $f ]] || continue
        while IFS= read -r m; do
            [[ -n $m ]] && decl[$m]=1
        done < <(grep -hoE '^[[:space:]]*module[[:space:]]+[A-Za-z_][A-Za-z0-9_$]*' "$f" \
                 | awk '{print $2}')
    done
    [[ ${#decl[@]} -eq 0 ]] && { echo ""; return; }

    # ung vien top = module KHONG duoc khoi tao o bat ky dau
    # (chi xet dong bat dau bang ten module -> tranh nham voi $dumpvars/log)
    local -a cand=()
    for m in "${!decl[@]}"; do
        local n
        n=$(grep -hE "^[[:space:]]*${m}[[:space:]]*(#|[A-Za-z_])" "${SRCS[@]}" 2>/dev/null | wc -l)
        [[ $n -eq 0 ]] && cand+=("$m")
    done
    [[ ${#cand[@]} -eq 0 ]] && { echo ""; return; }

    # uu tien ten quen thuoc
    local c p
    for p in tb testbench top; do
        for c in "${cand[@]}"; do [[ $c == "$p" ]] && { echo "$c"; return; }; done
    done
    for c in "${cand[@]}"; do
        [[ $c == *_tb || $c == tb_* ]] && { echo "$c"; return; }
    done
    [[ ${#cand[@]} -eq 1 ]] && { echo "${cand[0]}"; return; }
    echo ""
}

if [[ -z $TOP ]]; then
    if [[ ${#SRCS[@]} -gt 0 && ${#SRCS[@]} -le 300 ]]; then
        TOP="$(detect_top)"
    fi
    if [[ -n $TOP ]]; then
        info "Top module (tu doan): $B$TOP$N"
    else
        warn "Khong doan duoc top module - de Verilator tu quyet dinh."
        warn "Neu loi 'Multiple top level modules', chay lai voi: -t <ten_module>"
    fi
fi

# ---------- Build ----------
[[ $CLEAN -eq 1 ]] && { info "Xoa $BUILD_DIR"; rm -rf "$BUILD_DIR"; }

ARGS=(--binary --timing -j "$JOBS" -Wno-fatal -f "$NORM" -o "$EXE" --Mdir "$BUILD_DIR")
[[ -n $TOP ]] && ARGS+=(--top "$TOP")
if [[ $TRACE -eq 1 ]]; then
    if [[ $FST -eq 1 ]]; then
        ARGS+=(--trace-fst -DFST_TRACE)
    else
        ARGS+=(--trace)
    fi
fi

if [[ $VERBOSE -eq 1 ]]; then
    info "Lenh: verilator ${ARGS[*]} ${EXTRA[*]:-}"
fi

info "Dang build (co the mat vai chuc giay)..."
START=$(date +%s)
LOG="$TMPD/build.log"
verilator "${ARGS[@]}" "${EXTRA[@]}" 2>&1 | tee "$LOG"
BRC=${PIPESTATUS[0]}
END=$(date +%s)

if [[ $BRC -ne 0 || ! -x "$BUILD_DIR/$EXE" ]]; then
    printf '\n'
    warn "Build that bai (ma $BRC). Tom tat loi:"
    grep -E '^%Error|^%Warning' "$LOG" | head -n 15 | sed 's/^/      /'
    die "Xem lai filelist / top module. Goi y: ./run.sh -t <module>"
fi
ok "Build xong sau $((END - START))s -> $BUILD_DIR/$EXE"

# ---------- Chay mo phong ----------
printf '\n%s────────── SIMULATION ──────────%s\n' "$B" "$N"
"$BUILD_DIR/$EXE"
SIMRC=$?
printf '%s────────────────────────────────%s\n' "$B" "$N"

# ---------- Tim waveform ----------
WAVE="$(ls -t ./*.vcd ./*.fst 2>/dev/null | head -n1)"

if [[ $TRACE -eq 1 ]]; then
    if [[ -n $WAVE ]]; then
        ok "Waveform: $WAVE ($(du -h "$WAVE" | cut -f1))"
    else
        warn "Khong thay file waveform. Kiem tra testbench co \$dumpfile/\$dumpvars chua."
    fi
fi

if [[ $OPEN_WAVE -eq 1 && -n $WAVE ]]; then
    if command -v surfer >/dev/null 2>&1; then
        info "Mo Surfer: $WAVE"
        surfer "$WAVE" >/dev/null 2>&1 &
    elif command -v gtkwave >/dev/null 2>&1; then
        info "Mo GTKWave: $WAVE"
        gtkwave "$WAVE" >/dev/null 2>&1 &
    else
        warn "Khong co surfer/gtkwave de mo waveform"
    fi
fi

exit $SIMRC
