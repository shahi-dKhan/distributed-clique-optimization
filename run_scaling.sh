#!/usr/bin/env bash
# run_scaling.sh — Build parallel binary and measure strong scaling
#                  from 1 to 16 MPI ranks on a given input.
#
# Usage:
#   ./run_scaling.sh [input.txt]           (defaults to input_report.txt)
#
# Optional env overrides:
#   PROCS="1 2 4 8 16" ./run_scaling.sh   # which process counts to test
#   MPIRUN=srun ./run_scaling.sh           # MPI launcher (e.g. on SLURM)
#   REPS=3 ./run_scaling.sh               # repetitions per config (best kept)

set -euo pipefail

INPUT="${1:-input_report.txt}"
PROCS="${PROCS:-1 2 4 8 16}"
MPIRUN="${MPIRUN:-mpirun}"
REPS="${REPS:-1}"
PAR_SRC="main.cpp"
PAR_BIN="./par_bin"
LOG="scaling_results.csv"

# ── colour helpers ──────────────────────────────────────────────────────────
BOLD='\033[1m'; NC='\033[0m'

# ── build ───────────────────────────────────────────────────────────────────
echo "Compiling: mpic++ -O3 -std=c++17 $PAR_SRC"
mpic++ -O3 -std=c++17 "$PAR_SRC" -o "$PAR_BIN"
echo ""

# ── helper: best elapsed time over REPS runs ───────────────────────────────
best_time() {
    local np="$1" infile="$2" outfile="$3"
    local best=""
    for _ in $(seq 1 "$REPS"); do
        t=$( { /usr/bin/time -p $MPIRUN -np "$np" "$PAR_BIN" "$infile" "$outfile" 2>&1 >/dev/null; } 2>&1 \
             | awk '/^real/{print $2}' )
        if [ -z "$best" ] || python3 -c "exit(0 if float('$t') < float('$best') else 1)" 2>/dev/null; then
            best="$t"
        fi
    done
    echo "$best"
}

# ── print header ────────────────────────────────────────────────────────────
echo -e "${BOLD}Strong Scaling — input: $INPUT${NC}"
printf "%-8s  %-12s  %-10s  %-12s  %-8s\n" \
       "Procs" "Time (s)" "Speedup" "Efficiency" "Profit"
printf "%-8s  %-12s  %-10s  %-12s  %-8s\n" \
       "-----" "--------" "-------" "----------" "------"

echo "procs,time_s,speedup,efficiency,profit" > "$LOG"

BASE_TIME=""
BASE_PROFIT=""

for NP in $PROCS; do
    OUTFILE=$(mktemp /tmp/scaling_out_XXXXXX)
    trap "rm -f $OUTFILE" EXIT

    T=$(best_time "$NP" "$INPUT" "$OUTFILE")
    PROFIT=$(head -1 "$OUTFILE" 2>/dev/null || echo "?")

    if [ -z "$BASE_TIME" ]; then
        BASE_TIME="$T"
        BASE_PROFIT="$PROFIT"
    fi

    # Compute speedup and efficiency with python3
    read SPEEDUP EFFICIENCY < <(python3 -c "
t  = float('$T')
b  = float('$BASE_TIME')
np = int('$NP')
sp = b / t if t > 0 else float('inf')
ef = sp / np * 100
print(f'{sp:.3f} {ef:.1f}')
")

    printf "%-8s  %-12s  %-10s  %-11s%%  %-8s\n" \
           "$NP" "${T}s" "${SPEEDUP}x" "$EFFICIENCY" "$PROFIT"
    echo "$NP,$T,$SPEEDUP,$EFFICIENCY,$PROFIT" >> "$LOG"

    # Sanity-check: profit must match 1-process result
    if [ "$PROFIT" != "$BASE_PROFIT" ] && [ "$BASE_PROFIT" != "" ]; then
        echo "  WARNING: profit mismatch at np=$NP ($PROFIT vs $BASE_PROFIT)"
    fi
done

echo ""
echo "Results saved to: $LOG"
echo ""
echo "── Interpretation guide ───────────────────────────────────────────"
echo "  Speedup    : T(1) / T(P)  — ideal is P"
echo "  Efficiency : Speedup / P  × 100%  — ideal is 100%"
echo "  Values < 1x speedup indicate communication/imbalance overhead."
