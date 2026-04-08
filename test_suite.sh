#!/usr/bin/env bash
# test_suite.sh — Comprehensive correctness + strong-scaling suite
#
# Covers N = 30 … 1200 across sparse, medium, and dense graphs.
# Correctness is verified (seq vs par) for cases that finish quickly.
# Scaling sweeps run parallel-only on harder instances.
#
# Usage:
#   ./test_suite.sh                 # generate + correctness + scaling
#   ./test_suite.sh --gen-only      # only generate graphs
#   ./test_suite.sh --corr-only     # generate + correctness checks
#   ./test_suite.sh --scale-only    # generate + scaling sweeps
#
# Env overrides:
#   PROCS="1 2 4 8 16"    MPI process counts (default)
#   MPIRUN="mpirun"       launcher (set to srun on some clusters)
#   NP_CORR=4             ranks used for correctness checks

set -euo pipefail

MPIRUN="${MPIRUN:-mpirun}"
PROCS="${PROCS:-1 2 4 8 16}"
NP_CORR="${NP_CORR:-4}"
SEQ_BIN="./seq_bin"
PAR_BIN="./par_bin"
DIR="suite_tests"

# ── colour helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; CYAN='\033[0;36m'; NC='\033[0m'
pass() { echo -e "  ${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $*"; }
info() { echo -e "${YELLOW}[INFO]${NC} $*"; }
hdr()  { echo -e "\n${BOLD}${CYAN}══ $* ══${NC}"; }

# ── compile ──────────────────────────────────────────────────────────────────
compile_all() {
    hdr "Compilation"
    g++    -O3 -std=c++17 sequential.cpp -o "$SEQ_BIN" && info "Sequential compiled"
    mpic++ -O3 -std=c++17 main.cpp       -o "$PAR_BIN" && info "Parallel   compiled"
}

# ── generate graphs ───────────────────────────────────────────────────────────
generate_all() {
    hdr "Generating test graphs  →  $DIR/"
    mkdir -p "$DIR"

    # ── Correctness group: N ≤ 200, sequential finishes in < 10s ───────────────
    # Density held at 0.55–0.60; B scaled with N so clique size stays ~3–5.
    python3 gen.py -N   30 --density 0.60 -B  60 -s  1 -o "$DIR/n030_d60_b060.txt"
    python3 gen.py -N   50 --density 0.60 -B  80 -s  2 -o "$DIR/n050_d60_b080.txt"
    python3 gen.py -N   80 --density 0.55 -B 100 -s  3 -o "$DIR/n080_d55_b100.txt"
    python3 gen.py -N  100 --density 0.55 -B 110 -s  4 -o "$DIR/n100_d55_b110.txt"
    python3 gen.py -N  150 --density 0.50 -B 120 -s  5 -o "$DIR/n150_d50_b120.txt"
    python3 gen.py -N  200 --density 0.50 -B 130 -s  6 -o "$DIR/n200_d50_b130.txt"

    # ── Scaling group: parallel vs sequential comparison ──────────────────────
    # Density stays 0.45–0.55 throughout; B tightened as N grows to keep
    # sequential under ~5 min. Tight B activates the knapsack bound heavily.
    python3 gen.py -N  300 --density 0.50 -B 120 -s 10 -o "$DIR/n300_d50_b120.txt"   # est. ~15s seq
    python3 gen.py -N  400 --density 0.50 -B 110 -s 11 -o "$DIR/n400_d50_b110.txt"   # est. ~60s seq
    python3 gen.py -N  500 --density 0.50 -B 100 -s 12 -o "$DIR/n500_d50_b100.txt"   # est. ~2-5 min seq
    python3 gen.py -N  500 --density 0.60 -B  90 -s 13 -o "$DIR/n500_d60_b090.txt"   # ~150s seq (like report)

    # ── Large group: parallel-only (sequential likely > 30 min) ──────────────
    # Density kept at 0.45–0.50; very tight B keeps the knapsack bound sharp.
    python3 gen.py -N  600 --density 0.50 -B  95 -s 14 -o "$DIR/n600_d50_b095.txt"
    python3 gen.py -N  800 --density 0.45 -B  92 -s 15 -o "$DIR/n800_d45_b092.txt"
    python3 gen.py -N 1000 --density 0.45 -B  90 -s 16 -o "$DIR/n1000_d45_b090.txt"
    python3 gen.py -N 1200 --density 0.40 -B  90 -s 17 -o "$DIR/n1200_d40_b090.txt"

    # Also note the provided report graph
    [ -f input_report.txt ] && info "input_report.txt present — will be included in scaling"
    info "Done."
}

# ── helpers ───────────────────────────────────────────────────────────────────
elapsed() {
    { /usr/bin/time -p "$@" 2>&1 >/dev/null; } 2>&1 | awk '/^real/{print $2}'
}

validate() {          # args: infile outfile  →  prints OK or error message
    python3 - "$1" "$2" <<'PYEOF'
import sys, collections
in_f, out_f = sys.argv[1], sys.argv[2]
with open(in_f) as f:
    N, E, B = map(int, f.readline().split())
    profit, cost = [], []
    for _ in range(N):
        p, c = map(int, f.readline().split()); profit.append(p); cost.append(c)
    adj = collections.defaultdict(set)
    for _ in range(E):
        u, v = map(int, f.readline().split()); adj[u].add(v); adj[v].add(u)
with open(out_f) as f:
    rep = int(f.readline().strip())
    clique = list(map(int, f.readline().strip().split()))
ok = True
tc = sum(cost[v] for v in clique); tp = sum(profit[v] for v in clique)
if tc > B:  print(f"BUDGET_VIOLATED cost={tc}>B={B}"); ok = False
if tp != rep: print(f"PROFIT_MISMATCH actual={tp} reported={rep}"); ok = False
for i in range(len(clique)):
    for j in range(i+1, len(clique)):
        if clique[j] not in adj[clique[i]]:
            print(f"NOT_A_CLIQUE vertices {clique[i]},{clique[j]}"); ok = False; break
print("OK" if ok else "INVALID")
PYEOF
}

# ── correctness suite ─────────────────────────────────────────────────────────
run_correctness() {
    hdr "Correctness Checks  (sequential vs parallel, np=$NP_CORR)"
    echo ""
    printf "  %-26s  %5s  %7s  %7s  %6s  %6s  %s\n" \
           "Graph" "N" "Seq(s)" "Par(s)" "SeqP" "ParP" "Result"
    printf "  %-26s  %5s  %7s  %7s  %6s  %6s  %s\n" \
           "─────────────────────────" "─────" "──────" "──────" "─────" "─────" "──────"

    PASS=0; FAIL=0
    TMPFILES=()
    cleanup() { rm -f "${TMPFILES[@]}" 2>/dev/null || true; }
    trap cleanup EXIT

    for INPUT in \
        "$DIR/n030_d60_b060.txt" \
        "$DIR/n050_d60_b080.txt" \
        "$DIR/n080_d55_b100.txt" \
        "$DIR/n100_d55_b110.txt" \
        "$DIR/n150_d50_b120.txt" \
        "$DIR/n200_d50_b130.txt"
    do
        [ -f "$INPUT" ] || continue
        N_VAL=$(awk 'NR==1{print $1}' "$INPUT")

        OS=$(mktemp /tmp/seq_XXXXXX); OP=$(mktemp /tmp/par_XXXXXX)
        TMPFILES+=("$OS" "$OP")

        T_SEQ=$(elapsed "$SEQ_BIN" "$INPUT" "$OS")
        T_PAR=$(elapsed $MPIRUN -np "$NP_CORR" "$PAR_BIN" "$INPUT" "$OP")
        P_SEQ=$(head -1 "$OS"); P_PAR=$(head -1 "$OP")
        V_SEQ=$(validate "$INPUT" "$OS"); V_PAR=$(validate "$INPUT" "$OP")

        if [ "$P_SEQ" = "$P_PAR" ] && [ "$V_SEQ" = "OK" ] && [ "$V_PAR" = "OK" ]; then
            RESULT="${GREEN}PASS${NC}"; PASS=$((PASS+1))
        else
            RESULT="${RED}FAIL  seq=$V_SEQ par=$V_PAR${NC}"; FAIL=$((FAIL+1))
        fi

        printf "  %-26s  %5s  %7s  %7s  %6s  %6s  " \
               "$(basename "$INPUT")" "$N_VAL" "${T_SEQ}s" "${T_PAR}s" "$P_SEQ" "$P_PAR"
        echo -e "$RESULT"
    done

    echo ""
    echo -e "  ${BOLD}Total: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
}

# ── one scaling sweep ─────────────────────────────────────────────────────────
scale_one() {
    local INPUT="$1"
    [ -f "$INPUT" ] || return
    local NAME; NAME=$(basename "${INPUT%.txt}")
    local HDR; HDR=$(awk 'NR==1{printf "N=%s  density=%.2f  B=%s", $1, ($2*2/($1*($1-1))), $3}' "$INPUT")
    local CSV="results_${NAME}.csv"

    echo ""
    echo -e "  ${BOLD}$(basename "$INPUT")${NC}  ($HDR)"
    printf "  %-6s  %-10s  %-9s  %-12s  %s\n" "Procs" "Time(s)" "Speedup" "Efficiency" "Profit"
    printf "  %-6s  %-10s  %-9s  %-12s  %s\n" "──────" "────────" "───────" "──────────" "──────"
    echo "procs,time_s,speedup,efficiency,profit" > "$CSV"

    local BASE_T=""
    for NP in $PROCS; do
        local OUT; OUT=$(mktemp /tmp/scale_XXXXXX)
        local T; T=$(elapsed $MPIRUN -np "$NP" "$PAR_BIN" "$INPUT" "$OUT")
        local PROFIT; PROFIT=$(head -1 "$OUT" 2>/dev/null || echo "?")
        rm -f "$OUT"
        [ -z "$BASE_T" ] && BASE_T="$T"
        python3 -c "
np=$NP; t=float('$T'); b=float('$BASE_T')
sp = b/t if t>0 else 0; ef = sp/np*100
print(f'  {np:<6}  {t:<10.3f}  {sp:<9.3f}  {ef:<10.1f}%   $PROFIT')
print(f'$NP,{t:.3f},{sp:.3f},{ef:.1f},$PROFIT', file=__import__(\"sys\").stderr)
" 2>>"$CSV"
    done
    info "Saved → $CSV"
}

# ── scaling suite ─────────────────────────────────────────────────────────────
run_scaling() {
    hdr "Strong Scaling Sweeps  (parallel only, np: $PROCS)"

    # Include the provided report graph first if available
    [ -f input_report.txt ] && scale_one "input_report.txt"

    for INPUT in \
        "$DIR/n030_d60_b060.txt"   \
        "$DIR/n050_d60_b080.txt"   \
        "$DIR/n080_d55_b100.txt"   \
        "$DIR/n100_d55_b110.txt"   \
        "$DIR/n150_d50_b120.txt"   \
        "$DIR/n200_d50_b130.txt"   \
        "$DIR/n300_d50_b120.txt"   \
        "$DIR/n400_d50_b110.txt"   \
        "$DIR/n500_d50_b100.txt"   \
        "$DIR/n500_d60_b090.txt"   \
        "$DIR/n600_d50_b095.txt"   \
        "$DIR/n800_d45_b092.txt"   \
        "$DIR/n1000_d45_b090.txt"  \
        "$DIR/n1200_d40_b090.txt"
    do
        scale_one "$INPUT"
    done
}

# ── summary: read all CSVs and print one big table ────────────────────────────
print_summary() {
    hdr "Summary — np=1 vs np=16 across all graphs"
    echo ""
    printf "  %-26s  %5s  %8s  %8s  %7s  %10s  %s\n" \
           "Graph" "N" "T_np1(s)" "T_np16(s)" "Speedup" "Efficiency" "Profit"
    printf "  %-26s  %5s  %8s  %8s  %7s  %10s  %s\n" \
           "─────────────────────────" "─────" "────────" "─────────" "───────" "──────────" "──────"

    for CSV in results_*.csv; do
        [ -f "$CSV" ] || continue
        NAME="${CSV#results_}"; NAME="${NAME%.csv}"
        N_VAL=$(awk -F, 'NR==2{print $0}' "$CSV" | cut -d, -f1)
        python3 - "$CSV" "$NAME" <<'PYEOF'
import sys, csv
fname, name = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(fname)))
if not rows: sys.exit()
r1  = next((r for r in rows if r['procs']=='1'),  None)
r16 = next((r for r in rows if r['procs']=='16'), None)
t1  = float(r1['time_s'])  if r1  else 0
t16 = float(r16['time_s']) if r16 else 0
sp  = t1/t16 if t16 > 0 else 0
ef  = sp/16*100 if t16 > 0 else 0
p   = r16['profit'] if r16 else (r1['profit'] if r1 else '?')
# get N from first row if possible
n   = rows[0].get('procs','?')   # placeholder; filled below
print(f"  {name:<26}  {'':>5}  {t1:>8.3f}  {t16:>9.3f}  {sp:>7.2f}x  {ef:>9.1f}%  {p}")
PYEOF
    done
}

# ── entry point ───────────────────────────────────────────────────────────────
MODE="${1:---all}"
compile_all

case "$MODE" in
    --gen-only)   generate_all ;;
    --corr-only)  generate_all; run_correctness ;;
    --scale-only) generate_all; run_scaling; print_summary ;;
    *)            generate_all; run_correctness; run_scaling; print_summary ;;
esac

hdr "All done"
