#!/usr/bin/env bash
# test_suite.sh — Correctness + strong-scaling for all solver variants
#
# Variants tested:
#   seq_bin        — sequential, sort-based bounds
#   par_bin        — parallel, sort-based bounds  (main.cpp)
#   par_bin_bitset — parallel, bitset-accelerated (main_bitset.cpp)
#
# Graph groups:
#   Medium density  (d=0.40–0.60): good pruning benchmark, N up to 1200
#   High   density  (d=0.70–0.90): tests bitset advantage, N up to 1200
#
# Usage:
#   ./test_suite.sh                 # generate + correctness + scaling
#   ./test_suite.sh --gen-only      # only generate graphs
#   ./test_suite.sh --corr-only     # generate + correctness checks
#   ./test_suite.sh --scale-only    # generate + scaling sweeps
#
# Env overrides:
#   PROCS="1 2 4 8 16"    MPI process counts
#   MPIRUN="mpirun"       MPI launcher
#   NP_CORR=4             ranks used for correctness checks

set -euo pipefail

MPIRUN="${MPIRUN:-mpirun}"
PROCS="${PROCS:-1 2 4 8 16}"
NP_CORR="${NP_CORR:-4}"
SEQ_BIN="./seq_bin"
PAR_BIN="./par_bin"
PAR_BITSET="./par_bin_bitset"
DIR="suite_tests"

# ── colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; CYAN='\033[0;36m'; NC='\033[0m'
pass() { echo -e "  ${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $*"; }
info() { echo -e "${YELLOW}[INFO]${NC} $*"; }
hdr()  { echo -e "\n${BOLD}${CYAN}══ $* ══${NC}"; }

# ── compile ───────────────────────────────────────────────────────────────────
compile_all() {
    hdr "Compilation"
    g++    -O3 -std=c++17 sequential.cpp    -o "$SEQ_BIN"    && info "Sequential  compiled"
    mpic++ -O3 -std=c++17 main.cpp          -o "$PAR_BIN"    && info "Par(sort)   compiled"
    if [ -f main_bitset.cpp ]; then
        mpic++ -O3 -std=c++17 main_bitset.cpp -o "$PAR_BITSET" && info "Par(bitset) compiled"
    else
        info "main_bitset.cpp not found — bitset variant skipped"
        PAR_BITSET=""
    fi
}

# ── generate graphs ───────────────────────────────────────────────────────────
generate_all() {
    hdr "Generating test graphs  →  $DIR/"
    mkdir -p "$DIR"

    # ════════════════════════════════════════════════════════════════════════
    # MEDIUM-DENSITY GROUP  (d = 0.40–0.60)
    # Classic pruning benchmark.  B tightened with N to keep seq runtime sane.
    # ════════════════════════════════════════════════════════════════════════

    # Correctness (N ≤ 200, seq < 10 s)
    python3 gen.py -N   30 --density 0.60 -B  60 -s  1 -o "$DIR/n030_d60_b060.txt"
    python3 gen.py -N   50 --density 0.60 -B  80 -s  2 -o "$DIR/n050_d60_b080.txt"
    python3 gen.py -N   80 --density 0.55 -B 100 -s  3 -o "$DIR/n080_d55_b100.txt"
    python3 gen.py -N  100 --density 0.55 -B 110 -s  4 -o "$DIR/n100_d55_b110.txt"
    python3 gen.py -N  150 --density 0.50 -B 120 -s  5 -o "$DIR/n150_d50_b120.txt"
    python3 gen.py -N  200 --density 0.50 -B 130 -s  6 -o "$DIR/n200_d50_b130.txt"

    # Scaling (parallel vs sequential comparison)
    python3 gen.py -N  300 --density 0.50 -B 120 -s 10 -o "$DIR/n300_d50_b120.txt"
    python3 gen.py -N  400 --density 0.50 -B 110 -s 11 -o "$DIR/n400_d50_b110.txt"
    python3 gen.py -N  500 --density 0.50 -B 100 -s 12 -o "$DIR/n500_d50_b100.txt"
    python3 gen.py -N  500 --density 0.60 -B  90 -s 13 -o "$DIR/n500_d60_b090.txt"

    # Large (parallel-only; seq > 30 min)
    python3 gen.py -N  600 --density 0.50 -B  95 -s 14 -o "$DIR/n600_d50_b095.txt"
    python3 gen.py -N  800 --density 0.45 -B  92 -s 15 -o "$DIR/n800_d45_b092.txt"
    python3 gen.py -N 1000 --density 0.45 -B  90 -s 16 -o "$DIR/n1000_d45_b090.txt"
    python3 gen.py -N 1200 --density 0.40 -B  90 -s 17 -o "$DIR/n1200_d40_b090.txt"

    # ════════════════════════════════════════════════════════════════════════
    # HIGH-DENSITY GROUP  (d = 0.70–0.90)
    # Dense graphs → large cliques → tighter coloring bound → faster pruning.
    # Bitset C_next intersection shines here (k stays large, WORDS stays small).
    # ════════════════════════════════════════════════════════════════════════

    # Correctness (N ≤ 200)
    python3 gen.py -N   30 --density 0.90 -B  60 -s 21 -o "$DIR/n030_d90_b060.txt"
    python3 gen.py -N   50 --density 0.85 -B  70 -s 22 -o "$DIR/n050_d85_b070.txt"
    python3 gen.py -N   80 --density 0.80 -B  80 -s 23 -o "$DIR/n080_d80_b080.txt"
    python3 gen.py -N  100 --density 0.80 -B  90 -s 24 -o "$DIR/n100_d80_b090.txt"
    python3 gen.py -N  150 --density 0.75 -B 100 -s 25 -o "$DIR/n150_d75_b100.txt"
    python3 gen.py -N  200 --density 0.70 -B 110 -s 26 -o "$DIR/n200_d70_b110.txt"

    # Scaling (both seq and par viable at these densities)
    python3 gen.py -N  300 --density 0.80 -B 100 -s 30 -o "$DIR/n300_d80_b100.txt"
    python3 gen.py -N  500 --density 0.80 -B  90 -s 31 -o "$DIR/n500_d80_b090.txt"
    python3 gen.py -N  800 --density 0.90 -B  90 -s 32 -o "$DIR/n800_d90_b090.txt"
    python3 gen.py -N 1200 --density 0.90 -B  90 -s 33 -o "$DIR/n1200_d90_b090.txt"

    [ -f input_report.txt ] && info "input_report.txt present — will be included in scaling"
    info "Done."
}

# ── helpers ────────────────────────────────────────────────────────────────────
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

# ── correctness suite ──────────────────────────────────────────────────────────
run_correctness() {
    hdr "Correctness Checks  (seq vs par-sort vs par-bitset, np=$NP_CORR)"
    echo ""

    # Header
    printf "  %-26s  %5s  %7s  %7s  %7s  %6s  %6s  %6s  %s\n" \
           "Graph" "N" "Seq(s)" "Sort(s)" "Bset(s)" "SeqP" "SrtP" "BstP" "Result"
    printf "  %-26s  %5s  %7s  %7s  %7s  %6s  %6s  %6s  %s\n" \
           "─────────────────────────" "─────" "──────" "──────" "──────" "─────" "─────" "─────" "──────"

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
        "$DIR/n200_d50_b130.txt" \
        "$DIR/n030_d90_b060.txt" \
        "$DIR/n050_d85_b070.txt" \
        "$DIR/n080_d80_b080.txt" \
        "$DIR/n100_d80_b090.txt" \
        "$DIR/n150_d75_b100.txt" \
        "$DIR/n200_d70_b110.txt"
    do
        [ -f "$INPUT" ] || continue
        N_VAL=$(awk 'NR==1{print $1}' "$INPUT")

        OS=$(mktemp /tmp/seq_XXXXXX)
        OP=$(mktemp /tmp/sort_XXXXXX)
        OB=$(mktemp /tmp/bset_XXXXXX)
        TMPFILES+=("$OS" "$OP" "$OB")

        T_SEQ=$(elapsed "$SEQ_BIN" "$INPUT" "$OS")
        T_SRT=$(elapsed $MPIRUN -np "$NP_CORR" "$PAR_BIN" "$INPUT" "$OP")

        P_SEQ=$(head -1 "$OS"); P_SRT=$(head -1 "$OP")
        V_SEQ=$(validate "$INPUT" "$OS"); V_SRT=$(validate "$INPUT" "$OP")

        if [ -n "$PAR_BITSET" ] && [ -f "$PAR_BITSET" ]; then
            T_BST=$(elapsed $MPIRUN -np "$NP_CORR" "$PAR_BITSET" "$INPUT" "$OB")
            P_BST=$(head -1 "$OB")
            V_BST=$(validate "$INPUT" "$OB")
        else
            T_BST="N/A"; P_BST="N/A"; V_BST="OK"
        fi

        # PASS: all profits agree, all outputs valid
        if [ "$P_SEQ" = "$P_SRT" ] && \
           { [ "$P_BST" = "N/A" ] || [ "$P_SEQ" = "$P_BST" ]; } && \
           [ "$V_SEQ" = "OK" ] && [ "$V_SRT" = "OK" ] && [ "$V_BST" = "OK" ]; then
            RESULT="${GREEN}PASS${NC}"; PASS=$((PASS+1))
        else
            RESULT="${RED}FAIL seq=$V_SEQ srt=$V_SRT bst=$V_BST${NC}"; FAIL=$((FAIL+1))
        fi

        printf "  %-26s  %5s  %7s  %7s  %7s  %6s  %6s  %6s  " \
               "$(basename "$INPUT")" "$N_VAL" "${T_SEQ}s" "${T_SRT}s" "${T_BST}s" \
               "$P_SEQ" "$P_SRT" "$P_BST"
        echo -e "$RESULT"
    done

    echo ""
    echo -e "  ${BOLD}Total: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
}

# ── one scaling sweep (single binary) ─────────────────────────────────────────
# scale_one INPUT BIN SUFFIX
#   INPUT  — graph file
#   BIN    — binary to run (e.g. ./par_bin or ./par_bin_bitset)
#   SUFFIX — appended to CSV name (e.g. "" or "_bitset")
scale_one() {
    local INPUT="$1" BIN="$2" SUFFIX="${3:-}"
    [ -f "$INPUT" ] || return
    [ -f "$BIN"   ] || return

    local NAME; NAME=$(basename "${INPUT%.txt}")
    local HDR; HDR=$(awk 'NR==1{printf "N=%s  density=%.2f  B=%s", $1, ($2*2/($1*($1-1))), $3}' "$INPUT")
    local BNAME; BNAME=$(basename "$BIN")
    local CSV="results_${NAME}${SUFFIX}.csv"

    echo ""
    echo -e "  ${BOLD}$(basename "$INPUT")${NC}  ($HDR)  [${CYAN}${BNAME}${NC}]"
    printf "  %-6s  %-10s  %-9s  %-12s  %s\n" "Procs" "Time(s)" "Speedup" "Efficiency" "Profit"
    printf "  %-6s  %-10s  %-9s  %-12s  %s\n" "──────" "────────" "───────" "──────────" "──────"
    echo "procs,time_s,speedup,efficiency,profit" > "$CSV"

    local BASE_T=""
    for NP in $PROCS; do
        local OUT; OUT=$(mktemp /tmp/scale_XXXXXX)
        local T; T=$(elapsed $MPIRUN -np "$NP" "$BIN" "$INPUT" "$OUT")
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

# ── scaling suite ──────────────────────────────────────────────────────────────
run_scaling() {
    hdr "Strong Scaling Sweeps  (par-sort vs par-bitset, np: $PROCS)"

    [ -f input_report.txt ] && {
        scale_one "input_report.txt" "$PAR_BIN"    ""
        [ -n "$PAR_BITSET" ] && scale_one "input_report.txt" "$PAR_BITSET" "_bitset"
    }

    # ── medium-density scaling ────────────────────────────────────────────────
    echo ""
    echo -e "  ${BOLD}── Medium-density (d=0.40–0.60) ──${NC}"
    for INPUT in \
        "$DIR/n300_d50_b120.txt"   \
        "$DIR/n400_d50_b110.txt"   \
        "$DIR/n500_d50_b100.txt"   \
        "$DIR/n500_d60_b090.txt"   \
        "$DIR/n600_d50_b095.txt"   \
        "$DIR/n800_d45_b092.txt"   \
        "$DIR/n1000_d45_b090.txt"  \
        "$DIR/n1200_d40_b090.txt"
    do
        scale_one "$INPUT" "$PAR_BIN" ""
        [ -n "$PAR_BITSET" ] && scale_one "$INPUT" "$PAR_BITSET" "_bitset"
    done

    # ── high-density scaling ──────────────────────────────────────────────────
    echo ""
    echo -e "  ${BOLD}── High-density (d=0.70–0.90) ──${NC}"
    for INPUT in \
        "$DIR/n300_d80_b100.txt"  \
        "$DIR/n500_d80_b090.txt"  \
        "$DIR/n800_d90_b090.txt"  \
        "$DIR/n1200_d90_b090.txt"
    do
        scale_one "$INPUT" "$PAR_BIN" ""
        [ -n "$PAR_BITSET" ] && scale_one "$INPUT" "$PAR_BITSET" "_bitset"
    done
}

# ── summary: par-sort vs par-bitset comparison table ─────────────────────────
print_summary() {
    hdr "Summary — par-sort vs par-bitset (np=1 and np=16)"
    echo ""
    printf "  %-28s  %5s  %8s  %8s  %8s  %8s  %7s  %7s\n" \
           "Graph" "N" "T1_srt" "T16_srt" "T1_bst" "T16_bst" "Sp_srt" "Sp_bst"
    printf "  %-28s  %5s  %8s  %8s  %8s  %8s  %7s  %7s\n" \
           "────────────────────────────" "─────" "───────" "────────" "───────" "────────" "──────" "──────"

    python3 - "$DIR" <<'PYEOF'
import sys, csv, os, glob

def parse_csv(path):
    try:
        rows = list(csv.DictReader(open(path)))
    except Exception:
        return None, None, None, None
    r1  = next((r for r in rows if r['procs']=='1'),  None)
    r16 = next((r for r in rows if r['procs']=='16'), None)
    t1  = float(r1['time_s'])  if r1  else 0
    t16 = float(r16['time_s']) if r16 else 0
    sp  = t1/t16 if t16 > 0 else 0
    p   = (r16 or r1 or {}).get('profit','?')
    return t1, t16, sp, p

suite_dir = sys.argv[1]
# Collect all base names (without _bitset suffix)
base_csvs = sorted(f for f in glob.glob('results_*.csv') if '_bitset' not in f)

for base_csv in base_csvs:
    name = base_csv[len('results_'):-len('.csv')]
    bset_csv = f'results_{name}_bitset.csv'
    t1s, t16s, sps, ps = parse_csv(base_csv)
    t1b, t16b, spb, pb = parse_csv(bset_csv) if os.path.exists(bset_csv) else (None, None, None, None)

    # Get N from input file
    N = '?'
    inp = f'{suite_dir}/{name}.txt'
    if os.path.exists(inp):
        with open(inp) as f:
            N = f.readline().split()[0]

    t1s_s  = f'{t1s:.3f}s'  if t1s  is not None else 'N/A'
    t16s_s = f'{t16s:.3f}s' if t16s is not None else 'N/A'
    t1b_s  = f'{t1b:.3f}s'  if t1b  is not None else 'N/A'
    t16b_s = f'{t16b:.3f}s' if t16b is not None else 'N/A'
    sps_s  = f'{sps:.2f}x'  if sps  is not None else 'N/A'
    spb_s  = f'{spb:.2f}x'  if spb  is not None else 'N/A'

    print(f'  {name:<28}  {N:>5}  {t1s_s:>8}  {t16s_s:>8}  {t1b_s:>8}  {t16b_s:>8}  {sps_s:>7}  {spb_s:>7}')
PYEOF
}

# ── also include smaller graphs in correctness-only scaling if desired ────────
run_scaling_small() {
    hdr "Small-graph Scaling (correctness group, both variants)"
    for INPUT in \
        "$DIR/n030_d60_b060.txt" \
        "$DIR/n050_d60_b080.txt" \
        "$DIR/n080_d55_b100.txt" \
        "$DIR/n100_d55_b110.txt" \
        "$DIR/n030_d90_b060.txt" \
        "$DIR/n050_d85_b070.txt" \
        "$DIR/n080_d80_b080.txt" \
        "$DIR/n100_d80_b090.txt"
    do
        scale_one "$INPUT" "$PAR_BIN" ""
        [ -n "$PAR_BITSET" ] && scale_one "$INPUT" "$PAR_BITSET" "_bitset"
    done
}

# ── entry point ────────────────────────────────────────────────────────────────
MODE="${1:---all}"
compile_all

case "$MODE" in
    --gen-only)   generate_all ;;
    --corr-only)  generate_all; run_correctness ;;
    --scale-only) generate_all; run_scaling; print_summary ;;
    --small-scale) generate_all; run_scaling_small; print_summary ;;
    *)            generate_all; run_correctness; run_scaling; print_summary ;;
esac

hdr "All done"
