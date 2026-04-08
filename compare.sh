#!/usr/bin/env bash
# compare.sh — Build both programs, run them on one or more inputs,
#              validate correctness and print timing.
#
# Usage:
#   ./compare.sh [input1.txt input2.txt ...]   (defaults to input_report.txt)
#
# Optional env overrides:
#   NP=8 ./compare.sh test.txt      # number of MPI processes (default 4)
#   MPIRUN=srun ./compare.sh        # MPI launcher (default mpirun)

set -euo pipefail

NP="${NP:-4}"
MPIRUN="${MPIRUN:-mpirun}"

SEQ_SRC="sequential.cpp"
PAR_SRC="main.cpp"
SEQ_BIN="./seq_bin"
PAR_BIN="./par_bin"

# ── colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
pass()  { echo -e "${GREEN}[PASS]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }
info()  { echo -e "${YELLOW}[INFO]${NC} $*"; }

# ── build ───────────────────────────────────────────────────────────────────
info "Compiling sequential: g++ -O3 -std=c++17 $SEQ_SRC"
g++ -O3 -std=c++17 "$SEQ_SRC" -o "$SEQ_BIN"

info "Compiling parallel:   mpic++ -O3 -std=c++17 $PAR_SRC"
mpic++ -O3 -std=c++17 "$PAR_SRC" -o "$PAR_BIN"
echo ""

# ── helper: elapsed seconds using /usr/bin/time -p ─────────────────────────
elapsed_secs() {
    # $@ is the command; prints elapsed seconds to stdout
    { /usr/bin/time -p "$@" 2>&1 >/dev/null; } 2>&1 | awk '/^real/{print $2}'
}

# ── validate clique in output file ─────────────────────────────────────────
validate_output() {
    local infile="$1" outfile="$2" label="$3"
    python3 - "$infile" "$outfile" "$label" <<'PYEOF'
import sys, collections

in_f, out_f, label = sys.argv[1], sys.argv[2], sys.argv[3]

with open(in_f) as f:
    first = f.readline().split()
    N, E, B = int(first[0]), int(first[1]), int(first[2])
    profit = []; cost = []
    for _ in range(N):
        p, c = map(int, f.readline().split())
        profit.append(p); cost.append(c)
    adj = collections.defaultdict(set)
    for _ in range(E):
        u, v = map(int, f.readline().split())
        adj[u].add(v); adj[v].add(u)

with open(out_f) as f:
    reported_profit = int(f.readline().strip())
    clique_line = f.readline().strip()
    clique = list(map(int, clique_line.split())) if clique_line else []

# Check budget
total_cost = sum(cost[v] for v in clique)
total_profit = sum(profit[v] for v in clique)
ok = True

if total_cost > B:
    print(f"  [{label}] BUDGET VIOLATED: cost={total_cost} > B={B}")
    ok = False

if total_profit != reported_profit:
    print(f"  [{label}] PROFIT MISMATCH: actual={total_profit}, reported={reported_profit}")
    ok = False

for i in range(len(clique)):
    for j in range(i+1, len(clique)):
        u, v = clique[i], clique[j]
        if v not in adj[u]:
            print(f"  [{label}] NOT A CLIQUE: vertices {u},{v} not adjacent")
            ok = False
            break

if ok:
    print(f"  [{label}] Valid clique  |  size={len(clique)}, profit={total_profit}, cost={total_cost}/{B}")
PYEOF
}

# ── run over each input file ─────────────────────────────────────────────────
INPUTS=("${@:-input_report.txt}")
OVERALL_PASS=0
OVERALL_FAIL=0

for INPUT in "${INPUTS[@]}"; do
    echo "══════════════════════════════════════════════════════"
    echo "  Input: $INPUT"
    echo "══════════════════════════════════════════════════════"

    OUT_SEQ=$(mktemp /tmp/out_seq_XXXXXX)
    OUT_PAR=$(mktemp /tmp/out_par_XXXXXX)
    trap "rm -f $OUT_SEQ $OUT_PAR" EXIT

    # Run sequential
    printf "  Running sequential ...  "
    T_SEQ=$(elapsed_secs "$SEQ_BIN" "$INPUT" "$OUT_SEQ")
    PROFIT_SEQ=$(head -1 "$OUT_SEQ")
    echo "profit=$PROFIT_SEQ  time=${T_SEQ}s"
    validate_output "$INPUT" "$OUT_SEQ" "sequential"

    # Run parallel
    printf "  Running parallel (np=$NP) ...  "
    T_PAR=$(elapsed_secs $MPIRUN -np "$NP" "$PAR_BIN" "$INPUT" "$OUT_PAR")
    PROFIT_PAR=$(head -1 "$OUT_PAR")
    echo "profit=$PROFIT_PAR  time=${T_PAR}s"
    validate_output "$INPUT" "$OUT_PAR" "parallel  "

    # Compare profits
    echo ""
    if [ "$PROFIT_SEQ" = "$PROFIT_PAR" ]; then
        pass "Profits match: $PROFIT_SEQ"
        OVERALL_PASS=$((OVERALL_PASS + 1))
    else
        fail "Profit mismatch!  sequential=$PROFIT_SEQ  parallel=$PROFIT_PAR"
        OVERALL_FAIL=$((OVERALL_FAIL + 1))
    fi

    # Speedup
    if command -v python3 &>/dev/null && [ -n "$T_SEQ" ] && [ -n "$T_PAR" ]; then
        SPEEDUP=$(python3 -c "
seq, par = float('$T_SEQ'), float('$T_PAR')
if par > 0:
    print(f'Speedup ({seq:.3f}s / {par:.3f}s) = {seq/par:.2f}x')
else:
    print('(parallel time too small to measure)')
")
        info "$SPEEDUP"
    fi
    echo ""
done

echo "══════════════════════════════════════════════════════"
echo "  Summary: $OVERALL_PASS passed, $OVERALL_FAIL failed"
echo "══════════════════════════════════════════════════════"
[ "$OVERALL_FAIL" -eq 0 ]
