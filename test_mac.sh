#!/usr/bin/env bash
# test_mac.sh — Local Mac test suite
#
# Installs open-mpi if missing, generates small/medium test graphs,
# runs correctness checks and a scaling sweep (np = 1 2 4 8).
#
# Usage:  ./test_mac.sh

set -euo pipefail

SEQ_SRC="sequential.cpp"
PAR_SRC="main.cpp"
SEQ_BIN="./seq_bin"
PAR_BIN="./par_bin"
MAC_PROCS="1 2 4 8"

# ── colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; NC='\033[0m'
pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
info() { echo -e "${YELLOW}[INFO]${NC} $*"; }
hdr()  { echo -e "\n${BOLD}══ $* ══${NC}"; }

# ── 0. check / install open-mpi ────────────────────────────────────────────
hdr "Environment check"
if ! command -v mpic++ &>/dev/null; then
    info "mpic++ not found. Installing open-mpi via Homebrew..."
    brew install open-mpi
fi
info "MPI:  $(mpirun --version 2>&1 | head -1)"
info "GCC:  $(g++ --version | head -1)"
echo ""

# ── 1. compile ──────────────────────────────────────────────────────────────
hdr "Compilation"
g++    -O3 -std=c++17 "$SEQ_SRC" -o "$SEQ_BIN" && info "Sequential compiled OK"
mpic++ -O3 -std=c++17 "$PAR_SRC" -o "$PAR_BIN" && info "Parallel   compiled OK"

# ── 2. generate test graphs ─────────────────────────────────────────────────
hdr "Generating test graphs"
mkdir -p mac_tests

python3 gen.py -N 15  --density 0.8 -B 50  -s 1  -o mac_tests/tiny.txt
python3 gen.py -N 30  --density 0.6 -B 80  -s 2  -o mac_tests/small.txt
python3 gen.py -N 60  --density 0.5 -B 120 -s 3  -o mac_tests/medium.txt
python3 gen.py -N 100 --density 0.4 -B 150 -s 4  -o mac_tests/medium2.txt

# Also include the PDF example for sanity
cat > mac_tests/pdf_example.txt <<'EOF'
4 5 10
4 3
5 4
6 5
7 5
0 1
0 2
1 2
1 3
2 3
EOF
info "PDF example written (expected output: profit=13, clique=2 3)"

# ── helper: elapsed seconds ─────────────────────────────────────────────────
elapsed() {
    { /usr/bin/time -p "$@" 2>&1 >/dev/null; } 2>&1 | awk '/^real/{print $2}'
}

# ── helper: validate clique ─────────────────────────────────────────────────
validate() {
    local infile="$1" outfile="$2" label="$3"
    python3 - "$infile" "$outfile" "$label" <<'PYEOF'
import sys, collections
in_f, out_f, label = sys.argv[1], sys.argv[2], sys.argv[3]
with open(in_f) as f:
    N, E, B = map(int, f.readline().split())
    profit, cost = [], []
    for _ in range(N):
        p, c = map(int, f.readline().split()); profit.append(p); cost.append(c)
    adj = collections.defaultdict(set)
    for _ in range(E):
        u, v = map(int, f.readline().split()); adj[u].add(v); adj[v].add(u)
with open(out_f) as f:
    rep_profit = int(f.readline().strip())
    clique = list(map(int, f.readline().strip().split())) if True else []
ok = True
tot_cost   = sum(cost[v]   for v in clique)
tot_profit = sum(profit[v] for v in clique)
if tot_cost > B:
    print(f"    BUDGET VIOLATED: cost={tot_cost} > B={B}"); ok = False
if tot_profit != rep_profit:
    print(f"    PROFIT MISMATCH: actual={tot_profit} reported={rep_profit}"); ok = False
for i in range(len(clique)):
    for j in range(i+1, len(clique)):
        u, v = clique[i], clique[j]
        if v not in adj[u]:
            print(f"    NOT A CLIQUE: {u},{v} not adjacent"); ok = False; break
status = "valid " if ok else "INVALID"
print(f"    [{label}] {status}  |  size={len(clique)} profit={tot_profit} cost={tot_cost}/{B}")
PYEOF
}

# ── 3. correctness check over all test graphs ───────────────────────────────
hdr "Correctness checks (np=4)"
PASS=0; FAIL=0

# Collect temp files for cleanup at exit (macOS mktemp needs Xs at the very end)
TMPFILES=()
cleanup() { rm -f "${TMPFILES[@]}" 2>/dev/null || true; }
trap cleanup EXIT

for INPUT in mac_tests/*.txt; do
    echo ""
    echo "  ── $(basename $INPUT) ──"

    OUT_SEQ=$(mktemp /tmp/seq_XXXXXX)
    OUT_PAR=$(mktemp /tmp/par_XXXXXX)
    TMPFILES+=("$OUT_SEQ" "$OUT_PAR")

    T_SEQ=$(elapsed "$SEQ_BIN" "$INPUT" "$OUT_SEQ")
    P_SEQ=$(head -1 "$OUT_SEQ")
    printf "  Sequential: profit=%-6s  time=%ss\n" "$P_SEQ" "$T_SEQ"
    validate "$INPUT" "$OUT_SEQ" "seq"

    T_PAR=$(elapsed mpirun -np 4 "$PAR_BIN" "$INPUT" "$OUT_PAR")
    P_PAR=$(head -1 "$OUT_PAR")
    printf "  Parallel:   profit=%-6s  time=%ss\n" "$P_PAR" "$T_PAR"
    validate "$INPUT" "$OUT_PAR" "par"

    if [ "$P_SEQ" = "$P_PAR" ]; then
        pass "Profits match: $P_SEQ"
        PASS=$((PASS+1))
    else
        fail "MISMATCH — seq=$P_SEQ  par=$P_PAR"
        FAIL=$((FAIL+1))
    fi
done

# ── 4. scaling sweep on medium2 graph (larger = more interesting) ────────────
hdr "Scaling sweep — mac_tests/medium2.txt"
echo ""
printf "%-8s  %-12s  %-10s  %-12s\n" "Procs" "Time (s)" "Speedup" "Efficiency"
printf "%-8s  %-12s  %-10s  %-12s\n" "-----" "--------" "-------" "----------"

BASE_T=""
for NP in $MAC_PROCS; do
    OUTF=$(mktemp /tmp/scale_XXXXXX)
    TMPFILES+=("$OUTF")
    T=$(elapsed mpirun -np "$NP" "$PAR_BIN" mac_tests/medium2.txt "$OUTF")
    [ -z "$BASE_T" ] && BASE_T="$T"
    python3 -c "
np=$NP; t=float('$T'); b=float('$BASE_T')
sp = b/t if t>0 else 0; ef = sp/np*100
print(f'{np:<8}  {t:<12.3f}  {sp:<10.3f}  {ef:<10.1f}%')
"
done

# ── 5. summary ───────────────────────────────────────────────────────────────
hdr "Summary"
echo -e "  Correctness: ${GREEN}$PASS passed${NC}  ${RED}$FAIL failed${NC}"
[ "$FAIL" -eq 0 ] && echo -e "  ${GREEN}All tests passed!${NC}" || echo -e "  ${RED}Some tests failed.${NC}"
