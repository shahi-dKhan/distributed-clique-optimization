#!/usr/bin/env bash
# test_hpc.sh — HPC test suite (SLURM + OpenMPI)
#
# Two modes:
#   ./test_hpc.sh            → generates a SLURM job script and submits it
#   ./test_hpc.sh --direct   → runs directly (use inside an allocated node)
#
# Env overrides:
#   MPI_MODULE="amd/compilers/gcc/openmpi/5.0.8"  (OpenMPI module to load)
#   PARTITION="cpu"                                 (SLURM partition)
#   NTASKS=16                                       (max MPI ranks to test)

set -euo pipefail

MODE="${1:---slurm}"   # --slurm (default) | --direct

MPI_MODULE="${MPI_MODULE:-amd/compilers/gcc/openmpi/5.0.8}"
PARTITION="${PARTITION:-cpu}"
NTASKS="${NTASKS:-16}"
HPC_PROCS="1 2 4 8 16"

PAR_SRC="main.cpp"
SEQ_SRC="sequential.cpp"
PAR_BIN="./par_bin"
SEQ_BIN="./seq_bin"
WORKDIR="$(pwd)"

# ── colour helpers ──────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
info() { echo -e "${YELLOW}[INFO]${NC} $*"; }
hdr()  { echo -e "\n${BOLD}══ $* ══${NC}"; }

# ── generate test graphs (run on login node before submitting) ──────────────
generate_inputs() {
    hdr "Generating HPC test graphs"
    mkdir -p hpc_tests

    # medium graphs – correctness check (N≤100, finish quickly on any rank count)
    python3 gen.py -N 80  --density 0.5 -B 100 -s 10 -o hpc_tests/medium_a.txt
    python3 gen.py -N 80  --density 0.3 -B 150 -s 11 -o hpc_tests/medium_b.txt

    # large graphs – scaling benchmark (N=200-300, ~5-30s sequential)
    python3 gen.py -N 200 --density 0.6 -B 150 -s 20 -o hpc_tests/large_a.txt
    python3 gen.py -N 300 --density 0.5 -B 200 -s 21 -o hpc_tests/large_b.txt

    # the provided report graph
    if [ -f input_report.txt ]; then
        info "input_report.txt (N=500, E=74743, B=90) will be used for scaling"
    fi

    # ── EXTRA-LARGE graphs (commented out) ────────────────────────────────────
    # These push toward the upper end of the constraint space (N up to 1200).
    # Sequential may take many minutes; intended for 8-16 rank scaling runs on HPC.
    # Uncomment individually as needed.
    #
    # N=500, sparse (density 0.2) – many candidates, budget-constrained; B=500
    # python3 gen.py -N 500  --density 0.2 -B 500  -s 30 -o hpc_tests/xlarge_sparse.txt
    #
    # N=500, dense (density 0.6) – strong structural pruning, B=200
    # python3 gen.py -N 500  --density 0.6 -B 200  -s 31 -o hpc_tests/xlarge_dense.txt
    #
    # N=800, medium density – stresses both bounds equally, B=400
    # python3 gen.py -N 800  --density 0.4 -B 400  -s 40 -o hpc_tests/xxlarge_a.txt
    #
    # N=1000, sparse – knapsack bound dominates; expected within 30-min sequential
    # python3 gen.py -N 1000 --density 0.15 -B 600 -s 50 -o hpc_tests/xxlarge_b.txt
    #
    # N=1200, max constraint, very sparse – extreme scaling test
    # python3 gen.py -N 1200 --density 0.1  -B 800 -s 60 -o hpc_tests/max_scale.txt
    # ── end extra-large ───────────────────────────────────────────────────────

    info "Test inputs ready in hpc_tests/"
}

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
    clique = list(map(int, f.readline().strip().split()))
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
status = "VALID  " if ok else "INVALID"
print(f"    [{label}] {status}  size={len(clique)} profit={tot_profit} cost={tot_cost}/{B}")
PYEOF
}

# ── compile (called inside job or directly) ──────────────────────────────────
compile_all() {
    hdr "Compilation"
    module load "$MPI_MODULE" 2>/dev/null || true
    g++    -O3 -std=c++17 "$SEQ_SRC" -o "$SEQ_BIN" && info "Sequential compiled"
    mpic++ -O3 -std=c++17 "$PAR_SRC" -o "$PAR_BIN" && info "Parallel   compiled"
}

# ── correctness check ────────────────────────────────────────────────────────
run_correctness() {
    hdr "Correctness checks"
    PASS=0; FAIL=0

    for INPUT in hpc_tests/medium_a.txt hpc_tests/medium_b.txt; do
        echo ""
        echo "  ── $(basename $INPUT) ──"
        OUT_SEQ=$(mktemp /tmp/seq_XXXXXX); OUT_PAR=$(mktemp /tmp/par_XXXXXX)
        trap "rm -f $OUT_SEQ $OUT_PAR" EXIT

        T_SEQ=$(elapsed "$SEQ_BIN" "$INPUT" "$OUT_SEQ")
        P_SEQ=$(head -1 "$OUT_SEQ")
        printf "  Sequential: profit=%-6s  time=%ss\n" "$P_SEQ" "$T_SEQ"
        validate "$INPUT" "$OUT_SEQ" "seq"

        T_PAR=$(elapsed mpirun -np 4 "$PAR_BIN" "$INPUT" "$OUT_PAR")
        P_PAR=$(head -1 "$OUT_PAR")
        printf "  Parallel:   profit=%-6s  time=%ss\n" "$P_PAR" "$T_PAR"
        validate "$INPUT" "$OUT_PAR" "par"

        if [ "$P_SEQ" = "$P_PAR" ]; then
            echo -e "    ${GREEN}[PASS]${NC} Profits match: $P_SEQ"; PASS=$((PASS+1))
        else
            echo -e "    \033[0;31m[FAIL]\033[0m MISMATCH seq=$P_SEQ par=$P_PAR"; FAIL=$((FAIL+1))
        fi
    done
    echo ""
    echo "  Correctness: $PASS passed, $FAIL failed"
}

# ── scaling sweep ────────────────────────────────────────────────────────────
run_scaling() {
    local INPUT="${1:-input_report.txt}"
    hdr "Strong Scaling — $(basename $INPUT)"
    echo ""
    printf "%-8s  %-12s  %-10s  %-12s  %s\n" "Procs" "Time (s)" "Speedup" "Efficiency" "Profit"
    printf "%-8s  %-12s  %-10s  %-12s  %s\n" "-----" "--------" "-------" "----------" "------"

    CSV="scaling_$(basename ${INPUT%.txt}).csv"
    echo "procs,time_s,speedup,efficiency,profit" > "$CSV"

    BASE_T=""
    for NP in $HPC_PROCS; do
        [ "$NP" -gt "$NTASKS" ] && continue
        OUTF=$(mktemp /tmp/scale_XXXXXX); trap "rm -f $OUTF" EXIT
        T=$(elapsed mpirun -np "$NP" "$PAR_BIN" "$INPUT" "$OUTF")
        PROFIT=$(head -1 "$OUTF")
        [ -z "$BASE_T" ] && BASE_T="$T"
        python3 -c "
np=$NP; t=float('$T'); b=float('$BASE_T')
sp = b/t if t>0 else 0; ef = sp/np*100
print(f'{np:<8}  {t:<12.3f}  {sp:<10.3f}  {ef:<10.1f}%   $PROFIT')
"
        python3 -c "
np=$NP; t=float('$T'); b=float('$BASE_T')
sp = b/t if t>0 else 0; ef = sp/np*100
print(f'$NP,{t},{sp:.3f},{ef:.1f},$PROFIT')
" >> "$CSV"
    done
    echo ""
    info "Results saved to $CSV"
}

# ════════════════════════════════════════════════════════════════════════════
# SLURM MODE: generate test data here, then write + submit a job script
# ════════════════════════════════════════════════════════════════════════════
if [ "$MODE" = "--slurm" ]; then
    generate_inputs

    JOBSCRIPT="hpc_job.sh"
    cat > "$JOBSCRIPT" <<SLURM
#!/usr/bin/env bash
#SBATCH --job-name=mpi_clique
#SBATCH --output=hpc_job_%j.log
#SBATCH --error=hpc_job_%j.err
#SBATCH --ntasks=$NTASKS
#SBATCH --cpus-per-task=1
#SBATCH --partition=$PARTITION
#SBATCH --time=01:00:00

# Load MPI module (adjust to match your HPC's available module)
module load $MPI_MODULE

cd "$WORKDIR"

# Compile
g++    -O3 -std=c++17 $SEQ_SRC -o $SEQ_BIN
mpic++ -O3 -std=c++17 $PAR_SRC -o $PAR_BIN

# Run correctness checks
bash "$WORKDIR/test_hpc.sh" --direct --correctness-only

# Run scaling on the provided report graph
bash "$WORKDIR/test_hpc.sh" --direct --scaling-only input_report.txt

# Also scale on the large generated graph
bash "$WORKDIR/test_hpc.sh" --direct --scaling-only hpc_tests/large_a.txt
SLURM

    info "SLURM job script written: $JOBSCRIPT"
    echo ""
    echo "  To submit:  sbatch $JOBSCRIPT"
    echo "  To monitor: squeue -u \$USER"
    echo "  To view log after completion: cat hpc_job_<JOBID>.log"
    echo ""
    echo "  Alternatively, to run interactively on an allocated node:"
    echo "    salloc --ntasks=$NTASKS --partition=$PARTITION"
    echo "    ./test_hpc.sh --direct"
    exit 0
fi

# ════════════════════════════════════════════════════════════════════════════
# DIRECT MODE: run everything now (inside salloc or sbatch)
# ════════════════════════════════════════════════════════════════════════════
SUBMODE="${2:---all}"
EXTRA="${3:-input_report.txt}"

module load "$MPI_MODULE" 2>/dev/null || true

case "$SUBMODE" in
    --correctness-only)
        compile_all
        run_correctness
        ;;
    --scaling-only)
        compile_all
        run_scaling "$EXTRA"
        ;;
    *)
        compile_all
        run_correctness
        run_scaling "input_report.txt"
        [ -f hpc_tests/large_a.txt ] && run_scaling "hpc_tests/large_a.txt"
        hdr "All done"
        ;;
esac
