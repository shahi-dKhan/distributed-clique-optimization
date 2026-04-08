/*
 * main.cpp — MPI Parallel Branch and Bound for Budgeted Maximum Weight Clique
 *
 * Key optimizations over a naive sequential implementation:
 *
 *  1. Pre-sorted global arrays (profit_order, ratio_order).
 *     Sort once in main(); bound functions iterate the fixed global array
 *     and use a local O(1) membership bitset, eliminating all per-call sorts.
 *
 *  2. Local membership bitset inside each bound call.
 *     Each call to colorBound / knapsackBound builds a local vector<bool>
 *     of size N from the explicit candidate set.  Initializing N bools costs
 *     ~N/64 cache-line writes (negligible for N≤1200).  This avoids the
 *     global in_cand[] approach which required expensive clear/restore loops
 *     at every branching step.
 *
 *  3. Bitset-accelerated greedy coloring.
 *     Conflict tracking uses uint64_t bitsets (WORDS words per class) so
 *     conflict tests and neighbor-marking are word-parallel.
 *
 *  4. Asynchronous P_max sharing (MPI_Isend / MPI_Iprobe).
 *     Whenever a rank finds a new best it broadcasts the value to every other
 *     rank with a non-blocking send.  Every recursive call drains the inbox
 *     with MPI_Iprobe before computing bounds, so all ranks immediately
 *     benefit from each other's discoveries.
 *
 *  5. Static round-robin task distribution at the first branching level.
 *     All ranks iterate the same outer loop (keeping cand_copy in sync) so
 *     each rank can compute C_next independently without communication.
 *
 * Compile: mpic++ -O3 -std=c++17 main.cpp -o par_bin
 * Run:     mpirun -np <P> ./par_bin input.txt output.txt
 */

#include <mpi.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;

/* ── graph data (read once, shared across all calls) ──────────────────────── */
int N, E, B;
vector<int> profit;
vector<int> cost_v;
vector<vector<bool>> adj;
vector<vector<int>> nbrs;   // adjacency list (for fast neighbor iteration)
int WORDS = 0;              // ceil(N/64) — set after N is known

/* ── pre-sorted vertex orderings (built once in main) ─────────────────────── */
vector<int> profit_order;   // vertices sorted by profit descending
vector<int> ratio_order;    // vertices sorted by profit/cost descending

/* ── per-rank B&B state ─────────────────────────────────────────────────── */
int         P_max = 0;
vector<int> best_clique;
vector<int> curr_clique;

/* ── MPI async communication ─────────────────────────────────────────────── */
static const int TAG_PMAX  = 42;
static const int MAX_RANKS = 1024;
int mpi_rank_g = 0, mpi_size_g = 1;

static int         g_send_buf[MAX_RANKS];
static MPI_Request g_send_req[MAX_RANKS];

/* Non-blocking best-effort broadcast of a new P_max to every other rank. */
static void broadcast_pmax(int val) {
    for (int r = 0; r < mpi_size_g; r++) {
        if (r == mpi_rank_g) continue;
        if (g_send_req[r] != MPI_REQUEST_NULL) {
            int done = 0;
            MPI_Test(&g_send_req[r], &done, MPI_STATUS_IGNORE);
            if (!done) continue;
        }
        g_send_buf[r] = val;
        MPI_Isend(&g_send_buf[r], 1, MPI_INT, r, TAG_PMAX,
                  MPI_COMM_WORLD, &g_send_req[r]);
    }
}

static void check_inbox() {
    int flag = 1;
    MPI_Status st;
    while (flag) {
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_PMAX, MPI_COMM_WORLD, &flag, &st);
        if (flag) {
            int val;
            MPI_Recv(&val, 1, MPI_INT, st.MPI_SOURCE, TAG_PMAX,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (val > P_max) P_max = val;
        }
    }
}

/* ── bitset helpers (used by pre-sorted bounds below) ─────────────────────── */
inline void set_bit(vector<uint64_t>& b, int x) {
    b[x >> 6] |= (1ULL << (x & 63));
}
inline bool test_bit(const vector<uint64_t>& b, int x) {
    return (b[x >> 6] >> (x & 63)) & 1ULL;
}

/* ════════════════════════════════════════════════════════════════════════════
 * ACTIVE: sort-based bounds (identical to sequential.cpp)
 * These sort only the current candidate set (size k) on each call — O(k log k).
 * For k << N this is faster than iterating all N vertices in a pre-sorted array.
 * ════════════════════════════════════════════════════════════════════════════ */

/* ── Structural bound: greedy graph coloring ──────────────────────────────────
 * Sort candidates by profit descending, greedily assign to color classes.
 * Upper bound = sum of max profit per class.                                  */
static int colorBound(const vector<int>& cand) {
    vector<int> sc = cand;
    sort(sc.begin(), sc.end(), [](int a, int b){ return profit[a] > profit[b]; });

    vector<vector<int>> classes;
    vector<int>         class_max;
    classes.reserve(32);
    class_max.reserve(32);

    for (int v : sc) {
        bool placed = false;
        for (int i = 0; i < (int)classes.size(); i++) {
            bool ok = true;
            for (int u : classes[i]) if (adj[v][u]) { ok = false; break; }
            if (ok) {
                if (profit[v] > class_max[i]) class_max[i] = profit[v];
                classes[i].push_back(v);
                placed = true; break;
            }
        }
        if (!placed) { classes.push_back({v}); class_max.push_back(profit[v]); }
    }

    int U = 0;
    for (int mx : class_max) U += mx;
    return U;
}

/* ── Resource bound: fractional knapsack ──────────────────────────────────────
 * Sort candidates by p/c descending, greedily fill remaining budget.          */
static double knapsackBound(const vector<int>& cand, int rem_budget) {
    if (rem_budget <= 0) return 0.0;

    vector<int> sc = cand;
    sort(sc.begin(), sc.end(), [](int a, int b){
        return (double)profit[a]/cost_v[a] > (double)profit[b]/cost_v[b];
    });

    double U = 0.0; int rem = rem_budget;
    for (int v : sc) {
        if (rem <= 0) break;
        if (cost_v[v] <= rem) { U += profit[v]; rem -= cost_v[v]; }
        else { U += (double)profit[v] * rem / cost_v[v]; rem = 0; }
    }
    return U;
}

/* ════════════════════════════════════════════════════════════════════════════
 * ALTERNATIVE (commented out): pre-sorted bounds
 *
 * profit_order / ratio_order are sorted once in main() and never re-sorted.
 * Each bound call builds a local is_c(N) membership array and iterates the
 * global pre-sorted array — O(N) per call instead of O(k log k).
 *
 * Faster when k ≈ N (shallow levels of the tree); slower when k << N
 * (deep levels), which is why the sort-based version above wins overall.
 *
 * To enable: comment out the sort-based versions above and uncomment below.
 * ════════════════════════════════════════════════════════════════════════════

static int colorBound_presorted(const vector<int>& cand) {
    vector<bool> is_c(N, false);
    for (int v : cand) is_c[v] = true;

    vector<vector<uint64_t>> conflict;
    vector<int>              class_max;
    conflict.reserve(32);
    class_max.reserve(32);

    for (int v : profit_order) {       // global order, no sort per call
        if (!is_c[v]) continue;

        bool placed = false;
        for (int c = 0; c < (int)conflict.size(); c++) {
            if (test_bit(conflict[c], v)) continue;
            if (profit[v] > class_max[c]) class_max[c] = profit[v];
            for (int u : nbrs[v])
                if (is_c[u]) set_bit(conflict[c], u);
            placed = true; break;
        }
        if (!placed) {
            conflict.emplace_back(WORDS, 0ULL);
            class_max.push_back(profit[v]);
            for (int u : nbrs[v])
                if (is_c[u]) set_bit(conflict.back(), u);
        }
    }

    int U = 0;
    for (int mx : class_max) U += mx;
    return U;
}

static double knapsackBound_presorted(const vector<int>& cand, int rem_budget) {
    if (rem_budget <= 0) return 0.0;

    vector<bool> is_c(N, false);
    for (int v : cand) is_c[v] = true;

    double U = 0.0; int rem = rem_budget;
    for (int v : ratio_order) {        // global order, no sort per call
        if (!is_c[v]) continue;
        if (rem <= 0) break;
        if (cost_v[v] <= rem) { U += profit[v]; rem -= cost_v[v]; }
        else { U += (double)profit[v] * rem / cost_v[v]; rem = 0; }
    }
    return U;
}

 * ════════════════════════════════════════════════════════════════════════════
 * End of pre-sorted alternative
 * ════════════════════════════════════════════════════════════════════════════ */

/* ── Algorithm 1: Branch and Bound (exact, per rank) ─────────────────────── */
static void findClique(const vector<int>& cand, int P_curr, int W_curr) {
    /* ── inbox check for MPI progress and fresher global lower bound ── */
    if (mpi_size_g > 1)
        check_inbox();

    /* ── 1. Structural bound ── */
    if (P_curr + colorBound(cand) <= P_max) return;

    /* ── 2. Resource bound ── */
    if ((double)P_curr + knapsackBound(cand, B - W_curr) <= (double)P_max) return;

    /* ── 3. Branching ── */
    /* Work on a mutable copy so we can shrink it as we branch. */
    vector<int> rem(cand);
    while (!rem.empty()) {
        int v = rem.back();
        rem.pop_back();

        if (W_curr + cost_v[v] <= B) {
            if (P_curr + profit[v] > P_max) {
                P_max       = P_curr + profit[v];
                best_clique = curr_clique;
                best_clique.push_back(v);
                if (mpi_size_g > 1) broadcast_pmax(P_max);
            }

            /* C_next = remaining candidates ∩ Neighbors(v) */
            vector<int> next_cand;
            next_cand.reserve(rem.size());
            for (int u : rem)
                if (adj[v][u]) next_cand.push_back(u);

            curr_clique.push_back(v);
            findClique(next_cand, P_curr + profit[v], W_curr + cost_v[v]);
            curr_clique.pop_back();
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════ */
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_g);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_g);

    if (argc < 3) {
        if (mpi_rank_g == 0)
            cerr << "Usage: mpirun -np P ./par_bin <input> <output>\n";
        MPI_Finalize(); return 1;
    }

    for (int r = 0; r < MAX_RANKS; r++) g_send_req[r] = MPI_REQUEST_NULL;

    /* ── read graph (all ranks read independently) ── */
    {
        ifstream fin(argv[1]);
        if (!fin) {
            cerr << "[rank " << mpi_rank_g << "] Cannot open " << argv[1] << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fin >> N >> E >> B;
        profit.resize(N); cost_v.resize(N);
        adj.assign(N, vector<bool>(N, false));
        for (int i = 0; i < N; i++) fin >> profit[i] >> cost_v[i];
        for (int i = 0; i < E; i++) {
            int u, v; fin >> u >> v;
            adj[u][v] = adj[v][u] = true;
        }
    }

    /* ── build adjacency list and bitset word count ── */
    WORDS = (N + 63) >> 6;
    nbrs.resize(N);
    for (int u = 0; u < N; u++)
        for (int v = 0; v < N; v++)
            if (adj[u][v]) nbrs[u].push_back(v);

    /* ── build pre-sorted orderings (done ONCE) ── */
    profit_order.resize(N); iota(profit_order.begin(), profit_order.end(), 0);
    sort(profit_order.begin(), profit_order.end(),
         [](int a, int b){ return profit[a] > profit[b]; });

    ratio_order.resize(N); iota(ratio_order.begin(), ratio_order.end(), 0);
    sort(ratio_order.begin(), ratio_order.end(), [](int a, int b){
        return (double)profit[a] / cost_v[a] > (double)profit[b] / cost_v[b];
    });

    /* ── shared initial lower bound (single affordable vertex) ── */
    for (int i = 0; i < N; i++) {
        if (cost_v[i] <= B && profit[i] > P_max) {
            P_max       = profit[i];
            best_clique = {i};
        }
    }

    /* ── initial candidate set ── */
    vector<int> initial_cand;
    for (int i = 0; i < N; i++)
        if (cost_v[i] <= B) initial_cand.push_back(i);

    /* ── static round-robin distribution of first-level tasks ──────────────
     * All ranks run the same outer loop so cand_copy stays identical at every
     * iteration; each rank computes C_next independently without communication. */
    vector<int> cand_copy = initial_cand;
    int task_id = 0;

    while (!cand_copy.empty()) {
        int v = cand_copy.back();
        cand_copy.pop_back();

        if (task_id % mpi_size_g == mpi_rank_g) {
            if (mpi_size_g > 1) check_inbox();

            vector<int> next_cand;
            next_cand.reserve(cand_copy.size());
            for (int u : cand_copy)
                if (adj[v][u]) next_cand.push_back(u);

            if (profit[v] > P_max) {
                P_max       = profit[v];
                best_clique = {v};
                if (mpi_size_g > 1) broadcast_pmax(P_max);
            }

            curr_clique = {v};
            findClique(next_cand, profit[v], cost_v[v]);
            curr_clique.clear();
        }

        task_id++;
    }

    /* ── drain pending sends and inbox before collective ── */
    if (mpi_size_g > 1) {
        for (int r = 0; r < mpi_size_g; r++)
            if (r != mpi_rank_g && g_send_req[r] != MPI_REQUEST_NULL)
                MPI_Wait(&g_send_req[r], MPI_STATUS_IGNORE);
        check_inbox();
    }

    /* ── find global best across all ranks ──────────────────────────────────
     * Use the actual profit of best_clique (locally found) rather than P_max.
     * P_max may have been inflated by received broadcasts without updating
     * best_clique, so using P_max in MAXLOC could select a rank that holds
     * the wrong clique.                                                      */
    int local_found = 0;
    for (int v : best_clique) local_found += profit[v];
    struct { int val; int rnk; } local_in = {local_found, mpi_rank_g}, global_out;
    MPI_Allreduce(&local_in, &global_out, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
    P_max         = global_out.val;
    int best_rank = global_out.rnk;

    /* ── broadcast winning clique from best_rank to all, then rank 0 outputs ── */
    int clique_size = (mpi_rank_g == best_rank) ? (int)best_clique.size() : 0;
    MPI_Bcast(&clique_size, 1, MPI_INT, best_rank, MPI_COMM_WORLD);
    if (mpi_rank_g != best_rank) best_clique.resize(clique_size);
    if (clique_size > 0)
        MPI_Bcast(best_clique.data(), clique_size, MPI_INT, best_rank, MPI_COMM_WORLD);

    /* ── output (rank 0 only) ── */
    if (mpi_rank_g == 0) {
        sort(best_clique.begin(), best_clique.end());
        ofstream fout(argv[2]);
        if (!fout) { cerr << "Cannot open output: " << argv[2] << "\n"; MPI_Finalize(); return 1; }
        fout << P_max << "\n";
        for (int i = 0; i < clique_size; i++) {
            if (i > 0) fout << " ";
            fout << best_clique[i];
        }
        fout << "\n";
    }

    MPI_Finalize();
    return 0;
}
