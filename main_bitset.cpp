/*
 * main_bitset.cpp — MPI Parallel Branch and Bound (Bitset-accelerated variant)
 *
 *  1. Adjacency stored as fixed-size bitsets: array<uint64_t, MAX_WORDS>.
 *     N <= 1200 → MAX_WORDS = 19 (ceil(1200/64)).  Stack-allocated, fits in L1.
 *
 *  2. Candidate sets are Bits (array<uint64_t,19>) throughout findClique.
 *     C_next = rem_bits AND adj_bits[v]  — O(WORDS) ≈ O(N/64).
 *     No heap allocation in the hot recursive path — array lives on the stack.
 *
 *  3. colorBound: pre-sorted profit_order + bitset conflict tracking.
 *     Conflict classes use Bits arrays (stack-allocated, 152 bytes each).
 *
 *  4. knapsackBound: pre-sorted ratio_order, O(1) bit-test to skip non-candidates.
 *
 *  5. MPI async P_max sharing via MPI_Isend / MPI_Iprobe.
 *
 * Compile: mpic++ -O3 -std=c++17 main_bitset.cpp -o par_bin
 * Run:     mpirun -np <P> ./par_bin input.txt output.txt
 */

#include <mpi.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;

/* ── fixed-size bitset type ─────────────────────────────────────────────── */
/* N <= 1200 → ceil(1200/64) = 19 words max. Stack allocation, zero overhead. */
constexpr int MAX_WORDS = 19;
using Bits = array<uint64_t, MAX_WORDS>;

/* ── graph data ─────────────────────────────────────────────────────────── */
int N, E, B;
vector<int>  profit;
vector<int>  cost_v;
vector<Bits> adj_bits;   // adj_bits[u]: bitset of neighbors of u
int WORDS = 0;            // actual ceil(N/64) — loop bound for this graph

/* ── pre-sorted vertex orderings (built once in main) ────────────────────── */
vector<int> profit_order;   // sorted by profit descending
vector<int> ratio_order;    // sorted by profit/cost descending

/* ── per-rank B&B state ─────────────────────────────────────────────────── */
int         P_max = 0;
vector<int> best_clique;
vector<int> curr_clique;

/* ── MPI async communication ────────────────────────────────────────────── */
static const int TAG_PMAX  = 42;
static const int MAX_RANKS = 1024;
int mpi_rank_g = 0, mpi_size_g = 1;
static int         g_send_buf[MAX_RANKS];
static MPI_Request g_send_req[MAX_RANKS];

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

/* ── bitset helpers ─────────────────────────────────────────────────────── */
inline void set_bit(Bits& b, int x)             { b[x>>6] |=  (1ULL << (x&63)); }
inline bool test_bit(const Bits& b, int x)      { return (b[x>>6] >> (x&63)) & 1ULL; }

/* Extract candidate vertices from bitset (ascending order). */
static vector<int> bits_to_vec(const Bits& bits) {
    vector<int> res;
    res.reserve(64);
    for (int w = 0; w < WORDS; w++) {
        uint64_t word = bits[w];
        while (word) {
            res.push_back(w * 64 + __builtin_ctzll(word));
            word &= word - 1;
        }
    }
    return res;
}

/* ── Structural bound: pre-sorted greedy coloring with bitset conflicts ─── */
/*
 * Iterate vertices in profit_order (high → low).  Skip non-candidates.
 * Place each vertex in the first compatible color class (no adjacent member).
 * Conflict classes are Bits arrays — stack-allocated, 152 bytes each.
 * Upper bound = sum of max profit per class.
 */
static int colorBound(const Bits& cand_bits) {
    /* Use a fixed-size stack array for color classes (32 classes is ample). */
    static Bits  conflict[64];
    static int   class_max[64];
    int nclasses = 0;

    for (int v : profit_order) {
        if (!test_bit(cand_bits, v)) continue;

        bool placed = false;
        for (int c = 0; c < nclasses; c++) {
            if (test_bit(conflict[c], v)) continue;
            if (profit[v] > class_max[c]) class_max[c] = profit[v];
            for (int w = 0; w < WORDS; w++)
                conflict[c][w] |= adj_bits[v][w];
            placed = true; break;
        }
        if (!placed) {
            conflict[nclasses].fill(0);
            for (int w = 0; w < WORDS; w++)
                conflict[nclasses][w] = adj_bits[v][w];
            class_max[nclasses] = profit[v];
            nclasses++;
        }
    }

    int U = 0;
    for (int c = 0; c < nclasses; c++) U += class_max[c];
    return U;
}

/* ── Resource bound: pre-sorted fractional knapsack ────────────────────── */
static double knapsackBound(const Bits& cand_bits, int rem_budget) {
    if (rem_budget <= 0) return 0.0;
    double U = 0.0; int rem = rem_budget;
    for (int v : ratio_order) {
        if (!test_bit(cand_bits, v)) continue;
        if (rem <= 0) break;
        if (cost_v[v] <= rem) { U += profit[v]; rem -= cost_v[v]; }
        else { U += (double)profit[v] * rem / cost_v[v]; rem = 0; }
    }
    return U;
}

/* ── Branch and Bound ───────────────────────────────────────────────────── */
static void findClique(const Bits& cand_bits, int P_curr, int W_curr) {
    if (mpi_size_g > 1) check_inbox();

    /* ── 1. Structural bound ── */
    if (P_curr + colorBound(cand_bits) <= P_max) return;

    /* ── 2. Resource bound ── */
    if ((double)P_curr + knapsackBound(cand_bits, B - W_curr) <= (double)P_max) return;

    /* ── 3. Branching ── */
    vector<int> cand = bits_to_vec(cand_bits);

    Bits rem_bits = cand_bits;   // stack-allocated copy

    for (int i = (int)cand.size() - 1; i >= 0; i--) {
        int v = cand[i];
        rem_bits[v >> 6] &= ~(1ULL << (v & 63));

        if (W_curr + cost_v[v] <= B) {
            if (P_curr + profit[v] > P_max) {
                P_max       = P_curr + profit[v];
                best_clique = curr_clique;
                best_clique.push_back(v);
                if (mpi_size_g > 1) broadcast_pmax(P_max);
            }

            /* C_next = rem_bits AND adj_bits[v]  —  stack-allocated, O(WORDS) */
            Bits next_bits{};
            for (int w = 0; w < WORDS; w++)
                next_bits[w] = rem_bits[w] & adj_bits[v][w];

            curr_clique.push_back(v);
            findClique(next_bits, P_curr + profit[v], W_curr + cost_v[v]);
            curr_clique.pop_back();
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════ */
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

    /* ── read graph ── */
    {
        ifstream fin(argv[1]);
        if (!fin) {
            cerr << "[rank " << mpi_rank_g << "] Cannot open " << argv[1] << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fin >> N >> E >> B;
        WORDS = (N + 63) >> 6;
        profit.resize(N); cost_v.resize(N);
        Bits zero{}; adj_bits.assign(N, zero);
        for (int i = 0; i < N; i++) fin >> profit[i] >> cost_v[i];
        for (int i = 0; i < E; i++) {
            int u, v; fin >> u >> v;
            set_bit(adj_bits[u], v);
            set_bit(adj_bits[v], u);
        }
    }

    /* ── build pre-sorted orderings ── */
    profit_order.resize(N); iota(profit_order.begin(), profit_order.end(), 0);
    sort(profit_order.begin(), profit_order.end(),
         [](int a, int b){ return profit[a] > profit[b]; });

    ratio_order.resize(N); iota(ratio_order.begin(), ratio_order.end(), 0);
    sort(ratio_order.begin(), ratio_order.end(), [](int a, int b){
        return (double)profit[a]/cost_v[a] > (double)profit[b]/cost_v[b];
    });

    /* ── initial lower bound ── */
    for (int i = 0; i < N; i++) {
        if (cost_v[i] <= B && profit[i] > P_max) {
            P_max = profit[i]; best_clique = {i};
        }
    }

    /* ── initial candidate set ── */
    vector<int> initial_cand;
    for (int i = 0; i < N; i++)
        if (cost_v[i] <= B) initial_cand.push_back(i);

    Bits initial_bits{};
    for (int v : initial_cand) set_bit(initial_bits, v);

    /* ── static round-robin at first level ── */
    Bits rem_bits = initial_bits;
    int task_id = 0;

    for (int i = (int)initial_cand.size() - 1; i >= 0; i--) {
        int v = initial_cand[i];
        rem_bits[v >> 6] &= ~(1ULL << (v & 63));

        if (task_id % mpi_size_g == mpi_rank_g) {
            if (mpi_size_g > 1) check_inbox();

            if (profit[v] > P_max) {
                P_max = profit[v]; best_clique = {v};
                if (mpi_size_g > 1) broadcast_pmax(P_max);
            }

            Bits next_bits{};
            for (int w = 0; w < WORDS; w++)
                next_bits[w] = rem_bits[w] & adj_bits[v][w];

            curr_clique = {v};
            findClique(next_bits, profit[v], cost_v[v]);
            curr_clique.clear();
        }
        task_id++;
    }

    /* ── drain sends and inbox ── */
    if (mpi_size_g > 1) {
        for (int r = 0; r < mpi_size_g; r++)
            if (r != mpi_rank_g && g_send_req[r] != MPI_REQUEST_NULL)
                MPI_Wait(&g_send_req[r], MPI_STATUS_IGNORE);
        check_inbox();
    }

    /* ── global best via MPI_MAXLOC ── */
    int local_found = 0;
    for (int v : best_clique) local_found += profit[v];
    struct { int val; int rnk; } local_in = {local_found, mpi_rank_g}, global_out;
    MPI_Allreduce(&local_in, &global_out, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
    P_max         = global_out.val;
    int best_rank = global_out.rnk;

    int clique_size = (mpi_rank_g == best_rank) ? (int)best_clique.size() : 0;
    MPI_Bcast(&clique_size, 1, MPI_INT, best_rank, MPI_COMM_WORLD);
    if (mpi_rank_g != best_rank) best_clique.resize(clique_size);
    if (clique_size > 0)
        MPI_Bcast(best_clique.data(), clique_size, MPI_INT, best_rank, MPI_COMM_WORLD);

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
