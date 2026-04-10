/*
 * main_bitset.cpp — MPI Parallel Branch and Bound (Hybrid Bitset Variant)
 *
 * Combines dynamic per-call candidate sorting (O(k log k)) with 
 * ultra-fast word-parallel bitset adjacency tracking (O(WORDS)).
 *
 * Compile: mpic++ -O3 -std=c++17 main_bitset.cpp -o par_bin_bitset
 * Run:     mpirun -np <P> ./par_bin_bitset input.txt output.txt
 */

#include <mpi.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;

/* ── graph data ────────────────────────────────────────────────────────── */
int N, E, B;
vector<int> profit;
vector<int> cost_v;
vector<vector<uint64_t>> adj_bits;   // adj_bits[u]: WORDS-word bitset of neighbors
int WORDS = 0;                        // ceil(N / 64)

/* ── pre-sorted vertex orderings (built once in main) ─────────────────── */
vector<int> profit_order;   // sorted by profit descending
vector<int> ratio_order;    // sorted by profit/cost descending

/* ── per-rank B&B state ────────────────────────────────────────────────── */
int         P_max = 0;
vector<int> best_clique;
vector<int> curr_clique;

/* ── MPI async communication ───────────────────────────────────────────── */
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
inline void set_bit(vector<uint64_t>& b, int x) {
    b[x >> 6] |= (1ULL << (x & 63));
}
inline bool test_bit(const vector<uint64_t>& b, int x) {
    return (b[x >> 6] >> (x & 63)) & 1ULL;
}

/* Extract candidate vertices from bitset (ascending order). */
static vector<int> bits_to_vec(const vector<uint64_t>& bits) {
    vector<int> res;
    res.reserve(64);
    for (int w = 0; w < WORDS; w++) {
        uint64_t word = bits[w];
        while (word) {
            int bit = __builtin_ctzll(word);
            res.push_back(w * 64 + bit);
            word &= word - 1;   // clear lowest set bit
        }
    }
    return res;
}

/* ── Structural bound: greedy graph coloring (Hybrid Bitset) ──────────────────
 * Sorts ONLY the current candidates, but uses bitsets for conflict tracking.
 * conflict[c] tracks all neighbors of vertices placed in color class c.
 */
static int colorBound(const vector<int>& cand) {
    vector<int> sc = cand;
    sort(sc.begin(), sc.end(), [](int a, int b){ return profit[a] > profit[b]; });

    vector<vector<uint64_t>> conflict;
    vector<int> class_max;
    conflict.reserve(32);
    class_max.reserve(32);

    for (int v : sc) {
        bool placed = false;
        for (int c = 0; c < (int)conflict.size(); c++) {
            if (test_bit(conflict[c], v)) continue;   // v is adjacent to a member of class c

            if (profit[v] > class_max[c]) class_max[c] = profit[v];
            
            // Mark all of v's neighbors as conflicting with this class
            for (int w = 0; w < WORDS; w++)
                conflict[c][w] |= adj_bits[v][w];     
            
            placed = true; break;
        }
        if (!placed) {
            conflict.emplace_back(WORDS, 0ULL);
            class_max.push_back(profit[v]);
            for (int w = 0; w < WORDS; w++)
                conflict.back()[w] |= adj_bits[v][w];
        }
    }

    int U = 0;
    for (int mx : class_max) U += mx;
    return U;
}

/* ── Resource bound: fractional knapsack ────────────────────── */
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

/* ── Branch and Bound (bitset candidate sets) ───────────────────────────── */
static void findClique(const vector<uint64_t>& cand_bits, int P_curr, int W_curr) {
    if (mpi_size_g > 1) check_inbox();

    /* Extract sorted list for iteration order AND bounding functions (once per call) */
    vector<int> cand = bits_to_vec(cand_bits);   // ascending order
    if (cand.empty()) return;

    /* ── 1. Structural bound ── */
    if (P_curr + colorBound(cand) <= P_max) return;

    /* ── 2. Resource bound ── */
    if ((double)P_curr + knapsackBound(cand, B - W_curr) <= (double)P_max) return;

    /* ── 3. Branching ── */
    /* Process back-to-front (highest index first), matching sort-based behaviour. */
    vector<uint64_t> rem_bits = cand_bits;

    for (int i = (int)cand.size() - 1; i >= 0; i--) {
        int v = cand[i];
        /* Remove v from the remaining set BEFORE recursing. */
        rem_bits[v >> 6] &= ~(1ULL << (v & 63));

        if (W_curr + cost_v[v] <= B) {
            if (P_curr + profit[v] > P_max) {
                P_max       = P_curr + profit[v];
                best_clique = curr_clique;
                best_clique.push_back(v);
                if (mpi_size_g > 1) broadcast_pmax(P_max);
            }

            /* C_next = rem_bits AND adj_bits[v]  —  O(WORDS) */
            vector<uint64_t> next_bits(WORDS);
            for (int w = 0; w < WORDS; w++)
                next_bits[w] = rem_bits[w] & adj_bits[v][w];

            curr_clique.push_back(v);
            findClique(next_bits, P_curr + profit[v], W_curr + cost_v[v]);
            curr_clique.pop_back();
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════ */
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_g);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_g);

    if (argc < 3) {
        if (mpi_rank_g == 0)
            cerr << "Usage: mpirun -np P ./par_bin_bitset <input> <output>\n";
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
        
        WORDS = (N + 63) >> 6; // MUST BE CALCULATED BEFORE ASSIGNING ADJ_BITS
        
        profit.resize(N); 
        cost_v.resize(N);
        adj_bits.assign(N, vector<uint64_t>(WORDS, 0ULL)); 
        
        for (int i = 0; i < N; i++) fin >> profit[i] >> cost_v[i];
        for (int i = 0; i < E; i++) {
            int u, v; fin >> u >> v;
            
            // Populate the bitsets
            set_bit(adj_bits[u], v);
            set_bit(adj_bits[v], u);
        }
    }

    /* ── build pre-sorted orderings (once) ── */
    profit_order.resize(N); iota(profit_order.begin(), profit_order.end(), 0);
    sort(profit_order.begin(), profit_order.end(),
         [](int a, int b){ return profit[a] > profit[b]; });

    ratio_order.resize(N); iota(ratio_order.begin(), ratio_order.end(), 0);
    sort(ratio_order.begin(), ratio_order.end(), [](int a, int b){
        return (double)profit[a] / cost_v[a] > (double)profit[b] / cost_v[b];
    });

    /* ── shared initial lower bound ── */
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

    vector<uint64_t> initial_bits(WORDS, 0ULL);
    for (int v : initial_cand) set_bit(initial_bits, v);

    /* ── static round-robin at first level ──────────────────────────────────
     * All ranks iterate the same loop and keep rem_bits synchronized.
     * Each rank independently computes C_next for its assigned tasks.       */
    vector<uint64_t> rem_bits = initial_bits;
    int task_id = 0;

    for (int i = (int)initial_cand.size() - 1; i >= 0; i--) {
        int v = initial_cand[i];
        /* Remove v from the shared remaining set (all ranks do this). */
        rem_bits[v >> 6] &= ~(1ULL << (v & 63));

        if (task_id % mpi_size_g == mpi_rank_g) {
            if (mpi_size_g > 1) check_inbox();

            if (profit[v] > P_max) {
                P_max       = profit[v];
                best_clique = {v};
                if (mpi_size_g > 1) broadcast_pmax(P_max);
            }

            /* C_next = rem_bits AND adj_bits[v]  —  O(WORDS) */
            vector<uint64_t> next_bits(WORDS);
            for (int w = 0; w < WORDS; w++)
                next_bits[w] = rem_bits[w] & adj_bits[v][w];

            curr_clique = {v};
            findClique(next_bits, profit[v], cost_v[v]);
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
     * Use actual sum of profit[best_clique] — not P_max which may be inflated
     * by received broadcasts without a corresponding local best_clique update. */
    int local_found = 0;
    for (int v : best_clique) local_found += profit[v];
    struct { int val; int rnk; } local_in = {local_found, mpi_rank_g}, global_out;
    MPI_Allreduce(&local_in, &global_out, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
    P_max         = global_out.val;
    int best_rank = global_out.rnk;

    /* ── broadcast winning clique, rank 0 writes output ── */
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