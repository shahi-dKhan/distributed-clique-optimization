#include <mpi.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#include <cstring>
using namespace std;

constexpr int MAX_WORDS = 19;
using Bits = array<uint64_t, MAX_WORDS>;

int N, E, B;
vector<int>  profit;
vector<int>  cost_v;
int WORDS = 0;

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

/* ── precomputed data ───────────────────────────────────────────────────── */
static int bucket_data[101][1200];
static int bucket_sz[101];
uint64_t* adj_flat = nullptr;
uint64_t* conflict_flat = nullptr;
int* depth_cand_flat = nullptr;
vector<int> ratio_rank;
static int class_max[1200];

/* ── sort helpers ───────────────────────────────────────────────────────── */
static void sort_by_profit(int* cand_arr, int n) {
    if (n < 16) {
        sort(cand_arr, cand_arr + n, [](int a, int b){
            return profit[a] > profit[b];
        });
        return;
    }
    int lo = 101, hi = 0;
    for (int i = 0; i < n; i++) {
        int v = cand_arr[i];
        int p = profit[v];
        bucket_data[p][bucket_sz[p]++] = v;
        if (p < lo) lo = p;
        if (p > hi) hi = p;
    }
    int idx = 0;
    for (int p = hi; p >= lo; p--) {
        for (int j = 0; j < bucket_sz[p]; j++)
            cand_arr[idx++] = bucket_data[p][j];
        bucket_sz[p] = 0;
    }
}

/* ── MPI helpers ────────────────────────────────────────────────────────── */
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

/* ── bounds ─────────────────────────────────────────────────────────────── */
static int colorBound(const int* cand_arr, int n) {
    int nclasses = 0;

    for (int i = 0; i < n; i++) {
        int v = cand_arr[i];
        bool placed = false;
        const uint64_t* av = &adj_flat[v * WORDS];

        for (int c = 0; c < nclasses; c++) {
            uint64_t* cf = &conflict_flat[c * WORDS];
            if ((cf[v >> 6] >> (v & 63)) & 1ULL) continue;
            if (profit[v] > class_max[c]) class_max[c] = profit[v];
            for (int w = 0; w < WORDS; w++) cf[w] |= av[w];
            placed = true; break;
        }
        if (!placed) {
            uint64_t* cf = &conflict_flat[nclasses * WORDS];
            for (int w = 0; w < WORDS; w++) cf[w] = av[w];
            class_max[nclasses] = profit[v];
            nclasses++;
        }
    }

    int U = 0;
    for (int c = 0; c < nclasses; c++) U += class_max[c];
    for (int c = 0; c < nclasses; c++) {
        uint64_t* cf = &conflict_flat[c * WORDS];
        for (int w = 0; w < WORDS; w++) cf[w] = 0;
        class_max[c] = 0;
    }
    return U;
}

static double knapsackBound(const int* cand_arr, int n, int rem_budget) {
    if (rem_budget <= 0) return 0.0;
    double U = 0.0;
    int rem = rem_budget;
    for (int i = 0; i < n; i++) {
        int v = cand_arr[i];
        if (rem <= 0) break;
        if (cost_v[v] <= rem) { U += profit[v]; rem -= cost_v[v]; }
        else { U += (double)profit[v] * rem / cost_v[v]; rem = 0; }
    }
    return U;
}

/* ── branch and bound ───────────────────────────────────────────────────── */
static void findClique(const Bits& cand_bits, int P_curr, int W_curr, int depth) {
    if (mpi_size_g > 1) check_inbox();

    int* cand_arr = &depth_cand_flat[depth * 1200];
    int cand_sz = 0;
    for (int w = 0; w < WORDS; w++) {
        uint64_t word = cand_bits[w];
        while (word) {
            cand_arr[cand_sz++] = w * 64 + __builtin_ctzll(word);
            word &= word - 1;
        }
    }

    if (cand_sz == 0) return;
    if (cand_sz == 1) {
        int v = cand_arr[0];
        if (W_curr + cost_v[v] <= B && P_curr + profit[v] > P_max) {
            P_max = P_curr + profit[v];
            best_clique = curr_clique;
            best_clique.push_back(v);
            if (mpi_size_g > 1) broadcast_pmax(P_max);
        }
        return;
    }

    sort_by_profit(cand_arr, cand_sz);
    if (P_curr + colorBound(cand_arr, cand_sz) <= P_max) return;

    sort(cand_arr, cand_arr + cand_sz, [](int a, int b){
        return ratio_rank[a] < ratio_rank[b];
    });
    if ((double)P_curr + knapsackBound(cand_arr, cand_sz, B - W_curr) <= (double)P_max) return;

    Bits rem_bits = cand_bits;
    for (int i = 0; i < cand_sz; i++) {
        int v = cand_arr[i];
        rem_bits[v >> 6] &= ~(1ULL << (v & 63));
        if (W_curr + cost_v[v] <= B) {
            if (P_curr + profit[v] > P_max) {
                P_max = P_curr + profit[v];
                best_clique = curr_clique;
                best_clique.push_back(v);
                if (mpi_size_g > 1) broadcast_pmax(P_max);
            }
            Bits next_bits{};
            const uint64_t* av = &adj_flat[v * WORDS];
            for (int w = 0; w < WORDS; w++)
                next_bits[w] = rem_bits[w] & av[w];

            curr_clique.push_back(v);
            findClique(next_bits, P_curr + profit[v], W_curr + cost_v[v], depth + 1);
            curr_clique.pop_back();
        }
    }
}

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

    vector<Bits> adj_bits;
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

    /* ── build flat adjacency ───────────────────────────────────────────── */
    adj_flat = new uint64_t[N * WORDS]();
    for (int i = 0; i < N; i++)
        for (int w = 0; w < WORDS; w++)
            adj_flat[i * WORDS + w] = adj_bits[i][w];

    conflict_flat = new uint64_t[1200 * WORDS]();
    depth_cand_flat = new int[1200 * 1200]();

    /* ── precompute ratio rank ──────────────────────────────────────────── */
    ratio_rank.resize(N);
    {
        vector<int> order(N);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [](int a, int b){
            return (double)profit[a]/cost_v[a] > (double)profit[b]/cost_v[b];
        });
        for (int i = 0; i < N; i++) ratio_rank[order[i]] = i;
    }

    /* ── initial P_max from single vertices ─────────────────────────────── */
    for (int i = 0; i < N; i++) {
        if (cost_v[i] <= B && profit[i] > P_max) {
            P_max = profit[i]; best_clique = {i};
        }
    }

    /* ── build initial candidate set ────────────────────────────────────── */
    vector<int> initial_cand;
    for (int i = 0; i < N; i++)
        if (cost_v[i] <= B) initial_cand.push_back(i);

    Bits initial_bits{};
    for (int v : initial_cand) set_bit(initial_bits, v);

    /* ── precompute top-level task data ─────────────────────────────────── */
    int num_tasks = (int)initial_cand.size();
    vector<int>  task_vertex(num_tasks);
    vector<Bits> task_rembits(num_tasks);
    {
        Bits rb = initial_bits;
        for (int i = num_tasks - 1; i >= 0; i--) {
            int v = initial_cand[i];
            rb[v >> 6] &= ~(1ULL << (v & 63));
            task_vertex[i] = v;
            task_rembits[i] = rb;
        }
    }

    /* ── top-level branching with dynamic scheduling ────────────────────── */
    if (mpi_size_g == 1) {
        for (int t = num_tasks - 1; t >= 0; t--) {
            int v = task_vertex[t];
            if (profit[v] > P_max) {
                P_max = profit[v]; best_clique = {v};
            }
            Bits next_bits{};
            const uint64_t* av = &adj_flat[v * WORDS];
            for (int w = 0; w < WORDS; w++)
                next_bits[w] = task_rembits[t][w] & av[w];

            curr_clique = {v};
            findClique(next_bits, profit[v], cost_v[v], 1);
            curr_clique.clear();
        }
    } else {
        /* Dynamic self-scheduling: shared counter on rank 0 */
        int counter_val = num_tasks - 1;
        MPI_Win win;
        if (mpi_rank_g == 0) {
            MPI_Win_create(&counter_val, sizeof(int), sizeof(int),
                           MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        } else {
            MPI_Win_create(NULL, 0, sizeof(int),
                           MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        while (true) {
            int t;
            int decrement = -1;
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
            MPI_Fetch_and_op(&decrement, &t, MPI_INT, 0, 0, MPI_SUM, win);
            MPI_Win_unlock(0, win);

            if (t < 0) break;

            check_inbox();

            int v = task_vertex[t];
            if (profit[v] > P_max) {
                P_max = profit[v]; best_clique = {v};
                broadcast_pmax(P_max);
            }
            Bits next_bits{};
            const uint64_t* av = &adj_flat[v * WORDS];
            for (int w = 0; w < WORDS; w++)
                next_bits[w] = task_rembits[t][w] & av[w];

            curr_clique = {v};
            findClique(next_bits, profit[v], cost_v[v], 1);
            curr_clique.clear();
        }

        MPI_Win_free(&win);
    }

    if (mpi_size_g > 1) {
        for (int r = 0; r < mpi_size_g; r++)
            if (r != mpi_rank_g && g_send_req[r] != MPI_REQUEST_NULL)
                MPI_Wait(&g_send_req[r], MPI_STATUS_IGNORE);
        check_inbox();
    }

    /* ── global best via MPI_MAXLOC ─────────────────────────────────────── */
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

    delete[] adj_flat;
    delete[] conflict_flat;
    delete[] depth_cand_flat;
    MPI_Finalize();
    return 0;
}