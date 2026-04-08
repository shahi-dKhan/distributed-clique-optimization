#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

int N, E, B;
vector<int> profit;
vector<int> cost_v;
vector<vector<bool>> adj;

int P_max = 0;
vector<int> best_clique;
vector<int> curr_clique;

// Structural Bound: Greedy Graph Coloring
// Sort candidates by profit descending, greedily assign to color classes (independent sets).
// U_color = sum of max profit in each color class.
int colorBound(const vector<int>& cand) {
    vector<int> sorted_cand = cand;
    sort(sorted_cand.begin(), sorted_cand.end(), [](int a, int b) {
        return profit[a] > profit[b];
    });

    // Each entry: list of vertices in that color class
    vector<vector<int>> classes;
    vector<int> class_max; // max profit in each class

    for (int v : sorted_cand) {
        bool placed = false;
        for (int i = 0; i < (int)classes.size(); i++) {
            // Check if v is non-adjacent to all members of this class
            bool compatible = true;
            for (int u : classes[i]) {
                if (adj[v][u]) { compatible = false; break; }
            }
            if (compatible) {
                classes[i].push_back(v);
                // profit is sorted descending so class_max[i] is always the first element,
                // but we track it explicitly for safety
                class_max[i] = max(class_max[i], profit[v]);
                placed = true;
                break;
            }
        }
        if (!placed) {
            classes.push_back({v});
            class_max.push_back(profit[v]);
        }
    }

    int U = 0;
    for (int mx : class_max) U += mx;
    return U;
}

// Resource Bound: Fractional Knapsack
// Sort candidates by p(v)/c(v) descending, greedily fill remaining budget (fractional last item).
double knapsackBound(const vector<int>& cand, int rem_budget) {
    if (rem_budget <= 0) return 0.0;

    vector<int> sorted_cand = cand;
    sort(sorted_cand.begin(), sorted_cand.end(), [](int a, int b) {
        return (double)profit[a] / cost_v[a] > (double)profit[b] / cost_v[b];
    });

    double U = 0.0;
    int rem = rem_budget;
    for (int v : sorted_cand) {
        if (rem <= 0) break;
        if (cost_v[v] <= rem) {
            U += profit[v];
            rem -= cost_v[v];
        } else {
            // Fractional addition for the last item
            U += (double)profit[v] * rem / cost_v[v];
            rem = 0;
        }
    }
    return U;
}

// Algorithm 1: Sequential Branch and Bound for Budgeted Max Clique with Graph Coloring
// cand  : candidate vertices that can extend the current clique
// P_curr: total profit of current clique
// W_curr: total cost of current clique
void findClique(vector<int> cand, int P_curr, int W_curr) {
    // --- 1. Structural Bound (Greedy Graph Coloring) ---
    int U_color = colorBound(cand);
    if (P_curr + U_color <= P_max) return; // prune: graph structure prevents improvement

    // --- 2. Resource Bound (Fractional Knapsack) ---
    double U_knap = knapsackBound(cand, B - W_curr);
    if ((double)P_curr + U_knap <= (double)P_max) return; // prune: budget prevents improvement

    // --- 3. Branching ---
    while (!cand.empty()) {
        int v = cand.back();
        cand.pop_back();

        if (W_curr + cost_v[v] <= B) {
            // Adding v to the clique is feasible; update global best
            if (P_curr + profit[v] > P_max) {
                P_max = P_curr + profit[v];
                best_clique = curr_clique;
                best_clique.push_back(v);
            }

            // C_next = remaining cand ∩ Neighbors(v)  -- maintains clique property
            vector<int> next_cand;
            next_cand.reserve(cand.size());
            for (int u : cand) {
                if (adj[v][u]) next_cand.push_back(u);
            }

            curr_clique.push_back(v);
            findClique(next_cand, P_curr + profit[v], W_curr + cost_v[v]);
            curr_clique.pop_back();
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    ifstream fin(argv[1]);
    if (!fin) { cerr << "Cannot open input file: " << argv[1] << "\n"; return 1; }
    ofstream fout(argv[2]);
    if (!fout) { cerr << "Cannot open output file: " << argv[2] << "\n"; return 1; }

    fin >> N >> E >> B;

    profit.resize(N);
    cost_v.resize(N);
    adj.assign(N, vector<bool>(N, false));

    for (int i = 0; i < N; i++) {
        fin >> profit[i] >> cost_v[i];
    }

    for (int i = 0; i < E; i++) {
        int u, v;
        fin >> u >> v;
        adj[u][v] = adj[v][u] = true;
    }

    // Initial candidate set: only vertices affordable within the full budget
    vector<int> initial_cand;
    for (int i = 0; i < N; i++) {
        if (cost_v[i] <= B) initial_cand.push_back(i);
    }

    findClique(initial_cand, 0, 0);

    sort(best_clique.begin(), best_clique.end());

    fout << P_max << "\n";
    for (int i = 0; i < (int)best_clique.size(); i++) {
        if (i > 0) fout << " ";
        fout << best_clique[i];
    }
    fout << "\n";

    return 0;
}
