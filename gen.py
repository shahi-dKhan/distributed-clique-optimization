#!/usr/bin/env python3
"""
gen.py — Random graph generator for the Budgeted Max-Weight Clique problem.

Usage examples:
    python3 gen.py -N 50 -E 200 -B 100 -o test.txt
    python3 gen.py -N 100 --density 0.5 -B 200 -s 42 -o dense.txt
    python3 gen.py -N 500 --density 0.6 -B 90 -s 1 -o report_like.txt

Constraints (from assignment):
    10 <= N <= 1200
    0  <= E <= N*(N-1)/2
    10 <= B <= 1000
    1  <= p(v), c(v) <= 100
"""

import argparse
import random
import sys


def generate(N: int, E: int, B: int, seed, out_path):
    max_edges = N * (N - 1) // 2
    E = min(E, max_edges)

    if seed is not None:
        random.seed(seed)

    profits = [random.randint(1, 100) for _ in range(N)]
    costs   = [random.randint(1, 100) for _ in range(N)]

    # Sample E unique edges uniformly
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    random.shuffle(all_edges)
    edges = all_edges[:E]

    lines = [f"{N} {len(edges)} {B}"]
    for i in range(N):
        lines.append(f"{profits[i]} {costs[i]}")
    for u, v in edges:
        lines.append(f"{u} {v}")
    content = "\n".join(lines) + "\n"

    if out_path:
        with open(out_path, "w") as f:
            f.write(content)
        density = len(edges) / max_edges if max_edges > 0 else 0
        print(f"Generated: N={N}, E={len(edges)}, B={B}, density={density:.3f}  →  {out_path}")
    else:
        sys.stdout.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Generate random Budgeted Max-Clique instances."
    )
    parser.add_argument("-N", type=int, required=True,
                        help="Number of vertices (10–1200)")
    parser.add_argument("-B", type=int, required=True,
                        help="Budget (10–1000)")

    edge_grp = parser.add_mutually_exclusive_group(required=True)
    edge_grp.add_argument("-E", type=int,
                          help="Exact number of edges")
    edge_grp.add_argument("--density", type=float,
                          help="Edge density in [0, 1]  (fraction of max possible edges)")

    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file path (default: stdout)")
    args = parser.parse_args()

    max_edges = args.N * (args.N - 1) // 2
    if args.E is not None:
        E = args.E
    else:
        E = int(args.density * max_edges)

    generate(args.N, E, args.B, args.seed, args.output)


if __name__ == "__main__":
    main()
