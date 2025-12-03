#!/usr/bin/env python3

"""
TSP homework solver for 1000-node graphs (Colab-friendly).

Input format (both graphs):
    (optional) first line: "1000"
    second line: "Node1 Node2 Distance"
    following lines: i j d

Example:
    1000
    Node1 Node2 Distance
    1 2 26.98
    1 3 53.74
    ...

Algorithm:
    Nearest Neighbor (random start) + 2-opt local search under a time limit.
"""

import argparse
import random
import time
from typing import List, Tuple


# =========================================================
# Graph loading
# =========================================================

def load_edge_list_graph(path: str) -> List[List[float]]:
    """
    Load a complete graph from an edge list:

        Node1 Node2 Distance

    Works for both the Euclidean and random graph files.
    Handles an optional first line with just "1000".
    Returns an n x n symmetric distance matrix (0-based indices).
    """
    edges: List[Tuple[int, int, float]] = []
    n = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # If line is just "1000", treat it as node count
            if len(parts) == 1 and parts[0].isdigit():
                n = int(parts[0])
                continue

            # Skip header line
            if parts[0].lower() == "node1":
                continue

            if len(parts) >= 3:
                try:
                    i = int(parts[0])
                    j = int(parts[1])
                    d = float(parts[2])
                except ValueError:
                    continue
                edges.append((i, j, d))

    if not edges:
        raise ValueError("No edges read from file; check format.")

    if n is None:
        max_node = max(max(i, j) for i, j, _ in edges)
        n = max_node

    dist: List[List[float]] = [[0.0] * n for _ in range(n)]

    for i, j, d in edges:
        i0 = i - 1
        j0 = j - 1
        dist[i0][j0] = d
        dist[j0][i0] = d

    return dist


# =========================================================
# TSP helpers
# =========================================================

def tour_cost(tour: List[int], dist: List[List[float]]) -> float:
    """Cost of a tour that returns to the starting node."""
    total = 0.0
    n = len(tour)
    for k in range(n - 1):
        total += dist[tour[k]][tour[k + 1]]
    total += dist[tour[-1]][tour[0]]
    return total


def nearest_neighbor_tour(start: int, dist: List[List[float]]) -> List[int]:
    """Nearest Neighbor initial tour from a given start node."""
    n = len(dist)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start

    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[current][j])
        unvisited.remove(nxt)
        tour.append(nxt)
        current = nxt

    return tour


def two_opt_fast(tour, dist, time_limit, start_time, cycle_counter, initial_cost):
    """
    Faster 2-opt:
      - Uses O(1) delta cost instead of recomputing full tour cost.
      - Modifies 'tour' in-place.
      - Returns (improved_tour, improved_cost).
    """
    import time

    n = len(tour)
    best_cost = initial_cost
    improved = True

    while improved and (time.time() - start_time < time_limit):
        improved = False

        for i in range(n - 1):
            if time.time() - start_time >= time_limit:
                break

            a = tour[i]
            b = tour[(i + 1) % n]

            for j in range(i + 2, n):
                # avoid swapping the first and last edge (would break the cycle)
                if i == 0 and j == n - 1:
                    continue

                c = tour[j]
                d = tour[(j + 1) % n]

                # Δ cost of replacing (a,b) and (c,d) with (a,c) and (b,d)
                delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])

                # count this candidate as a visited "cycle"
                cycle_counter[0] += 1

                if delta < -1e-9:   # improvement
                    # apply the reversal in-place: reverse segment (i+1 .. j)
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    best_cost += delta
                    improved = True
                    break

            if improved:
                break

    return tour, best_cost


# =========================================================
# Main solving loop
# =========================================================

def solve_tsp(dist: List[List[float]], time_limit: float, seed: int = 0):
    """
    Repeated NN + 2-opt restarts until time_limit is reached.
    Returns best_tour, best_cost, cycles_evaluated, elapsed_time.
    """
    random.seed(seed)
    n = len(dist)
    start_time = time.time()

    best_global_tour: List[int] = []
    best_global_cost = float("inf")
    cycles = [0]

    # main loop
    while time.time() - start_time < time_limit:
        start_node = random.randrange(n)
        tour = nearest_neighbor_tour(start_node, dist)
        cost_nn = tour_cost(tour, dist)
        cycles[0] += 1

        tour, cost = two_opt_fast(
            tour,
            dist,
            time_limit=time_limit,
            start_time=start_time,
            cycle_counter=cycles,
            initial_cost=cost_nn,
        )


        if cost < best_global_cost:
            best_global_cost = cost
            best_global_tour = tour

    elapsed = time.time() - start_time
    return best_global_tour, best_global_cost, cycles[0], elapsed


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Heuristic TSP solver for graphs A and B."
    )
    parser.add_argument("--graph", choices=["A", "B"], required=True,
                        help="Graph type (A=Euclidean, B=Random) – used for file naming.")
    parser.add_argument("--input", required=True,
                        help="Input filename (edge list).")
    parser.add_argument("--sid", required=True,
                        help="Student ID for solution file name.")
    parser.add_argument("--time", type=float, default=40.0,
                        help="Time limit in seconds (default: 40; set to ~55 for final run).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")

    args = parser.parse_args()

    print(f"Loading graph from {args.input} ...")
    dist = load_edge_list_graph(args.input)
    n = len(dist)
    print(f"Loaded graph with {n} nodes.")

    if args.graph == "A":
        sol_file = f"solutionA_{args.sid}.txt"
    else:
        sol_file = f"solutionB_{args.sid}.txt"

    print(f"Starting solver with time limit = {args.time} seconds ...")
    best_tour, best_cost, cycles, elapsed = solve_tsp(
        dist, time_limit=args.time, seed=args.seed
    )

    print("=== RESULT ===")
    print(f"TIME   : {elapsed:.2f} sec")
    print(f"CYCLES : {cycles:.3e}")
    print(f"COST   : {best_cost:.2f}")

    # Save 1-based indices to match Node numbers in input
    with open(sol_file, "w") as f:
        f.write(", ".join(str(i + 1) for i in best_tour) + "\n")

    print(f"Best tour written to {sol_file}")


if __name__ == "__main__":
    main()
