#!/usr/bin/env python3
"""Exact maximum-toroidal-distance (MTD) code search.

Given n, q >= 2, find a code C subset (Z/qZ)^n with |C|=2**n,
maximizing the minimum *Euclidean* toroidal distance

    d_T(x,y) = sqrt(sum(min(|x_i-y_i|,q-|x_i-y_i|)**2)).

The optimization is exact for runs that finish. It is exponential (or worse)
in the worst case; n=16, q=4 is far outside feasible exhaustive search.

Method: translate an optimum to include the zero word; for each proposed
integer squared distance D, make a compatibility graph whose vertices are
words at distance squared >= D from zero, with edges for pair distances >= D.
Then a code exists exactly when the graph contains a clique of size 2**n-1.
A bitset branch-and-bound maximum-clique decision algorithm with greedy
coloring is used, and binary search determines the largest feasible D.

The search has a default wall-clock limit of five days. If this limit is
reached, the computation stops with a TIME LIMIT message. Any completed
feasibility tests remain valid, but an unfinished search is NOT an exact
optimality certificate.

Only the Python standard library is required.
"""

from __future__ import annotations

# =========================================================
# 用户设置：只需要修改下面的 n 和 q，然后在 IDLE 中按 F5 运行
# =========================================================
n = 4              # 码长，例如 n = 2、4
q = 7              # 模数，例如 q = 4、6、8
MAX_DAYS = 5       # 整次搜索最多运行 5 天（120 小时）
MAX_POINTS = 100000 # 搜索空间规模上限；与时间限制相互独立
# =========================================================

import argparse
import itertools
import math
import sys
import time


class SearchTimeLimitExceeded(TimeoutError):
    """Raised only when the search exceeds its wall-clock budget."""


def squared_toroidal_distance(x: tuple[int, ...], y: tuple[int, ...], q: int) -> int:
    """Return the exact integer squared Euclidean toroidal distance."""
    if len(x) != len(y):
        raise ValueError("The two words must have the same length.")
    return sum(min(abs(a - b), q - abs(a - b)) ** 2 for a, b in zip(x, y))


def all_squared_distances(n: int, q: int) -> list[int]:
    """All attainable squared distances between two distinct words."""
    residues = {min(a, q - a) ** 2 for a in range(q)}
    sums = {0}
    for _ in range(n):
        sums = {s + d for s in sums for d in residues}
    return sorted(sums - {0})


def _color_sort(
    available: int, neighbors: list[int], check_time
) -> tuple[list[int], list[int]]:
    """Greedy color ordering. Every prefix's chromatic bound is valid."""
    order: list[int] = []
    colors: list[int] = []
    uncolored = available
    color = 0
    while uncolored:
        check_time()
        color += 1
        independent = uncolored
        while independent:
            check_time()
            bit = independent & -independent
            v = bit.bit_length() - 1
            order.append(v)
            colors.append(color)
            uncolored ^= bit
            independent ^= bit
            independent &= ~neighbors[v]
    return order, colors


def _target_clique(neighbors: list[int], target: int, check_time) -> list[int] | None:
    """Exact clique feasibility with coloring upper bounds, using Python ints."""
    count = len(neighbors)
    if target == 0:
        return []
    if count < target:
        return None

    # Quick attempt to obtain a target clique; never used to certify failure.
    full = (1 << count) - 1
    deg = [row.bit_count() for row in neighbors]
    starts = sorted(range(count), key=lambda v: (-deg[v], v))[: min(count, 32)]
    for start in starts:
        check_time()
        chosen = [start]
        candidates = neighbors[start]
        while candidates and len(chosen) < target:
            check_time()
            # Select a high-degree vertex in the remaining induced graph.
            c = candidates
            best_v, best_score = -1, -1
            while c:
                check_time()
                bit = c & -c
                v = bit.bit_length() - 1
                score = (neighbors[v] & candidates).bit_count()
                if score > best_score:
                    best_v, best_score = v, score
                c ^= bit
            chosen.append(best_v)
            candidates &= neighbors[best_v]
        if len(chosen) == target:
            return chosen

    selected: list[int] = []

    def dfs(available: int) -> list[int] | None:
        check_time()
        need = target - len(selected)
        if need == 0:
            return selected.copy()
        if available.bit_count() < need:
            return None

        order, colors = _color_sort(available, neighbors, check_time)
        # Reversed order: removing a vertex leaves a prefix, whose color
        # upper bound is its last color label.
        for i in range(len(order) - 1, -1, -1):
            check_time()
            if colors[i] < need:
                break
            vertex = order[i]
            bit = 1 << vertex
            remaining = available & neighbors[vertex]
            if remaining.bit_count() >= need - 1:
                selected.append(vertex)
                answer = dfs(remaining)
                if answer is not None:
                    return answer
                selected.pop()
            available &= ~bit
        return None

    return dfs(full)


def clique_code_for_threshold(
    n: int,
    q: int,
    threshold_sq: int,
    words: list[tuple[int, ...]],
    norms_sq: list[int],
    check_time,
) -> list[tuple[int, ...]] | None:
    """Return a code of size 2**n at threshold_sq, or prove none exists."""
    size = 1 << n
    zero = (0,) * n
    if threshold_sq <= 0:
        return words[:size] if len(words) >= size else None

    indices = [i for i, dsq in enumerate(norms_sq) if dsq >= threshold_sq]
    if len(indices) < size - 1:
        return None

    # Form graph on the remaining words; bit i represents vertex i.
    vertex_words = [words[i] for i in indices]
    k = len(vertex_words)
    neighbors = [0] * k
    one_dim_sq = [[min(abs(a - b), q - abs(a - b)) ** 2
                   for b in range(q)] for a in range(q)]
    for i, xi in enumerate(vertex_words):
        check_time()
        for j in range(i + 1, k):
            if (j & 1023) == 0:
                check_time()
            xj = vertex_words[j]
            dsq = sum(one_dim_sq[a][b] for a, b in zip(xi, xj))
            if dsq >= threshold_sq:
                neighbors[i] |= 1 << j
                neighbors[j] |= 1 << i

    result = _target_clique(neighbors, size - 1, check_time)
    if result is None:
        return None
    return [zero] + [vertex_words[v] for v in result]


def verify_code(code: list[tuple[int, ...]], n: int, q: int, check_time) -> int:
    """Validate the final code while still respecting the time limit."""
    if len(code) != 1 << n or len(set(code)) != len(code):
        raise AssertionError("Wrong code cardinality or duplicate words.")
    if any(len(word) != n or any(not 0 <= x < q for x in word) for word in code):
        raise AssertionError("A codeword is not in (Z/qZ)^n.")
    best = None
    for i, x in enumerate(code):
        check_time()
        for j in range(i + 1, len(code)):
            if (j & 1023) == 0:
                check_time()
            distance = squared_toroidal_distance(x, code[j], q)
            if best is None or distance < best:
                best = distance
    assert best is not None
    return best


def find_mtd(
    n: int, q: int, max_points: int = MAX_POINTS,
    verbose: bool = True, max_days: float = MAX_DAYS,
) -> dict:
    if n < 1 or q < 2:
        raise ValueError("Require n >= 1 and q >= 2.")
    if max_points < 1 or not math.isfinite(max_days) or max_days <= 0:
        raise ValueError("Require max_points >= 1 and max_days > 0 (finite).")
    total = q ** n
    size = 1 << n
    if total > max_points:
        raise ValueError(
            f"q^n={total:,} exceeds --max-points={max_points:,}. "
            "Increase the cap only if you understand the potentially enormous runtime."
        )

    start = time.perf_counter()
    limit_seconds = max_days * 24 * 60 * 60
    deadline = start + limit_seconds

    def check_time() -> None:
        if time.perf_counter() >= deadline:
            raise SearchTimeLimitExceeded("Search reached its configured time limit.")

    tests_completed = 0
    active_threshold = None
    best_code: list[tuple[int, ...]] | None = None
    best_sq: int | None = None
    try:
        check_time()
        words = list(itertools.product(range(q), repeat=n))
        norms_sq = [squared_toroidal_distance(w, (0,) * n, q) for w in words]
        distances = all_squared_distances(n, q)
        lo, hi = 0, len(distances) - 1
        while lo <= hi:
            check_time()
            mid = (lo + hi) // 2
            threshold = distances[mid]
            active_threshold = threshold
            if verbose:
                print(f"[test {tests_completed + 1}] Can {size} words have d_T^2 >= {threshold}?",
                      file=sys.stderr, flush=True)
            result = clique_code_for_threshold(n, q, threshold, words, norms_sq,
                                               check_time)
            tests_completed += 1
            active_threshold = None
            if result is not None:
                best_code, best_sq = result, threshold
                lo = mid + 1
                if verbose:
                    print("           feasible", file=sys.stderr, flush=True)
            else:
                hi = mid - 1
                if verbose:
                    print("           impossible (exhaustively proved)",
                          file=sys.stderr, flush=True)
        assert best_code is not None and best_sq is not None
        true_min_sq = verify_code(best_code, n, q, check_time)
        if true_min_sq != best_sq:
            raise AssertionError(f"Algorithm error: expected {best_sq}, got {true_min_sq}.")
        check_time()
        return {
            "n": n,
            "q": q,
            "code_size": size,
            "minimum_toroidal_distance_squared": best_sq,
            "minimum_toroidal_distance": math.sqrt(best_sq),
            "codebook": [list(w) for w in sorted(best_code)],
            "certified_globally_optimal": True,
            "threshold_tests": tests_completed,
            "elapsed_seconds": round(time.perf_counter() - start, 4),
        }
    except SearchTimeLimitExceeded as exc:
        exc.elapsed_seconds = time.perf_counter() - start
        exc.limit_days = max_days
        exc.tests_completed = tests_completed
        exc.active_threshold = active_threshold
        exc.best_feasible_sq = best_sq
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("n", nargs="?", type=int, default=n,
                   help="Optional code length (default: n set at the top of this file)")
    p.add_argument("q", nargs="?", type=int, default=q,
                   help="Optional modulus (default: q set at the top of this file)")
    p.add_argument("--max-points", type=int, default=MAX_POINTS,
                   help=f"Refuse q^n above this cap (default {MAX_POINTS})")
    p.add_argument("--max-days", type=float, default=MAX_DAYS,
                   help=f"Wall-clock search limit in days (default {MAX_DAYS})")
    p.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    args = p.parse_args()
    try:
        result = find_mtd(args.n, args.q, max_points=args.max_points,
                          verbose=not args.quiet, max_days=args.max_days)
    except SearchTimeLimitExceeded as exc:
        print(f"TIME LIMIT REACHED: stopped after {exc.elapsed_seconds:.1f} seconds "
              f"(limit: {exc.limit_days:g} days).")
        print(f"Completed threshold tests: {exc.tests_completed}.")
        if exc.active_threshold is not None:
            print(f"Unfinished squared-distance threshold: {exc.active_threshold}.")
        if exc.best_feasible_sq is not None:
            print(f"Known feasible lower bound: d_T^2 >= {exc.best_feasible_sq} "
                  "(NOT proven optimal).")
        print("NO EXACT MTD RESULT: the exhaustive search did not finish.")
        return
    except ValueError as e:
        p.error(str(e))
    print(f"n={args.n}, q={args.q}, |C|={result['code_size']}")
    print(f"Optimal d_T^2 = {result['minimum_toroidal_distance_squared']}")
    print(f"Optimal d_T   = {result['minimum_toroidal_distance']:.12g}")
    print(f"Certified exact optimum; tested {result['threshold_tests']} thresholds.")
    print("Optimal codebook:")
    for word in result["codebook"]:
        print(" ", word)
    print(f"Total search time: {result['elapsed_seconds']:.4f} seconds")


if __name__ == "__main__":
    main()
