"""Exact pairwise minimum toroidal distance for the complete Z_q-linear code.

C = {mG (mod q) : m in Z_q^k}, where k is the number of rows of G.
This program generates ALL q^k messages, keeps distinct codewords, and
compares EVERY unordered pair of distinct codewords (not just a subset).

For the supplied matrix, q=4, k=8, n=16, M=4**8=65536, and the number
of comparisons is M*(M-1)//2 = 2147450880.

Requires Python 3, NumPy and Numba:
    pip install numpy numba

Run:
    python Example_7_full_Z4_pairwise_distance.py

Only q, M, n, d_T(C)^2, d_T(C), and elapsed time are printed.
"""

from itertools import product
from math import sqrt
from time import perf_counter

import numpy as np
from numba import njit, prange

# ====================== USER INPUT ======================
q = 4
G = [
    [1,0,0,0,0,0,0,0, 2,1,1,1,1,1,1,1],
    [0,1,0,0,0,0,0,0, 1,2,3,1,2,1,0,0],
    [0,0,1,0,0,0,0,0, 1,0,2,3,1,2,1,0],
    [0,0,0,1,0,0,0,0, 1,0,0,2,3,1,2,1],
    [0,0,0,0,1,0,0,0, 1,1,0,0,2,3,1,2],
    [0,0,0,0,0,1,0,0, 1,2,1,0,0,2,3,1],
    [0,0,0,0,0,0,1,0, 1,1,2,1,0,0,2,3],
    [0,0,0,0,0,0,0,1, 1,3,1,2,1,0,0,2],
]
# ========================================================


def generate_codewords():
    """Enumerate all q-ary messages and keep all distinct codewords."""
    if not isinstance(q, int) or q < 2:
        raise ValueError("q must be an integer >= 2")
    if not G or not G[0] or any(len(row) != len(G[0]) for row in G):
        raise ValueError("G must be a nonempty rectangular matrix")
    if any(not isinstance(v, int) for row in G for v in row):
        raise ValueError("G must contain integers")

    k = len(G)
    matrix = np.asarray(G, dtype=np.int32)
    messages = np.asarray(list(product(range(q), repeat=k)), dtype=np.int32)
    codewords = (messages @ matrix) % q
    codewords = np.unique(codewords, axis=0)
    # uint8 saves memory/cache traffic for q=4.
    return np.ascontiguousarray(codewords.astype(np.uint8 if q <= 256 else np.int32))


@njit(parallel=True)
def pairwise_minimum_squared(codewords, cost):
    """Check each distinct unordered pair exactly once, using all coordinates.

    For balanced parallel execution, each job handles row i and its
    mirrored row M-1-i; together they have approximately M-1 pairs.
    """
    M, n = codewords.shape
    per_job = np.empty((M + 1) // 2, dtype=np.int32)

    for t in prange((M + 1) // 2):
        best = np.int32(2147483647)
        i = t
        for j in range(i + 1, M):
            d2 = 0
            for h in range(n):
                d = np.int32(codewords[i, h]) - np.int32(codewords[j, h])
                if d < 0:
                    d = -d
                d2 += cost[d]
            if d2 < best:
                best = d2

        mirror = M - 1 - t
        if mirror != i:
            for j in range(mirror + 1, M):
                d2 = 0
                for h in range(n):
                    d = np.int32(codewords[mirror, h]) - np.int32(codewords[j, h])
                    if d < 0:
                        d = -d
                    d2 += cost[d]
                if d2 < best:
                    best = d2
        per_job[t] = best

    return int(np.min(per_job))


def main():
    start = perf_counter()
    codewords = generate_codewords()
    M, n = codewords.shape
    if M < 2:
        raise ValueError("The code must contain at least two distinct codewords")

    # cost[d] = min(d, q-d)^2 for d = |x_i-y_i|.
    cost = np.asarray([min(d, q - d) ** 2 for d in range(q)], dtype=np.int32)
    minimum_squared = pairwise_minimum_squared(codewords, cost)
    elapsed = perf_counter() - start

    print(f"q = {q}")
    print(f"M = {M}")
    print(f"n = {n}")
    print(f"d_T(C)^2 = {minimum_squared}")
    print(f"d_T(C) = {sqrt(minimum_squared):.10f}")
    print(f"Elapsed time = {elapsed:.6f} s")


if __name__ == "__main__":
    main()
