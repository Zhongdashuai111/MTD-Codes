"""Generate and numerically check periodic Voronoi constraints from codewords."""
from collections import Counter
import itertools
import numpy as np
from scipy.spatial import Voronoi, HalfspaceIntersection
from scipy.optimize import linprog
from .codebooks import Q

def periodic_neighbors(C):
    ell = C.shape[1]
    shifts = np.array(list(itertools.product([-1,0,1], repeat=ell)))
    points = (C[None,:,:] + Q*shifts[:,None,:]).reshape(-1,ell)
    start = int(np.flatnonzero((shifts == 0).all(axis=1))[0])*len(C)
    vor = Voronoi(points)
    neighbors = [set() for _ in C]
    for a,b in vor.ridge_points:
        for i,j in ((a,b),(b,a)):
            if start <= i < start+len(C):
                neighbors[i-start].add(tuple(points[j]-points[i]))
    counts = Counter()
    checks = []
    for c, H in zip(C, neighbors):
        H = sorted(H)
        counts.update(H)
        h = np.asarray(H, dtype=np.int64)
        rhs = np.sum(h*h, axis=1)/2
        for axis in range(ell):
            for sign in (-1,1):
                obj = np.zeros(ell); obj[axis] = sign
                r = linprog(obj, A_ub=h, b_ub=rhs,
                            bounds=[(None,None)]*ell, method="highs")
                if not r.success or -r.fun > Q/2+1e-6:
                    raise RuntimeError("The proposed Voronoi cell is not certified by the numerical cube check.")
        vertices = HalfspaceIntersection(np.column_stack([h,-rhs]), np.zeros(ell)).intersections
        allh = points-c
        violation = float(np.max(vertices@allh.T - np.sum(allh*allh,axis=1)/2))
        if violation > 1e-6:
            raise RuntimeError("A candidate periodic constraint is missing.")
        checks.append(dict(vertices=len(vertices), max_constraint_violation=violation))
    hs = sorted(counts)
    return np.asarray(hs, dtype=np.int64), np.array([counts[h] for h in hs]), {
        "different_directions":len(hs), "total_multiplicity":sum(counts.values()),
        "checks":checks, "status":"floating-point geometry check, not an exact rational certificate"}
