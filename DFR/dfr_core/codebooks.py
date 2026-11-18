"""The four fixed codebooks used in the DFR comparison table."""
import itertools
import numpy as np

Q = 3329
PARAMETERS = dict(q=3329, n=256, k=4, eta1=2, eta2=2, du=11, dv=5)
CASES = {
    "LiuSakzad_l2_gamma446": (2, "reference", 2967048),
    "ThisWork_l2_gamma446": (2, "ours", 2967812),
    "LiuSakzad_l4_scale554": (4, "reference", 3682992),
    "ThisWork_l4_gamma554": (4, "ours", 3689644),
}

def codebook(case):
    ell, family, expected_distance = CASES[case]
    a = Q // 2
    if ell == 2:
        g = 446
        if family == "ours":
            C = [[0, 0], [g, a], [a, Q-g], [a+g, a-g]]
        else:
            C = [[0, 0], [g, a], [a, g], [a+g, a+g]]
    elif family == "ours":
        g = 554
        G = np.array([[a,g,g,g], [Q-g,a,g,Q-g],
                      [Q-g,Q-g,a,g], [Q-g,g,Q-g,a]], dtype=np.int64)
        C = np.array(list(itertools.product([0,1], repeat=4))) @ G % Q
    else:
        C = 554 * np.array([
            [0,0,0,0], [4,2,4,0], [3,3,3,3], [2,0,4,2],
            [2,4,2,0], [3,1,1,1], [1,1,3,5], [1,5,1,3],
            [0,2,2,2], [3,5,5,5], [1,3,5,1], [0,4,4,4],
            [4,0,2,4], [4,4,0,2], [5,1,5,3], [5,5,3,1]])
    C = np.asarray(C, dtype=np.int64)
    if len(np.unique(C, axis=0)) != 2**ell:
        raise ValueError("The codebook has the wrong cardinality.")
    differences = np.abs(C[:,None,:] - C[None,:,:])
    distances = np.minimum(differences, Q-differences)
    squared = np.sum(distances**2, axis=2)
    np.fill_diagonal(squared, np.iinfo(np.int64).max)
    if int(squared.min()) != expected_distance:
        raise ValueError("The minimum distance differs from the manuscript.")
    return C
