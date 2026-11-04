"""Exact local ball--Plotkin bound for toroidal-distance codes.

    P_q(x) = sum_{r=0}^{floor(q/2)} m_r*x**(r*r),
    V_q(n,rho) = sum_{j=0}^rho [x^j] P_q(x)**n,
    L_rho = ceil(M*V_q(n,rho)/q**n).

The optimized upper bound on d_T(C)**2 is
    min_{rho in S, L_rho >= 2} 2*rho*L_rho/(L_rho-1),
where S = {r_1**2 + ... + r_n**2 : 0 <= r_i <= floor(q/2)}.

It is unnecessary to consider other radii: V_q(n,rho) changes only at
rho in S, is constant between consecutive elements of S, and equals
q**n at and beyond n*floor(q/2)**2. The objective increases with rho
when L_rho is fixed.

All counts, ceilings, comparisons, and optimal bounds use exact
integer/Fraction arithmetic. No third-party packages are required.

Edit n, q, M below and press F5 in Python IDLE.
"""

from fractions import Fraction
from math import sqrt

# ============================================================================
# USER SETTINGS
# ============================================================================
n = 16                # Code length
q = 5                # Alphabet modulus Z_q
M = 2**n            # Code size (e.g. M = 2**n)

TOP_K = 8           # Print the 8 strongest admissible radii by default
SHOW_FULL_TABLE = False  # True: show ALL admissible radii in increasing order
# ============================================================================


def validate(n, q, M):
    for name, value in (("n", n), ("q", q), ("M", M)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}.")
    if n < 1:
        raise ValueError("n must be >= 1.")
    if q < 2:
        raise ValueError("q must be >= 2.")
    if M < 2:
        raise ValueError("M must be >= 2.")
    if M > q**n:
        raise ValueError(f"M = {M} exceeds the ambient space size q^n = {q**n}.")
    if not isinstance(TOP_K, int) or TOP_K < 1:
        raise ValueError("TOP_K must be a positive integer.")


def lee_multiplicities(q):
    """Return m_r = #{a in Z_q : d_L(a,0) = r}."""
    a = q // 2
    return [1 if (r == 0 or (q % 2 == 0 and r == a)) else 2
            for r in range(a + 1)]


def toroidal_ball_volumes(n, q):
    """Return (rho, V_q(n,rho)) only at attainable squared radii.

    Sparse DP computes the nonzero coefficients of P_q(x)**n.
    The returned radii are exactly the integers of the form
    r_1**2 + ... + r_n**2, with 0 <= r_i <= floor(q/2),
    in strictly increasing order.
    """
    steps = [(r*r, count) for r, count in enumerate(lee_multiplicities(q))]
    coeff = {0: 1}
    for _ in range(n):
        next_coeff = {}
        for rho, count in coeff.items():
            for distance2, multiplicity in steps:
                new_rho = rho + distance2
                next_coeff[new_rho] = (next_coeff.get(new_rho, 0)
                                       + count * multiplicity)
        coeff = next_coeff

    cumulative = 0
    volumes = []
    for rho in sorted(coeff):
        cumulative += coeff[rho]
        volumes.append((rho, cumulative))

    assert cumulative == q**n, "Ball counts must sum to q^n."
    assert volumes[-1][0] == n*(q//2)**2
    return volumes


def compute_local_plotkin(n, q, M):
    """Return (exact best bound, best rho, admissible rows).

    Each row is (rho, volume, L_rho, bound). Rows contain ONLY
    attainable squared radii for which L_rho >= 2.
    """
    validate(n, q, M)
    ambient = q**n
    rows = []
    best, best_rho = None, None

    for rho, volume in toroidal_ball_volumes(n, q):
        L = (M*volume + ambient - 1)//ambient  # exact ceiling
        if L < 2:
            continue
        bound = Fraction(2*rho*L, L-1)
        rows.append((rho, volume, L, bound))
        if best is None or bound < best:
            best, best_rho = bound, rho

    assert best is not None, "The maximal radius always has L_rho = M >= 2."
    return best, best_rho, rows


def fmt_number(value):
    return f"{float(value):.10f}"


def fmt_int(value):
    return f"{value:,}"


def display_table(rows, best_rho, header):
    print("\n" + header)
    print("-" * 90)
    print(f"{'rho':>6} | {'V_q(n,rho)':>16} | {'L_rho':>10} | "
          f"{'Local bound on d_T^2':>20} | {'floor':>5} | {'note'}")
    print("-" * 90)
    for rho, volume, L, bound in rows:
        integer_bound = bound.numerator // bound.denominator
        note = "<-- BEST" if rho == best_rho else ""
        print(f"{rho:>6} | {fmt_int(volume):>16} | {fmt_int(L):>10} | "
              f"{fmt_number(bound):>20} | {integer_bound:>5} | {note}")
    print("-" * 90)


def main():
    best, best_rho, rows = compute_local_plotkin(n, q, M)
    floor_bound = best.numerator // best.denominator
    max_dist_squared = n*(q//2)**2

    print("=" * 90)
    print("LOCAL BALL--PLOTKIN BOUND")
    print("=" * 90)
    print(f"n = {n}, q = {q}, M = {fmt_int(M)} (q^n = {fmt_int(q**n)})")
    print(f"Only rho = sum_i r_i^2, 0 <= r_i <= {q//2}, "
          f"0 <= rho <= {max_dist_squared}, with L_rho >= 2 are used.")
    print(f"Number of admissible squared radii: {len(rows)}")
    print("All calculations use exact integer / rational arithmetic.")

    top = sorted(rows, key=lambda row: (row[3], row[0]))[:TOP_K]
    display_table(top, best_rho,
                  f"Top {len(top)} admissible radii (by strongest bound):")

    if SHOW_FULL_TABLE:
        display_table(rows, best_rho,
                      "All admissible squared radii (in increasing order):")

    print("\nFINAL RESULT")
    print(f"Optimal squared radius rho = {best_rho}")
    print(f"Exact bound on d_T(C)^2 = {best}")
    print(f"Decimal bound on d_T(C)^2 = {float(best):.12f}")
    print(f"Integer-valued distance bound: d_T(C)^2 <= {floor_bound}")
    print(f"Therefore d_T(C) <= sqrt({floor_bound}) = {sqrt(floor_bound):.12f}")
    print("This is an upper bound, not a claim that a code achieves it.")


if __name__ == "__main__":
    main()
