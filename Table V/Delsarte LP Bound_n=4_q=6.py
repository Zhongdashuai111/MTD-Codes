#!/usr/bin/env python3


import argparse
import csv
from math import comb, cos, pi, sqrt, isfinite
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog


# ============================================================
# USER SETTINGS / 用户参数设置：只需修改这里，然后按 F5 运行
# ============================================================
# q = 模数（q >= 2）
q = 6

# n = 码长 / 维数（n >= 1）
n = 4

# M = 码字数量。None 表示自动使用 M = 2**n。
# 如果希望计算其它码字数量，例如 M=1000，可以改为 M = 1000。
M = None
# 自动找到临界值 delta_0 后，显示其前后各几个 delta。
WINDOW = 3

# DELTA = None：自动搜索 delta_0，并打印 delta_0 附近的表格。
# DELTA = 17：只计算 delta=17 这一行。
DELTA = None

# CSV 输出。None 表示不保存文件；也可改为 "lp_results.csv"。
CSV_OUTPUT = None

# 求解器浮点容差（通常不需要修改）。
TOLERANCE = 1e-9

# 目标值是否接近整数的绝对容差（只是数值判断，不是严格数学证明）。
# 例如 255.99999999999983 在 1e-7 容差下被判为接近整数 256。
INTEGER_TOLERANCE = 1e-7

# 为数值上接近整数的 LP 最优值输出非零 B_alpha 与目标函数展开。
SHOW_INTEGER_SOLUTIONS = True
# 仅用于忽略解向量中的浮点零（不影响求解过程）。
ZERO_DISPLAY_TOLERANCE = 1e-10

# Automatically attempt exact primal/dual verification near an integer.
EXACT_CHECK = True
# Largest denominator accepted during rational reconstruction.
EXACT_MAX_DENOMINATOR = 1_000_000
# Skip potentially expensive exact calculations above this many LP variables.
EXACT_MAX_VARIABLES = 200
# Optional complete exact certificate report (e.g., 'exact_lp_report.txt').
EXACT_REPORT = None
# ============================================================

DEFAULT_SETTINGS = {"n": n, "q": q, "M": M}


def lee_compositions(n, k):
    """Generate all nonnegative k-tuples summing to n."""
    if k == 1:
        yield (n,)
        return
    for a in range(n + 1):
        for rest in lee_compositions(n - a, k - 1):
            yield (a,) + rest


def lee_data(q):
    """Lee-class sizes m[s], and P[r,s] for 0 <= r,s <= floor(q/2)."""
    t = q // 2
    m = [0] * (t + 1)
    P = np.zeros((t + 1, t + 1), dtype=float)
    for a in range(q):
        s = min(a, q - a)
        m[s] += 1
        for r in range(t + 1):
            P[r, s] += cos(2 * pi * a * r / q)
    P[np.abs(P) < 1e-12] = 0.0
    return m, P


def valency(alpha, multiplicities):
    """Exact number of words having Lee composition alpha."""
    remaining = sum(alpha)
    result = 1
    for a, m in zip(alpha, multiplicities):
        result *= comb(remaining, a) * m**a
        remaining -= a
    return result


def create_lp_data(q, n):
    """Build the row-normalized Lee--Krawtchouk matrix and LP metadata."""
    if q < 2 or n < 1:
        raise ValueError('Require q >= 2 and n >= 1.')
    t = q // 2
    k = t + 1
    multiplicities, P = lee_data(q)
    compositions = list(lee_compositions(n, k))
    index = {alpha: j for j, alpha in enumerate(compositions)}
    v = np.array([valency(a, multiplicities) for a in compositions], dtype=float)
    D = np.array([sum(r*r*a[r] for r in range(1, k)) for a in compositions], dtype=int)
    zero = index[(n,) + (0,) * t]

    # Polynomial coefficients for each alpha (keeps only the previous degree).
    cached = {(0,) * k: {(0,) * k: 1.0}}
    for degree in range(1, n + 1):
        for alpha in lee_compositions(degree, k):
            r = next(i for i, count in enumerate(alpha) if count)
            prev = list(alpha)
            prev[r] -= 1
            old = cached[tuple(prev)]
            polynomial = {}
            for beta, coefficient in old.items():
                for s in range(k):
                    new_beta = list(beta)
                    new_beta[s] += 1
                    new_beta = tuple(new_beta)
                    polynomial[new_beta] = (polynomial.get(new_beta, 0.0)
                                            + coefficient * P[r, s])
            cached[alpha] = polynomial
        if degree >= 2:
            for prev in lee_compositions(degree - 2, k):
                cached.pop(prev, None)

    K = np.zeros((len(compositions), len(compositions)), dtype=float)
    for column, alpha in enumerate(compositions):
        for beta, coefficient in cached[alpha].items():
            K[index[beta], column] = coefficient
    K /= v[:, None]  # Positive row normalization; preserves inequalities.
    if not np.allclose(K[:, zero], 1.0, atol=2e-8):
        raise RuntimeError('Lee--Krawtchouk normalization check failed.')
    if not np.all(np.isfinite(K)):
        raise RuntimeError('Lee--Krawtchouk matrix has nonfinite entries.')
    return compositions, D, v, K, zero


def solve_lp(delta, D, v, K, zero, tol):
    """Compute numerical L_q(n,delta), min normalized positivity slack."""
    bounds = []
    for j, (dist, count) in enumerate(zip(D, v)):
        if j == zero:
            bounds.append((1.0, 1.0))
        elif 0 < dist < delta:
            bounds.append((0.0, 0.0))
        else:
            bounds.append((0.0, float(count)))
    others = np.arange(len(D)) != zero  # Exclude redundant beta=0 constraint.
    result = linprog(
        -np.ones(len(D)),
        A_ub=-K[others, :],
        b_ub=np.zeros(int(np.sum(others))),
        bounds=bounds,
        method='highs',
        options={'primal_feasibility_tolerance': tol,
                 'dual_feasibility_tolerance': tol},
    )
    if not result.success:
        raise RuntimeError(f'LP delta={delta}: {result.message} (status {result.status})')
    value = -float(result.fun)
    slack = float(np.min(K @ result.x))
    return value, slack, result


# Only these moduli have rational cosine values at every relevant root
# of unity; do NOT apply rational reconstruction to irrational K entries.
_RATIONAL_COSINES = {
    2: (1, -1),
    3: (1, Fraction(-1, 2), Fraction(-1, 2)),
    4: (1, 0, -1, 0),
    6: (1, Fraction(1, 2), Fraction(-1, 2), -1,
        Fraction(-1, 2), Fraction(1, 2)),
}


def exact_krawtchouk(q, n, compositions, max_variables):
    """Construct rational K exactly from polynomial coefficients.

    Returns row-normalized Fraction matrix, exact valencies, or a reason to
    skip. No numbers in the exact K are inferred from the floating solution.
    """
    if q not in _RATIONAL_COSINES:
        return None, None, (
            f'q={q} not supported for rational certificates '
            '(Lee eigenvalues may be algebraic irrational).'
        )
    if len(compositions) > max_variables:
        return None, None, (
            f'{len(compositions)} variables exceed exact limit '
            f'{max_variables}; increase --exact-max-variables if desired.'
        )

    t = q // 2
    k = t + 1
    cos_table = _RATIONAL_COSINES[q]
    m = [0] * k
    P = [[Fraction(0) for _ in range(k)] for _ in range(k)]
    for a in range(q):
        s = min(a, q - a)
        m[s] += 1
        for r in range(k):
            P[r][s] += cos_table[(a * r) % q]
    # These P entries are in fact integers for these q.
    assert all(x.denominator == 1 for row in P for x in row)
    P = [[int(x) for x in row] for row in P]

    index = {alpha: j for j, alpha in enumerate(compositions)}
    v_exact = [valency(a, m) for a in compositions]
    cached = {(0,) * k: {(0,) * k: 1}}
    K_int = [[0] * len(compositions) for _ in compositions]
    for degree in range(1, n + 1):
        for alpha in lee_compositions(degree, k):
            r = next(i for i, count in enumerate(alpha) if count)
            prev = list(alpha)
            prev[r] -= 1
            old = cached[tuple(prev)]
            poly = {}
            for beta, coeff in old.items():
                for s in range(k):
                    if P[r][s] == 0:
                        continue
                    next_beta = list(beta)
                    next_beta[s] += 1
                    next_beta = tuple(next_beta)
                    poly[next_beta] = poly.get(next_beta, 0) + coeff * P[r][s]
            cached[alpha] = poly
            if degree == n:
                for beta, coeff in poly.items():
                    K_int[index[beta]][index[alpha]] = coeff
        if degree >= 2:
            for prev in lee_compositions(degree - 2, k):
                cached.pop(prev, None)

    K_exact = [
        [Fraction(coeff, v_exact[i]) for coeff in row]
        for i, row in enumerate(K_int)
    ]
    return K_exact, v_exact, None


def certify_exact(delta, compositions, D, K_float, zero, result,
                  K_exact, valencies, max_denominator):
    """Certify primal and dual solutions exactly, returning a full certificate.

    Variables B:  max sum B; lower <= B <= upper, K B >= 0.
    Dual y,t,s >= 0: 1 = -K.T y + t - s, objective upper*t-lower*s.
    With exact feasibility and matching objectives, the reconstructed value
    is the exact optimal L, not merely an integer-like primal objective.
    """
    m = len(compositions)
    if K_exact is None:
        return None, 'Exact Lee--Krawtchouk matrix unavailable.'
    # Ensure exact construction agrees with the numerical LP being certified.
    numerical_difference = max(
        abs(float(K_exact[i][j]) - float(K_float[i, j]))
        for i in range(m) for j in range(m)
    )
    if numerical_difference > 1e-8:
        return None, ('Exact coefficient matrix does not match the '
                      f'numerical one (error {numerical_difference:.3g}).')

    to_rat = lambda a: Fraction(float(a)).limit_denominator(max_denominator)
    B = [to_rat(a) for a in result.x]
    lower = [1 if j == zero else 0 for j in range(m)]
    upper = [1 if j == zero else 0 if 0 < int(D[j]) < delta else
             int(valencies[j]) for j in range(m)]
    if any(not (lower[j] <= B[j] <= upper[j]) for j in range(m)):
        return None, 'Reconstructed primal B violates exact variable bounds.'
    p_slack = [
        sum((K_exact[i][j] * B[j] for j in range(m)), Fraction(0))
        for i in range(m)
    ]
    if min(p_slack) < 0:
        return None, 'Reconstructed primal B violates exact positivity constraints.'
    primal = sum(B, Fraction(0))

    # HiGHS reports marginals for its MINIMIZATION formulation, in which
    # A_ub=-K[others], objective=-sum(B). Flip signs as appropriate.
    y = [Fraction(0)] * m
    others = [j for j in range(m) if j != zero]
    if not (len(others) == len(result.ineqlin.marginals)):
        return None, 'Unexpected solver dual multiplier dimensions.'
    for i, multiplier in zip(others, result.ineqlin.marginals):
        y[i] = to_rat(-multiplier)
    t = [to_rat(-multiplier) for multiplier in result.upper.marginals]
    s = [to_rat(multiplier) for multiplier in result.lower.marginals]
    if min(y + t + s) < 0:
        return None, 'Reconstructed dual multiplier is negative.'
    # Exact stationarity for every LP variable.
    dual_residuals = [
        Fraction(1) +
        sum((K_exact[i][j] * y[i] for i in range(m)), Fraction(0))
        + s[j] - t[j]
        for j in range(m)
    ]
    if any(dual_residuals):
        return None, 'Reconstructed dual multipliers fail exact stationarity.'
    dual = sum((upper[j] * t[j] - lower[j] * s[j]
                for j in range(m)), Fraction(0))
    if primal != dual:
        return None, (f'Exact primal value {primal} and exact dual '
                      f'upper bound {dual} do not match.')
    return {
        'delta': delta,
        'objective': primal,
        'integer': primal.denominator == 1,
        'B': B, 'y': y, 'upper_mult': t, 'lower_mult': s,
        'min_primal_slack': min(p_slack),
        'max_dual_residual': max(abs(x) for x in dual_residuals),
        'composition_count': len(compositions),
    }, None


def print_exact_certificate(certificate, compositions, show_terms=30):
    """Display exact symbolic expression for L from nonzero B coefficients."""
    d = certificate['delta']
    pairs = [(compositions[i], v) for i, v in enumerate(certificate['B']) if v]
    terms = [str(v) for _, v in pairs]
    expression = ' + '.join(terms)
    print(f'\nEXACT CERTIFICATE for delta={d}:')
    print(f'  L = {expression} = {certificate["objective"]}'
          if len(expression) <= 1200 else
          f'  L = sum of {len(terms)} rational B_alpha = {certificate["objective"]}')
    print(f'  Integer? {"YES (EXACT)" if certificate["integer"] else "NO (EXACT)"}')
    print(f'  Exact primal and dual objectives match: {certificate["objective"]}')
    print(f'  Exact primal positivity min: {certificate["min_primal_slack"]}')
    print('  Nonzero B_alpha (alpha -> exact fraction):')
    for a, v in pairs[:show_terms]:
        print(f'    {a}: {v}')
    if len(pairs) > show_terms:
        print(f'    ... {len(pairs)-show_terms} more terms (use --exact-report for all).')


def exact_report_text(certificates, compositions):
    out = [
        'Exact primal and dual LP certificates using Fraction arithmetic',
        'B_alpha = primal variables, y_beta = multipliers of K B >=0,',
        't_alpha = upper-bound multipliers, s_alpha = lower-bound multipliers.',
        'All fractions are exact; equality of feasible objectives proves optimality.',
        '',
    ]
    for cert in certificates:
        out.extend([f'delta = {cert["delta"]}',
                    f'L = {cert["objective"]}',
                    f'Integer = {cert["integer"]}',
                    f'Minimum exact primal slack = {cert["min_primal_slack"]}',
                    'Nonzero primal variables B_alpha:'])
        out += [f'  {compositions[i]} : {v}'
                for i, v in enumerate(cert['B']) if v]
        for label, key in [('y_beta', 'y'), ('t_alpha', 'upper_mult'),
                           ('s_alpha', 'lower_mult')]:
            out.append(f'Nonzero exact dual multipliers {label}:')
            out += [f'  {compositions[i]} : {v}'
                    for i, v in enumerate(cert[key]) if v]
        out.append('')
    return '\n'.join(out)


def check_near_integer(value, int_tol):
    """Numerically test closeness to an integer; not an exact LP certificate.

    Returns (status, nearest_integer, absolute_error).
    Double precision cannot resolve integer gaps reliably at >= 2**53.
    """
    if not isfinite(value):
        return 'UNKNOWN', None, float('nan')
    nearest_integer = round(value)
    error = abs(value - nearest_integer)
    if abs(value) >= 2**53:
        return 'UNRELIABLE', nearest_integer, error
    return ('YES' if error <= int_tol else 'NO'), nearest_integer, error


def integer_status(value, certificate, tolerance):
    """Separate certified integer optimal values from numerical near-integers."""
    if certificate is not None:
        return 'YES (exact)' if certificate['integer'] else 'NO (exact)'
    near, _, _ = check_near_integer(value, tolerance)
    if near == 'YES':
        return 'YES (num)'
    if near == 'NO':
        return 'NO'
    return 'N/A'


def print_table(rows, q, n, M, delta0=None, integer_tol=INTEGER_TOLERANCE):
    """Report the LP objective, its comparison with M, and integrality."""
    names = ('delta', f'L_{q}({n},delta)', 'L vs M', 'Integer?')
    body = []
    for delta, value, slack, status, certificate, failure in rows:
        if status.endswith(' <'):
            relation = '<'
        elif status.endswith(' >'):
            relation = '>'
        elif status.endswith(' ='):
            relation = '='
        else:
            relation = '~'
        body.append((str(delta), f'{value:.10f}', relation,
                     integer_status(value, certificate, integer_tol)))
    widths = [max(len(names[j]), *(len(row[j]) for row in body))
              for j in range(len(names))]
    pattern = '  '.join(f'{{:<{w}}}' for w in widths)
    print(f'\nDelsarte LP results (M = {M}):')
    print(pattern.format(*names))
    print('  '.join('-' * w for w in widths))
    for row in body:
        print(pattern.format(*row))


def print_integer_solutions(rows, cache, compositions, q, n, integer_tol):
    """For each integer-valued LP optimum, show objective and nonzero B_alpha."""
    for delta, value, slack, status, certificate, failure in rows:
        status_integer = integer_status(value, certificate, integer_tol)
        if not status_integer.startswith('YES'):
            continue

        print(f'\nLP solution for delta = {delta}  [{status_integer}]')
        print(f'  Objective: maximize sum_alpha B_alpha = L_{q}({n},{delta})')
        if certificate is not None:
            items = [(alpha, b) for alpha, b in zip(compositions, certificate['B']) if b]
            objective = certificate['objective']
            print(f'  Optimal objective value: {objective}')
            values = [str(b) for _, b in items]
        else:
            x = cache[delta]['result'].x
            items = [(alpha, float(b)) for alpha, b in zip(compositions, x)
                     if abs(float(b)) > ZERO_DISPLAY_TOLERANCE]
            print(f'  Optimal objective value: {value:.12f}  (rounded integer: {round(value)})')
            values = [f'{b:.12g}' for _, b in items]

        print('  Objective expansion:')
        for i in range(0, len(values), 6):
            chunk = ' + '.join(values[i:i+6])
            if i + 6 < len(values):
                chunk += ' +'
            print('    ' + chunk)
        print(f'    = {objective if certificate is not None else f"{value:.12f}"}')
        print('  Nonzero LP variables B_alpha:')
        for alpha, b in items:
            shown = str(b) if certificate is not None else f'{b:.12g}'
            print(f'    B_{alpha} = {shown}')



def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n', type=int, default=DEFAULT_SETTINGS["n"], help='code length (default: value of n in USER SETTINGS)')
    parser.add_argument('--q', type=int, default=DEFAULT_SETTINGS["q"], help='modulus (default: value of q in USER SETTINGS)')
    parser.add_argument('--M', type=int, default=DEFAULT_SETTINGS["M"], help='code size (default: 2**n if M=None)')
    parser.add_argument('--delta', type=int, default=DELTA,
                        help='compute one threshold only; omit to search delta_0')
    parser.add_argument('--window', type=int, default=WINDOW,
                        help='show delta_0 +/- this many deltas (default: 3)')
    parser.add_argument('--start-delta', type=int, default=None,
                        help='starting threshold for directed search')
    parser.add_argument('--tol', type=float, default=TOLERANCE, help='HiGHS tolerance')
    parser.add_argument('--integer-tol', type=float, default=INTEGER_TOLERANCE,
                        help='absolute tolerance for numerical near-integer detection (default: 1e-7)')
    parser.add_argument('--no-exact', action='store_true',
                        help='disable automatic exact rational certification')
    parser.add_argument('--max-denominator', type=int, default=EXACT_MAX_DENOMINATOR,
                        help='max denominator for rational reconstruction (default: 1000000)')
    parser.add_argument('--exact-max-variables', type=int, default=EXACT_MAX_VARIABLES,
                        help='maximum LP variable count for exact arithmetic (default: 200)')
    parser.add_argument('--exact-report', type=str, default=EXACT_REPORT,
                        help='optional text file with exact B_alpha and dual multipliers')
    parser.add_argument('--csv', type=str, default=CSV_OUTPUT,
                        help='optional CSV output file for the displayed table')
    args = parser.parse_args()
    if args.max_denominator < 1 or args.exact_max_variables < 1:
        parser.error('Require max-denominator >= 1 and exact-max-variables >= 1.')
    if args.q < 2 or args.n < 1 or args.window < 0 or args.tol <= 0 or not isfinite(args.integer_tol) or not 0 < args.integer_tol < 0.5:
        parser.error('Require q>=2, n>=1, window>=0, tol>0 and 0<integer-tol<0.5.')
    M = args.M if args.M is not None else 2**args.n
    if not 2 <= M <= args.q**args.n:
        parser.error('Require 2 <= M <= q**n.')

    dmax = args.n * (args.q//2)**2
    if args.delta is not None and not 1 <= args.delta <= dmax+1:
        parser.error(f'--delta must be in [1, {dmax+1}].')
    start = args.start_delta if args.start_delta is not None else max(1, dmax//4)
    if not 1 <= start <= dmax+1:
        parser.error(f'--start-delta must be in [1, {dmax+1}].')

    print(f'Parameters: n={args.n}, q={args.q}, M={M}')
    print('Building Lee--Krawtchouk matrix...', flush=True)
    comp, D, v, K, zero = create_lp_data(args.q, args.n)
    print(f'Lee compositions / LP variables: {len(comp)}')
    print(f'Maximum possible squared distance: {dmax}')

    cache = {}
    exact_matrix = None
    exact_valencies = None
    exact_skip_reason = None
    exact_initialized = False

    def get_exact_data():
        nonlocal exact_matrix, exact_valencies, exact_skip_reason, exact_initialized
        if not exact_initialized:
            exact_initialized = True
            exact_matrix, exact_valencies, exact_skip_reason = exact_krawtchouk(
                args.q, args.n, comp, args.exact_max_variables)
        return exact_matrix, exact_valencies, exact_skip_reason

    def evaluate(delta, force_exact=False):
        if delta not in cache:
            value, slack, result = solve_lp(delta, D, v, K, zero, args.tol)
            cache[delta] = {'value':value, 'slack':slack,
                            'result':result, 'cert':None,
                            'attempted':False, 'failure':None}
        entry = cache[delta]
        near_integer = check_near_integer(entry['value'], args.integer_tol)[0] == 'YES'
        near_m = abs(entry['value'] - M) <= max(1e-5, 1e-9*M)
        if (EXACT_CHECK and not args.no_exact and (near_integer or near_m or force_exact)
                and not entry['attempted']):
            entry['attempted'] = True
            mat, vals, skip = get_exact_data()
            if skip:
                entry['failure'] = skip
            else:
                entry['cert'], entry['failure'] = certify_exact(
                    delta, comp, D, K, zero, entry['result'],
                    mat, vals, args.max_denominator)
        return entry

    margin = max(1e-5, 1e-9*M)
    def classification(delta):
        item = evaluate(delta)
        if item['cert'] is not None:
            val = item['cert']['objective']
            return ('EXACT <' if val < M else 'EXACT >' if val > M else 'EXACT =')
        val = item['value']
        if val < M - margin:
            return 'NUM <'
        if val > M + margin:
            return 'NUM >'
        return 'NEAR'

    def excluded(delta):
        return classification(delta) in ('EXACT <', 'NUM <')

    if args.delta is not None:
        delta0 = None
        selected = [args.delta]
    else:
        print('Searching numerical threshold delta_0...', flush=True)
        if excluded(start):
            high, step = start, 1
            low = 0
            while True:
                candidate = max(1, high - step)
                if not excluded(candidate):
                    low = candidate
                    break
                if candidate == 1:
                    raise RuntimeError('Unexpected threshold at delta=1.')
                high = candidate
                step *= 2
        else:
            low, step = start, 1
            high = dmax + 1
            while True:
                candidate = min(dmax+1, low + step)
                if excluded(candidate):
                    high = candidate
                    break
                if candidate == dmax+1:
                    raise RuntimeError('No excluded delta; check numerical solver.')
                low = candidate
                step *= 2

        while low + 1 < high:
            mid = (low + high)//2
            if excluded(mid):
                high = mid
            else:
                low = mid
        delta0 = high
        # The two boundary values, not just integer-looking values, determine
        # delta_0. Try to certify both exactly whenever supported.
        evaluate(delta0, force_exact=True)
        if delta0 > 1:
            evaluate(delta0-1, force_exact=True)
        selected = list(range(max(1, delta0-args.window),
                              min(dmax+1, delta0+args.window)+1))

    rows = []
    certificates = []
    for delta in selected:
        item = evaluate(delta)
        status = classification(delta)
        rows.append((delta, item['value'], item['slack'], status,
                     item['cert'], item['failure']))
        if item['cert'] is not None:
            certificates.append(item['cert'])
    print_table(rows, args.q, args.n, M, delta0, args.integer_tol)
    if SHOW_INTEGER_SOLUTIONS:
        print_integer_solutions(rows, cache, comp, args.q, args.n,
                                args.integer_tol)

    if delta0 is not None:
        print(f'\nLP threshold delta_0 = {delta0}')
        print(f'Computed distance bound: d_T(C)^2 <= {delta0-1}; '
              f'd_T(C) <= sqrt({delta0-1}) = {sqrt(delta0-1):.10f}')

    if args.exact_report:
        with open(args.exact_report, 'w', encoding='utf-8') as fh:
            fh.write(exact_report_text(certificates, comp))
        print(f'Exact report saved to: {args.exact_report}')

    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as output:
            writer = csv.writer(output)
            writer.writerow(['n', 'q', 'M', 'delta', 'L_q(n,delta)',
                             'L_vs_M', 'is_delta_0', 'Integer?'])
            for d, val, slack, status, cert, failure in rows:
                relation = '<' if status.endswith(' <') else (
                    '>' if status.endswith(' >') else (
                        '=' if status.endswith(' =') else '~'))
                writer.writerow([args.n, args.q, M, d, f'{val:.16g}',
                                 relation, int(d == delta0),
                                 integer_status(val, cert, args.integer_tol)])
        print(f'Table saved to: {args.csv}')


if __name__ == '__main__':
    main()
