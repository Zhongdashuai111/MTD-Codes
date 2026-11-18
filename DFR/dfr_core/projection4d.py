"""Paiva-style 1D projections, stabilized by exponential tilting.
The target distribution is discrete and retains polynomial-product dependence.
This is an independent numerical implementation, not certified interval arithmetic.
"""
import itertools, json, math, time
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp
from .noise_model import compression, cgf
CBD = np.array([1, 4, 6, 4, 1], dtype=float) / 16
A = np.array(list(itertools.product(range(-2, 3), repeat=4)), dtype=np.int64)
AW = np.prod(CBD[A + 2], axis=1)
PU, OU = compression(11)
PV, OV = compression(5)
B = [(CBD, -2), (np.convolve(CBD, PU), OU - 2)]
E = (np.convolve(CBD, PV), OV - 2)
BV = []
for p, lo in B:
    v = np.array(list(itertools.product(range(lo, lo + len(p)), repeat=4)), dtype=np.int64)
    BV.append((v, np.prod(p[v - lo], axis=1)))

def make_parts(h):
    """Enumerate scalar projections of the two dependent product vectors and final coordinate noise."""
    M = np.array([[h[(i + j) % 4] * (1 if i + j < 4 else -1) for j in range(4)] for i in range(4)], dtype=np.int64)
    W = np.ascontiguousarray(A @ M)
    out = []
    for (p, lo), (v, pb) in zip(B, BV):
        rad = int(np.abs(W).sum(1).max()) * max(abs(lo), abs(lo + len(p) - 1))
        pbase = np.zeros(2 * rad + 1)
        indices = (W @ v.T + rad).ravel()
        weights = (AW[:, None] * pb[None, :]).ravel()
        pbase = np.bincount(indices, weights=weights, minlength=2 * rad + 1)
        inds = np.flatnonzero(pbase)
        begin, end = (int(inds[0]), int(inds[-1]) + 1)
        pbase = pbase[begin:end]
        assert abs(pbase.sum() - 1) < 2e-14
        out.append((pbase, begin - rad, 256))
    p, lo = E
    for a in h:
        if not a:
            continue
        out.append((p, (np.arange(len(p), dtype=np.int64) + lo) * int(a), 1))
    return [(p, np.arange(len(p), dtype=np.int64) + o if np.isscalar(o) else o, r) for p, o, r in out]

def evaluate(parts, T, tiltfactor=1.0, nmult=1, full=False):
    """Evaluate a discrete tail by exponential tilting and FFT; record alias and truncation controls."""

    def cumul(t):
        K = m = v = 0.0
        for p, x, r in parts:
            y = t * x
            peak = float(y.max())
            w = p * np.exp(y - peak)
            z = w.sum()
            pnew = w / z
            mean = float(pnew @ x)
            var = float(pnew @ (x - mean) ** 2)
            K += r * (math.log(z) + peak)
            m += r * mean
            v += r * var
        return (K, m, v)
    scale = max((abs(int(x.min())) for p, x, r in parts)) + max((abs(int(x.max())) for p, x, r in parts))
    hi = 1 / scale
    while cumul(hi)[1] < T:
        hi *= 2
    saddle = brentq(lambda t: cumul(t)[1] - T, 0, hi, xtol=1e-16)
    tilt = saddle * tiltfactor
    K, mean, var = cumul(tilt)
    window = math.ceil(48 / tilt)
    supportlo = sum((int(x.min()) * r for p, x, r in parts))
    supporthi = sum((int(x.max()) * r for p, x, r in parts))
    if full:
        N = next_fast_len(supporthi - supportlo + 1, real=True)
    else:
        N = next_fast_len(math.ceil(14 * math.sqrt(var) + 2 * window + 2 * abs(mean - T)), real=True) * nmult
    assert N > window
    freq = np.ones(N // 2 + 1, dtype=np.complex128)
    for p, x, r in parts:
        y = tilt * x
        w = p * np.exp(y - y.max())
        w /= w.sum()
        pad = np.zeros(N)
        np.add.at(pad, x % N, w)
        freq *= rfft(pad, workers=1) ** r
    tilted = irfft(freq, n=N, workers=1)
    s = np.arange(T, min(supporthi, T + window) + 1, dtype=np.int64)
    weighted = float(tilted[s % N] @ np.exp(-tilt * (s - T)))
    assert weighted > 0
    logtail = (K - tilt * T + math.log(weighted)) / math.log(2)

    def tilted_chernoff(bound, direction):
        if bound < supportlo or bound > supporthi:
            return -math.inf

        def f(u):
            return cumul(tilt + u)[0] - K - u * bound
        lim = 16 / scale
        r = minimize_scalar(f, bounds=(-lim, 0) if direction < 0 else (0, lim), method='bounded', options={'xatol': 1e-15})
        return r.fun / math.log(2)
    if full:
        alias = -math.inf
    else:
        alias = float(logsumexp(np.array([tilted_chernoff(T + window - N, -1), tilted_chernoff(T + N, 1)]) * math.log(2)) / math.log(2))
    return dict(log2_tail=logtail, tilt=tilt, fft_length=N, full_support_length=supporthi - supportlo + 1, relative_alias_bound_log2=None if math.isinf(alias) else alias - math.log2(weighted), relative_truncated_weight_bound_log2=-tilt * window / math.log(2) - math.log2(weighted), tilted_mass=float(tilted.sum()), weighted_tail=weighted)

def run(label, hs, mult, output_file):
    """Evaluate all specified directions and retain distant contributions as Chernoff remainders."""
    start = time.time()
    records = []
    for h, m in zip(hs, mult):
        g = int(np.gcd.reduce(h))
        hr = h // g
        T = (int(h @ h) + 2 * g - 1) // (2 * g)
        scaled = hr / 1024.0
        M = np.array([[scaled[(i + j) % 4] * (1 if i + j < 4 else -1) for j in range(4)] for i in range(4)])
        W = A @ M
        target = T / 1024.0
        hi = 1.0
        while cgf(hi, W, scaled, AW, B, E, 4)[1] < target:
            hi *= 2
        t = brentq(lambda t: cgf(t, W, scaled, AW, B, E, 4)[1] - target, 0, hi)
        K = cgf(t, W, scaled, AW, B, E, 4)[0]
        records.append(dict(h=h.tolist(), primitive_h=hr.tolist(), threshold=T, multiplicity=int(m), chernoff_log2=(K - t * target) / math.log(2)))
    largest = max((r['chernoff_log2'] for r in records))
    partial = []
    remaining = []
    for i, r in enumerate(records):
        if r['chernoff_log2'] < largest - 65:
            r['method'] = 'Chernoff remainder'
            remaining.append(r['chernoff_log2'] + math.log2(r['multiplicity']))
        else:
            ps = make_parts(np.array(r['primitive_h'], dtype=np.int64))
            r['method'] = 'tilted one-dimensional FFT'
            r['fft'] = evaluate(ps, r['threshold'])
            if i in [0, 1, 2] or i % 20 == 0:
                r['check'] = evaluate(ps, r['threshold'], 0.94, 2)
            partial.append(r['fft']['log2_tail'] + math.log2(r['multiplicity']))
        if i % 10 == 0:
            print(label, i, len(records), round(time.time() - start, 1), r.get('fft', {}).get('log2_tail'), flush=True)
        data = dict(label=label, parameters=dict(q=3329, n=256, k=4, eta1=2, eta2=2, du=11, dv=5), complete=i == len(records) - 1, directions=records[:i + 1], elapsed_seconds=time.time() - start)
        if partial:
            data['log2_evaluated_sum'] = 2 + float(logsumexp(np.array(partial) * math.log(2)) / math.log(2))
        if remaining:
            data['log2_chernoff_remainder'] = 2 + float(logsumexp(np.array(remaining) * math.log(2)) / math.log(2))
        if partial or remaining:
            data['log2_combined_bound'] = 2 + float(logsumexp(np.array(partial + remaining) * math.log(2)) / math.log(2))
        output_file.write_text(json.dumps(data, indent=2))
    print('DONE', label, data['log2_combined_bound'], time.time() - start, flush=True)
    return data
