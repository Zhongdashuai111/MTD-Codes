"""Discrete noise PMFs and cumulants; numerical arithmetic is floating point."""
import itertools, json, struct, sys, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import brentq
CBD = np.array([1, 4, 6, 4, 1], dtype=float) / 16

def compression(d, q=3329):
    x = np.arange(q, dtype=np.int64)
    y = (2 * (x << d) + q) // (2 * q) % (1 << d)
    z = (2 * y * q + (1 << d)) // (2 * (1 << d))
    e = (z - x + q // 2) % q - q // 2
    lo, hi = (int(e.min()), int(e.max()))
    return (np.bincount(e - lo, minlength=hi - lo + 1) / q, lo)

def cgf(t, W, h, Aweights, B, E, ell, details=False):
    K = K1 = K2 = 0.0
    samplers = []
    reps = 1024 // ell
    for p, o in B:
        vals = np.arange(len(p)) + o
        v = np.exp(t * W[:, :, None] * vals[None, None, :]) * p
        m = v.sum(axis=2)
        condp = v / m[:, :, None]
        first = (condp * vals).sum(axis=2) * W
        second = (condp * vals ** 2).sum(axis=2) * W ** 2 - first ** 2
        logs = np.log(Aweights) + np.log(m).sum(axis=1)
        logz = logsumexp(logs)
        pa = np.exp(logs - logz)
        mu = first.sum(axis=1)
        var = second.sum(axis=1)
        m1 = pa @ mu
        m2 = pa @ (var + mu ** 2) - m1 ** 2
        K += reps * logz
        K1 += reps * m1
        K2 += reps * m2
        if details:
            samplers.append((pa, np.cumsum(condp, axis=2)))
    p, o = E
    vals = np.arange(len(p)) + o
    v = np.exp(t * h[:, None] * vals[None, :]) * p
    m = v.sum(axis=1)
    pv = v / m[:, None]
    mu = (pv * vals).sum(axis=1) * h
    var = (pv * vals ** 2).sum(axis=1) * h * h - mu * mu
    K += np.log(m).sum()
    K1 += mu.sum()
    K2 += var.sum()
    if details:
        return (K, K1, K2, samplers, np.cumsum(pv, axis=1))
    return (K, K1, K2)
