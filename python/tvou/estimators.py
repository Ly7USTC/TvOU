"""Estimators of the drift coefficient Q of (time-varying) OU processes."""

import warnings

import numpy as np


def jump_filter_estimate(y, delta, exponent, filter="abs"):
    """Jump-filtering estimator

        Q_hat = - sum_i dY_i 1{filter_i} / (delta * sum_{i=0}^{N-1} Y_i),

    filter="abs":      keep |dY_i| <= delta^exponent  (eq. 5 of the paper)
    filter="positive": keep  dY_i  <= delta^exponent  (removes only positive jumps)
    """
    y = np.asarray(y, float)
    dy = np.diff(y)
    v = delta**exponent
    if filter == "abs":
        keep = np.abs(dy) <= v
    elif filter == "positive":
        keep = dy <= v
    else:
        raise ValueError("filter must be 'abs' or 'positive'")
    return -dy[keep].sum() / (delta * y[:-1].sum())


def bpv_estimate(y, delta, r=(0.5, 0.5), denominator="paper"):
    """Bipower-variation estimator

        Q_hat = sum_j |dY_{j-1}|^r1 |dY_j|^r2 / D,

    D = delta * sum_{i=0}^{N-1} Y_i^(r1+r2)      (denominator="paper")
    D = delta * sum_j Y_{j-1}^r1 Y_j^r2           (denominator="product")

    The ratio behaves like Q^(r1+r2) delta^(r1+r2-1), so r1 + r2 = 1 is required
    for it to estimate Q.
    """
    r1, r2 = r
    if abs(r1 + r2 - 1) > 1e-12:
        warnings.warn("r1 + r2 != 1: the estimator then targets Q^(r1+r2) delta^(r1+r2-1)", stacklevel=2)
    y = np.asarray(y, float)
    a = np.abs(np.diff(y))
    num = np.sum(a[:-1] ** r1 * a[1:] ** r2)
    if denominator == "paper":
        den = delta * np.sum(y[:-1] ** (r1 + r2))
    elif denominator == "product":
        den = delta * np.sum(y[:-2] ** r1 * y[1:-1] ** r2)
    else:
        raise ValueError("denominator must be 'paper' or 'product'")
    return num / den


def window_slice(y, delta, t0, bandwidth):
    """Observations in [t0 - bandwidth, t0 + bandwidth] of a path observed at 0, delta, ..., T."""
    centre = int(round(t0 / delta))
    half = int(round(bandwidth / delta))
    lo, hi = centre - half, centre + half
    if lo < 0 or hi >= len(y):
        raise ValueError("window extends beyond the observed path")
    return np.asarray(y)[lo:hi + 1]


def local_estimate(y, delta, t0, bandwidth, method="jump_filter", **kwargs):
    """Estimate Q(t0) from the observations in [t0 - bandwidth, t0 + bandwidth].

    method="jump_filter" passes kwargs (exponent, filter) to jump_filter_estimate;
    method="bpv" passes kwargs (r, denominator) to bpv_estimate.
    """
    w = window_slice(y, delta, t0, bandwidth)
    if method == "jump_filter":
        return jump_filter_estimate(w, delta, **kwargs)
    if method == "bpv":
        return bpv_estimate(w, delta, **kwargs)
    raise ValueError("method must be 'jump_filter' or 'bpv'")


def local_curve(y, delta, t0s, bandwidth, method="jump_filter", **kwargs):
    """local_estimate at each t0 in t0s."""
    return np.array([local_estimate(y, delta, t0, bandwidth, method, **kwargs) for t0 in t0s])
