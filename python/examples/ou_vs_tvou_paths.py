"""OU vs time-varying OU paths driven by the same noise, zooming in at t = 5.

Usage:  python examples/ou_vs_tvou_paths.py [gamma|brownian]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

import tvou

driver = sys.argv[1] if len(sys.argv) > 1 else "gamma"
T, delta = 10, 1e-5
N = int(T / delta)
Qfun = lambda t: 1 + 0.5 * np.sin(10 * t)
Q5 = Qfun(5.0)                                   # the OU uses the tv coefficient at t = 5
rng = np.random.default_rng(1)

if driver == "gamma":
    inc = tvou.gamma_increments(N, 10, 10, delta, rng)   # L_1 ~ Gamma(10, 10)
    y0 = 10 / (10 * Q5)                                  # stationary mean
else:
    inc = tvou.brownian_increments(N, delta, rng)
    y0 = 0.0

ou = tvou.euler_ou(inc, y0, delta, Q5)
tv = tvou.euler_ou(inc, y0, delta, Qfun)
t = np.arange(N + 1) * delta

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (lo, hi) in zip(axes.flat, [(0, 10), (4, 6), (4.9, 5.1), (4.99, 5.01)]):
    m = (t >= lo) & (t <= hi)
    ax.plot(t[m], tv[m], color="blue", lw=0.6, label="tv-OU")
    ax.plot(t[m], ou[m], color="grey", lw=0.6, ls="--", label="OU")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("time")
    ax.legend(loc="upper left", fontsize=8)
fig.suptitle(f"OU (Q = Q(5)) vs tv-OU (Q(t) = 1 + 0.5 sin 10t), {driver} driver")
fig.tight_layout()
fig.savefig(f"ou_vs_tvou_{driver}.png", dpi=130)
