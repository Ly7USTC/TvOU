"""Local estimation of Q(t) = 1 + 0.5 sin(0.8 t) over [0, 10].

Each path is simulated once and Q is estimated at every t0 on that path.
Usage:  python examples/time_varying_study.py [M]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

import tvou

M = int(sys.argv[1]) if len(sys.argv) > 1 else 100
Qfun = lambda t: 1 + 0.5 * np.sin(0.8 * t)
delta, bandwidth = 0.001, 0.1
t0s = np.round(np.arange(0.2, 9.81, 0.2), 10)
rng = np.random.default_rng(2023)

settings = {
    "jump filter, exponent 0.4": dict(method="jump_filter", exponent=0.4),
    "jump filter, exponent 0.5": dict(method="jump_filter", exponent=0.5),
    "BPV, r = (0.5, 0.5)": dict(method="bpv", r=(0.5, 0.5), denominator="product"),
    "BPV, r = (0.3, 0.7)": dict(method="bpv", r=(0.3, 0.7), denominator="product"),
}
est = {k: [] for k in settings}
for _ in range(M):
    y = tvou.simulate_gamma_ou(1, 1, 2, 10, delta, Qfun, rng=rng)
    for k, kw in settings.items():
        est[k].append(tvou.local_curve(y, delta, t0s, bandwidth, **kw))

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
grid = np.linspace(0, 10, 400)
for ax, (k, e) in zip(axes.flat, est.items()):
    e = np.array(e)
    m, s = e.mean(0), e.std(0, ddof=1)
    ax.fill_between(t0s, m - s, m + s, color="0.8")
    ax.plot(grid, Qfun(grid), color="black", lw=1)
    ax.plot(t0s, m, color="red", lw=1)
    ax.set_title(k, fontsize=10)
    ax.set_xlabel("time")
    print(f"{k:<28} mean abs bias {np.mean(np.abs(m - Qfun(t0s))):.3f}   mean sd {s.mean():.3f}")
fig.tight_layout()
fig.savefig("time_varying_estimates.png", dpi=130)
