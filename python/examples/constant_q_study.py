"""Constant coefficient: mean path vs E[Y_t | Y_0], and estimator accuracy
across sampling frequencies and threshold exponents.

Usage:  python examples/constant_q_study.py [M]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

import tvou

M = int(sys.argv[1]) if len(sys.argv) > 1 else 100
Q = 1 + 0.5 * np.sin(50)
rng = np.random.default_rng(2023)

# 1. mean of M paths vs the conditional mean (alpha = beta = 10, Y_0 = 10, delta = 0.01)
paths = np.array([tvou.simulate_gamma_ou(10, 10, 10, 10, 0.01, Q, rng=rng) for _ in range(M)])
t = np.arange(paths.shape[1]) * 0.01
theory = tvou.conditional_mean(t, 10, Q, 10, 10)
print(f"mean path vs E[Y_t | Y_0]: MSE = {np.mean((paths.mean(0) - theory) ** 2):.2e}")
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(t, paths.mean(0), color="black", label="mean of paths")
ax.plot(t, theory, color="grey", label="E[Y_t | Y_0]")
ax.legend()
fig.savefig("mean_path.png", dpi=130, bbox_inches="tight")


def study(delta, label, **kw):
    est = np.array([tvou.jump_filter_estimate(tvou.simulate_gamma_ou(1, 1, 2, 10, delta, Q, rng=rng),
                                              delta, **kw) for _ in range(M)])
    print(f"{label:<34} mean {est.mean():.4f}  sd {est.std(ddof=1):.4f}  MSE {np.mean((est - Q)**2):.2e}")


# 2. sampling frequency (alpha = beta = 1, threshold exponent 0.3)
for delta in (1e-2, 1e-3, 1e-4):
    study(delta, f"delta = {delta:g}, exponent 0.3", exponent=0.3)
# 3. threshold exponent (delta = 0.001)
for b in (0.1, 0.3, 0.4, 0.5):
    study(1e-3, f"delta = 0.001, exponent {b}", exponent=b)
# 4. exponent 1: |dY| filter vs positive-jump filter
study(1e-3, "exponent 1, |dY| filter", exponent=1.0, filter="abs")
study(1e-3, "exponent 1, positive filter", exponent=1.0, filter="positive")
