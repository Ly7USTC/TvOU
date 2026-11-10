# Jump-filtering estimation of time-varying Ornstein–Uhlenbeck processes

Simulation and estimation of the drift coefficient of Lévy-driven Ornstein–Uhlenbeck
(OU) processes, including the time-varying case. Python package with an R reference
implementation; the Python code is cross-checked against R in the test suite.

## Model

$$
dY_t = -Q(t)\,Y_t\,dt + dL_t ,
$$

where $L$ is a gamma subordinator with $L_1 \sim \Gamma(\alpha, \beta)$ (shape α, rate β).
With constant $Q$ this is the classical gamma-driven OU process; with a smooth
function $Q(t)$ it is a time-varying OU (tv-OU) process, which is not stationary
but is locally stationary.

## Estimators

For observations $Y_0, Y_\delta, \dots, Y_{N\delta}$ with increments $\Delta_i Y = Y_{(i+1)\delta} - Y_{i\delta}$:

**Jump filtering.** The continuous part satisfies $dY^c_t = -Q Y_t\,dt$, so small increments carry the drift:

$$
\hat Q_N = -\,\frac{\sum_{i} \Delta_i Y\,\mathbf 1_{\lbrace |\Delta_i Y| \le \delta^{b}\rbrace}}{\delta \sum_{i} Y_{i\delta}} .
$$

`filter="positive"` keeps all increments with $\Delta_i Y \le \delta^b$, removing only
upward jumps, which matters when the threshold is small.

**Bipower variation.** A threshold-free alternative, for $r_1 + r_2 = 1$:

$$
\hat Q^{BPV} = \frac{\sum_{j} |\Delta_{j-1} Y|^{r_1}\,|\Delta_j Y|^{r_2}}{\delta \sum_{i} Y_{i\delta}} .
$$

**Local estimation.** For a time-varying coefficient, either estimator is applied to
the observations in a window $[t_0 - b, t_0 + b]$ to estimate $Q(t_0)$.

## Simulation

* `simulate_gamma_ou`: exact simulation of the gamma-driven OU process
  (Qu, Dassios & Zhao, 2019). If `Q` is a function or an array, Q is frozen over each step,
  which gives the approximate tv-OU scheme.
* `euler_ou`: Euler scheme for any driving increments (gamma process or Brownian motion).
* `conditional_mean`, `stationary_moments`: closed-form moments for checks.

## Repository layout

```
python/tvou/          Python package
  simulation.py       exact / approximate / Euler simulation, moments
  estimators.py       jump filtering, bipower variation, local estimation
python/tests/         pytest suite (cross-checks against R + statistical tests)
python/examples/      OU vs tv-OU paths, constant-Q study, time-varying study
R/tvou.R              R reference implementation
tests/r/              R checks against the original scripts; writes the fixtures
tests/fixtures/       reference values produced by R
```

## Quick start

```bash
cd python
pip install -e ".[test]"
pytest
```

```python
import numpy as np
import tvou

Q = lambda t: 1 + 0.5 * np.sin(0.8 * t)
y = tvou.simulate_gamma_ou(alpha=1, beta=1, y0=2, T=10, delta=0.001, Q=Q, rng=1)

t0s = np.arange(0.2, 9.81, 0.2)
q_hat = tvou.local_curve(y, 0.001, t0s, bandwidth=0.1, method="jump_filter", exponent=0.5)
```

## R reference

```bash
Rscript tests/r/make_fixtures.R
```

This checks the clean R code against the original simulation and estimation
scripts (identical output for the same seed), then writes the fixtures used by
`python/tests/test_against_r.py`. `tests/r/compare_distribution.R` together with
`python/tests/compare_distribution.py` compares estimator distributions from R- and
Python-simulated paths.

## References

* H. Mai (2014). Efficient maximum likelihood estimation for Lévy-driven Ornstein–Uhlenbeck processes. *Bernoulli*, 20(2), 919–957.
* Y. Qu, A. Dassios and H. Zhao (2019). Exact simulation of gamma-driven Ornstein–Uhlenbeck processes with finite and infinite activity jumps. *Journal of the Operational Research Society*.
* O. E. Barndorff-Nielsen, N. Shephard and M. Winkel (2006). Limit theorems for multipower variation in the presence of jumps. *Stochastic Processes and their Applications*, 116(5), 796–806.
* A. Bitter, R. Stelzer and B. Ströh (2023). Continuous-time locally stationary time series models. *Advances in Applied Probability*, 55(3), 965–998.

## Author

Yuan Li, Department of Mathematics, Imperial College London
