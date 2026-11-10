"""Simulation of gamma-driven (time-varying) Ornstein-Uhlenbeck processes.

    dY_t = -Q(t) Y_t dt + dL_t,   L a gamma subordinator with L_1 ~ Gamma(alpha, rate beta).
"""

import numpy as np


def q_on_steps(Q, N, delta):
    """Q value used on each of the N steps of length delta.

    Q may be a number, an array of length N, or a function of t. A function is
    evaluated at the right end of each step, Q(i * delta), i = 1..N, as in the
    original scripts.
    """
    if callable(Q):
        return np.asarray(Q(np.arange(1, N + 1) * delta), dtype=float)
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 0:
        return np.full(N, float(Q))
    if Q.shape != (N,):
        raise ValueError(f"Q must be a number, a function, or an array of length {N}")
    return Q


def path_from_innovations(y0, phi, eps):
    """Y_i = phi_i Y_{i-1} + eps_i, i = 1..N, with Y_0 = y0."""
    y = np.empty(len(phi) + 1)
    y[0] = y0
    for i in range(len(phi)):
        y[i + 1] = phi[i] * y[i] + eps[i]
    return y


def gamma_ou_path_from_draws(y0, q, delta, gamma_tilde, jump_sum):
    """Rebuild a path from the step innovations of the exact scheme."""
    q = np.asarray(q, float)
    return path_from_innovations(y0, np.exp(-q * delta), np.asarray(gamma_tilde) + np.asarray(jump_sum))


def simulate_gamma_ou(alpha, beta, y0, T, delta, Q, rng=None, return_draws=False):
    """Simulate a gamma-driven OU process on 0, delta, ..., T (Qu, Dassios & Zhao, 2019).

    Each step of length tau = delta:
        Y_{t+tau} = exp(-Q tau) Y_t + Gamma~ + sum_{k <= N~} S_k,
        Gamma~ ~ Gamma(alpha tau, rate beta exp(Q tau)),
        N~ ~ Poisson(alpha Q tau^2 / 2),
        S_k ~ Exp(rate beta exp(Q tau sqrt(U_k))),  U_k ~ U[0, 1].

    With constant Q the scheme is exact. When Q varies (array or function), Q is
    frozen over each step, which gives the approximate time-varying scheme.
    """
    rng = np.random.default_rng(rng)
    N = int(round(T / delta))
    q = q_on_steps(Q, N, delta)
    gamma_tilde = rng.gamma(shape=alpha * delta, scale=1.0 / (beta * np.exp(q * delta)))
    n_jumps = rng.poisson(0.5 * alpha * q * delta**2)
    step = np.repeat(np.arange(N), n_jumps)
    u = rng.uniform(size=step.size)
    s = rng.exponential(scale=1.0 / (beta * np.exp(q[step] * delta * np.sqrt(u))))
    jump_sum = np.bincount(step, weights=s, minlength=N)
    y = gamma_ou_path_from_draws(y0, q, delta, gamma_tilde, jump_sum)
    if return_draws:
        return y, dict(Q=q, gamma_tilde=gamma_tilde, jump_sum=jump_sum)
    return y


def euler_ou(increments, y0, delta, Q):
    """Euler scheme Y_i = Y_{i-1} - Q_i Y_{i-1} delta + dL_i for given increments dL."""
    increments = np.asarray(increments, float)
    q = q_on_steps(Q, len(increments), delta)
    return path_from_innovations(y0, 1.0 - q * delta, increments)


def gamma_increments(N, alpha, beta, delta, rng=None):
    """Increments of a gamma process with L_1 ~ Gamma(alpha, rate beta)."""
    return np.random.default_rng(rng).gamma(shape=alpha * delta, scale=1.0 / beta, size=N)


def brownian_increments(N, delta, rng=None):
    return np.random.default_rng(rng).normal(0.0, np.sqrt(delta), size=N)


def conditional_mean(t, y0, Q, alpha, beta):
    """E[Y_t | Y_0] = e^{-Qt} Y_0 + (1 - e^{-Qt}) alpha / (Q beta)  (constant Q)."""
    t = np.asarray(t, float)
    return np.exp(-Q * t) * y0 + (1 - np.exp(-Q * t)) * alpha / (Q * beta)


def stationary_moments(Q, alpha, beta):
    """Mean and variance of the stationary gamma-driven OU process."""
    return alpha / (Q * beta), alpha / (2 * Q * beta**2)
