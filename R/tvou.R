# =============================================================================
# Jump-filtering estimation of (time-varying) Ornstein-Uhlenbeck processes
# Clean R reference implementation.
#
#   dY_t = -Q(t) Y_t dt + dL_t,   L a gamma subordinator with L_1 ~ Gamma(alpha, rate beta)
#
# Contents
#   1. Simulation: exact gamma-OU (Qu, Dassios & Zhao 2019), approximate
#      time-varying version, Euler scheme with given driving increments
#   2. Conditional mean E[Y_t | Y_0]
#   3. Estimators: jump filtering (global / local), bipower variation
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Simulation
# -----------------------------------------------------------------------------

# Q values used on each of the N steps of length delta.
#   Q: a number (constant coefficient), a vector of length N, or a function of t.
#   For a function, step i (from (i-1)delta to i delta) uses Q(i delta), as in the
#   original scripts.
q_on_steps <- function(Q, N, delta) {
  if (is.function(Q)) return(Q((1:N) * delta))
  if (length(Q) == 1) return(rep(Q, N))
  stopifnot(length(Q) == N)
  Q
}

# One step of the exact simulation (Qu, Dassios & Zhao 2019) over a step tau:
#   Y_{t+tau} = exp(-Q tau) Y_t + Gamma~ + sum_{k <= N~} S_k,
#   Gamma~ ~ Gamma(alpha tau, rate beta exp(Q tau)),  N~ ~ Poisson(alpha Q tau^2 / 2),
#   S_k ~ Exp(rate beta exp(Q tau sqrt(U_k))),  U_k ~ U[0, 1].
# With a constant Q this is exact; with Q varying between steps it is the
# approximation of the paper (Q frozen over each step).
# The draws follow the same order as the original exactGammaOU_3 /
# appro_TVgammaOU_2, so set.seed() reproduces those functions.
simulate_gamma_ou <- function(alpha, beta, y0, T, delta, Q) {
  N <- round(T / delta)
  q <- q_on_steps(Q, N, delta)
  y <- numeric(N + 1); y[1] <- y0
  gamma_tilde <- numeric(N); jump_sum <- numeric(N)
  for (i in 1:N) {
    gamma_tilde[i] <- rgamma(1, alpha * delta, beta * exp(q[i] * delta))
    n_jumps <- rpois(1, 0.5 * alpha * q[i] * delta^2)
    s <- 0
    for (j in seq_len(n_jumps)) {
      u <- runif(1)
      s <- s + rexp(1, beta * exp(q[i] * delta * sqrt(u)))
    }
    jump_sum[i] <- s
    y[i + 1] <- exp(-q[i] * delta) * y[i] + gamma_tilde[i] + s
  }
  list(path = y, gamma_tilde = gamma_tilde, jump_sum = jump_sum, Q = q)
}

# Path from given step innovations (deterministic; used to cross-check Python).
gamma_ou_path_from_draws <- function(y0, q, delta, gamma_tilde, jump_sum) {
  y <- numeric(length(q) + 1); y[1] <- y0
  for (i in seq_along(q)) y[i + 1] <- exp(-q[i] * delta) * y[i] + gamma_tilde[i] + jump_sum[i]
  y
}

# Euler scheme Y_i = Y_{i-1} - Q_i Y_{i-1} delta + dL_i for given increments dL.
euler_ou <- function(increments, y0, delta, Q) {
  N <- length(increments)
  q <- q_on_steps(Q, N, delta)
  y <- numeric(N + 1); y[1] <- y0
  for (i in 1:N) y[i + 1] <- y[i] - q[i] * y[i] * delta + increments[i]
  y
}

gamma_increments <- function(N, alpha, beta, delta) rgamma(N, alpha * delta, beta)
brownian_increments <- function(N, delta) rnorm(N, 0, sqrt(delta))

# -----------------------------------------------------------------------------
# 2. Conditional mean (constant Q):  E[Y_t | Y_0] = e^{-Qt} Y_0 + (1 - e^{-Qt}) alpha / (Q beta)
# -----------------------------------------------------------------------------
conditional_mean <- function(t, y0, Q, alpha, beta) {
  exp(-Q * t) * y0 + (1 - exp(-Q * t)) * alpha / (Q * beta)
}

# -----------------------------------------------------------------------------
# 3. Estimators
# -----------------------------------------------------------------------------

# Jump-filtering estimator
#   Q_hat = - sum_i dY_i 1{filter} / (delta sum_{i=0}^{N-1} Y_i)
#   filter = "abs":      |dY_i| <= delta^exponent   (paper, eq. 5)
#   filter = "positive": dY_i <= delta^exponent     (only positive jumps removed, Example 4.5)
jump_filter_estimate <- function(y, delta, exponent, filter = c("abs", "positive")) {
  filter <- match.arg(filter)
  dy <- diff(y)
  v <- delta^exponent
  keep <- if (filter == "abs") abs(dy) <= v else dy <= v
  -sum(dy[keep]) / (delta * sum(head(y, -1)))
}

# Bipower-variation estimator (needs r1 + r2 = 1)
#   numerator   sum_j |dY_{j-1}|^r1 |dY_j|^r2
#   denominator "paper":   delta sum_{i=0}^{N-1} Y_i^(r1 + r2)
#               "product": delta sum_j Y_{j-1}^r1 Y_j^r2   (used in the simulation scripts)
bpv_estimate <- function(y, delta, r = c(0.5, 0.5), denominator = c("paper", "product")) {
  denominator <- match.arg(denominator)
  if (abs(sum(r) - 1) > 1e-12) warning("r1 + r2 != 1: the estimator then targets Q^(r1+r2) delta^(r1+r2-1)")
  a <- abs(diff(y))
  n <- length(y)
  num <- sum(head(a, -1)^r[1] * tail(a, -1)^r[2])
  den <- if (denominator == "paper") delta * sum(head(y, -1)^sum(r))
         else delta * sum(y[1:(n - 2)]^r[1] * y[2:(n - 1)]^r[2])
  num / den
}

# Observations in the window [t0 - b_N, t0 + b_N] (path observed at 0, delta, ..., T)
window_slice <- function(y, delta, t0, bandwidth) {
  centre <- round(t0 / delta) + 1
  half <- round(bandwidth / delta)
  idx <- (centre - half):(centre + half)
  if (min(idx) < 1 || max(idx) > length(y)) stop("window extends beyond the observed path")
  y[idx]
}

local_estimate <- function(y, delta, t0, bandwidth, method = c("jump_filter", "bpv"), ...) {
  method <- match.arg(method)
  w <- window_slice(y, delta, t0, bandwidth)
  if (method == "jump_filter") jump_filter_estimate(w, delta, ...) else bpv_estimate(w, delta, ...)
}
