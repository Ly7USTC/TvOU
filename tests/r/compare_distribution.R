# 200 R-simulated paths -> estimates; python/tests/compare_distribution.py compares
# them with Python-simulated paths (different RNGs, same distribution expected).
source("R/tvou.R")
set.seed(32)
Qu <- 1 + 0.5 * sin(50)
est <- t(replicate(1000, {
  y <- simulate_gamma_ou(1, 1, 2, 10, 0.001, Qu)$path
  c(jf = jump_filter_estimate(y, 0.001, 0.5), bpv = bpv_estimate(y, 0.001, c(0.5, 0.5), "product"),
    y_end = tail(y, 1))
}))
write.csv(est, "tests/fixtures/distribution_r.csv", row.names = FALSE)
