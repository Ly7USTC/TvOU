# Checks the clean R code against the original scripts, then writes fixtures
# for the Python tests.  Run from the repository root:  Rscript tests/r/make_fixtures.R
source("R/tvou.R")
source("tests/r/draft_functions.R")
out <- "tests/fixtures"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
wcsv <- function(x, f) write.csv(x, file.path(out, f), row.names = FALSE)
check <- function(label, a, b, tol = 1e-12) {
  err <- max(abs(a - b))
  cat(sprintf("%-60s max abs diff = %.2e  %s\n", label, err, if (err < tol) "OK" else "FAIL"))
  if (err >= tol) quit(status = 1)
}
Qu <- 1 + 0.5 * sin(50)

# ---- 1. exact gamma-OU: clean vs draft (same seed) -------------------------
set.seed(1); draft <- exactGammaOU_3(1, 1, 2, 10, 0.001)
set.seed(1); clean <- simulate_gamma_ou(1, 1, 2, 10, 0.001, Qu)
check("simulate_gamma_ou (constant Q) vs draft exactGammaOU_3", clean$path, draft)
check("path rebuilt from draws equals simulated path",
      gamma_ou_path_from_draws(2, clean$Q, 0.001, clean$gamma_tilde, clean$jump_sum), clean$path)
wcsv(data.frame(Q = clean$Q, gamma_tilde = clean$gamma_tilde, jump_sum = clean$jump_sum), "ou_draws.csv")
wcsv(data.frame(y = clean$path), "ou_path.csv")

# ---- 2. time-varying: clean vs draft ---------------------------------------
Qfun <- function(t) 1 + 0.5 * sin(0.8 * t)
set.seed(2); draft_tv <- appro_TVgammaOU_2(1, 1, 2, 10, 0.001, Qfun(seq(0.001, 10, by = 0.001)))
set.seed(2); clean_tv <- simulate_gamma_ou(1, 1, 2, 10, 0.001, Qfun)
check("simulate_gamma_ou (Q function) vs draft appro_TVgammaOU_2", clean_tv$path, draft_tv)
wcsv(data.frame(Q = clean_tv$Q, gamma_tilde = clean_tv$gamma_tilde, jump_sum = clean_tv$jump_sum), "tv_draws.csv")
wcsv(data.frame(y = clean_tv$path), "tv_path.csv")

# ---- 3. conditional mean ---------------------------------------------------
tt <- seq(0, 10, by = 0.01)
check("conditional_mean vs draft theorectical_gammaOU",
      conditional_mean(tt, 10, Qu, 10, 10), theorectical_gammaOU(10, 10, 10, 10, 0.01))
wcsv(data.frame(t = tt, mean = conditional_mean(tt, 10, Qu, 10, 10)), "conditional_mean.csv")

# ---- 4. estimators: clean vs draft -----------------------------------------
y <- clean$path; d <- 0.001
est <- data.frame()
for (b in c(0.1, 0.3, 0.4, 0.5, 1)) {
  check(sprintf("jump filter (b = %.1f) vs draft jump_filtering_1storder", b),
        jump_filter_estimate(y, d, b, "abs"), jump_filtering_1storder(y, d, b))
  est <- rbind(est, data.frame(kind = "jf_abs", p1 = b, p2 = NA, value = jump_filter_estimate(y, d, b, "abs")),
               data.frame(kind = "jf_pos", p1 = b, p2 = NA, value = jump_filter_estimate(y, d, b, "positive")))
}
for (r in list(c(0.5, 0.5), c(0.3, 0.7))) {
  check(sprintf("BPV 'paper' r = (%.1f, %.1f) vs draft BPV_estimator", r[1], r[2]),
        bpv_estimate(y, d, r, "paper"), BPV_estimator(y, d, r))
  check(sprintf("BPV 'product' r = (%.1f, %.1f) vs draft BPV_estimator_m", r[1], r[2]),
        bpv_estimate(y, d, r, "product"), BPV_estimator_m(y, d, r))
  est <- rbind(est, data.frame(kind = "bpv_paper", p1 = r[1], p2 = r[2], value = bpv_estimate(y, d, r, "paper")),
               data.frame(kind = "bpv_product", p1 = r[1], p2 = r[2], value = bpv_estimate(y, d, r, "product")))
}
wcsv(est, "estimates.csv")

# ---- 5. local estimates on the time-varying path ---------------------------
t0s <- seq(0.2, 9.8, by = 0.2)
yt <- clean_tv$path
wcsv(data.frame(t0 = t0s,
                jf = sapply(t0s, function(s) local_estimate(yt, d, s, 0.1, "jump_filter", exponent = 0.4)),
                bpv = sapply(t0s, function(s) local_estimate(yt, d, s, 0.1, "bpv", r = c(0.5, 0.5), denominator = "product"))),
     "local_estimates.csv")

# ---- 6. Euler scheme --------------------------------------------------------
set.seed(6); inc <- gamma_increments(2000, 10, 10, 0.005)
wcsv(data.frame(inc = inc), "euler_increments.csv")
wcsv(data.frame(y = euler_ou(inc, 1, 0.005, function(t) 1 + 0.5 * sin(10 * t))), "euler_path.csv")

cat("All R reference checks passed; fixtures written to", out, "\n")
