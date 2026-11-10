"""Statistical and property tests that do not depend on R."""

import numpy as np
import pytest
from scipy import stats

import tvou

QU = 1 + 0.5 * np.sin(50)          # Q(5) for Q(t) = 1 + 0.5 sin(10 t)
QFUN = lambda t: 1 + 0.5 * np.sin(0.8 * t)


# ---------------------------------------------------------------- simulation
def test_one_step_distribution_matches_conditional_mean_and_variance():
    # many independent single steps of length t from the same Y_0
    rng = np.random.default_rng(1)
    Q, a, b, t, y0 = 0.9, 2.0, 3.0, 0.5, 1.0
    ys = np.array([tvou.simulate_gamma_ou(a, b, y0, t, t, Q, rng=rng)[-1] for _ in range(40000)])
    mean = tvou.conditional_mean(t, y0, Q, a, b)
    var = a / (2 * Q * b**2) * (1 - np.exp(-2 * Q * t))
    assert ys.mean() == pytest.approx(mean, rel=0.01)
    assert ys.var() == pytest.approx(var, rel=0.04)


def test_mean_of_paths_follows_conditional_mean():
    # Example 4.1 setting: alpha = beta = 10, Y_0 = 10, delta = 0.01
    rng = np.random.default_rng(2)
    paths = np.array([tvou.simulate_gamma_ou(10, 10, 10, 10, 0.01, QU, rng=rng) for _ in range(300)])
    t = np.arange(paths.shape[1]) * 0.01
    np.testing.assert_allclose(paths.mean(0), tvou.conditional_mean(t, 10, QU, 10, 10), atol=0.06)


def test_stationary_moments():
    rng = np.random.default_rng(3)
    m, v = tvou.stationary_moments(QU, 1, 1)
    y = tvou.simulate_gamma_ou(1, 1, m, 3000, 0.01, QU, rng=rng)
    assert y.mean() == pytest.approx(m, rel=0.05)
    assert y.var() == pytest.approx(v, rel=0.1)


def test_paths_stay_positive():
    y = tvou.simulate_gamma_ou(1, 1, 2, 10, 0.001, QFUN, rng=4)
    assert np.all(y > 0) and len(y) == 10001


def test_zero_jump_steps_add_nothing():
    # the first draft added two spurious jumps whenever N~ = 0
    _, d = tvou.simulate_gamma_ou(1, 1, 2, 1, 0.001, QU, rng=5, return_draws=True)
    assert np.all(d["jump_sum"] >= 0)
    assert np.mean(d["jump_sum"] == 0) > 0.99      # P(N~ > 0) ~ alpha Q delta^2 / 2


def test_q_on_steps_validation():
    with pytest.raises(ValueError):
        tvou.q_on_steps(np.ones(5), 10, 0.1)


def test_euler_with_gamma_increments_is_close_to_exact_mean():
    rng = np.random.default_rng(6)
    ends = [tvou.euler_ou(tvou.gamma_increments(1000, 2, 2, 0.001, rng), 3.0, 0.001, 1.0)[-1]
            for _ in range(3000)]
    assert np.mean(ends) == pytest.approx(tvou.conditional_mean(1.0, 3.0, 1.0, 2, 2), rel=0.02)


# ---------------------------------------------------------------- estimators
def test_jump_filter_without_jumps_recovers_q():
    # with no jumps, dY = (e^{-Q d} - 1) Y, so the estimator is (1 - e^{-Q d}) / d ~ Q
    d, Q = 0.001, 0.8
    y = 5 * np.exp(-Q * d * np.arange(2001))
    assert tvou.jump_filter_estimate(y, d, 0.4) == pytest.approx((1 - np.exp(-Q * d)) / d, rel=1e-10)
    r = tvou.bpv_estimate(y, d)
    assert r == pytest.approx(Q, rel=2e-3)


def test_filter_removes_large_jumps():
    d = 0.001
    y = 5 * np.exp(-0.8 * d * np.arange(2001))
    y[1000:] += 0.5                       # one jump of size 0.5 >> delta^0.4
    clean = 5 * np.exp(-0.8 * d * np.arange(2001))
    est = tvou.jump_filter_estimate(y, d, 0.4)
    # the jump step (index 999) is removed together with its drift; all others are kept
    kept = np.delete(np.diff(clean), 999)
    assert est == pytest.approx(-kept.sum() / (d * y[:-1].sum()), rel=1e-12)


@pytest.fixture(scope="module")
def ou_paths():
    rng = np.random.default_rng(7)
    return [tvou.simulate_gamma_ou(1, 1, 2, 10, 0.001, QU, rng=rng) for _ in range(60)]


def test_jump_filter_consistency(ou_paths):
    est = np.array([tvou.jump_filter_estimate(y, 0.001, 0.5) for y in ou_paths])
    assert np.median(est) == pytest.approx(QU, abs=0.05)


def test_positive_filter_is_more_accurate_with_small_threshold(ou_paths):
    # Example 4.5: with exponent 1 the |dY| filter also removes drift increments
    abs_f = np.array([tvou.jump_filter_estimate(y, 0.001, 1.0, "abs") for y in ou_paths])
    pos_f = np.array([tvou.jump_filter_estimate(y, 0.001, 1.0, "positive") for y in ou_paths])
    assert np.mean((pos_f - QU) ** 2) < np.mean((abs_f - QU) ** 2)
    assert np.median(pos_f) == pytest.approx(QU, abs=0.03)


def test_bpv_consistency(ou_paths):
    for den in ("paper", "product"):
        est = np.array([tvou.bpv_estimate(y, 0.001, (0.5, 0.5), den) for y in ou_paths])
        assert np.median(est) == pytest.approx(QU, abs=0.1)


def test_bpv_warns_if_powers_do_not_sum_to_one(ou_paths):
    with pytest.warns(UserWarning):
        tvou.bpv_estimate(ou_paths[0], 0.001, (0.5, 0.7))


def test_finer_sampling_reduces_error():
    rng = np.random.default_rng(8)
    errs = {}
    for d in (0.01, 0.0005):
        est = [tvou.jump_filter_estimate(tvou.simulate_gamma_ou(1, 1, 2, 10, d, QU, rng=rng), d, 0.5)
               for _ in range(40)]
        errs[d] = np.median(np.abs(np.array(est) - QU))
    assert errs[0.0005] < errs[0.01]


def test_local_estimates_track_q_of_t():
    rng = np.random.default_rng(9)
    t0s = np.arange(0.2, 9.81, 0.4)
    curves = np.array([tvou.local_curve(tvou.simulate_gamma_ou(1, 1, 2, 10, 0.001, QFUN, rng=rng),
                                        0.001, t0s, 0.1, "jump_filter", exponent=0.5)
                       for _ in range(40)])
    med = np.median(curves, axis=0)
    assert stats.pearsonr(med, QFUN(t0s))[0] > 0.95
    assert np.max(np.abs(med - QFUN(t0s))) < 0.25


def test_window_slice_bounds():
    y = np.arange(10001.0)
    w = tvou.window_slice(y, 0.001, 5.0, 0.1)
    assert w[0] == 4900 and w[-1] == 5100 and len(w) == 201
    with pytest.raises(ValueError):
        tvou.window_slice(y, 0.001, 0.05, 0.1)
