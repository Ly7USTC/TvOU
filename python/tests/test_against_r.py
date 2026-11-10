"""Exact cross-checks against the R reference (fixtures from tests/r/make_fixtures.R,
which first checks the clean R code against the original scripts)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tvou

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
QU = 1 + 0.5 * np.sin(50)
QFUN = lambda t: 1 + 0.5 * np.sin(0.8 * t)


def load(name):
    return pd.read_csv(FIX / name)


@pytest.mark.parametrize("draws, path", [("ou_draws.csv", "ou_path.csv"), ("tv_draws.csv", "tv_path.csv")])
def test_path_from_r_draws(draws, path):
    d = load(draws)
    y = tvou.gamma_ou_path_from_draws(2.0, d.Q, 0.001, d.gamma_tilde, d.jump_sum)
    np.testing.assert_allclose(y, load(path).y, rtol=1e-12, atol=1e-14)


def test_q_on_steps_matches_r():
    np.testing.assert_allclose(tvou.q_on_steps(QFUN, 10000, 0.001), load("tv_draws.csv").Q, rtol=1e-14)
    np.testing.assert_allclose(tvou.q_on_steps(QU, 10000, 0.001), load("ou_draws.csv").Q, rtol=1e-14)


def test_conditional_mean_matches_r():
    df = load("conditional_mean.csv")
    np.testing.assert_allclose(tvou.conditional_mean(df.t, 10, QU, 10, 10), df["mean"], rtol=1e-13)


def test_global_estimators_match_r():
    y = load("ou_path.csv").y.to_numpy()
    for row in load("estimates.csv").itertuples():
        if row.kind == "jf_abs":
            val = tvou.jump_filter_estimate(y, 0.001, row.p1, "abs")
        elif row.kind == "jf_pos":
            val = tvou.jump_filter_estimate(y, 0.001, row.p1, "positive")
        else:
            val = tvou.bpv_estimate(y, 0.001, (row.p1, row.p2), row.kind.split("_")[1])
        assert val == pytest.approx(row.value, rel=1e-12), row


def test_local_estimators_match_r():
    y = load("tv_path.csv").y.to_numpy()
    df = load("local_estimates.csv")
    jf = tvou.local_curve(y, 0.001, df.t0, 0.1, "jump_filter", exponent=0.4)
    bpv = tvou.local_curve(y, 0.001, df.t0, 0.1, "bpv", r=(0.5, 0.5), denominator="product")
    # atol covers the 15 significant digits written by R's write.csv
    np.testing.assert_allclose(jf, df.jf, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bpv, df.bpv, rtol=1e-12, atol=1e-12)


def test_euler_matches_r():
    inc = load("euler_increments.csv").inc.to_numpy()
    y = tvou.euler_ou(inc, 1.0, 0.005, lambda t: 1 + 0.5 * np.sin(10 * t))
    np.testing.assert_allclose(y, load("euler_path.csv").y, rtol=1e-12)
