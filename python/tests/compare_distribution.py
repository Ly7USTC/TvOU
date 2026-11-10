"""Compare estimator distributions from R- and Python-simulated paths (two-sample KS)."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
import tvou

r = pd.read_csv(Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "distribution_r.csv")
QU = 1 + 0.5 * np.sin(50)
rng = np.random.default_rng(32)
rows = []
for _ in range(1000):
    y = tvou.simulate_gamma_ou(1, 1, 2, 10, 0.001, QU, rng=rng)
    rows.append((tvou.jump_filter_estimate(y, 0.001, 0.5), tvou.bpv_estimate(y, 0.001, (0.5, 0.5), "product"), y[-1]))
py = pd.DataFrame(rows, columns=["jf", "bpv", "y_end"])
for c in py:
    print(f"{c:6s} R mean {r[c].mean():.4f} sd {r[c].std():.4f} | Python mean {py[c].mean():.4f} "
          f"sd {py[c].std():.4f} | KS p = {ks_2samp(r[c], py[c]).pvalue:.2f}")
