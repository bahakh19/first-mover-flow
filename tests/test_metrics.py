import numpy as np
import pandas as pd
import pytest

from fmflow.analytics.metrics import capm, drawdown, factor_attribution, perf_stats, stress_table


def test_perf_stats_known_values():
    r = pd.Series([0.01, -0.02, 0.01, 0.01] * 63)
    s = perf_stats(r)
    eq = (1 + r).cumprod()
    assert s["CAGR"] == pytest.approx(eq.iloc[-1] ** (252 / len(r)) - 1)
    assert s["Max drawdown"] == pytest.approx(drawdown(r).min())
    assert s["Calmar"] == pytest.approx(s["CAGR"] / abs(s["Max drawdown"]))


def test_capm_recovers_beta_and_alpha():
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0.0004, 0.01, 2520))
    y = 0.6 * m + 0.0004 + pd.Series(rng.normal(0, 0.003, 2520))
    c = capm(y, m)
    assert c["beta"] == pytest.approx(0.6, abs=0.03) and c["alpha_t"] > 3
    fa = factor_attribution(y, {"mkt": m})
    assert fa["b_mkt"] == pytest.approx(0.6, abs=0.03)


def test_stress_table():
    idx = pd.bdate_range("2020-01-01", periods=10)
    t = stress_table({"x": pd.Series(0.01, index=idx)}, {"ep": ("2020-01-02", "2020-01-03")})
    assert t.loc["x", "ep"] == pytest.approx(1.01 ** 2 - 1)
