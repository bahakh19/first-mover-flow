import numpy as np
import pandas as pd
import pytest

from fmflow.config import DataConfig
from fmflow.risk.garch import garch_forecast_panel, gjr_variance_path


def test_variance_recursion():
    x = np.array([1.0, -2.0, 0.5])
    s2 = gjr_variance_path(x, 0.0, 0.1, 0.05, 0.1, 0.8, 1.0)
    assert s2[1] == pytest.approx(0.1 + 0.05 * 1 + 0.8 * 1.0)
    assert s2[2] == pytest.approx(0.1 + (0.05 + 0.1) * 4 + 0.8 * s2[1])   # leverage term only for negative shocks


@pytest.mark.slow
def test_garch_is_causal_and_positive():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.standard_t(5, (420, 2)) * 0.01, index=pd.bdate_range("2020-01-01", periods=420), columns=["a", "b"])
    cfg = DataConfig(garch_min_obs=250, garch_refit=21)
    full, _ = garch_forecast_panel(r, cfg)
    part, _ = garch_forecast_panel(r.iloc[:330], cfg)
    assert (full.dropna() > 0).all().all()
    pd.testing.assert_frame_equal(part, full.iloc[:330], check_freq=False, atol=1e-12, rtol=0)


def _simulate_gjr(n, mu=0.05, omega=0.05, alpha=0.04, gamma=0.08, beta=0.88, nu=6.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))
    x, s2 = np.empty(n), omega / (1 - alpha - gamma / 2 - beta)
    for t in range(n):
        x[t] = mu + np.sqrt(s2) * z[t]
        e = x[t] - mu
        s2 = omega + (alpha + gamma * (e < 0)) * e * e + beta * s2
    return x


def test_grid_fit_recovers_a_persistent_gjr_process():
    from fmflow.risk.garch import grid_fit
    mu, omega, a, g, b, nu = grid_fit(_simulate_gjr(3000))
    assert 0.80 <= b <= 0.94 and g > a and 4 <= nu <= 10
    assert a + g / 2 + b < 1 and omega > 0


def test_grid_fit_is_deterministic():
    from fmflow.risk.garch import grid_fit
    x = _simulate_gjr(600, seed=3)
    assert grid_fit(x) == grid_fit(x.copy())


def test_slsqp_fit_never_prints_convergence_warnings():
    """arch re-enables its ConvergenceWarning inside fit(); the wrapper must keep diagnostics in the log only."""
    import warnings
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.standard_t(3, (300, 1)) * 0.02, index=pd.bdate_range("2020-01-01", periods=300), columns=["a"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, log = garch_forecast_panel(r, DataConfig(garch_min_obs=250, garch_refit=21))
    assert not [x for x in w if "Convergence" in type(x.message).__name__ or "optimizer" in str(x.message)]
    assert {"optimizer_status", "new_estimate"} <= set(log.columns)


def test_unknown_garch_method_is_rejected():
    r = pd.DataFrame({"a": np.zeros(10)}, index=pd.bdate_range("2020-01-01", periods=10))
    with pytest.raises(ValueError):
        garch_forecast_panel(r, DataConfig(garch_method="bfgs"))
