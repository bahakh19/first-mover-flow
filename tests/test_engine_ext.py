import numpy as np
import pandas as pd

from fmflow.backtest.engine import backtest, breakeven_cost_bps, cost_sensitivity


def _toy(n=300, k=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    ret = pd.DataFrame(rng.normal(0.0005, 0.01, (n, k)), index=idx, columns=list("abcd"))
    w = pd.DataFrame(rng.normal(0, 0.2, (n, k)), index=idx, columns=list("abcd"))
    return w.sub(w.mean(axis=1), axis=0), ret


def test_borrow_is_charged_on_short_notional_only():
    w, r = _toy()
    a, b = backtest(w, r, 0.0, 0.0), backtest(w, r, 0.0, 100.0)
    expected = (100e-4 / 252) * (-w.shift(1)).clip(lower=0).sum(axis=1)
    assert np.allclose((a.net - b.net).fillna(0), expected.fillna(0))


def test_execution_lag_shifts_the_held_book():
    w, r = _toy()
    lagged = backtest(w, r, 0.0, 0.0, execution_lag=2)
    assert np.allclose(lagged.gross.iloc[5:], (w.shift(3) * r).sum(axis=1).iloc[5:])


def test_breakeven_cost_zeroes_mean_net_return():
    w, r = _toy()
    res = backtest(w, r, 0.0)
    be = breakeven_cost_bps(res)
    assert abs(backtest(w, r, be).net.mean()) < 1e-12


def test_cost_sensitivity_monotone_in_costs():
    w, r = _toy()
    g = cost_sensitivity(w, r, cost_grid=(0, 10, 50), borrow_grid=(0,), execution_lags=(0,))
    assert g["ann_return"].is_monotonic_decreasing
