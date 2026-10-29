import numpy as np
import pandas as pd
import pytest

from fmflow.backtest.engine import annual_turnover, backtest


def test_one_session_lag_and_costs():
    idx = pd.bdate_range("2022-01-03", periods=4)
    W = pd.DataFrame({"a": [0.5, 0.5, -0.5, -0.5], "b": [-0.5, -0.5, 0.5, 0.5]}, index=idx)
    R = pd.DataFrame({"a": [0.10, 0.02, 0.03, -0.01], "b": [0.00, -0.02, 0.01, 0.01]}, index=idx)
    bt = backtest(W, R, cost_bps=10)
    assert bt.gross.iloc[0] == 0                                    # no position before the first close
    assert bt.gross.iloc[1] == pytest.approx(0.5 * 0.02 - 0.5 * -0.02)  # weights at t-1 earn the return at t
    assert bt.traded.iloc[2] == pytest.approx(2.0)                   # full reversal: |dw| = 1 + 1
    assert bt.costs.iloc[2] == pytest.approx(10e-4 * 2.0)
    assert np.allclose(bt.net, bt.gross - bt.costs)


def test_annual_turnover():
    tr = pd.Series([0.02] * 252)
    assert annual_turnover(tr) == pytest.approx(0.5 * 0.02 * 252)
