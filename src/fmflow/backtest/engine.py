"""Vectorised daily backtest with linear transaction costs, short-borrow fees and an optional execution delay.

Timing. Weights decided at the close of t are executed `execution_lag` sessions later (0 = traded at the close of t,
the research convention; 1 = traded at the close of t+1, i.e. one full session of implementation shortfall) and earn
the following session's return:

    held_t        = w_{t - execution_lag}
    gross_{t+1}   = sum_i held_{i,t} * R_{i,t+1}
    cost_t        = (cost_bps / 1e4) * sum_i |held_{i,t} - held_{i,t-1}|      (held_{i,-1} = 0: opening the book costs)
    borrow_{t+1}  = (borrow_bps / 1e4 / 252) * sum_i max(-held_{i,t}, 0)      (annual fee on short notional)
    net           = gross - cost - borrow

Missing weights are flat positions. A held name whose return is missing (no price beyond the forward-fill limit)
contributes zero to that session's gross return.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Held weights, gross / cost / borrow / net daily returns and traded notional of a backtest."""
    weights: pd.DataFrame      # held (post-lag) weights
    gross: pd.Series
    costs: pd.Series           # transaction costs
    net: pd.Series
    traded: pd.Series          # sum_i |dw_i| per session
    borrow: pd.Series | None = None

    def window(self, start=None, end=None) -> BacktestResult:
        """The same result restricted to the sessions from `start` to `end` (inclusive, either may be None)."""
        sl = slice(start, end)
        b = None if self.borrow is None else self.borrow.loc[sl]
        return BacktestResult(self.weights.loc[sl], self.gross.loc[sl], self.costs.loc[sl], self.net.loc[sl],
                              self.traded.loc[sl], b)


def backtest(weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 5.0, borrow_bps: float = 0.0,
             execution_lag: int = 0) -> BacktestResult:
    """Run the daily backtest: held = weights lagged by `execution_lag`; net = gross - costs - borrow."""
    if execution_lag < 0:
        raise ValueError("execution_lag must be >= 0")
    held = weights.shift(execution_lag) if execution_lag else weights
    held = held.fillna(0.0)
    ret = returns.reindex(index=held.index, columns=held.columns)
    gross = (held.shift(1) * ret).sum(axis=1)
    traded = held.diff().fillna(held).abs().sum(axis=1)
    costs = cost_bps * 1e-4 * traded
    borrow = (borrow_bps * 1e-4 / 252) * (-held.shift(1)).clip(lower=0).sum(axis=1)
    return BacktestResult(weights=held, gross=gross, costs=costs, net=gross - costs - borrow, traded=traded, borrow=borrow)


def annual_turnover(traded: pd.Series, periods: int = 252) -> float:
    """One-way turnover per year (half the traded notional)."""
    return float(0.5 * traded.sum() / (len(traded) / periods)) if len(traded) else float("nan")


def breakeven_cost_bps(result: BacktestResult, start=None, end=None) -> float:
    """Cost per unit traded (bps) at which the mean net return would be zero:  mean(gross - borrow) / mean(traded)."""
    r = result.window(start, end)
    b = 0.0 if r.borrow is None else r.borrow
    tr = r.traded.mean()
    return float((r.gross - b).mean() / tr * 1e4) if tr > 0 else float("nan")


def cost_sensitivity(weights: pd.DataFrame, returns: pd.DataFrame, cost_grid=(0, 2.5, 5, 10, 20, 30, 50),
                     borrow_grid=(0, 25, 50, 100), execution_lags=(0, 1), window: slice | None = None) -> pd.DataFrame:
    """Annualised net Sharpe over a grid of trading costs, borrow fees and execution delays."""
    rows = []
    for lag in execution_lags:
        for bb in borrow_grid:
            for c in cost_grid:
                n = backtest(weights, returns, c, bb, lag).net
                n = n.loc[window] if window is not None else n
                s = n.std()
                rows.append({"execution_lag": lag, "borrow_bps": bb, "cost_bps": c,
                             "Sharpe": float(n.mean() / s * np.sqrt(252)) if s > 0 else np.nan, "ann_return": float(n.mean() * 252)})
    return pd.DataFrame(rows)
