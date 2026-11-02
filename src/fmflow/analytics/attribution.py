"""Where does the P&L come from? Identity versus timing.

In a universe chosen with hindsight (today's large caps), a book that is *on average* long the stocks that later did
well earns money without any skill in timing. The circular time-shift placebo cannot rule this out because it keeps
each stock's habitual rank. Two decompositions separate the two sources.

lo_decomposition   Lo (2008): E[sum_i w_i r_i] = sum_i Cov(w_i, r_i) + sum_i E[w_i] E[r_i]
                   (active = covariance between positions and subsequent returns; passive = average tilt x average
                   return). Uses in-window means, so it is a diagnostic, not a tradable split.
static_timing_books  the CAUSAL version: the static book holds each stock's expanding mean of PAST weights (known at
                   t); the timing book holds the deviation of today's weight from it. Both are tradable; their sum is
                   the strategy. A positive, significant timing book is skill in WHEN, not WHICH.
stock_contributions  per-stock P&L contribution with HAC standard errors (input to Bayesian partial pooling).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def lo_decomposition(weights: pd.DataFrame, returns: pd.DataFrame, window: slice | None = None,
                     periods: int = 252) -> dict:
    """Lo (2008) split of the mean annualised gross return into passive (average tilt) and active (timing) parts."""
    w = weights.shift(1)
    r = returns.reindex_like(weights)
    if window is not None:
        w, r = w.loc[window], r.loc[window]
    total = float((w * r).sum(axis=1).mean() * periods)
    passive = float((w.mean() * r.mean()).sum() * periods)
    return {"total": total, "passive": passive, "active": total - passive,
            "active_share": (total - passive) / total if total != 0 else np.nan}


def static_timing_books(weights: pd.DataFrame, returns: pd.DataFrame, min_periods: int = 21) -> pd.DataFrame:
    """Daily returns of the causal static-tilt book, the timing book and the strategy (gross)."""
    r = returns.reindex_like(weights)
    wbar = weights.expanding(min_periods=min_periods).mean()
    stat = (wbar.shift(1) * r).sum(axis=1)
    timing = ((weights - wbar).shift(1) * r).sum(axis=1)
    return pd.DataFrame({"static": stat, "timing": timing, "total": (weights.shift(1) * r).sum(axis=1)})


def _hac_se(x: np.ndarray, lags: int) -> float:
    x = x - x.mean()
    T = len(x)
    g0 = (x @ x) / T
    v = g0 + 2 * sum((1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / T for k in range(1, lags + 1))
    return float(np.sqrt(max(v, 1e-30) / T))


def mean_with_hac(x: pd.Series, lags: int = 10, periods: int = 252) -> dict:
    """Annualised mean of a daily series with a Newey-West t-statistic."""
    a = x.dropna().to_numpy(float)
    se = _hac_se(a, lags)
    return {"ann_mean": float(a.mean() * periods), "t_hac": float(a.mean() / se) if se > 0 else np.nan, "se_daily": se}


def stock_contributions(weights: pd.DataFrame, returns: pd.DataFrame, window: slice | None = None,
                        lags: int = 10, periods: int = 252) -> pd.DataFrame:
    """Per-stock annualised P&L contribution with Newey-West t-statistics, average weight and share of days held."""
    c = weights.shift(1) * returns.reindex_like(weights)
    if window is not None:
        c = c.loc[window]
    rows = {}
    for s in c.columns:
        h = mean_with_hac(c[s], lags, periods)
        rows[s] = {"ann_contribution": h["ann_mean"], "t_hac": h["t_hac"], "se_ann": h["se_daily"] * periods,
                   "avg_weight": float(weights[s].loc[c.index].mean()),
                   "held_share": float((weights[s].shift(1).loc[c.index].abs() > 1e-6).mean())}
    return pd.DataFrame(rows).T.sort_values("ann_contribution", ascending=False)
