"""Institutional comparables: the signals and allocations most widely used by equity and QIS desks, built only from
prices and analyst targets so they can be compared with FMF-N on any dataset that satisfies the data contract.

`comparable_books(out)` runs every cross-sectional comparable through the SAME construction as the strategy in `out`
(legs, allocator, neutralisation, cap, tranches, volatility target, costs, borrow, execution delay), dollar-neutral,
which isolates the signal, and adds allocation benchmarks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fmflow.backtest.engine import backtest
from fmflow.pipeline import StrategyOutput, construct_book
from fmflow.signals.analyst import analyst_signals
from fmflow.signals.breadth import breadth
from fmflow.signals.composite import composite_score
from fmflow.signals.price import price_signals

PRICE_COMPARABLES = {
    "12-1 momentum (Jegadeesh-Titman)": "mom_12_1", "6-1 momentum": "mom_6_1", "Short-term reversal (1m)": "rev_1m",
    "Low volatility (60d)": "low_vol_60", "Betting-against-beta (low beta)": "low_beta_252",
    "Residual momentum (Blitz et al.)": "resid_mom_12_1", "52-week-high proximity": "high_52w"}
ANALYST_COMPARABLES = {"Target-price upside (consensus)": "tp_upside", "Target-price momentum (63d)": "cons_change_63",
                       "Low analyst dispersion": "low_dispersion"}


def comparable_signals(px: pd.DataFrame, snap) -> dict[str, pd.DataFrame]:
    """The 13 institutional cross-sectional signals, oriented higher = more attractive."""
    ps, an = price_signals(px), analyst_signals(px, snap)
    b = breadth(snap).reindex(index=px.index, columns=px.columns)
    sig = {**{k: ps[v] for k, v in PRICE_COMPARABLES.items()}, **{k: an[v] for k, v in ANALYST_COMPARABLES.items()},
           "Revision breadth (classic)": b}
    sig["Multi-factor: mom + low vol + reversal"] = composite_score(ps, ["mom_12_1", "low_vol_60", "rev_1m"], index=px.index)
    sig["Momentum + revisions blend"] = composite_score({"m": ps["mom_12_1"], "b": b}, ["m", "b"], index=px.index)
    return sig


def comparable_books(out: StrategyOutput, include_allocations: bool = True) -> dict[str, pd.Series]:
    """Net daily returns of every comparable, same construction as `out.config`."""
    cfg, px, ret = out.config, out.prices, out.returns
    books = {}
    for name, sc in comparable_signals(px, out.snapshot).items():
        _, w = construct_book(sc, ret, cfg, out.sig_vol, net_exposure=0.0)
        books[name] = backtest(w, ret, cfg.cost_bps, cfg.borrow_bps, cfg.execution_lag).net
    if include_allocations:
        mkt = ret.mean(axis=1)
        iv = 1 / ret.rolling(60, min_periods=20).std()
        iv = iv.div(iv.sum(axis=1), axis=0)
        mom = price_signals(px)["mom_12_1"]
        # 1/N over the names with a signal at t: dividing by every column of the file would depend on how many stocks
        # list later
        ts = np.sign(mom).div(mom.notna().sum(axis=1).where(lambda n: n > 0), axis=0).fillna(0.0)
        ts = ts.where(np.repeat((np.arange(len(px)) % 21 == 0)[:, None], px.shape[1], axis=1)).ffill().fillna(0.0)
        books["Equal-weight long-only (market)"] = mkt
        books["Inverse-vol long-only (naive risk parity)"] = (iv.shift(1) * ret).sum(axis=1)
        books["Time-series momentum 12-1 (CTA-style)"] = backtest(ts, ret, cfg.cost_bps).net
    return books
