"""Tabular reports used by the CLI, the scripts and the notebooks."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from fmflow.analytics.inference import psr
from fmflow.analytics.metrics import capm, perf_stats
from fmflow.backtest.engine import annual_turnover


def summary_table(books: Mapping[str, pd.Series], windows: Mapping[str, slice], market: pd.Series,
                  reference: pd.Series | None = None) -> pd.DataFrame:
    """Performance, CAPM and PSR statistics of several books over several windows."""
    rows = {}
    for name, r in books.items():
        for wn, w in windows.items():
            rr = r.loc[w].dropna()
            if len(rr) < 30:
                continue
            s = perf_stats(rr)
            c = capm(rr, market.loc[w])
            row = {"CAGR": s["CAGR"], "Vol": s["Ann. vol"], "Sharpe": s["Sharpe"], "MaxDD": s["Max drawdown"],
                   "Calmar": s["Calmar"], "beta": c["beta"], "alpha": c["alpha_ann"], "alpha t": c["alpha_t"], "PSR": psr(rr.values)}
            if reference is not None:
                a, b = rr.align(reference.loc[w], join="inner")
                row["corr ref"] = float(np.corrcoef(a.fillna(0), b.fillna(0))[0, 1])
            rows[(name, wn)] = row
    return pd.DataFrame(rows).T


def strategy_report(out, windows: Mapping[str, slice]) -> pd.DataFrame:
    """Headline statistics of a StrategyOutput over named windows."""
    mkt = out.returns.mean(axis=1)
    tab = summary_table({out.config.name: out.net_returns}, windows, mkt)
    tab["turnover"] = [annual_turnover(out.result.traded.loc[windows[w]]) for _, w in tab.index]
    return tab


def default_windows(index: pd.DatetimeIndex, warmup: int = 252) -> dict[str, slice]:
    """Whole sample after a warm-up, and its two halves (the warm-up shrinks to a quarter of short samples)."""
    if len(index) < 2:
        raise ValueError("default_windows needs at least two sessions")
    if len(index) <= warmup + 60:
        warmup = max(0, len(index) // 4)
    ev = index[warmup:]
    mid = ev[len(ev) // 2]
    return {"evaluation": slice(ev[0], ev[-1]), "first half": slice(ev[0], mid), "second half": slice(mid, ev[-1])}
