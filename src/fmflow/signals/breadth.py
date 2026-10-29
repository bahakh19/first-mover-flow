"""Revision-breadth family.

breadth(W)          = sum(up - down) / sum(up + down) over the last W sessions (fresh revisions only, by knowledge
                      date); defined when at least `min_n` revisions arrived. Sign, not size: robust to fat tails.
event_breadth(W)    = the same ratio using only revisions published on information-event days (>= 5 same-day ratings).
breadth_resid_mom   = residual of rank(breadth_63) on rank(6-1 momentum), date by date: breadth net of price chasing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fmflow.data.snapshot import TPSnapshot
from fmflow.signals.composite import cs_rank


def breadth(snap: TPSnapshot, window: int = 63, min_n: int = 3) -> pd.DataFrame:
    """Revision breadth: (up - down) / (up + down) fresh revisions over `window` sessions."""
    up = snap.n_up.rolling(window, min_periods=1).sum()
    dn = snap.n_down.rolling(window, min_periods=1).sum()
    return ((up - dn) / (up + dn)).where(up + dn >= min_n)


def event_breadth(snap: TPSnapshot, sessions: pd.DatetimeIndex, stocks, window: int = 63, min_n: int = 2) -> pd.DataFrame:
    """Breadth computed only from revisions published on information-event days."""
    ev = snap.events

    def arrivals(mask: pd.Series) -> pd.DataFrame:
        out = ev.assign(_v=mask.astype(float)).groupby(["knowledge_date", "stock"])["_v"].sum().unstack("stock")
        return out.reindex(index=sessions, columns=list(stocks)).fillna(0.0)

    up = arrivals((ev["kind"] == "up") & ev["event_day"]).rolling(window, min_periods=1).sum()
    dn = arrivals((ev["kind"] == "down") & ev["event_day"]).rolling(window, min_periods=1).sum()
    return ((up - dn) / (up + dn)).where(up + dn >= min_n)


def momentum_6_1(px: pd.DataFrame) -> pd.DataFrame:
    """Log price change from t-126 to t-21."""
    logp = np.log(px)
    return logp.shift(21) - logp.shift(126)


def breadth_resid_mom(breadth_panel: pd.DataFrame, px: pd.DataFrame, min_names: int = 5) -> pd.DataFrame:
    """Breadth rank orthogonalised to 6-1 momentum rank cross-sectionally each day."""
    y, x = cs_rank(breadth_panel, min_names), cs_rank(momentum_6_1(px), min_names)
    ok = y.notna() & x.notna()
    xc = x.where(ok).sub(x.where(ok).mean(axis=1), axis=0)
    yc = y.where(ok).sub(y.where(ok).mean(axis=1), axis=0)
    beta = (xc * yc).sum(axis=1) / (xc ** 2).sum(axis=1)
    # Known limitation, kept for exact parity with the published results: residuals of ranks can be mathematically
    # tied, and the floating-point row sums give such ties last-bit (1e-17) differences that depend on where the
    # missing values sit, including columns of stocks that only list later. The next ranking may then order two exactly
    # tied stocks differently. Rounding the residuals would remove this but changes the development weights by up to
    # 0.003, because the reference results contain such ties (see CHANGELOG 2.2.3).
    return yc.sub(xc.mul(beta, axis=0)).where(ok.sum(axis=1) >= min_names)
