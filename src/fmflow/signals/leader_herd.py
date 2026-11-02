"""Leader-minus-herd (LMH): the arrival-time asymmetry of analyst revisions.

Leader  : a fresh revision published on an information-event day, or one that breaks >= `leader_gap` sessions with no
          other fresh revision on the stock (first mover; private research effort).
Herder  : a non-leader revision 3..15 sessions after the stock's latest event cluster, in the cluster's direction
          (copycat / career-concern revision; arrives when the move is largely done).

LMH_{i,t} = [sum of leader directions - sum of herder directions] / (# leader + # herder revisions), trailing
`window` sessions, defined when at least `min_n` classified revisions occurred. Every classification uses only
revisions with knowledge date <= t (verified by the truncation tests in tests/test_causality.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fmflow.data.snapshot import TPSnapshot


def leader_minus_herd(snap: TPSnapshot, sessions: pd.DatetimeIndex, stocks, leader_gap: int = 10, herd_lo: int = 3,
                      herd_hi: int = 15, window: int = 126, min_n: int = 3, return_parts: bool = False):
    """Net direction of leader minus herder revisions over `window` sessions (see module docstring)."""
    stocks = list(stocks)
    S = len(sessions)
    ev = snap.events.reset_index(drop=True)
    fr = ev[ev["kind"].isin(["up", "down"])].sort_values(["stock", "k_pos", "analyst"]).copy()
    empty = pd.DataFrame(np.nan, index=sessions, columns=stocks)
    if fr.empty:
        return {"leader_minus_herd": empty, "leaders_only": empty, "herders_only": empty} if return_parts else empty
    gap = fr["k_pos"] - fr.groupby("stock")["k_pos"].shift(1)
    same_day_first = fr["k_pos"] != fr.groupby("stock")["k_pos"].shift(1)
    is_event = fr["event_day"].astype(bool)
    fr["leader"] = is_event | (same_day_first & (gap.isna() | (gap >= leader_gap)))
    evd = (fr[is_event].groupby(["k_pos", "stock"])["revision"].sum().unstack("stock")
           .reindex(index=np.arange(S), columns=stocks))
    last_dir = np.sign(evd).where(evd.notna()).ffill()
    last_pos = pd.DataFrame(np.where(evd.notna().values, np.arange(S)[:, None], np.nan), columns=stocks).ffill()
    ci = fr["stock"].map({s: i for i, s in enumerate(stocks)}).values
    kp = fr["k_pos"].values.astype(int)
    since = kp - last_pos.to_numpy()[kp, ci]
    herd = (~fr["leader"].values) & (since >= herd_lo) & (since <= herd_hi) & \
           (np.sign(fr["revision"].values) == last_dir.to_numpy()[kp, ci])
    sgn = np.sign(fr["revision"]).values
    fr2 = fr.assign(_lead=np.where(fr["leader"], sgn, 0.0), _herd=np.where(herd, sgn, 0.0),
                    _n=(fr["leader"].values | herd).astype(float), _nl=fr["leader"].astype(float).values,
                    _nh=herd.astype(float))

    def roll(col: str) -> pd.DataFrame:
        o = (fr2.assign(k=sessions[fr2["k_pos"].values.astype(int)]).groupby(["k", "stock"])[col].sum()
             .unstack("stock").reindex(index=sessions, columns=stocks).fillna(0.0))
        return o.rolling(window, min_periods=1).sum()

    lead, hrd, nn, nl, nh = roll("_lead"), roll("_herd"), roll("_n"), roll("_nl"), roll("_nh")
    lmh = ((lead - hrd) / nn).where(nn >= min_n)
    if return_parts:
        return {"leader_minus_herd": lmh, "leaders_only": (lead / nl).where(nl >= 2), "herders_only": (hrd / nh).where(nh >= 2)}
    return lmh
