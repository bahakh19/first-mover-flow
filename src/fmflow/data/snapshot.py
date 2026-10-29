"""Point-in-time snapshot of analyst targets on the trading calendar.

At session t the snapshot contains, for every (stock, analyst) pair, the latest target with knowledge date <= t whose
age is < `tp_expiry_days`, plus per-stock arrival counts of fresh up / down revisions, reiterations and initiations.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fmflow.config import DataConfig


@dataclass
class TPSnapshot:
    """Point-in-time analyst state on the trading calendar (targets, ages, consensus, revision arrivals)."""
    tp: pd.DataFrame            # sessions x (stock, analyst): active target
    age: pd.DataFrame           # sessions x (stock, analyst): age (sessions) of the active target
    n_active: pd.DataFrame      # sessions x stock: number of active targets
    cons_median: pd.DataFrame
    cons_mean: pd.DataFrame
    dispersion: pd.DataFrame    # cross-analyst std of log targets
    n_up: pd.DataFrame          # arrivals (by knowledge date) of fresh up revisions
    n_down: pd.DataFrame
    n_reit: pd.DataFrame
    n_init: pd.DataFrame
    rev_sum: pd.DataFrame
    events: pd.DataFrame        # rating-level table with k_pos and kind in {initiation, reiteration, up, down}


def classify_ratings(rt: pd.DataFrame, sessions: pd.DatetimeIndex, cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Add session positions and the fresh-revision / reiteration / (re-)initiation classification."""
    ev = rt.sort_values(["stock", "analyst", "knowledge_date", "pub_date"]).copy()
    ev["k_pos"] = sessions.get_indexer(ev["knowledge_date"])
    if not (ev["k_pos"] >= 0).all():
        raise RuntimeError("internal error: a knowledge date lies outside the trading calendar")
    prev_pos = ev.groupby(["stock", "analyst"])["k_pos"].shift(1)
    prev_active = (ev["k_pos"] - prev_pos) < cfg.tp_expiry_days
    ev["kind"] = np.select(
        [ev["prev_tp"].isna() | ~prev_active, ev["is_reiteration"], ev["revision"] > 0, ev["revision"] < 0],
        ["initiation", "reiteration", "up", "down"], default="reiteration")
    ev["fresh_revision"] = np.where(ev["kind"].isin(["up", "down"]), ev["revision"], np.nan)
    return ev


def build_tp_snapshot(rt: pd.DataFrame, sessions: pd.DatetimeIndex, stocks: Sequence[str],
                      cfg: DataConfig = DataConfig()) -> TPSnapshot:
    """Build the point-in-time snapshot from knowledge-date-stamped ratings (targets expire after tp_expiry_days)."""
    ev = classify_ratings(rt, sessions, cfg)
    last = ev.drop_duplicates(["knowledge_date", "stock", "analyst"], keep="last")
    wide = last.pivot(index="knowledge_date", columns=["stock", "analyst"], values="tp").reindex(sessions).sort_index(axis=1)
    pos = pd.Series(np.arange(len(sessions), dtype=float), index=sessions)
    k_pos = wide.notna().mul(pos, axis=0).where(wide.notna())
    limit = cfg.tp_expiry_days - 1
    if limit > 0:                                 # pandas rejects limit=0: a 1-session expiry means no carry-forward
        wide, k_pos = wide.ffill(limit=limit), k_pos.ffill(limit=limit)
    tp = wide
    age = k_pos.rsub(pos, axis=0)

    def per_stock(panel: pd.DataFrame, how: str) -> pd.DataFrame:
        return getattr(panel.T.groupby(level="stock"), how)().T.reindex(columns=list(stocks))

    n_active = per_stock(tp.notna().astype(float), "sum").fillna(0.0)
    enough = n_active >= cfg.min_analysts

    def arrivals(mask: pd.Series, values: pd.Series | None = None) -> pd.DataFrame:
        v = mask.astype(float) if values is None else values.where(mask, 0.0)
        out = (ev.assign(_v=v).groupby(["knowledge_date", "stock"])["_v"].sum()
               .unstack("stock").reindex(index=sessions, columns=list(stocks)))
        return out.fillna(0.0)

    return TPSnapshot(
        tp=tp, age=age, n_active=n_active,
        cons_median=per_stock(tp, "median").where(enough), cons_mean=per_stock(tp, "mean").where(enough),
        dispersion=per_stock(np.log(tp), "std").where(enough),
        n_up=arrivals(ev["kind"] == "up"), n_down=arrivals(ev["kind"] == "down"),
        n_reit=arrivals(ev["kind"] == "reiteration"), n_init=arrivals(ev["kind"] == "initiation"),
        rev_sum=arrivals(ev["kind"].isin(["up", "down"]), ev["fresh_revision"].fillna(0.0)),
        events=ev)
