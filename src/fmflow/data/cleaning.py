"""Causal cleaning of the raw price and rating tables.

Rules (all fixed ex ante, documented in DATA.md):
  prices  : parse, drop non-positive / duplicate keys, keep dates on which >= `min_calendar_coverage` of the active
            stocks print (exchange calendar inferred from the data, causally), drop all-stale rows, forward-fill
            isolated gaps (bounded, never backward).
  ratings : parse, drop unparseable / non-positive targets, missing analyst ids, unknown stocks, duplicates;
            stamp the KNOWLEDGE DATE = first session strictly after publication (+ lag - 1); attach the as-of
            reference price; drop implausible targets (target/price outside bounds) and revisions inconsistent with
            the price change since the analyst's previous target; derive revision, initiation, reiteration and
            same-day-cluster flags.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from fmflow.config import DataConfig


def to_calendar_dates(values) -> pd.Series:
    """Parse dates and reduce them to timezone-naive calendar days.

    Times of day are dropped on purpose: a rating published at 08:30 on day D must become usable on the session AFTER
    D, never at D's close, whatever timestamps the vendor attaches to prices and ratings. Timezone-aware values keep
    their local calendar day. Unparseable values become NaT (and are logged by the callers)."""
    s = pd.Series(values)
    if s.empty:
        return pd.Series(pd.DatetimeIndex([]), index=s.index)
    d = _parse_dates(s)
    if d.isna().sum() > s.isna().sum():          # the inferred format did not fit every row: retry row by row
        d = _parse_dates(s, mixed=True)
    return d.dt.normalize()


def _parse_dates(s: pd.Series, mixed: bool = False) -> pd.Series:
    """`pd.to_datetime` that never raises and always returns timezone-naive local calendar times."""
    with warnings.catch_warnings():
        # pandas 2.x warns (FutureWarning) on mixed UTC offsets and (UserWarning) when it cannot infer a format;
        # both cases are handled below, so the warnings would only be noise for the user
        warnings.simplefilter("ignore")
        try:
            d = pd.to_datetime(s, errors="coerce", format="mixed" if mixed else None)
        except (ValueError, TypeError):          # pandas 3: different UTC offsets in one column raise
            d = None
        if d is None or not pd.api.types.is_datetime64_any_dtype(d):
            return pd.to_datetime(s.map(_naive_timestamp), errors="coerce")
    if d.dt.tz is not None:
        d = d.dt.tz_localize(None)
    return d


def _naive_timestamp(value) -> pd.Timestamp:
    t = pd.to_datetime(value, errors="coerce")
    if t is None or pd.isna(t):
        return pd.NaT
    return t.tz_localize(None) if t.tzinfo is not None else t


def to_float(values) -> pd.Series:
    """Numbers as plain float64, NaN for anything unparseable, whatever the input dtype (text, pandas' nullable
    Float64 / Int64 / string, pyarrow-backed columns): nullable dtypes would otherwise carry pd.NA into the pipeline."""
    s = pd.Series(values)
    x = pd.to_numeric(s, errors="coerce")
    return pd.Series(x.to_numpy(dtype="float64", na_value=np.nan), index=s.index)


def natural_sort_key(name: str):
    """Sort 'stock_2' before 'stock_10'."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(name))]


def to_long(panel: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Wide (session x stock) panel to a long (date, stock, value) table."""
    out = panel.rename_axis(index="date", columns="stock").reset_index()
    return out.melt(id_vars="date", var_name="stock", value_name=value_name)


class AuditLog:
    """Append-only log of cleaning actions (exported to the research report)."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(self, dataset: str, check: str, finding: str, n_rows: int, decision: str) -> None:
        """Record one cleaning rule: what it checked, what it found, how many rows, and what was done."""
        self._rows.append(dict(dataset=dataset, check=check, finding=finding, n_rows=int(n_rows), decision=decision))

    def to_frame(self) -> pd.DataFrame:
        """The log as a table (dataset, check, finding, n_rows, decision)."""
        return pd.DataFrame(self._rows, columns=["dataset", "check", "finding", "n_rows", "decision"])


def clean_prices(prices_raw: pd.DataFrame, cfg: DataConfig = DataConfig(), log: AuditLog | None = None) -> pd.DataFrame:
    """Long price table -> clean (session x stock) panel on the inferred trading calendar."""
    log = log if log is not None else AuditLog()
    df = prices_raw.copy()
    df.columns = df.columns.str.strip().str.lower()
    missing = {"date", "stock", "price"} - set(df.columns)
    if missing:
        raise ValueError(f"prices: missing columns {missing}")
    df["date"] = to_calendar_dates(df["date"]).to_numpy()
    df["price"] = to_float(df["price"])
    df["stock"] = df["stock"].astype("string").str.strip()

    bad_date = df["date"].isna()
    log.add("prices", "clean", "Unparseable dates dropped", bad_date.sum(), "drop")
    df = df.loc[~bad_date]
    no_stock = df["stock"].isna() | (df["stock"] == "")
    log.add("prices", "clean", "Missing stock identifier", no_stock.sum(), "drop")
    df = df.loc[~no_stock].astype({"stock": object})
    if df["price"].gt(0).sum() == 0:
        raise ValueError("prices: no row has a parseable date, a stock identifier and a positive price")
    non_pos = df["price"] <= 0
    log.add("prices", "clean", "Non-positive prices set to NaN", non_pos.sum(), "set NaN")
    df.loc[non_pos, "price"] = np.nan
    dup = df.duplicated(["date", "stock"], keep="last")
    log.add("prices", "clean", "Duplicate (date, stock) keys", dup.sum(), "keep last")
    df = df.loc[~dup]

    px = df.pivot(index="date", columns="stock", values="price").sort_index()
    px = px[sorted(px.columns, key=natural_sort_key)]
    # A date is a session if enough of the stocks ACTIVE at that date print on it. "Active" = printed in the trailing
    # `calendar_active_days` calendar days, which uses data up to the date only: dividing by every stock of the file
    # would let stocks that list later remove past sessions, i.e. make the calendar look ahead. Inside the file's
    # first window there is no history yet, so the stocks that print anywhere in that window count as active
    # (otherwise a stray first row, e.g. a holiday on which two stocks print, would always pass).
    printed = px.notna()
    window = pd.Timedelta(days=cfg.calendar_active_days)
    active = printed.astype(float).rolling(window).max().sum(axis=1)
    first = px.index < px.index[0] + window
    active[first] = int(printed.loc[first].any().sum())
    session = printed.sum(axis=1) >= cfg.min_calendar_coverage * active
    log.add("prices", "clean", f"Non-sessions removed (coverage < {cfg.min_calendar_coverage:.0%} of active stocks)",
            (~session).sum(), "drop date")
    px = px.loc[session]
    if px.empty:
        raise ValueError(f"prices: no date is covered by at least {cfg.min_calendar_coverage:.0%} of the stocks "
                         "(lower DataConfig.min_calendar_coverage if the file legitimately has sparse coverage)")
    unchanged = (px.pct_change(fill_method=None).abs() < 1e-12) | px.isna()
    stale_row = unchanged.all(axis=1) & px.notna().any(axis=1)
    stale_row.iloc[0] = False
    log.add("prices", "clean", "Stale rows (all prices unchanged)", stale_row.sum(), "drop date")
    px = px.loc[~stale_row]
    n_before = px.isna().sum().sum()
    if cfg.price_ffill_limit > 0:                 # pandas rejects limit=0; 0 means "no forward fill"
        px = px.ffill(limit=cfg.price_ffill_limit)
    log.add("prices", "clean", f"Isolated gaps forward-filled (limit {cfg.price_ffill_limit})", n_before - px.isna().sum().sum(), "ffill")
    px.index.name, px.columns.name = "date", "stock"
    if not (px.index.is_monotonic_increasing and px.index.is_unique):        # explicit checks: asserts vanish under -O
        raise RuntimeError("internal error: the price calendar is not sorted and unique after cleaning")
    if (px <= 0).any().any():
        raise RuntimeError("internal error: non-positive prices survived cleaning")
    return px


def clean_ratings(ratings_raw: pd.DataFrame, px: pd.DataFrame, cfg: DataConfig = DataConfig(),
                  log: AuditLog | None = None) -> pd.DataFrame:
    """Long ratings table -> one row per accepted target, stamped with its point-in-time knowledge date."""
    log = log if log is not None else AuditLog()
    df = ratings_raw.copy()
    df.columns = df.columns.str.strip().str.lower()
    missing = {"date", "estimate_value", "stock", "analyst"} - set(df.columns)
    if missing:
        raise ValueError(f"ratings: missing columns {missing}")
    df = df.rename(columns={"estimate_value": "tp"})
    df["pub_date"] = to_calendar_dates(df["date"]).to_numpy()
    df["tp"] = to_float(df["tp"])
    for col in ["stock", "analyst"]:
        df[col] = df[col].where(df[col].notna()).astype(object)
        df.loc[df[col].notna(), col] = df.loc[df[col].notna(), col].astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = np.nan

    def _drop(mask: pd.Series, check: str, finding: str, decision: str = "drop") -> None:
        nonlocal df
        log.add("ratings", check, finding, int(mask.sum()), decision)
        df = df.loc[~mask]

    _drop(df["tp"].isna() | (df["tp"] <= 0), "R1", "Unparseable / non-positive target")
    _drop(df["pub_date"].isna(), "R1", "Unparseable publication date")
    _drop(df["analyst"].isna(), "R2", "Missing analyst identifier")
    _drop(df["stock"].isna() | ~df["stock"].isin(px.columns), "clean", "Stock missing or absent from price file")
    _drop(df.duplicated(["pub_date", "stock", "analyst"], keep="last"), "clean", "Duplicate (date, stock, analyst)", "keep last")

    sessions = px.index
    pos = sessions.searchsorted(df["pub_date"].to_numpy(), side="right") + (cfg.knowledge_lag - 1)
    late = pos >= len(sessions)
    _drop(pd.Series(late, index=df.index), "R6", "Published after the last session (not yet usable)")
    df["knowledge_date"] = sessions[pos[~late]]

    px_long = to_long(px, "ref_price").dropna().rename(columns={"date": "px_date"}).sort_values("px_date")
    px_long["stock"] = px_long["stock"].astype(object)
    df["stock"] = df["stock"].astype(object)
    df = pd.merge_asof(df.sort_values("pub_date"), px_long, left_on="pub_date", right_on="px_date", by="stock",
                       direction="backward")
    _drop(df["ref_price"].isna(), "clean", "No price available at publication")
    df["tp_ratio"] = df["tp"] / df["ref_price"]
    lo, hi = cfg.tp_ratio_bounds
    _drop(~df["tp_ratio"].between(lo, hi, inclusive="neither"), "R3", f"TP/P outside ({lo}, {hi})")

    df = df.sort_values(["stock", "analyst", "pub_date"]).reset_index(drop=True)
    g = df.groupby(["stock", "analyst"])
    excess = (df["tp"] / g["tp"].shift(1)) / (df["ref_price"] / g["ref_price"].shift(1))
    elo, ehi = cfg.revision_excess_bounds
    _drop(excess.notna() & ~excess.between(elo, ehi), "R4",
          f"Target change inconsistent with price change (excess outside ({elo:.2f}, {ehi:.1f}))")

    df = df.sort_values(["stock", "analyst", "pub_date"]).reset_index(drop=True)
    df["prev_tp"] = df.groupby(["stock", "analyst"])["tp"].shift(1)
    df["revision"] = np.log(df["tp"] / df["prev_tp"])
    df["is_initiation"] = df["prev_tp"].isna()
    df["is_reiteration"] = np.isclose(df["tp"], df["prev_tp"])
    df["n_same_day"] = df.groupby(["stock", "pub_date"])["tp"].transform("size")
    df["event_day"] = df["n_same_day"] >= cfg.event_min_ratings
    cols = ["knowledge_date", "pub_date", "stock", "analyst", "tp", "ref_price", "tp_ratio", "prev_tp", "revision",
            "is_initiation", "is_reiteration", "n_same_day", "event_day"]
    out = df[cols].sort_values(["knowledge_date", "stock", "analyst"]).reset_index(drop=True)
    if not (out["knowledge_date"] > out["pub_date"]).all():
        raise RuntimeError("look-ahead: a target would be usable on or before its publication date (knowledge_lag must be >= 1)")
    return out
