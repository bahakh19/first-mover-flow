"""Turn a raw (ticker, broker, dollar) universe into the anonymised, normalised format of `data/open/`.

The open dataset was built in two steps (see `notebooks/Data_Gathering.ipynb` and DATA_SOURCES.md):

1. download analyst target prices (Kaggle) and split-adjusted closes (Yahoo Finance) for a list of tickers;
2. `normalize_universe`: anonymise and rescale, so that levels and names are not shipped, while every signal is
   unchanged (all signals use ratios: returns, target / price, target / previous target).

Rules (deterministic, so anyone can rebuild the shipped files from the same raw downloads):

* stocks are numbered in alphabetical order of their identifier: stock_1, stock_2, ...
* analysts (brokers in the open data) are numbered in alphabetical order of their name: analyst_1, analyst_2, ...
* each stock's prices AND targets are divided by the stock's first available price, then rounded to `decimals`.

Rounding to 4 decimals is harmless for normal stocks but quantises the price of a stock that falls far below its
starting level (a price of 0.02 carries a rounding error of up to 0.25%); `decimals=None` keeps full precision.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fmflow.data.cleaning import to_float


@dataclass
class NormalizedUniverse:
    """Normalised tables plus the private keys needed to map them back (never ship the keys)."""
    prices: pd.DataFrame            # date, stock, price
    ratings: pd.DataFrame           # date, estimate_value, stock, analyst
    stock_map: pd.Series            # original identifier -> stock_i
    analyst_map: pd.Series          # original analyst / broker name -> analyst_j
    base_price: pd.Series           # original identifier -> divisor (first available price)


def _numbered(names, prefix: str) -> pd.Series:
    u = sorted({str(x) for x in names})
    return pd.Series([f"{prefix}{i + 1}" for i in range(len(u))], index=u, name=prefix.rstrip("_"))


def normalize_universe(prices: pd.DataFrame, ratings: pd.DataFrame, decimals: int | None = 4,
                       stock_prefix: str = "stock_", analyst_prefix: str = "analyst_") -> NormalizedUniverse:
    """Anonymise identifiers and divide each stock's prices and targets by its first price.

    `prices` needs columns date, stock, price; `ratings` needs date, estimate_value, stock, analyst. Ratings on a
    stock without any price are dropped (they could not be scaled). Rows keep their input order for ratings; prices
    are sorted by (date, stock id) like the shipped file."""
    p = prices[["date", "stock", "price"]].copy()
    r = ratings[["date", "estimate_value", "stock", "analyst"]].copy()
    p["stock"], r["stock"] = p["stock"].astype(str), r["stock"].astype(str)
    has_analyst = r["analyst"].notna()                   # a missing analyst stays missing (cleaning rule R2 drops it)
    r["analyst"] = r["analyst"].where(has_analyst, None).astype(object)
    r.loc[has_analyst, "analyst"] = r.loc[has_analyst, "analyst"].astype(str)
    p["price"] = to_float(p["price"])
    r["estimate_value"] = to_float(r["estimate_value"])
    p = p.dropna(subset=["price"])
    p = p[p["price"] > 0]
    order = pd.to_datetime(p["date"])
    base = p.assign(_d=order).sort_values(["stock", "_d"]).groupby("stock")["price"].first()
    smap = _numbered(base.index, stock_prefix)
    r = r[r["stock"].isin(base.index)]
    amap = _numbered(r["analyst"].dropna().unique(), analyst_prefix)

    def rnd(x):
        return x.round(decimals) if decimals is not None else x

    p_out = pd.DataFrame({"date": p["date"].values, "stock": p["stock"].map(smap).values,
                          "price": rnd(p["price"] / p["stock"].map(base)).values})
    p_out = p_out.assign(_d=pd.to_datetime(p_out["date"])).sort_values(["_d", "stock"]).drop(columns="_d")
    r_out = pd.DataFrame({"date": r["date"].values, "estimate_value": rnd(r["estimate_value"] / r["stock"].map(base)).values,
                          "stock": r["stock"].map(smap).values, "analyst": r["analyst"].map(amap).values})
    return NormalizedUniverse(p_out.reset_index(drop=True), r_out.reset_index(drop=True), smap, amap, base)


def compare_tables(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], value: str, tol: float = 1.5e-4) -> dict:
    """Row-set and value agreement of two long tables (order-insensitive). Used to check a rebuilt dataset against
    the shipped one: `tol` absorbs the last-digit differences of 4-decimal rounding."""
    m = a.merge(b, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)
    both = m[m["_merge"] == "both"]
    diff = (both[f"{value}_a"] - both[f"{value}_b"]).abs()
    return {"rows_a": len(a), "rows_b": len(b), "only_in_a": int((m["_merge"] == "left_only").sum()),
            "only_in_b": int((m["_merge"] == "right_only").sum()), "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
            "share_within_tol": float((diff <= tol).mean()) if len(diff) else 1.0}
