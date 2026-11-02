"""The input data contract (see DATA.md for the full outline).

Two long-format tables are required:

prices   : date (ISO yyyy-mm-dd), stock (str id), price (float > 0; any common scale per stock, e.g. normalised to 1)
ratings  : date (publication date), stock (str id, must appear in prices), analyst (str id),
           estimate_value (analyst target price, SAME SCALE as that stock's price series)

`validate_*` raise a `DataContractError` with an actionable message instead of failing deep inside the pipeline.
Validation is structural (columns, types, scale consistency); semantic cleaning is done by `fmflow.data.cleaning`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fmflow.data.cleaning import to_calendar_dates, to_float

PRICE_COLUMNS = ("date", "stock", "price")
RATING_COLUMNS = ("date", "estimate_value", "stock", "analyst")


class DataContractError(ValueError):
    """Raised when an input table does not satisfy the documented contract."""


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of `df` with column names stripped and lower-cased (' Date ' -> 'date')."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


_normalise_columns = normalise_columns      # backwards compatibility


def _stock_ids(s: pd.Series) -> pd.Series:
    """Stock identifiers as stripped text (object dtype), so integer ids in one table match text ids in the other."""
    out = s.astype("string").str.strip()
    return out.where(out.notna() & (out != ""), None).astype(object)


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Check the price table; returns a column-normalised copy."""
    df = normalise_columns(prices)
    missing = set(PRICE_COLUMNS) - set(df.columns)
    if missing:
        raise DataContractError(f"prices: missing columns {sorted(missing)}; expected {PRICE_COLUMNS}")
    if df.empty:
        raise DataContractError("prices: table is empty")
    dates = to_calendar_dates(df["date"])
    if dates.isna().mean() > 0.01:
        raise DataContractError("prices: more than 1% of dates are unparseable (use ISO yyyy-mm-dd)")
    px = to_float(df["price"])
    if (px.dropna() <= 0).mean() > 0.01:
        raise DataContractError("prices: more than 1% of prices are non-positive")
    if df["stock"].nunique() < 6:
        raise DataContractError("prices: need at least 6 stocks for a long/short cross-section")
    return df


def validate_ratings(ratings: pd.DataFrame, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Check the ratings table (and, if prices are given, that targets are on the price scale)."""
    df = normalise_columns(ratings)
    missing = set(RATING_COLUMNS) - set(df.columns)
    if missing:
        raise DataContractError(f"ratings: missing columns {sorted(missing)}; expected {RATING_COLUMNS}")
    if df.empty:
        raise DataContractError("ratings: table is empty")
    if to_calendar_dates(df["date"]).isna().mean() > 0.05:
        raise DataContractError("ratings: more than 5% of dates are unparseable")
    if prices is not None:
        p = normalise_columns(prices)
        rated, priced = _stock_ids(df["stock"]), _stock_ids(p["stock"])
        if not set(rated.dropna()) & set(priced.dropna()):
            raise DataContractError("ratings: no rated stock appears in the price table (check stock ids)")
        # scale check: median target/price ratio should be of order 1 (targets on the same scale as prices)
        p2 = pd.DataFrame({"date": to_calendar_dates(p["date"]).to_numpy(), "stock": priced.to_numpy(),
                           "price": to_float(p["price"]).to_numpy()})
        r2 = pd.DataFrame({"date": to_calendar_dates(df["date"]).to_numpy(), "stock": rated.to_numpy(),
                           "estimate_value": to_float(df["estimate_value"]).to_numpy()}
                          ).dropna(subset=["date", "stock"])
        m = pd.merge_asof(r2.sort_values("date"), p2.dropna().sort_values("date")[["date", "stock", "price"]],
                          on="date", by="stock", direction="backward")
        ratio = (m["estimate_value"] / m["price"]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) and not 0.3 < ratio.median() < 3.5:
            raise DataContractError(f"ratings: median target/price = {ratio.median():.3g}; targets must be expressed "
                                    "on the same scale as prices (e.g. both normalised by the first price)")
    return df
