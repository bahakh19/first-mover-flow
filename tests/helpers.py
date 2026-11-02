"""Hand-built rating tables for exact unit tests of the snapshot and the signals."""
import numpy as np
import pandas as pd

from fmflow.config import DataConfig
from fmflow.data.cleaning import clean_prices, clean_ratings
from fmflow.data.snapshot import build_tp_snapshot


def flat_prices(n_stocks=8, n_days=400, start="2020-01-01"):
    """Prices that stay within 1 +/- 0.1% (so targets/price ~ targets) but are never stale (stale rows are cleaned)."""
    dates = pd.bdate_range(start, periods=n_days)
    wiggle = 1.0 + 1e-3 * np.sin(np.arange(n_days * n_stocks) * 0.7)
    return pd.DataFrame({"date": np.repeat(dates.strftime("%Y-%m-%d"), n_stocks),
                         "stock": np.tile([f"s{i}" for i in range(n_stocks)], n_days), "price": wiggle})


def make(rows, n_stocks=8, n_days=400, cfg=DataConfig()):
    """rows: list of (session_index, stock, analyst, target). Prices are ~1.0 throughout."""
    p = flat_prices(n_stocks, n_days)
    px = clean_prices(p, cfg)
    d = px.index
    r = pd.DataFrame([(d[i].strftime("%Y-%m-%d"), str(tp), s, a) for i, s, a, tp in rows],
                     columns=["date", "estimate_value", "stock", "analyst"])
    rt = clean_ratings(r, px, cfg)
    return px, rt, build_tp_snapshot(rt, px.index, px.columns, cfg)
