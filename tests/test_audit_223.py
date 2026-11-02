"""Regression tests for the defects fixed in 2.2.3: results must not depend on stocks that list later."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fmflow import run_strategy
from fmflow.robustness import truncation_check


def _late_listing(synth, after: int = 500):
    """Two stocks of the synthetic universe only start trading `after` sessions in."""
    p, r = synth
    dates = sorted(p["date"].unique())
    late = p["stock"].isin(["stock_9", "stock_10"]) & (p["date"] < dates[after])
    return p[~late], r, dates


def test_volatility_target_ignores_stocks_that_list_later(synth, ew_cfg):
    p, r, dates = _late_listing(synth)
    t = truncation_check(p, r, ew_cfg.with_(vol_target=0.10), [dates[400]])
    assert t["causal"].all(), t[~t["causal"]]


def test_trend_benchmark_is_one_over_n_of_the_names_with_a_signal(synth, ew_cfg):
    from fmflow.benchmarks import comparable_books
    p, r, _ = _late_listing(synth)
    full = comparable_books(run_strategy(p, r, ew_cfg))["Time-series momentum 12-1 (CTA-style)"]
    without = comparable_books(run_strategy(p[~p["stock"].isin(["stock_9", "stock_10"])],
                                            r[~r["stock"].isin(["stock_9", "stock_10"])], ew_cfg))["Time-series momentum 12-1 (CTA-style)"]
    early = full.index[300:450]          # before the late names have 12 months of history: same names, same book
    assert np.allclose(full.loc[early], without.loc[early])


def test_trading_calendar_does_not_depend_on_stocks_that_list_later():
    from fmflow.data.cleaning import clean_prices
    dates = pd.bdate_range("2021-01-04", periods=80)
    rows = [(d, s, 1.0 + 0.01 * i + 0.001 * j) for i, d in enumerate(dates) for j, s in enumerate("abc")]
    rows = [x for x in rows if not (x[0] == dates[20] and x[1] == "c")]            # day 20: 2 of the 3 listed stocks print
    rows += [(d, s, 1.0 + 0.01 * i) for i, d in enumerate(dates) if i >= 60 for s in "defg"]   # four later listings
    p = pd.DataFrame(rows, columns=["date", "stock", "price"])
    full = clean_prices(p)
    cut = clean_prices(p[p["date"] <= dates[40]])
    assert dates[20] in full.index                                 # 2 of 3 active stocks printed: a session
    assert full.index[full.index <= dates[40]].equals(cut.index)  # the same calendar with or without the later stocks


def test_a_stray_first_date_is_still_not_a_session():
    from fmflow.data.cleaning import clean_prices
    dates = pd.bdate_range("2021-01-04", periods=60)
    rows = [("2021-01-01", s, 1.0) for s in "ab"]                        # a holiday row on which two stocks print
    rows += [(d, s, 1.0 + 0.01 * i + 0.001 * j) for i, d in enumerate(dates) for j, s in enumerate("abcdefgh")]
    px = clean_prices(pd.DataFrame(rows, columns=["date", "stock", "price"]))
    assert pd.Timestamp("2021-01-01") not in px.index and px.index[0] == dates[0]


def test_zero_forward_fill_and_one_session_expiry_are_accepted(synth, ew_cfg):
    from dataclasses import replace
    for dc in (dict(price_ffill_limit=0), dict(tp_expiry_days=1)):
        cfg = ew_cfg.with_(data=replace(ew_cfg.data, **dc))
        out = run_strategy(*synth, cfg)
        assert out.weights.notna().all().all()


def test_short_beta_window_works_and_one_session_is_rejected(synth, ew_cfg):
    import pytest
    out = run_strategy(*synth, ew_cfg.with_(neutralize="beta", beta_window=20))
    assert out.weights.notna().all().all() and out.weights.abs().sum(axis=1).max() > 0
    with pytest.raises(ValueError, match="beta_window"):
        ew_cfg.with_(beta_window=1).validate()
