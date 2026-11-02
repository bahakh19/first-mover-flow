import numpy as np
import pandas as pd
import pytest

from fmflow.signals.breadth import breadth, breadth_resid_mom, event_breadth
from fmflow.signals.composite import centred_rank, composite_score, cs_rank

from helpers import make


def test_breadth_counts_signs_over_window():
    rows = [(5, "s1", f"a{j}", 1.1) for j in range(4)] + [(25, "s1", "a0", 1.2), (25, "s1", "a1", 1.3),
                                                           (25, "s1", "a2", 1.0), (100, "s1", "a3", 1.2)]
    px, rt, snap = make(rows)
    b = breadth(snap, 63, 3)
    d = px.index
    assert b.loc[d[26], "s1"] == pytest.approx((2 - 1) / 3)   # 2 up, 1 down known at session 26
    assert np.isnan(b.loc[d[26], "s2"])                         # fewer than 3 revisions -> undefined
    assert np.isnan(b.loc[d[26 + 63], "s1"])                     # the three revisions left the window


def test_event_breadth_uses_only_event_days():
    rows = [(5, "s1", f"a{j}", 1.1) for j in range(6)] + [(40, "s1", f"a{j}", 1.2) for j in range(5)] + \
           [(45, "s1", "a5", 0.9)]                                   # off-event down revision must be ignored
    px, rt, snap = make(rows)
    eb = event_breadth(snap, px.index, px.columns, 63, 2)
    assert eb.loc[px.index[50], "s1"] == pytest.approx(1.0)
    assert breadth(snap).loc[px.index[50], "s1"] == pytest.approx((5 - 1) / 6)


def test_resid_breadth_orthogonal_to_momentum():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame(np.cumprod(1 + rng.normal(0, 0.02, (300, 10)), axis=0), index=idx, columns=[f"s{i}" for i in range(10)])
    b = pd.DataFrame(rng.normal(size=(300, 10)), index=idx, columns=px.columns)
    res = breadth_resid_mom(b, px)
    mom = np.log(px).shift(21) - np.log(px).shift(126)
    x = cs_rank(mom)
    ok = res.notna().all(axis=1)
    xc = x[ok].sub(x[ok].mean(axis=1), axis=0)
    assert np.abs((res[ok] * xc).sum(axis=1)).max() < 1e-10       # residual is orthogonal date by date


def test_rank_helpers():
    p = pd.DataFrame([[1.0, 2, 3, 4, 5, np.nan], [1, 2, np.nan, np.nan, np.nan, np.nan]])
    r = cs_rank(p)
    assert r.iloc[1].isna().all() and r.iloc[0].notna().sum() == 5
    assert abs(centred_rank(p).iloc[0].sum()) < 1e-12
    c = composite_score({"x": p, "y": -p}, ["x", "y"])
    row = c.iloc[0].dropna()
    assert np.allclose(row, row.iloc[0])                            # equal and opposite signals leave no ranking
