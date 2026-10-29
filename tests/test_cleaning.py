import numpy as np
import pandas as pd

from fmflow.config import DataConfig
from fmflow.data.cleaning import AuditLog, clean_prices, clean_ratings

from conftest import toy_prices


def test_holidays_removed_and_gaps_bounded():
    p = toy_prices(n_stocks=8, n_days=60)
    holiday = p["date"].unique()[10]
    p = p[~((p["date"] == holiday) & (p["stock"] != "s0"))]          # only 1/8 stocks print -> not a session
    gap_days = p["date"].unique()[20:28]                               # an 8-session gap for one stock
    p = p[~((p["date"].isin(gap_days)) & (p["stock"] == "s3"))]
    px = clean_prices(p)
    assert pd.Timestamp(holiday) not in px.index
    s3 = px["s3"].loc[pd.Timestamp(gap_days[0]):pd.Timestamp(gap_days[-1])]
    assert s3.notna().sum() == DataConfig().price_ffill_limit       # forward fill is bounded, never backward
    assert px.index.is_monotonic_increasing


def test_knowledge_date_is_next_session_and_filters():
    p = toy_prices(n_stocks=8, n_days=120)
    px = clean_prices(p)
    d = px.index
    r = pd.DataFrame({
        "date": [d[10], d[10], d[20], d[30], d[40], d[50], d[50], d[-1] + pd.Timedelta(days=3)],
        "estimate_value": [px["s1"].iloc[10] * 1.2, px["s2"].iloc[10] * 1e6, px["s1"].iloc[20] * 1.25, "Not Available",
                           px["s1"].iloc[40] * 1.3, px["s4"].iloc[50] * 1.1, px["s4"].iloc[50] * 1.1, 1.0],
        "stock": ["s1", "s2", "s1", "s1", "s1", "s4", "s4", "s5"],
        "analyst": ["a", "b", "a", "a", np.nan, "c", "c", "d"]})
    r["date"] = pd.to_datetime(r["date"]).dt.strftime("%Y-%m-%d")
    log = AuditLog()
    rt = clean_ratings(r.astype({"estimate_value": str}), px, DataConfig(), log)
    assert (rt["knowledge_date"] > rt["pub_date"]).all()
    first = rt[(rt["stock"] == "s1") & (rt["pub_date"] == d[10])]
    assert first["knowledge_date"].iloc[0] == d[11]                   # usable the next session, never the same day
    f = log.to_frame().set_index("finding")["n_rows"]
    assert f["TP/P outside (0.2, 5.0)"] == 1 and f["Missing analyst identifier"] == 1
    assert f["Unparseable / non-positive target"] == 1 and f["Duplicate (date, stock, analyst)"] == 1
    assert f["Published after the last session (not yet usable)"] == 1
    assert rt.loc[(rt["stock"] == "s1") & (rt["pub_date"] == d[20]), "is_initiation"].iloc[0] == False  # noqa: E712


def test_inconsistent_revision_removed():
    p = toy_prices(n_stocks=8, n_days=80)
    px = clean_prices(p)
    d = px.index
    r = pd.DataFrame({"date": [d[5].strftime("%Y-%m-%d"), d[15].strftime("%Y-%m-%d")],
                      "estimate_value": [str(px["s1"].iloc[5] * 1.1), str(px["s1"].iloc[15] * 1.1 * 4.5)],
                      "stock": ["s1", "s1"], "analyst": ["a", "a"]})
    log = AuditLog()
    rt = clean_ratings(r, px, DataConfig(), log)
    assert len(rt) == 1 and log.to_frame().query("check == 'R4'")["n_rows"].iloc[0] == 1
