"""Daily production workflow for a portfolio manager: today's book, the trades to get there, and the risk and health
checks to run before sending orders.

    out   = run_strategy(prices, ratings, cfg)          # with data up to today's close
    book  = latest_book(out)                            # target and held weights, scores, legs, signal values
    trades = trade_list(book["held_weight"], current_shares, last_prices, nav=50e6, lot_size=1)
    risk  = risk_report(out)                            # exposures, ex-ante vol and beta, concentration, data freshness
    health = signal_monitor(out)                        # realised rolling IC of the composite and of LMH
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fmflow.analytics.inference import forward_log_return, rank_ic
from fmflow.portfolio.construction import ledoit_wolf_cov, rolling_betas


def _asof(out, date=None) -> pd.Timestamp:
    idx = out.weights.index
    if date is None:
        return idx[-1]
    before = idx[idx <= pd.Timestamp(date)]
    if before.empty:
        raise ValueError(f"no session on or before {pd.Timestamp(date).date()} (the data start on {idx[0].date()})")
    return before[-1]


def latest_book(out, date=None) -> pd.DataFrame:
    """Per-stock view of the book on `date` (default: the last session)."""
    t = _asof(out, date)
    df = pd.DataFrame({"held_weight": out.weights.loc[t], "target_weight": out.target.loc[t], "score": out.score.loc[t]})
    df["rank"] = df["score"].rank(ascending=False, method="first")
    df["leg"] = np.select([df["target_weight"] > 1e-12, df["target_weight"] < -1e-12], ["long", "short"], "flat")
    for s in out.config.signals:
        df[s] = out.signals[s].loc[t]
    if out.sig_vol is not None:
        df["vol_forecast_ann"] = out.sig_vol.loc[t] * np.sqrt(252)
    df.attrs["date"] = t
    return df.sort_values("score", ascending=False)


def trade_list(target_weights: pd.Series, current_shares: pd.Series | None, prices: pd.Series, nav: float,
               lot_size: int = 1, min_trade_notional: float = 0.0) -> pd.DataFrame:
    """Orders that move the current holdings to `target_weights` x `nav`, rounded to lots.

    prices must be in the SAME units as the holdings (real currency prices, not normalised series)."""
    if not nav > 0:
        raise ValueError("nav must be positive")
    if lot_size < 1:
        raise ValueError("lot_size must be >= 1")
    names = target_weights.index.union(current_shares.index if current_shares is not None else [])
    w = target_weights.reindex(names).fillna(0.0)
    cur = (current_shares.reindex(names).fillna(0.0) if current_shares is not None else pd.Series(0.0, index=names))
    px = prices.reindex(names)
    if px.isna().any():
        raise ValueError(f"no price for {list(px[px.isna()].index)}: every name in the target book or in the current "
                         "holdings needs a price in currency (a holding outside the book is sold)")
    tgt = np.round(w * nav / px / lot_size) * lot_size
    d = tgt - cur
    notional = d * px
    keep = notional.abs() >= min_trade_notional
    out = pd.DataFrame({"current_shares": cur, "target_shares": tgt, "trade_shares": d.where(keep, 0.0),
                        "trade_notional": notional.where(keep, 0.0), "side": np.where(d > 0, "BUY", np.where(d < 0, "SELL", "-"))})
    out.loc[~keep, "side"] = "-"
    out.attrs["gross_traded"] = float(out["trade_notional"].abs().sum())
    out.attrs["turnover_vs_nav"] = out.attrs["gross_traded"] / nav
    return out[out["side"] != "-"].sort_values("trade_notional", key=np.abs, ascending=False) if keep.any() else out.iloc[:0]


def risk_report(out, date=None, lookback: int = 126) -> pd.Series:
    """Exposures, ex-ante volatility and beta, concentration and data freshness of the book on `date`."""
    t = _asof(out, date)
    w = out.weights.loc[t]
    hist = out.returns.loc[:t, w.index]
    cols = (hist.notna().any() | (w != 0)).to_numpy()     # stocks that have traded by t (see ex_ante_vol)
    R = hist.iloc[-lookback:].fillna(0.0).to_numpy()[:, cols]
    wv = w.to_numpy()[cols]
    vol = float(np.sqrt(max(wv @ ledoit_wolf_cov(R) @ wv, 0) * 252))
    beta = rolling_betas(out.returns.loc[:t], 252, 126).iloc[-1].reindex(w.index).fillna(1.0)
    last_rev = out.ratings.groupby("stock")["knowledge_date"].max().reindex(w.index)
    sessions_since = pd.Series({s: int((out.prices.index[(out.prices.index > d) & (out.prices.index <= t)]).size)
                                if pd.notna(d) else np.nan for s, d in last_rev.items()})
    aw = w.abs().sort_values(ascending=False)
    return pd.Series({
        "date": t.date(), "gross": float(w.abs().sum()), "net": float(w.sum()), "long names": int((w > 1e-9).sum()),
        "short names": int((w < -1e-9).sum()), "largest |w|": float(aw.iloc[0]),
        "top-3 share of gross": float(aw.iloc[:3].sum() / aw.sum()) if aw.sum() else np.nan,
        "ex-ante vol (ann.)": vol, "ex-ante beta (EW market)": float((w * beta).sum()),
        "names with valid score": int(out.score.loc[t].notna().sum()),
        "median sessions since last rating": float(sessions_since.median()),
        "max sessions since last rating (held names)": float(sessions_since[w.abs() > 1e-9].max()) if (w.abs() > 1e-9).any() else np.nan,
    })


def signal_monitor(out, h: int = 21, window: int = 126, signals=("composite", "leader_minus_herd")) -> pd.DataFrame:
    """Rolling mean of the REALISED rank IC (labels end h sessions before each date, so the series is known at t).

    A sustained negative value is the early-warning signal a PM monitors for alpha decay."""
    fwd = forward_log_return(out.prices, h)
    res = {}
    for s in signals:
        f = out.score if s == "composite" else out.signals[s]
        ic = rank_ic(f, fwd).shift(h)                    # IC of the signal at t-h, whose label is complete at t
        res[f"{s} IC{h} rolling {window}"] = ic.rolling(window, min_periods=window // 2).mean()
    return pd.DataFrame(res)
