"""Price-based signals: the four controls used in all attribution tests and the institutional comparables.

All signals are computed from closes up to t (no look-ahead) and oriented so that higher = more attractive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def price_signals(px: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Price-based controls and comparables (momentum, reversal, low vol, 52-week high, low beta, residual momentum)."""
    logp = np.log(px)
    ret = px.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)
    out = {
        "mom_12_1": logp.shift(21) - logp.shift(252),          # Jegadeesh-Titman 12-1 momentum
        "mom_6_1": logp.shift(21) - logp.shift(126),
        "rev_1m": -(logp - logp.shift(21)),                     # short-term reversal
        "low_vol_60": -ret.rolling(60).std(),                   # low-volatility anomaly
        "high_52w": px / px.rolling(252, min_periods=126).max(),  # George-Hwang 52-week-high proximity
    }
    # low beta (Frazzini-Pedersen betting-against-beta, cross-sectional version): -beta to the equal-weight market
    var_m = mkt.rolling(252, min_periods=126).var()
    beta = pd.DataFrame({c: ret[c].rolling(252, min_periods=126).cov(mkt) / var_m for c in px.columns})
    out["low_beta_252"] = -beta
    # residual momentum (Blitz-Huij-Martens): 12-1 cumulative market-residual return scaled by residual volatility
    resid = ret - beta.shift(1) * mkt.values[:, None]
    r = resid.rolling(231, min_periods=180)
    out["resid_mom_12_1"] = (r.sum() / r.std()).shift(21)
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in out.items()}
