"""Portfolio construction.

Target book (formed every session from the score known at the close):
    long the k highest-ranked names, short the k lowest-ranked; each leg holds half of gross capital, allocated
    equally ("EW") or in proportion to 1/sigma_hat ("IV-GJR"); a constant net exposure c tilts the leg budgets to
    0.5 + c/2 (long) and 0.5 - c/2 (short). c = 0 gives a dollar-neutral book.
Held book (staggered overlapping tranches):
    w_t = (1/f) * sum_{s=t-f+1..t} target_s,
    identical to holding f sub-portfolios, one opened each session and held f sessions (Jegadeesh-Titman). This removes
    rebalancing-anchor luck and makes turnover smooth.
Optional layers (all causal, all off by default so FMF-N is unchanged):
    neutralize="beta"  leg budgets solved so that the TARGET book has zero ex-ante beta to the equal-weight market
                       (rolling betas known at t), instead of zero net dollars;
    max_weight         cap on |w_i| within the target book (excess redistributed inside the leg);
    vol_target         scale the HELD book to an annualised ex-ante volatility (Ledoit-Wolf shrunk covariance of the
                       trailing `vol_lookback` returns), capped at `max_leverage`;
    sector_neutralize  demean the composite score within user-supplied sectors before ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _leg_masks(score: pd.DataFrame, k: int | None, leg_fraction: float | None, k_min: int):
    if (k is None) == (leg_fraction is None):
        raise ValueError("set exactly one of k (names per leg) or leg_fraction (share of valid names per leg)")
    n = score.notna().sum(axis=1)
    if leg_fraction is None:
        kk = pd.Series(float(k), index=score.index)
        valid = n >= 2 * k
    else:
        kk = np.maximum(k_min, np.floor(leg_fraction * n)).astype(float)
        valid = (n >= 2 * kk) & (n >= 2 * k_min)
    rd = score.rank(axis=1, ascending=False, method="first")
    long_ = rd.le(kk, axis=0) & valid.values[:, None]
    ra = score.where(~long_).rank(axis=1, ascending=True, method="first")
    short_ = ra.le(kk, axis=0) & valid.values[:, None]
    return long_, short_, valid.astype(float)


def target_weights(score: pd.DataFrame, k: int | None = 4, leg_fraction: float | None = None, k_min: int = 2,
                   allocator: str = "EW", sig_vol: pd.DataFrame | None = None, ret: pd.DataFrame | None = None,
                   net_exposure: float = 0.0) -> pd.DataFrame:
    """Daily target weights (sessions x stocks). Rows with too few valid names are flat (all zeros)."""
    long_, short_, valid = _leg_masks(score, k, leg_fraction, k_min)
    if allocator == "EW":
        base = pd.DataFrame(1.0, index=score.index, columns=score.columns)
    elif allocator == "IV-GJR":
        if sig_vol is None or ret is None:
            raise ValueError("IV-GJR allocation needs sig_vol and ret")
        vol = sig_vol.reindex(index=score.index, columns=score.columns).fillna(ret.rolling(60, min_periods=2).std())
        base = 1.0 / vol.where(vol > 0)
        # a selected name with no usable volatility estimate (too little history, zero variance) gets the day's median
        # inverse volatility instead of silently dropping out of its leg and unbalancing the book
        base = base.where(np.isfinite(base))
        base = base.where(base.notna(), base.median(axis=1), axis=0).fillna(1.0)
    else:
        raise ValueError(f"unknown allocator {allocator!r}")
    aL = base.where(long_).div(base.where(long_).sum(axis=1), axis=0).fillna(0.0)
    aS = base.where(short_).div(base.where(short_).sum(axis=1), axis=0).fillna(0.0)
    tau = (net_exposure / 2.0) * valid
    return aL.mul(0.5 + tau, axis=0) - aS.mul(0.5 - tau, axis=0)


def tranche_weights(target: pd.DataFrame, period: int) -> pd.DataFrame:
    """Average of the last `period` daily targets (staggered overlapping tranches)."""
    return target.rolling(period, min_periods=1).sum() / period


def rolling_betas(ret: pd.DataFrame, window: int = 252, min_periods: int = 126) -> pd.DataFrame:
    """Causal beta of each stock to the equal-weight market over the trailing `window` sessions (known at t).
    `min_periods` is capped at `window`, so a short window works without adjusting it."""
    min_periods = min(min_periods, window)
    mkt = ret.mean(axis=1)
    var = mkt.rolling(window, min_periods=min_periods).var()
    return pd.DataFrame({c: ret[c].rolling(window, min_periods=min_periods).cov(mkt) / var for c in ret.columns})


def beta_neutral_budgets(target: pd.DataFrame, betas: pd.DataFrame, gross: float = 1.0, floor: float = 0.2) -> pd.DataFrame:
    """Rescale long and short legs so that b_L * beta_L = b_S * beta_S with b_L + b_S = gross (target book beta = 0).

    beta_L, beta_S are the weighted betas of the unit-normalised legs. Leg budgets are kept inside [floor, 1 - floor]
    x gross so a single extreme beta estimate cannot put the whole book on one side."""
    L, S = target.clip(lower=0), (-target).clip(lower=0)
    sl, ss = L.sum(axis=1), S.sum(axis=1)
    bL = (L.div(sl.where(sl > 0), axis=0) * betas).sum(axis=1, min_count=1)
    bS = (S.div(ss.where(ss > 0), axis=0) * betas).sum(axis=1, min_count=1)
    share_long = (bS / (bL + bS)).where((bL > 0) & (bS > 0)).fillna(0.5).clip(floor, 1 - floor)
    out = L.div(sl.where(sl > 0), axis=0).mul(gross * share_long, axis=0) - S.div(ss.where(ss > 0), axis=0).mul(gross * (1 - share_long), axis=0)
    return out.fillna(0.0)


def cap_weights(target: pd.DataFrame, max_weight: float) -> pd.DataFrame:
    """Cap |w_i| at `max_weight`, redistributing the excess pro rata over the uncapped names of the same leg.

    Leg budgets are unchanged whenever the cap is feasible (names in the leg x max_weight >= leg budget). When it is
    not, every name is held at the cap and the leg's gross shrinks: the cap is a hard limit and is never exceeded."""
    tol = 1e-12
    out = []
    for sign in (1, -1):
        w = (sign * target).clip(lower=0)
        for _ in range(w.shape[1] + 1):          # each pass caps at least one more name, so this always terminates
            if not (w > max_weight + tol).to_numpy().any():
                break
            excess = (w - max_weight).clip(lower=0).sum(axis=1)
            w = w.clip(upper=max_weight)
            free = w.where((w > 0) & (w < max_weight - tol), 0.0)
            fs = free.sum(axis=1)
            w = w + free.div(fs.where(fs > 0), axis=0).fillna(0.0).mul(excess, axis=0)
        out.append(sign * w.clip(upper=max_weight))
    return out[0] + out[1]


def ledoit_wolf_cov(x: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance towards a scaled identity."""
    x = x - x.mean(0)
    n, p = x.shape
    S = x.T @ x / n
    mu = np.trace(S) / p
    d2 = ((S - mu * np.eye(p)) ** 2).sum() / p
    # sum_r ||r r' - S||_F^2 = sum_r ||r||^4 - n ||S||_F^2   (vectorised form of the Ledoit-Wolf b-bar term)
    b2 = (((x ** 2).sum(axis=1) ** 2).sum() - n * (S ** 2).sum()) / (n ** 2 * p)
    shrink = min(b2, d2) / d2 if d2 > 0 else 1.0
    return shrink * mu * np.eye(p) + (1 - shrink) * S


def ex_ante_vol(weights: pd.DataFrame, ret: pd.DataFrame, lookback: int = 126, periods: int = 252,
                every: int = 1) -> pd.Series:
    """Annualised ex-ante volatility of the book held at t, from returns up to t (Ledoit-Wolf covariance).

    The covariance is estimated on the stocks that have traded by t. Including a stock that only lists later (a column
    of zeros) would change the Ledoit-Wolf shrinkage, so the estimate would depend on the future composition of the
    universe."""
    out = pd.Series(np.nan, index=weights.index)
    R = ret.reindex(index=weights.index, columns=weights.columns)
    traded = R.notna().cummax().to_numpy()               # has had a return by t (known at t)
    for i in range(lookback, len(weights), every):
        w = weights.iloc[i].to_numpy()
        if not np.abs(w).sum():
            continue
        cols = traded[i] | (w != 0)
        x = R.iloc[i - lookback + 1: i + 1].to_numpy()[:, cols]
        x = np.nan_to_num(x, nan=0.0)
        out.iloc[i] = float(np.sqrt(max(w[cols] @ ledoit_wolf_cov(x) @ w[cols], 0.0) * periods))
    return out.ffill() if every > 1 else out


def vol_target_weights(weights: pd.DataFrame, ret: pd.DataFrame, target: float, lookback: int = 126,
                       max_leverage: float = 2.0, every: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """Scale the held book to `target` annualised ex-ante volatility (scale known at t, capped at max_leverage)."""
    vol = ex_ante_vol(weights, ret, lookback, every=every)
    scale = (target / vol).clip(upper=max_leverage).fillna(0.0)
    return weights.mul(scale, axis=0), scale


def sector_demean(score: pd.DataFrame, sectors: dict) -> pd.DataFrame:
    """Subtract the within-sector cross-sectional mean of the score (names without a sector are left unchanged)."""
    out = score.copy()
    groups: dict = {}
    for stock, sec in sectors.items():
        if stock in score.columns:
            groups.setdefault(sec, []).append(stock)
    for cols in groups.values():
        if len(cols) > 1:
            out[cols] = score[cols].sub(score[cols].mean(axis=1), axis=0)
    return out
