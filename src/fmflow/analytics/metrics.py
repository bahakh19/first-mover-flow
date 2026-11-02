"""Performance and attribution metrics.

Definitions match the research notebook, with one exception: the Sortino ratio divides by the downside deviation
sqrt(mean(min(r, 0)^2)) over ALL sessions (Sortino and Price, 1994), not by the standard deviation of the negative
returns alone."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats


def drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown path (equity / running peak - 1) of a daily return series."""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0


def perf_stats(returns: pd.Series, periods: int = 252) -> pd.Series:
    """CAGR, volatility, Sharpe, Sortino (downside deviation, target 0), max drawdown, Calmar (CAGR/|MaxDD|), tails,
    hit rate. NaN-only or single-observation inputs give an empty Series."""
    r = returns.dropna()
    n = len(r)
    if n < 2:
        return pd.Series(dtype=float)
    equity = (1.0 + r).cumprod()
    cagr = equity.iloc[-1] ** (periods / n) - 1.0 if equity.iloc[-1] > 0 else -1.0   # a wiped-out book stays at -100%
    vol = r.std(ddof=1) * np.sqrt(periods)
    downside = float(np.sqrt((np.minimum(r, 0.0) ** 2).mean() * periods))
    dd = equity / equity.cummax() - 1.0
    mdd = dd.min()
    return pd.Series({
        "CAGR": cagr, "Ann. vol": vol, "Sharpe": r.mean() / r.std(ddof=1) * np.sqrt(periods) if r.std() > 0 else np.nan,
        "Sortino": r.mean() * periods / downside if downside > 0 else np.nan, "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan, "Skew": stats.skew(r), "Excess kurtosis": stats.kurtosis(r),
        "VaR 5%": -np.quantile(r, 0.05), "Hit rate": (r > 0).mean(), "N sessions": n,
    })


def _weekly(r: pd.Series) -> pd.Series:
    return r.groupby(np.arange(len(r)) // 5).sum()


def capm(strategy: pd.Series, market: pd.Series, hac_lags: int = 4) -> dict:
    """Weekly (non-overlapping 5-session sums) regression on the market with HAC standard errors."""
    import statsmodels.api as sm
    a, b = strategy.align(market, join="inner")
    y, x = _weekly(a), _weekly(b)
    f = sm.OLS(y.values, sm.add_constant(x.values)).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    degenerate = f.ssr <= 1e-12 * max(float(((y - y.mean()) ** 2).sum()), 1e-300)   # strategy == market: t is noise
    return {"beta": float(f.params[1]), "alpha_ann": float(f.params[0] * 52),
            "alpha_t": float("nan") if degenerate else float(f.tvalues[0])}


def factor_attribution(strategy: pd.Series, factors: Mapping[str, pd.Series], hac_lags: int = 4) -> dict:
    """Weekly regression of the strategy on several factor-return series; returns alpha, t, betas, R2."""
    import statsmodels.api as sm
    df = pd.concat({"y": strategy, **factors}, axis=1).dropna()
    W = df.groupby(np.arange(len(df)) // 5).sum()
    f = sm.OLS(W["y"].values, sm.add_constant(W.drop(columns="y").values)).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    out = {"alpha_ann": float(f.params[0] * 52), "alpha_t": float(f.tvalues[0]), "R2": float(f.rsquared)}
    out.update({f"b_{c}": float(v) for c, v in zip(W.drop(columns="y").columns, f.params[1:], strict=True)})
    return out


def stress_table(books: Mapping[str, pd.Series], episodes: Mapping[str, tuple]) -> pd.DataFrame:
    """Cumulative return of each book over named episodes {name: (start, end)}."""
    return pd.DataFrame({name: {ep: float((1 + r.loc[s:e].fillna(0)).prod() - 1) for ep, (s, e) in episodes.items()}
                         for name, r in books.items()}).T


def lo_adjusted_sharpe(returns: pd.Series, periods: int = 252, max_lag: int = 10) -> dict:
    """Annualised Sharpe ratio corrected for serial correlation (Lo, 2002).

    sqrt(periods) scaling assumes IID returns. With autocorrelations rho_k the correct factor is
    eta = q / sqrt(q + 2 * sum_{k=1}^{L} (q - k) rho_k), q = periods, truncated at `max_lag` (L) to limit noise.
    Positive autocorrelation (smoothed P&L) inflates the naive ratio; negative autocorrelation deflates it."""
    r = returns.dropna()
    sr_d = r.mean() / r.std(ddof=1)
    rho = [r.autocorr(k) for k in range(1, max_lag + 1)]
    q = periods
    eta = q / np.sqrt(q + 2 * sum((q - k) * p for k, p in enumerate(rho, start=1)))
    return {"sharpe_naive": float(sr_d * np.sqrt(q)), "sharpe_lo": float(sr_d * eta), "rho_1": float(rho[0]),
            "sum_rho_1_to_L": float(np.sum(rho))}


def conditional_betas(strategy: pd.Series, market: pd.Series, hac_lags: int = 4) -> dict:
    """Weekly up-market and down-market betas (Henriksson-Merton style split) and the downside correlation."""
    import statsmodels.api as sm
    a, b = strategy.align(market, join="inner")
    y, x = _weekly(a), _weekly(b)
    X = np.column_stack([np.ones(len(x)), np.where(x > 0, x, 0.0), np.where(x <= 0, x, 0.0)])
    f = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    down = x <= x.quantile(0.1)
    return {"beta_up": float(f.params[1]), "beta_down": float(f.params[2]), "t_up": float(f.tvalues[1]),
            "t_down": float(f.tvalues[2]), "mean_in_worst_market_decile_weeks": float(y[down].mean()),
            "corr_in_worst_decile": float(np.corrcoef(y[down], x[down])[0, 1]) if down.sum() > 3 else np.nan}
