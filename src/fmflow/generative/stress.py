"""Path-level stress statistics from simulated return paths (n_paths, horizon, C), channel 0 = the strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd


def path_stats(paths: np.ndarray, channel: int = 0, periods: int = 252) -> pd.DataFrame:
    """CAGR, max drawdown, Calmar, Sharpe and total return of each simulated path."""
    r = paths[:, :, channel]
    eq = np.cumprod(1 + r, axis=1)
    years = r.shape[1] / periods
    cagr = eq[:, -1] ** (1 / years) - 1
    mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(1)
    sr = r.mean(1) / r.std(1, ddof=1) * np.sqrt(periods)
    return pd.DataFrame({"CAGR": cagr, "MaxDD": mdd, "Calmar": cagr / np.abs(mdd), "Sharpe": sr,
                         "total return": eq[:, -1] - 1})


def summarise(stats_: pd.DataFrame, q=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
    """Quantiles of path statistics and the probability that each is negative."""
    out = stats_.quantile(list(q)).T
    out.columns = [f"q{int(x * 100):02d}" for x in q]
    out["P(<0)"] = (stats_ < 0).mean()
    return out


def conditional_on(paths: np.ndarray, condition: np.ndarray, channel: int = 0) -> pd.DataFrame:
    """Strategy path statistics on the subset of simulated paths where `condition` (boolean, n_paths) holds."""
    return summarise(path_stats(paths[condition], channel))


def market_drawdown(paths: np.ndarray, channel: int = 1) -> np.ndarray:
    """Maximum drawdown of a channel (default: the market) along each simulated path."""
    eq = np.cumprod(1 + paths[:, :, channel], axis=1)
    return (eq / np.maximum.accumulate(eq, axis=1) - 1).min(1)
