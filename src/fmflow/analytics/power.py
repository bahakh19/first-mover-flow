"""Statistical power in small cross-sections: how large an edge can this sample detect, and how long must one wait?

mde_ic             minimum detectable mean IC (one-sided test, non-overlapping horizon-h observations)
ic_power           power of that test for a given true IC
simulate_power     Monte Carlo power of the full pipeline on the synthetic generator: for each (stocks, years, strength)
                   cell, the share of simulated worlds in which the strategy's net Sharpe is significant. This answers
                   "would we have found the effect if it were there?" -- the question a small-sample null result raises.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mde_ic(n_stocks: int, n_obs: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true mean IC detectable with `power`: (z_a + z_b) / (sqrt(N - 1) * sqrt(M)), M independent dates."""
    return float((stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) / (np.sqrt(n_stocks - 1) * np.sqrt(n_obs)))


def ic_power(true_ic: float, n_stocks: int, n_obs: int, alpha: float = 0.05) -> float:
    """Power of the one-sided IC test for a true mean IC with `n_stocks` names and `n_obs` independent dates."""
    se = 1 / (np.sqrt(n_stocks - 1) * np.sqrt(n_obs))
    return float(stats.norm.sf(stats.norm.ppf(1 - alpha) - true_ic / se))


def simulate_power(n_stocks=(10, 30, 100), years=(3, 5, 10), strengths=(0.0, 0.2, 0.4), seeds: int = 10,
                   allocator: str = "EW", alpha: float = 0.05) -> pd.DataFrame:
    """Share of synthetic worlds where the one-sided Sharpe test (PSR > 1 - alpha) rejects. strength = 0 gives size."""
    from fmflow import StrategyConfig, run_strategy
    from fmflow.analytics.inference import psr
    from fmflow.synthetic import SyntheticConfig, generate
    rows = []
    for n in n_stocks:
        cfg = StrategyConfig.fmf_n().with_(allocator=allocator, k=None, leg_fraction=0.4)
        for y in years:
            for s in strengths:
                rej, srs = [], []
                for seed in range(1, seeds + 1):
                    sc = SyntheticConfig(n_stocks=n, n_sessions=252 * (y + 1), seed=seed, strength=s, stock_drift_sd=0.0,
                                         market_drift=0.0, analysts_per_stock=(10, 40))
                    net = run_strategy(*generate(sc), cfg).net_returns.iloc[252:]
                    rej.append(psr(net.values) > 1 - alpha)
                    srs.append(net.mean() / net.std() * np.sqrt(252))
                rows.append({"stocks": n, "years": y, "strength": s, "power": float(np.mean(rej)),
                             "mean_sharpe": float(np.mean(srs)), "seeds": seeds})
    return pd.DataFrame(rows)
