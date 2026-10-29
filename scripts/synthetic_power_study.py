"""Does the pipeline find alpha only when it exists? A Monte Carlo on the synthetic generator.

For each seed, three worlds are simulated with the development dataset's moments:
  null         no analyst information, no drift dispersion, no premium -> Sharpe should be ~0
  drift only   persistent winners and losers, analysts uninformed       -> momentum-like, price-chasing breadth works
  informed     analysts (first movers) see the latent fundamental state -> the LMH channel works

    python scripts/synthetic_power_study.py --seeds 40 [--allocator EW]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fmflow import StrategyConfig, run_strategy
from fmflow.analytics.metrics import perf_stats
from fmflow.synthetic import SyntheticConfig, generate

WORLDS = {"null": dict(strength=0.0, stock_drift_sd=0.0, market_drift=0.0), "drift only": dict(strength=0.0),
          "informed": dict(stock_drift_sd=0.0, strength=0.4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--allocator", default="EW", choices=["EW", "IV-GJR"])
    a = ap.parse_args()
    cfg = StrategyConfig.fmf_n().with_(allocator=a.allocator)
    rows = []
    for name, kw in WORLDS.items():
        for s in range(1, a.seeds + 1):
            out = run_strategy(*generate(SyntheticConfig(seed=s, **kw)), cfg)
            st = perf_stats(out.net_returns.iloc[252:])
            rows.append({"world": name, "seed": s, "Sharpe": st["Sharpe"], "Calmar": st["Calmar"]})
            print(f"{name:11s} seed {s}: Sharpe {st['Sharpe']:+.2f}")
    df = pd.DataFrame(rows)
    summ = df.groupby("world")["Sharpe"].agg(["mean", "std", "count"])
    summ["t (mean = 0)"] = summ["mean"] / (summ["std"] / np.sqrt(summ["count"]))
    print("\n", summ.loc[list(WORLDS)].round(2).to_string())


if __name__ == "__main__":
    main()
