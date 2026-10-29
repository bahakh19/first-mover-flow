"""Reproduce the development / holdout results (Sections 5-7 of the paper) from the original 10-stock files.

The development dataset is NOT distributed with this repository. Place MarketData.csv and Ratings.csv (same contract
as DATA.md) in data/private/ and run:

    python scripts/reproduce_development.py [--fast] [--placebo] [--loo]

--fast reuses GARCH forecasts cached in results/development/sig_gjr.csv.gz after a first full run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fmflow import StrategyConfig, load_prices, load_ratings, run_strategy
from fmflow.benchmarks import comparable_books
from fmflow.report import summary_table
from fmflow.robustness import leave_one_out, signal_ic_table, strategy_placebo

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {"DEV 2017-19": slice("2017-01-01", "2019-12-31"), "Holdout 2020-21": slice("2020-01-01", "2021-12-31"),
           "Full 2017-21": slice("2017-01-01", "2021-12-31")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--loo", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "development"))
    a = ap.parse_args()
    src = ROOT / "data" / "private"
    if not (src / "MarketData.csv").exists():
        raise SystemExit("data/private/MarketData.csv and Ratings.csv are required (not distributed; see DATA.md)")
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    cache = od / "sig_gjr.csv.gz"
    sig = pd.read_csv(cache, index_col=0, parse_dates=True) if (a.fast and cache.exists()) else None

    prices = load_prices(src / "MarketData.csv")
    ratings = load_ratings(src / "Ratings.csv", prices)
    fmf = run_strategy(prices, ratings, StrategyConfig.fmf_n(), sig_vol=sig)
    fmf.sig_vol.to_csv(cache)
    b20 = run_strategy(prices, ratings, StrategyConfig.fmf_b20(), sig_vol=fmf.sig_vol)
    mkt = fmf.returns.mean(axis=1)

    tables = {
        "performance": summary_table({"FMF-N": fmf.net_returns, "FMF-B20": b20.net_returns, "Equal-weight market": mkt}, WINDOWS, mkt),
        "signal_ic21_dev": signal_ic_table(fmf, 21, WINDOWS["DEV 2017-19"]),
        "comparables": summary_table({"FMF-N": fmf.net_returns, **comparable_books(fmf)}, {"Full 2017-21": WINDOWS["Full 2017-21"]}, mkt,
                                     reference=fmf.net_returns).droplevel(1).sort_values("Sharpe", ascending=False),
    }
    if a.placebo:
        tables["placebo"] = strategy_placebo(fmf, WINDOWS)
    if a.loo:
        tables["leave_one_out"] = leave_one_out(prices, ratings, StrategyConfig.fmf_n(), WINDOWS, sig_vol=fmf.sig_vol)
    for n, t in tables.items():
        t.to_csv(od / f"{n}.csv")
        print(f"\n== {n} ==\n{t.round(3).to_string()}")

    ref = pd.read_csv(ROOT / "tests" / "fixtures" / "development_reference_weights_fmf_n.csv.gz", index_col=0, parse_dates=True)
    x, y = fmf.weights.align(ref, join="outer")
    print(f"\nparity with the research notebook: max |w_package - w_notebook| = {float((x - y).abs().max().max()):.2e}")


if __name__ == "__main__":
    main()
