"""Reproduce Section 9 of the paper (Section 16 of the research notebook): the frozen FMF-N on the open dataset (never-seen period;
20 of its 30 names never used in development, see DATA_SOURCES.md).

    python scripts/reproduce_open_universe.py                 # re-estimates GJR-GARCH (a few minutes)
    python scripts/reproduce_open_universe.py --fast          # uses the notebook's GARCH forecasts (tests/fixtures)
    python scripts/reproduce_open_universe.py --placebo       # adds the three placebo nulls (slow)

Steps: verify the pre-registration fingerprint -> run the primary (leg fraction 0.4) and secondary (k = 4) rules ->
performance on the pre-registered evaluation window and its halves -> signal ICs -> institutional comparables ->
(optional) placebos -> exploratory FMF-B20 -> parity check against the notebook's reference weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fmflow import StrategyConfig, load_prices, load_ratings, run_strategy
from fmflow.benchmarks import comparable_books
from fmflow.prereg import verify
from fmflow.report import summary_table
from fmflow.robustness import book_from_score, signal_ic_table, strategy_placebo

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="use the notebook's GJR-GARCH forecasts instead of re-estimating")
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "open_universe"))
    a = ap.parse_args()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)

    reg = verify(ROOT / "configs" / "oos_open_universe_preregistration.json")
    print(f"pre-registration {reg['sha256_16']} (registered {reg['registered_utc']}) verified unchanged")
    import re
    start, end = re.findall(r"\d{4}-\d{2}-\d{2}", reg["transfer_choices"]["evaluation_window"])[:2]   # "2024-01-05 .. 2026-08-05"

    prices = load_prices(ROOT / "data" / "open" / "MarketData_Normalized.csv")
    ratings = load_ratings(ROOT / "data" / "open" / "Ratings_Normalized.csv", prices)
    sig = pd.read_csv(ROOT / "tests" / "fixtures" / "open_sig_gjr.csv.gz", index_col=0, parse_dates=True) if a.fast else None

    primary = run_strategy(prices, ratings, StrategyConfig.fmf_n().with_(name="FMF-N primary (leg fraction 0.4)", k=None, leg_fraction=0.4),
                           sig_vol=sig)
    secondary = run_strategy(prices, ratings, StrategyConfig.fmf_n().with_(name="FMF-N secondary (k = 4)"), sig_vol=primary.sig_vol)
    idx = primary.prices.index
    ev = idx[(idx >= start) & (idx <= end)]
    mid = ev[len(ev) // 2]
    win = {"evaluation": slice(ev[0], ev[-1]), "first half": slice(ev[0], mid), "second half": slice(mid, ev[-1])}
    mkt = primary.returns.mean(axis=1)
    b20 = book_from_score(primary.score, primary, net_exposure=0.2)                  # exploratory, not pre-registered

    perf = summary_table({primary.config.name: primary.net_returns, secondary.config.name: secondary.net_returns,
                          "FMF-B20 (exploratory, not pre-registered)": b20, "Equal-weight market": mkt}, win, mkt)
    ic = signal_ic_table(primary, 21, win["evaluation"])
    comp = summary_table({primary.config.name: primary.net_returns, **comparable_books(primary)}, {"evaluation": win["evaluation"]},
                         mkt, reference=primary.net_returns).droplevel(1).sort_values("Sharpe", ascending=False)
    tables = {"performance": perf, "signal_ic21": ic, "comparables": comp}
    if a.placebo:
        tables["placebo"] = strategy_placebo(primary, {"evaluation": win["evaluation"]}, n_iid=300, n_perm=500, shift_step=3)
    for n, t in tables.items():
        t.to_csv(od / f"{n}.csv")
        print(f"\n== {n} ==\n{t.round(3).to_string()}")

    ref = pd.read_csv(ROOT / "tests" / "fixtures" / "open_reference_weights_fmf_n_leg040.csv.gz", index_col=0, parse_dates=True)
    x, y = primary.weights.align(ref, join="outer")
    print(f"\nparity with the research notebook: max |w_package - w_notebook| = {float((x - y).abs().max().max()):.2e}")
    primary.weights.to_csv(od / "weights_primary.csv")


if __name__ == "__main__":
    main()
