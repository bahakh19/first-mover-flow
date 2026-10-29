"""Does the out-of-sample result depend on the ten development names that re-appear in the open universe?

The open universe (2024-01-05 to 2026-08-05) contains the ten development stocks plus twenty names never used in
development (DATA_SOURCES.md, section 3). This script re-runs the frozen specification on the full universe, on the
twenty new names and on the ten development names, with the shipped GARCH forecasts, and reports performance and the
information coefficient of each signal. It is a post-hoc check, not part of the pre-registration.

    python scripts/overlap_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fmflow import StrategyConfig, load_prices, load_ratings, run_strategy
from fmflow.robustness import signal_ic_table

ROOT = Path(__file__).resolve().parents[1]
EV = slice("2024-01-05", "2026-08-05")
# stock ids of the development names in the open data (alphabetical map, DATA_SOURCES.md section 2.3):
# AAPL, BA, CHGG, INTC, JPM, MSFT, NKE, NVDA, PANW, QCOM
DEVELOPMENT_IDS = [f"stock_{i}" for i in (1, 6, 7, 15, 16, 20, 22, 23, 25, 26)]


def perf(r: pd.Series) -> dict:
    r = r.loc[EV]
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    cagr = float(eq.iloc[-1] ** (252 / len(r)) - 1)
    return {"Sharpe": r.mean() / r.std() * 252 ** 0.5, "CAGR": cagr, "MaxDD": dd, "Calmar": cagr / -dd}


def main() -> int:
    P = load_prices(ROOT / "data/open/MarketData_Normalized.csv")
    R = load_ratings(ROOT / "data/open/Ratings_Normalized.csv", P)
    sig = pd.read_csv(ROOT / "tests/fixtures/open_sig_gjr.csv.gz", index_col=0, parse_dates=True)
    cfg = StrategyConfig.fmf_n_scalable()
    new_ids = sorted(set(P["stock"]) - set(DEVELOPMENT_IDS), key=lambda s: int(s.split("_")[1]))
    rows, ics = {}, {}
    for name, ids in (("all 30 names (pre-registered)", sorted(set(P["stock"]))), ("20 new names", new_ids),
                      ("10 development names", DEVELOPMENT_IDS)):
        out = run_strategy(P[P["stock"].isin(ids)], R[R["stock"].isin(ids)], cfg, sig_vol=sig[ids])
        rows[name] = perf(out.net_returns)
        t = signal_ic_table(out, 21, EV)
        ics[name] = t["IC"]
        rows[name]["LMH IC21"] = t.loc["leader_minus_herd", "IC"]
        rows[name]["LMH p (one-sided)"] = t.loc["leader_minus_herd", "p (one-sided)"]
    pd.set_option("display.width", 160)
    print(pd.DataFrame(rows).T.round(3).to_string())
    print("\nIC21 of every signal:\n" + pd.DataFrame(ics).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
