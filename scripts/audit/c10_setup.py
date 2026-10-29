# Section 17.0 — Audit set-up: the companion package `fmflow` re-runs the frozen strategy from the RAW tables, so every
# audit below is computed on exactly the book the notebook built (parity asserted when the notebook objects exist).
import sys, importlib.util
from pathlib import Path
if importlib.util.find_spec("fmflow") is None:                      # repository layout: notebooks/ next to src/
    for _p in (Path("../src"), Path("src")):
        if (_p / "fmflow").exists():
            sys.path.insert(0, str(_p.resolve()))
from fmflow import StrategyConfig, run_strategy
from fmflow.analytics.metrics import perf_stats, capm, lo_adjusted_sharpe, conditional_betas
from fmflow.benchmarks import comparable_books

AUD = {}                                                              # every audit result, for the paper
W_AUD = {"DEV": slice("2017-01-01", "2019-12-31"), "HOL": slice("2020-01-01", "2021-12-31"), "FULL": slice("2017-01-01", "2021-12-31")}
W_OOS = {"OOS": slice("2024-01-05", "2026-08-05")}
DEV_OUT = run_strategy(prices_raw, ratings_raw, StrategyConfig.fmf_n(), sig_vol=SIG_GJR)
OOS_OUT = run_strategy(OOS_PRICES_RAW, OOS_RATINGS_RAW, StrategyConfig.fmf_n().with_(k=None, leg_fraction=0.4), sig_vol=OOS_SIG)
if "W_V2" in globals():
    _a, _b = DEV_OUT.weights.align(W_V2, join="inner")
    assert float((_a - _b).abs().max().max()) < 1e-9, "package and notebook books differ"
    print("Parity: package book == notebook book (development).")
for _nm, _o, _w in (("development", DEV_OUT, W_AUD["FULL"]), ("never-seen", OOS_OUT, W_OOS["OOS"])):
    _s = perf_stats(_o.net_returns.loc[_w])
    print(f"{_nm:12s} Sharpe {_s['Sharpe']:.2f}  Calmar {_s['Calmar']:.2f}")
