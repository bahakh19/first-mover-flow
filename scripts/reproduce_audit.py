"""Reproduce Sections 10-11 of the paper (Section 17 of the research notebook): every anticipated objection and every
small-sample test -- identity vs timing, implementation realism, the multiverse, data-snooping-robust tests, Bayesian
inference, power, and generative stress testing.

    python scripts/reproduce_audit.py [--blocks 11 12 13 14 15 16] [--out results/audit]

Needs data/private/MarketData.csv and Ratings.csv (the development universe, not distributed) and the shipped open
universe. The blocks in scripts/audit/ are byte-identical to the notebook cells; this runner only recreates the globals
the notebook would have (raw tables and GARCH forecasts, cached on disk after the first run). Section 16 needs JAX.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from fmflow.cache import cached_garch
from fmflow.config import DataConfig
from fmflow.data.cleaning import clean_prices

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = {10: "c10_setup.py", 11: "c11_identity.py", 12: "c12_implementation.py", 13: "c13_multiverse.py",
          14: "c14_bayes.py", 15: "c15_power.py", 16: "c16_generative.py"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="*", type=int, default=[11, 12, 13, 14, 15, 16])
    ap.add_argument("--out", default=str(ROOT / "results" / "audit"))
    a = ap.parse_args()
    if not (ROOT / "data/private/MarketData.csv").exists():
        raise SystemExit("data/private/MarketData.csv and Ratings.csv are required (see DATA.md)")
    g: dict = {"display": lambda x: print(x.round(3).to_string() if hasattr(x, "round") else x)}
    g["prices_raw"] = pd.read_csv(ROOT / "data/private/MarketData.csv")
    g["ratings_raw"] = pd.read_csv(ROOT / "data/private/Ratings.csv", dtype=str)
    g["OOS_PRICES_RAW"] = pd.read_csv(ROOT / "data/open/MarketData_Normalized.csv")
    g["OOS_RATINGS_RAW"] = pd.read_csv(ROOT / "data/open/Ratings_Normalized.csv", dtype=str)
    cfg = DataConfig()
    g["SIG_GJR"] = cached_garch(clean_prices(g["prices_raw"], cfg).pct_change(fill_method=None), cfg, ROOT / ".fmflow_cache")
    g["OOS_SIG"] = pd.read_csv(ROOT / "tests/fixtures/open_sig_gjr.csv.gz", index_col=0, parse_dates=True)
    exec("import numpy as np, pandas as pd", g)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    order = [b for b in (10, 11, 12, 13, 14, 15, 16) if b == 10 or b in a.blocks]
    if 16 in order and 14 not in order:
        order.insert(order.index(16), 14)            # the generative block draws drift from the Block-14 meta-analysis
    for b in order:
        print(f"\n######## Section 17.{b - 10}: {BLOCKS[b]}", flush=True)
        exec((ROOT / "scripts" / "audit" / BLOCKS[b]).read_text(encoding="utf-8"), g)
        (od / "audit_results.pkl").write_bytes(pickle.dumps(g["AUD"]))
    print(f"\nwrote {od}/audit_results.pkl")


if __name__ == "__main__":
    main()
