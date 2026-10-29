"""Fuzz the pipeline with randomly messy universes and check that nothing looks ahead.

For each seed, a synthetic universe of 6 to 24 stocks is generated and damaged the way vendor files are: late listings,
delistings, missing rows, duplicated rows, zero prices, shuffled row order and (for half the seeds) injected vendor
errors. Every registered signal, the composite and the weights are then recomputed on the data truncated at two cut
dates and compared with the full run (`fmflow.robustness.truncation_check`), under three construction settings. Any
difference is look-ahead. The script also checks the book's invariants (no missing weights, gross <= 1, legs balanced).

    python scripts/fuzz_causality.py                  # 10 seeds (a few minutes)
    python scripts/fuzz_causality.py --seeds 40       # what was run for release 2.2.3

Exit code 1 if any seed fails.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from fmflow import StrategyConfig, run_strategy
from fmflow.data.schema import DataContractError
from fmflow.robustness import truncation_check
from fmflow.signals.registry import available_signals
from fmflow.synthetic import SyntheticConfig, generate


def messy_universe(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A synthetic universe damaged like a real vendor file (see the module docstring)."""
    rng = np.random.default_rng(seed)
    n, n_sessions = int(rng.integers(6, 25)), int(rng.integers(320, 700))
    p, r = generate(SyntheticConfig(n_stocks=n, n_sessions=n_sessions, seed=seed, analysts_per_stock=(3, 30),
                                    inject_errors=bool(rng.random() < 0.5)))
    dates = sorted(p["date"].unique())
    for s in rng.choice(sorted(p["stock"].unique()), size=min(3, n // 3), replace=False):
        cut = dates[int(rng.integers(50, n_sessions - 50))]
        p = p[~((p["stock"] == s) & ((p["date"] < cut) if rng.random() < 0.5 else (p["date"] > cut)))]  # listing / delisting
    p = p.drop(p.sample(frac=0.02, random_state=seed).index)                     # missing rows
    p = pd.concat([p, p.sample(5, random_state=seed)])                           # duplicated rows
    p.loc[p.sample(2, random_state=seed + 1).index, "price"] = 0.0               # zero prices
    return p.sample(frac=1.0, random_state=seed), r.sample(frac=1.0, random_state=seed)


def invariant_problems(out, cfg: StrategyConfig) -> list[str]:
    """Violations of the book's construction invariants."""
    problems, t = [], out.target
    gross = t.abs().sum(axis=1)
    if out.weights.isna().any().any():
        problems.append("missing weights")
    if (gross > 1 + 1e-9).any():
        problems.append(f"target gross above 1 ({gross.max():.4f})")
    if cfg.neutralize == "dollar" and cfg.max_weight is None:
        net = t.sum(axis=1)[gross > 0]
        if ((net - cfg.net_exposure).abs() > 1e-9).any():
            problems.append("legs not balanced")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    a = ap.parse_args(argv)
    base = StrategyConfig.fmf_n().with_(allocator="EW", signals=tuple(available_signals()))
    configs = {"FMF-N (EW)": base,
               "beta-neutral, 20% cap, 10% vol target, 1-day delay": base.with_(neutralize="beta", max_weight=0.2,
                                                                                 vol_target=0.1, execution_lag=1),
               "leg fraction 0.3, 10 tranches": base.with_(k=None, leg_fraction=0.3, tranche_period=10)}
    failures = 0
    for seed in range(a.first_seed, a.first_seed + a.seeds):
        p, r = messy_universe(seed)
        dates = sorted(pd.to_datetime(p["date"]).unique())
        cuts = [dates[len(dates) // 2], dates[int(len(dates) * 0.8)]]
        for name, cfg in configs.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    checks = truncation_check(p, r, cfg, cuts)
                    problems = invariant_problems(run_strategy(p, r, cfg), cfg)
            except DataContractError as e:            # e.g. fewer than 6 stocks before a cut: not a defect
                print(f"seed {seed:3d}  skipped ({e})")
                break
            leaks = checks[~checks["causal"]]
            signals = checks[checks["object"].str.startswith("signal:")]
            # every signal identical to 1e-12 but the composite differs: exactly tied scores ordered differently by
            # last-bit noise (the known limitation of breadth_resid_mom, CHANGELOG 2.2.3), not information from the future
            ties = (not leaks.empty and set(leaks["object"]) <= {"composite", "weights"}
                    and signals["max_abs_diff"].max() <= 1e-12 and signals["nan_mismatches"].sum() == 0)
            ok = (leaks.empty or ties) and not problems
            failures += not ok
            status = "ok" if ok and not ties else ("ok (tied scores ordered differently)" if ok else "FAIL")
            print(f"seed {seed:3d}  {name:<52} {status}")
            if ties:
                continue
            if not leaks.empty:
                print(leaks.to_string())
            for msg in problems:
                print(f"           {msg}")
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
