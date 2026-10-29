import numpy as np
import pytest

from fmflow import StrategyConfig, run_strategy
from fmflow.analytics.metrics import perf_stats
from fmflow.synthetic import SyntheticConfig, describe_dataset, generate


def test_reproducible_and_contract():
    p1, r1 = generate(SyntheticConfig(seed=3, n_sessions=300))
    p2, r2 = generate(SyntheticConfig(seed=3, n_sessions=300))
    assert p1.equals(p2) and r1.equals(r2)
    assert list(p1.columns) == ["date", "stock", "price"] and list(r1.columns) == ["date", "estimate_value", "stock", "analyst"]


def test_calibrated_moments():
    d = describe_dataset(*generate(SyntheticConfig()))
    assert 0.3 < d["avg pairwise corr"] < 0.5                       # development data: 0.40
    assert 0.9 < d["median target/price"] < 1.4                      # 1.17
    assert 0.15 < d["share reiteration"] < 0.35 and d["share up"] > d["share down"]
    assert 120 < d["event clusters (>=5 same-day)"] < 320            # 214
    o = describe_dataset(*generate(SyntheticConfig.open_universe_like()))
    assert o["stocks"] == 30 and o["stocks per analyst (median)"] >= 3   # sparse, multi-stock coverage
    assert 1500 < o["ratings"] < 3500                                 # open universe: 2,363
    assert 8 <= o["analysts per stock (median)"] <= 18               # 13
    assert 60 < o["event clusters (>=5 same-day)"] < 220             # 109


def test_injected_errors_are_cleaned():
    p, r = generate(SyntheticConfig(n_sessions=400, inject_errors=True))
    out = run_strategy(p, r, StrategyConfig.fmf_n().with_(allocator="EW"))
    a = out.audit.set_index("finding")["n_rows"]
    assert a["TP/P outside (0.2, 5.0)"] > 0 and a["Missing analyst identifier"] > 0
    assert a["Unparseable / non-positive target"] > 0 and a["Duplicate (date, stock, analyst)"] > 0


@pytest.mark.slow
def test_finds_alpha_only_when_it_exists():
    cfg = StrategyConfig.fmf_n().with_(allocator="EW")
    def sharpe(c):
        return perf_stats(run_strategy(*generate(c), cfg).net_returns.iloc[252:])["Sharpe"]
    null = [sharpe(SyntheticConfig.null(seed=s)) for s in range(1, 7)]
    info = [sharpe(SyntheticConfig(stock_drift_sd=0.0, strength=0.4, seed=s)) for s in range(1, 7)]
    assert abs(np.mean(null)) < 0.5                                 # no alpha found where none exists
    assert np.mean(info) > np.mean(null) + 0.4                       # alpha found where analysts are informed
