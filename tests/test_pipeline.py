import numpy as np
import pandas as pd

from fmflow import run_strategy
from fmflow.benchmarks import comparable_books
from fmflow.report import default_windows, strategy_report


def test_output_contract(synth_out):
    o = synth_out
    assert o.weights.shape == o.prices.shape and np.isfinite(o.weights.values).all()
    assert np.allclose(o.weights.sum(axis=1), 0.0, atol=1e-12)              # dollar-neutral every session
    assert (o.weights.abs().sum(axis=1) <= 1 + 1e-12).all()
    assert np.allclose(o.result.net, o.result.gross - o.result.costs)
    assert set(o.signals) >= {"breadth_63", "event_breadth_63", "breadth_resid_mom", "leader_minus_herd"}
    rep = strategy_report(o, default_windows(o.prices.index))
    assert {"Sharpe", "Calmar", "MaxDD", "beta", "alpha t", "turnover"} <= set(rep.columns)


def test_beta_budget_and_leg_fraction(synth, ew_cfg, synth_out):
    o = run_strategy(*synth, ew_cfg.with_(net_exposure=0.2))
    live = synth_out.weights.abs().sum(axis=1) > 0.99
    assert np.allclose(o.weights.sum(axis=1)[live], 0.2, atol=0.02)
    f = run_strategy(*synth, ew_cfg.with_(k=None, leg_fraction=0.4))
    assert np.allclose(f.weights.sum(axis=1), 0.0, atol=1e-12)


def test_comparables(synth_out):
    books = comparable_books(synth_out)
    assert len(books) == 16 and all(isinstance(v, pd.Series) for v in books.values())
    assert all(np.isfinite(v.dropna()).all() for v in books.values())
