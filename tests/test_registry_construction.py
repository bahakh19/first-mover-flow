import numpy as np
import pandas as pd
import pytest

from fmflow import StrategyConfig, run_strategy
from fmflow.portfolio.construction import beta_neutral_budgets, cap_weights, ex_ante_vol, ledoit_wolf_cov, rolling_betas, sector_demean
from fmflow.signals.registry import available_signals, register_signal, unregister_signal


def test_builtin_registry_contains_fmf_and_comparables():
    s = set(available_signals())
    assert {"breadth_63", "leader_minus_herd", "mom_12_1", "tp_upside"} <= s


def test_custom_signal_end_to_end(synth, ew_cfg):
    @register_signal("test_neg_rev", overwrite=True)
    def _(ctx):
        return np.log(ctx.px).diff(21)
    try:
        out = run_strategy(*synth, ew_cfg.with_(signals=("leader_minus_herd", "test_neg_rev")))
        assert "test_neg_rev" in out.signals and out.weights.abs().sum(axis=1).max() > 0
    finally:
        unregister_signal("test_neg_rev")
    with pytest.raises(ValueError):
        StrategyConfig.fmf_n().with_(signals=("does_not_exist",)).validate()


def test_beta_neutral_budgets_zero_target_beta():
    idx = pd.bdate_range("2021-01-01", periods=3)
    t = pd.DataFrame([[0.25, 0.25, -0.25, -0.25]] * 3, index=idx, columns=list("abcd"))
    b = pd.DataFrame([[1.5, 1.5, 0.8, 0.8]] * 3, index=idx, columns=list("abcd"))
    w = beta_neutral_budgets(t, b)
    assert np.allclose((w * b).sum(axis=1), 0) and np.allclose(w.abs().sum(axis=1), 1.0)


def test_cap_weights_respects_cap_and_leg_budgets():
    idx = pd.bdate_range("2021-01-01", periods=2)
    t = pd.DataFrame([[0.4, 0.05, 0.05, -0.3, -0.1, -0.1]] * 2, index=idx, columns=list("abcdef"))
    w = cap_weights(t, 0.2)
    assert (w.abs() <= 0.2 + 1e-12).all().all()
    assert np.allclose(w.clip(lower=0).sum(axis=1), 0.5) and np.allclose((-w).clip(lower=0).sum(axis=1), 0.5)


def test_ledoit_wolf_is_psd_and_vol_scales_linearly():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, (60, 8))
    assert np.linalg.eigvalsh(ledoit_wolf_cov(x)).min() > 0
    idx = pd.bdate_range("2021-01-01", periods=200)
    ret = pd.DataFrame(rng.normal(0, 0.01, (200, 4)), index=idx, columns=list("abcd"))
    w = pd.DataFrame([[0.25, 0.25, -0.25, -0.25]] * 200, index=idx, columns=list("abcd"))
    v1, v2 = ex_ante_vol(w, ret, 60), ex_ante_vol(2 * w, ret, 60)
    assert np.allclose(v2.dropna(), 2 * v1.dropna())


def test_rolling_betas_are_causal():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=400)
    ret = pd.DataFrame(rng.normal(0, 0.01, (400, 5)), index=idx)
    full, cut = rolling_betas(ret), rolling_betas(ret.iloc[:300])
    pd.testing.assert_frame_equal(full.iloc[:300], cut)


def test_sector_demean():
    s = pd.DataFrame([[1.0, 3.0, 5.0, 7.0]], columns=list("abcd"))
    out = sector_demean(s, {"a": "x", "b": "x", "c": "y", "d": "y"})
    assert out.iloc[0].tolist() == [-1.0, 1.0, -1.0, 1.0]


def test_layers_change_the_book_but_default_is_unchanged(synth, ew_cfg, synth_out):
    base = run_strategy(*synth, ew_cfg)
    pd.testing.assert_frame_equal(base.weights, synth_out.weights)
    vt = run_strategy(*synth, ew_cfg.with_(vol_target=0.05, max_leverage=3.0))
    assert not np.allclose(vt.weights.fillna(0).values, base.weights.fillna(0).values)
    bn = run_strategy(*synth, ew_cfg.with_(neutralize="beta"))
    assert np.allclose(bn.target.abs().sum(axis=1)[bn.target.abs().sum(axis=1) > 0], 1.0)
