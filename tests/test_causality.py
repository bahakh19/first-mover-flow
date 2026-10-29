"""No look-ahead: everything computed on data truncated at T must equal the full-sample result up to T."""
import numpy as np
import pandas as pd
import pytest

from fmflow import run_strategy
from fmflow.pipeline import build_signal_panels, prepare_data


def _truncate(prices, ratings, T):
    return prices[pd.to_datetime(prices["date"]) <= T], ratings[pd.to_datetime(ratings["date"]) <= T]


@pytest.mark.parametrize("frac", [0.45, 0.8])
def test_weights_and_signals_are_causal(synth, ew_cfg, synth_out, frac):
    T = synth_out.prices.index[int(frac * len(synth_out.prices))]
    out_T = run_strategy(*_truncate(*synth, T), ew_cfg)
    pd.testing.assert_frame_equal(out_T.weights, synth_out.weights.loc[:T], check_freq=False, atol=1e-12, rtol=0)
    for k in ("breadth_63", "event_breadth_63", "breadth_resid_mom", "leader_minus_herd"):
        pd.testing.assert_frame_equal(out_T.signals[k], synth_out.signals[k].loc[:T], check_freq=False, atol=1e-12, rtol=0)


def test_truncation_test_has_power(synth, ew_cfg, synth_out):
    """Negative control: a centred (two-sided) smoothing of a signal must FAIL the same test."""
    T = synth_out.prices.index[int(0.6 * len(synth_out.prices))]
    px, rt, snap, _ = prepare_data(*_truncate(*synth, T), ew_cfg)
    leak_T = build_signal_panels(px, snap, ew_cfg)["leader_minus_herd"].rolling(21, center=True, min_periods=1).mean()
    leak = synth_out.signals["leader_minus_herd"].rolling(21, center=True, min_periods=1).mean().loc[:T]
    assert not np.allclose(leak_T.fillna(0).values, leak.fillna(0).values)


def test_truncation_check_api_flags_a_leaky_custom_signal(synth, ew_cfg):
    from fmflow.robustness import truncation_check
    from fmflow.signals.registry import register_signal, unregister_signal
    register_signal("leaky_future", lambda ctx: ctx.px.shift(-5) / ctx.px, overwrite=True)      # uses future prices
    try:
        idx = sorted(synth[0]["date"].unique())
        ok = truncation_check(*synth, ew_cfg, [idx[500]])
        assert ok["causal"].all()
        bad = truncation_check(*synth, ew_cfg.with_(signals=("leader_minus_herd", "leaky_future")), [idx[500]])
        assert not bad.set_index("object").loc["signal:leaky_future", "causal"]
    finally:
        unregister_signal("leaky_future")
