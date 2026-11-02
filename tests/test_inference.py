import numpy as np
import pandas as pd
import pytest

from fmflow.analytics.inference import (
    cscv_pbo,
    deflated_sharpe,
    holm,
    ic_phase_test,
    memmel_test,
    min_track_record,
    psr,
    stationary_bootstrap_indices,
)


def test_psr_and_dsr():
    rng = np.random.default_rng(1)
    assert psr(rng.normal(0, 0.01, 756)) < 0.95
    assert psr(rng.normal(0.0012, 0.01, 756)) > 0.99
    trials = pd.DataFrame(rng.normal(0, 0.01, (756, 200)))
    best = trials.loc[:, (trials.mean() / trials.std()).idxmax()]
    d = deflated_sharpe(best.values, trials=trials)
    assert d["PSR"] > d["DSR"] and d["DSR"] < 0.9                    # the best of 200 noise strategies does not survive
    assert min_track_record(rng.normal(0.001, 0.01, 756)) < np.inf


def test_holm_monotone_and_conservative():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    h = holm(p)
    assert (h.values == np.maximum.accumulate(h.values)).all() and (h >= p[h.index]).all()
    assert h["a"] == pytest.approx(0.04)


def test_phase_test_size_under_overlap():
    rng = np.random.default_rng(5)
    rej = 0
    for _ in range(300):
        e = rng.normal(size=760 + 21)
        ic = pd.Series(np.convolve(e, np.ones(21) / 21, mode="valid")[:760] * 0.3)   # 21-session overlapping labels
        rej += ic_phase_test(ic, 21)["p"] < 0.05
    assert rej / 300 < 0.09                                          # calibrated; Newey-West with 21 lags is ~0.09 here


def test_pbo_noise_and_memmel():
    rng = np.random.default_rng(2)
    noise = pd.DataFrame(rng.normal(0, 0.01, (800, 12)))
    assert 0.2 < cscv_pbo(noise, n_blocks=8)["PBO"] < 0.8
    a, b = pd.Series(rng.normal(0.001, 0.01, 1000)), pd.Series(rng.normal(0.0, 0.01, 1000))
    z1, p1 = memmel_test(a, b)
    z2, p2 = memmel_test(b, a)
    assert z1 == pytest.approx(-z2) and p1 == pytest.approx(p2)


def test_bootstrap_indices():
    idx = stationary_bootstrap_indices(100, 50, 10, seed=0)
    assert idx.shape == (50, 100) and idx.min() >= 0 and idx.max() < 100
