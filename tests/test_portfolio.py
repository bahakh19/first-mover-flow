import numpy as np
import pandas as pd
import pytest

from fmflow.portfolio.construction import target_weights, tranche_weights


@pytest.fixture
def score():
    rng = np.random.default_rng(3)
    s = pd.DataFrame(rng.normal(size=(200, 10)), index=pd.bdate_range("2021-01-01", periods=200), columns=[f"s{i}" for i in range(10)])
    s.iloc[:5] = np.nan                       # warm-up rows -> flat book
    s.iloc[50:60, :4] = np.nan                # only 6 valid names -> k = 4 infeasible -> flat
    return s


def test_dollar_neutral_gross_and_leg_size(score):
    W = target_weights(score, k=4)
    valid = score.notna().sum(axis=1) >= 8
    assert np.allclose(W.sum(axis=1), 0.0, atol=1e-12)
    assert np.allclose(W.abs().sum(axis=1)[valid], 1.0) and (W[~valid] == 0).all().all()
    assert ((W > 0).sum(axis=1)[valid] == 4).all() and ((W < 0).sum(axis=1)[valid] == 4).all()
    top = score.rank(axis=1, ascending=False) <= 4
    assert (W[valid][top[valid]] > 0).sum().sum() == 4 * valid.sum()   # longs are exactly the top-ranked names


def test_inverse_vol_weights_within_leg(score):
    vol = pd.DataFrame(np.tile(np.linspace(0.01, 0.03, 10), (200, 1)), index=score.index, columns=score.columns)
    W = target_weights(score, k=4, allocator="IV-GJR", sig_vol=vol, ret=vol * 0)
    row = W.iloc[100]
    longs = row[row > 0]
    assert np.allclose(longs * vol.iloc[100][longs.index], (longs * vol.iloc[100][longs.index]).iloc[0])  # w ~ 1/vol
    assert longs.sum() == pytest.approx(0.5)


def test_net_exposure_and_leg_fraction(score):
    W = target_weights(score, k=4, net_exposure=0.2)
    valid = score.notna().sum(axis=1) >= 8
    assert np.allclose(W.sum(axis=1)[valid], 0.2)
    Wf = target_weights(score, k=None, leg_fraction=0.4, k_min=2)
    n = score.notna().sum(axis=1)
    kk = np.maximum(2, np.floor(0.4 * n))
    ok = (n >= 2 * kk) & (n >= 4)
    assert ((Wf > 0).sum(axis=1)[ok] == kk[ok]).all()
    full = n == 10
    assert np.allclose(Wf[full], target_weights(score, k=4)[full])   # identical to k = 4 when 10 names are valid


def test_tranche_identity(score):
    T = target_weights(score, k=4).fillna(0)
    f = 7

    def held(o):
        W = T.copy()
        W.loc[(np.arange(len(W)) - o) % f != 0] = np.nan
        return W.ffill().fillna(0.0)

    explicit = sum(held(o) for o in range(f)) / f
    assert np.allclose(explicit, tranche_weights(T, f), atol=1e-12)
