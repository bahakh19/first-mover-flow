import numpy as np
import pandas as pd
import pytest

from fmflow.generative import fidelity as F
from fmflow.generative import jax_available, jax_status
from fmflow.generative import stress as S
from fmflow.generative.baselines import block_bootstrap_paths
from fmflow.generative.windows import VolScaled, make_windows


def _frames(seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for n in (400, 300):
        idx = pd.bdate_range("2020-01-01", periods=n)
        m = rng.standard_t(4, n) * 0.01
        out.append(pd.DataFrame({"S": 0.3 * m + rng.normal(0, 0.005, n), "M": m}, index=idx))
    return out


def test_windows_and_volscaled_roundtrip():
    wd = make_windows(_frames(), length=32)
    assert wd.X.shape == (369 + 269, 32, 2)
    vs = VolScaled(wd.X)
    back = vs.decode(vs.encode(wd.X))
    assert np.allclose(back.std(1), wd.X.std(1), rtol=1e-4)


def test_bootstrap_paths_and_fidelity_metrics():
    f = pd.concat(_frames())
    P = block_bootstrap_paths(f, 50, 252, 21, 0)
    assert P.shape == (50, 252, 2)
    W = P[:, :224].reshape(-1, 32, 2)
    sf = F.stylised_facts(W, ["S", "M"])
    assert abs(sf["corr S-M"] - f.corr().iloc[0, 1]) < 0.15
    real = make_windows(_frames(), 32).X
    assert 0 <= F.discriminative_score(real, real[::-1].copy()) <= 0.5
    assert F.memorisation_ratio(real[:300], real[400:], real[:300]) < 0.05       # copying the training set is flagged
    st = S.path_stats(P)
    assert {"Calmar", "MaxDD", "Sharpe"} <= set(st.columns) and (st["MaxDD"] <= 0).all()


NEEDS_JAX = pytest.mark.skipif(not jax_available(), reason=jax_status()[1])


def test_jax_status_is_a_clean_diagnosis():
    ok, msg = jax_status()
    assert isinstance(ok, bool) and isinstance(msg, str) and msg


@pytest.mark.slow
@NEEDS_JAX
def test_ddpm_learns_a_gaussian_and_conditions():
    from fmflow.generative.ddpm import DDPM
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 8, 2)).astype(np.float32)
    X[:, :, 1] = 0.8 * X[:, :, 0] + 0.6 * X[:, :, 1]                    # correlated channels
    m = DDPM(8, 2, T=100, hidden=128, n_blocks=2, whiten=False).fit(X, steps=1500)
    G = m.sample(1000)
    assert abs(G.std() - X.std()) < 0.15 and abs(np.corrcoef(G[:, :, 0].ravel(), G[:, :, 1].ravel())[0, 1] - 0.8) < 0.15
    known = np.zeros((8, 2), np.float32)
    known[:, 0] = 2.0
    mask = np.zeros((8, 2), bool)
    mask[:, 0] = True
    Gc = m.sample(500, known=known, known_mask=mask)
    assert np.allclose(Gc[:, :, 0], 2.0) and Gc[:, :, 1].mean() > 0.8           # E[c1 | c0 = 2] = 1.6


@pytest.mark.slow
@NEEDS_JAX
def test_diffusion_fhs_runs():
    from fmflow.generative.diffusion_fhs import DiffusionFHS
    g = DiffusionFHS(["S", "M"], T=50, hidden=64, n_blocks=2).fit(_frames(), steps=400)
    P = g.simulate(20, 100, seed=1, steps=10)
    assert P.shape == (20, 100, 2) and np.isfinite(P).all()
