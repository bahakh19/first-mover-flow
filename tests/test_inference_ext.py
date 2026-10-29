import numpy as np
import pandas as pd

from fmflow.analytics.attribution import lo_decomposition, mean_with_hac, static_timing_books, stock_contributions
from fmflow.analytics.bayes import bayes_factor_normal, bayesian_sharpe, random_effects_meta, shrink_normal_means
from fmflow.analytics.metrics import conditional_betas, lo_adjusted_sharpe
from fmflow.analytics.power import ic_power, mde_ic
from fmflow.analytics.snooping import reality_check, romano_wolf, sharpe_diff_test, spa_test


def _family(seed=0, T=800, K=6, edge=0.0015):
    rng = np.random.default_rng(seed)
    D = pd.DataFrame(rng.normal(0, 0.01, (T, K)))
    D[0] += edge
    return D


def test_spa_detects_a_real_edge_and_not_noise():
    assert spa_test(_family(), n_boot=400)["p_value"] < 0.05
    assert spa_test(_family(edge=0.0, seed=3), n_boot=400)["p_value"] > 0.05
    assert reality_check(_family(), n_boot=400)["p_value"] < 0.05


def test_romano_wolf_controls_family_and_rejects_true_edge():
    rw = romano_wolf(_family(), n_boot=400)
    assert bool(rw.loc[0, "reject"]) and (rw.drop(index=0)["p_romano_wolf"] > 0.05).all()
    assert (rw["p_romano_wolf"] >= rw["p_raw"] - 1e-12).all()


def test_sharpe_diff_test_symmetric_and_powerful_for_correlated_pairs():
    D = _family()
    r1 = D[0]
    r2 = r1 - 0.0012 + 0.002 * D[2]                      # same risk, lower mean: a precisely measurable difference
    a, b = sharpe_diff_test(r1, r2, n_boot=300), sharpe_diff_test(r2, r1, n_boot=300)
    assert np.isclose(a["sharpe_diff"], -b["sharpe_diff"]) and a["p_value"] < 0.05
    assert sharpe_diff_test(D[1], D[2], n_boot=300)["p_value"] > 0.05


def test_bayesian_sharpe_prior_shrinks_towards_zero():
    r = _family()[0]
    flat = bayesian_sharpe(r, prior_sr_sd=None, n_draws=5000).summary()
    scep = bayesian_sharpe(r, prior_sr_sd=0.5, n_draws=5000).summary()
    naive = r.mean() / r.std() * np.sqrt(252)
    assert abs(flat["mean"] - naive) < 0.6 and scep["mean"] < flat["mean"] and 0 <= scep["P(>0)"] <= 1


def test_meta_analysis_pools_and_predicts():
    m = random_effects_meta([0.1, 0.12, 0.09], [0.05, 0.05, 0.05], ["a", "b", "c"], n_draws=5000)
    s = m["mu"].summary()
    assert 0.07 < s["mean"] < 0.13 and m["P(mu>0)"] > 0.97 and m["P(theta_new>0)"] <= m["P(mu>0)"] + 1e-9


def test_shrinkage_and_bayes_factor():
    t = pd.Series([2.5, 0.3, -0.4, 1.0, -1.2, 0.1, 0.8, -2.0])
    sh = shrink_normal_means(t)
    m = t.mean()
    assert ((sh["shrunk_t"] - m).abs() <= (sh["t"] - m).abs() + 1e-12).all()      # shrinks towards the family mean
    assert bayes_factor_normal(0.1, 0.03, 0.1) > 10 > bayes_factor_normal(0.0, 0.03, 0.1)


def test_lo_decomposition_and_static_timing_identity():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.DataFrame(rng.normal(0.0005, 0.01, (300, 4)), index=idx, columns=list("abcd"))
    w = pd.DataFrame(rng.normal(0, 0.1, (300, 4)), index=idx, columns=list("abcd")) + [0.2, 0.1, -0.1, -0.2]
    d = lo_decomposition(w, r)
    assert np.isclose(d["passive"] + d["active"], d["total"])
    b = static_timing_books(w, r)
    assert np.allclose((b["static"] + b["timing"]).iloc[30:], b["total"].iloc[30:])
    assert "held_share" in stock_contributions(w, r).columns and np.isfinite(mean_with_hac(b["total"])["t_hac"])


def test_lo_adjusted_sharpe_and_conditional_betas():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=600)
    m = pd.Series(rng.normal(0, 0.01, 600), index=idx)
    s = 0.5 * m.clip(upper=0) + pd.Series(rng.normal(0.0005, 0.005, 600), index=idx)
    cb = conditional_betas(s, m)
    assert cb["beta_down"] > cb["beta_up"]
    assert np.isfinite(lo_adjusted_sharpe(s)["sharpe_lo"])


def test_power_formulas():
    assert mde_ic(10, 36) > mde_ic(30, 36) > mde_ic(30, 120)
    assert abs(ic_power(mde_ic(20, 60), 20, 60) - 0.8) < 1e-9
