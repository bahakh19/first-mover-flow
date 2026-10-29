# Section 17.4 — Objection 6: "Ten stocks and five years cannot prove anything." Small-sample inference done properly:
# (a) data-snooping-robust tests against the institutional toolkit, (b) Bayesian posteriors of the Sharpe ratio under
# sceptical priors, (c) random-effects meta-analysis across the three independent samples, (d) empirical-Bayes
# shrinkage of the 18-candidate search, (e) Bayes factors.
from fmflow.analytics.snooping import romano_wolf, spa_test, sharpe_diff_test
from fmflow.analytics.bayes import bayesian_sharpe, random_effects_meta, shrink_normal_means, bayes_factor_normal
from fmflow.analytics.inference import psr
from fmflow.robustness import signal_ic_table
from scipy import stats as _st

# (a) FMF-N against each institutional book: H0_k "FMF-N does not beat book k" (vol-matched returns), FWER-controlled.
SNOOP = {}
for nm, o, w in (("dev FULL", DEV_OUT, W_AUD["FULL"]), ("oos", OOS_OUT, W_OOS["OOS"])):
    books = comparable_books(o)
    f = o.net_returns.loc[w]
    scale = lambda x: x / x.std() * f.std()                       # equal-volatility comparison (Sharpe-difference test)
    D = pd.DataFrame({k: f - scale(v.loc[w]) for k, v in books.items()}).dropna()
    rw = romano_wolf(D, n_boot=2000)
    spa_rev = spa_test(pd.DataFrame({k: scale(v.loc[w]) - f for k, v in books.items()}).dropna(), n_boot=2000)
    SNOOP[nm] = {"romano_wolf": rw, "SPA p (H0: no comparable beats FMF-N)": spa_rev["p_value"]}
    print(f"\n{nm}: FMF-N minus each comparable (equal vol). Romano-Wolf FWER-adjusted p-values; "
          f"SPA p for 'some comparable beats FMF-N' = {spa_rev['p_value']:.3f}")
    display(rw.round(3))
AUD["SNOOP"] = SNOOP

# (a2) The marginal value of the NEW signal: FMF-N against the same book without leader-minus-herd (three breadth
# signals), same construction. Correlation between the two books is high by design, which makes the test powerful.
LMH_MARGINAL = {}
for nm, o, w, P_, R_, sv in (("dev FULL", DEV_OUT, W_AUD["FULL"], prices_raw, ratings_raw, SIG_GJR),
                             ("oos", OOS_OUT, W_OOS["OOS"], OOS_PRICES_RAW, OOS_RATINGS_RAW, OOS_SIG)):
    no_lmh = run_strategy(P_, R_, o.config.with_(signals=("breadth_63", "event_breadth_63", "breadth_resid_mom")), sig_vol=sv)
    a, b = o.net_returns.loc[w], no_lmh.net_returns.loc[w]
    t_ = sharpe_diff_test(a, b, n_boot=2000)
    LMH_MARGINAL[nm] = {"Sharpe FMF-N": perf_stats(a)["Sharpe"], "Sharpe without LMH": perf_stats(b)["Sharpe"],
                        "correlation": float(a.corr(b)), "Sharpe difference": t_["sharpe_diff"], "95% CI low": t_["ci95"][0],
                        "95% CI high": t_["ci95"][1], "p (two-sided)": t_["p_value"]}
LMH_MARGINAL = pd.DataFrame(LMH_MARGINAL).T
_se = (LMH_MARGINAL["95% CI high"] - LMH_MARGINAL["95% CI low"]) / 3.92
_m = random_effects_meta(LMH_MARGINAL["Sharpe difference"], _se, list(LMH_MARGINAL.index), tau_scale=0.2)
LMH_MARGINAL.loc["pooled (random effects)", ["Sharpe difference", "posterior P(diff > 0)"]] = [_m["mu"].summary()["mean"], _m["P(mu>0)"]]
AUD["LMH_MARGINAL"] = LMH_MARGINAL
display(LMH_MARGINAL.round(3))
print(f"P(LMH adds Sharpe | both samples) = {_m['P(mu>0)']:.3f}")

# (b) Bayesian Sharpe ratios (Bayesian bootstrap of weekly returns) under three priors
BSR = {}
for nm, r in (("dev DEV", DEV_OUT.net_returns.loc[W_AUD["DEV"]]), ("dev HOL", DEV_OUT.net_returns.loc[W_AUD["HOL"]]),
              ("dev FULL", DEV_OUT.net_returns.loc[W_AUD["FULL"]]), ("OOS", OOS_OUT.net_returns.loc[W_OOS["OOS"]])):
    for pr_nm, pr in (("flat", None), ("sceptical N(0,1)", 1.0), ("very sceptical N(0,0.5)", 0.5)):
        BSR[(nm, pr_nm)] = bayesian_sharpe(r, prior_sr_sd=pr, n_draws=40000, seed=1).summary()
BSR = pd.DataFrame(BSR).T
AUD["BSR"] = BSR
display(BSR.round(3))

# (c) Random-effects meta-analysis across development, holdout and the never-seen universe
def _ic(o, w):
    t = signal_ic_table(o, 21, w)
    return t
ICS = {"DEV": _ic(DEV_OUT, W_AUD["DEV"]), "HOL": _ic(DEV_OUT, W_AUD["HOL"]), "OOS": _ic(OOS_OUT, W_OOS["OOS"])}
META = {}
for sig in ("leader_minus_herd", "composite"):
    est = [ICS[k].loc[sig, "IC"] for k in ICS]
    se = [ICS[k].loc[sig, "IC"] / ICS[k].loc[sig, "t_phase"] for k in ICS]
    m = random_effects_meta(est, se, list(ICS), tau_scale=0.05)
    META[sig] = m
    z = sum(_st.norm.isf(ICS[k].loc[sig, "p (one-sided)"]) for k in ICS) / np.sqrt(3)
    print(f"\n{sig}: IC by sample {np.round(est, 3)}, se {np.round(se, 3)}")
    print(f"  pooled mean IC {m['mu'].summary()['mean']:.3f} (90% HDI {m['mu'].summary()['hdi90_low']:.3f} to "
          f"{m['mu'].summary()['hdi90_high']:.3f}); P(mean IC > 0) = {m['P(mu>0)']:.3f}; "
          f"P(IC > 0 in the NEXT universe) = {m['P(theta_new>0)']:.3f}; Stouffer p = {_st.norm.sf(z):.4f}")
    for k_, e_, s_ in zip(ICS, est, se):
        print(f"  BF10 (IC>0 vs 0, prior sd 0.1) {k_}: {bayes_factor_normal(e_, s_, 0.1):.2f}")
# Sharpe ratios as effects: se from the PSR variance formula (non-normality adjusted)
def _sr_se(r):
    r = r.dropna().values; sr = r.mean() / r.std(ddof=1)
    v = (1 - _st.skew(r) * sr + (_st.kurtosis(r, fisher=False) - 1) / 4 * sr ** 2) / (len(r) - 1)
    return sr * np.sqrt(252), np.sqrt(v * 252)
_srs = [_sr_se(DEV_OUT.net_returns.loc[W_AUD["DEV"]]), _sr_se(DEV_OUT.net_returns.loc[W_AUD["HOL"]]),
        _sr_se(OOS_OUT.net_returns.loc[W_OOS["OOS"]])]
m = random_effects_meta([a for a, _ in _srs], [b for _, b in _srs], ["DEV", "HOL", "OOS"], tau_scale=0.5)
META["sharpe"] = m
print(f"\nSharpe ratio: pooled {m['mu'].summary()['mean']:.2f} (90% HDI {m['mu'].summary()['hdi90_low']:.2f} to "
      f"{m['mu'].summary()['hdi90_high']:.2f}); P(SR > 0) = {m['P(mu>0)']:.3f}; P(SR > 0 next universe) = {m['P(theta_new>0)']:.3f}; "
      f"P(SR_next > 0.5) = {(m['theta_new'].draws > 0.5).mean():.3f}")
AUD["META"] = {k: {"mu": v["mu"].summary(), "tau": v["tau"].summary(), "theta_new": v["theta_new"].summary(),
                   "table": v["table"], "P(mu>0)": v["P(mu>0)"], "P(theta_new>0)": v["P(theta_new>0)"]} for k, v in META.items()}
AUD["ICS"] = ICS

# (d) Empirical-Bayes shrinkage of the 18-candidate development search (Fama-MacBeth t over four price controls)
LIB18 = pd.Series({"fresh_upside_63": 0.6104, "fresh_upside_126": 0.7158, "breadth_21": 0.7863, "breadth_42": 0.0671,
                   "breadth_126": 0.7130, "diffusion_63": -0.1300, "breadth_ts_z": -0.8186, "cons_chg_resid_px": -2.2678,
                   "disp_chg": 1.2685, "leader_minus_herd": 2.4581, "high52": -2.8266, "resid_mom_12_1": 0.4476,
                   "neg_ivol_60": 0.0622, "neg_max5": -1.9934, "low_beta_126": 0.3775, "rev_5": -0.6555,
                   "ear_drift": 0.5478, "analyst_conviction_shock": 1.0700})
SHRINK = shrink_normal_means(LIB18)
AUD["SHRINK"] = SHRINK
display(SHRINK.round(3).head(6))
