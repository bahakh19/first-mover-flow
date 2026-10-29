# Section 17.3 — Objection 5: "Your parameters were chosen to make it work" (researcher degrees of freedom).
# A multiverse (specification-curve) analysis re-runs the full pipeline over every reasonable choice the researcher could
# have made -- signal windows, leader/herder definitions, target expiry (including the quarterly 63-session calibration),
# event size, legs, tranches, allocator, signal set -- one at a time around the frozen specification and as 300 random
# joint draws. If the result were a knife-edge, the curve would collapse away from the chosen point.
from fmflow.robustness import multiverse
from fmflow.analytics.snooping import spa_test
from fmflow.analytics.inference import cscv_pbo, deflated_sharpe

GRID = {"leader_gap": [5, 10, 15, 20], "herd_window": [(2, 10), (3, 15), (5, 20)], "lmh_window": [63, 126, 189, 252],
        "breadth_window": [42, 63, 126], "tranche_period": [21, 42, 63, 84], "allocator": ["EW", "IV-GJR"],
        "signals": [("breadth_63", "event_breadth_63", "breadth_resid_mom", "leader_minus_herd"), ("leader_minus_herd",),
                    ("breadth_63", "leader_minus_herd"), ("breadth_63", "event_breadth_63", "breadth_resid_mom")],
        "data.tp_expiry_days": [63, 126, 252], "data.event_min_ratings": [4, 5, 6], "data.knowledge_lag": [1, 2]}
MV = {}
for nm, P_, R_, base, sv, W, legs in (("dev", prices_raw, ratings_raw, StrategyConfig.fmf_n(), SIG_GJR, W_AUD, [3, 4, 5]),
                                      ("oos", OOS_PRICES_RAW, OOS_RATINGS_RAW, StrategyConfig.fmf_n().with_(k=None, leg_fraction=0.4),
                                       OOS_SIG, W_OOS, [0.3, 0.4, 0.5])):
    g = {**GRID, "legs": legs}
    oat, oat_nets = multiverse(P_, R_, base, g, W, sig_vol=sv, mode="oat", n_jobs=2)
    rnd, rnd_nets = multiverse(P_, R_, base, g, W, sig_vol=sv, mode="random", n_random=300, seed=7, n_jobs=2)
    MV[nm] = {"oat": oat, "random": rnd, "random_nets": rnd_nets}
    key = list(W)[-1]
    wkey = list(W.values())[0] if nm == "dev" else W["OOS"]     # snooping tests on the window where choices were made
    fam = rnd_nets.loc[wkey].dropna(axis=1, how="all").fillna(0.0)
    spa = spa_test(fam, n_boot=2000)
    pbo = cscv_pbo(fam, n_blocks=12)
    s = rnd[f"Sharpe {key}"].dropna()
    MV[nm]["summary"] = {"specifications": len(s), "share Sharpe > 0": float((s > 0).mean()),
                         "share Sharpe > 0.5": float((s > 0.5).mean()), "median Sharpe": float(s.median()),
                         "10th pct": float(s.quantile(0.1)), "90th pct": float(s.quantile(0.9)),
                         "frozen spec Sharpe": float(rnd.loc["base", f"Sharpe {key}"]),
                         "frozen spec percentile": float((s < rnd.loc["base", f"Sharpe {key}"]).mean()),
                         f"SPA p ({'DEV' if nm == 'dev' else 'OOS'}, H0: no spec beats 0)": spa["p_value"],
                         "PBO across the family": pbo["PBO"]}
    print(f"\n== {nm}: multiverse of {len(s)} random joint specifications, window {key} ==")
    print(pd.Series(MV[nm]["summary"]).round(3).to_string())
    print("\none-at-a-time sensitivity (Sharpe):")
    display(oat[[c for c in oat.columns if c.startswith("Sharpe")]].round(2))
MV_SUMMARY = pd.DataFrame({k: v["summary"] for k, v in MV.items()})
AUD["MV"] = {k: {"oat": v["oat"], "random": v["random"], "summary": v["summary"]} for k, v in MV.items()}
display(MV_SUMMARY.round(3))

# The multiverse ENSEMBLE: hold every specification at once (equal capital). Its return cannot be the product of
# picking the best specification, so it is a selection-free estimate of what the idea is worth.
from fmflow.analytics.attribution import mean_with_hac
ENS = {}
for nm, W in (("dev", W_AUD), ("oos", W_OOS)):
    ens = MV[nm]["random_nets"].mean(axis=1)
    for wn, w in W.items():
        s, h = perf_stats(ens.loc[w]), mean_with_hac(ens.loc[w])
        fam = MV[nm]["random_nets"].loc[w].dropna(axis=1, how="all").fillna(0.0)
        ENS[(nm, wn)] = {"ensemble Sharpe": s["Sharpe"], "ensemble Calmar": s["Calmar"], "ensemble t(HAC)": h["t_hac"],
                         "SPA p (family, this window)": spa_test(fam, n_boot=2000)["p_value"]}
ENS = pd.DataFrame(ENS).T
AUD["MV_ENS"] = ENS
display(ENS.round(3))
