# Section 17.1 — Objection 1: "In a hindsight-selected universe the book is just long the winners."
# Lo (2008) splits mean P&L into passive (average tilt x average return) and active (covariance of positions with
# subsequent returns). The causal version holds the expanding mean of PAST weights (static book) and the deviation from
# it (timing book); both are tradable and sum to the strategy.
from fmflow.analytics.attribution import lo_decomposition, static_timing_books, mean_with_hac, stock_contributions
from fmflow.analytics.bayes import random_effects_meta
from fmflow.robustness import book_from_score

rows = {}
for nm, o, W in (("dev", DEV_OUT, W_AUD), ("oos", OOS_OUT, W_OOS)):
    stb = static_timing_books(o.weights, o.returns)
    for wn, w in W.items():
        lo = lo_decomposition(o.weights, o.returns, w)
        st, tm = mean_with_hac(stb["static"].loc[w]), mean_with_hac(stb["timing"].loc[w])
        rows[(nm, wn)] = {"ann. gross return": lo["total"], "Lo passive": lo["passive"], "Lo active": lo["active"],
                          "active share": lo["active_share"],
                          "static book: ann. return": st["ann_mean"], "static t(HAC)": st["t_hac"],
                          "static Sharpe": perf_stats(stb["static"].loc[w])["Sharpe"],
                          "timing book: ann. return": tm["ann_mean"], "timing t(HAC)": tm["t_hac"],
                          "timing Sharpe": perf_stats(stb["timing"].loc[w])["Sharpe"]}
    for L in (126, 252):   # identity-neutral diagnostic: each stock's score minus its own trailing mean, re-ranked
        b = book_from_score(o.score - o.score.rolling(L, min_periods=63).mean(), o)
        for wn, w in W.items():
            rows[(nm, wn)][f"identity-neutral({L}) Sharpe"] = perf_stats(b.loc[w])["Sharpe"]
IDENT = pd.DataFrame(rows).T
AUD["IDENT"] = IDENT
display(IDENT.round(3))

# Bayesian partial pooling of per-stock contributions: does the result rest on one or two names?
POOL = {}
for nm, o, w in (("dev FULL", DEV_OUT, W_AUD["FULL"]), ("oos", OOS_OUT, W_OOS["OOS"])):
    sc = stock_contributions(o.weights, o.returns, w)
    sc = sc[(sc["se_ann"] > 0) & (sc["held_share"] >= 0.2)]          # names actually traded (>= 20% of days)
    m = random_effects_meta(sc["ann_contribution"], sc["se_ann"], sc.index)
    POOL[nm] = {"table": m["table"], "mu": m["mu"].summary(), "P(mu>0)": m["P(mu>0)"], "tau": m["tau"].summary()}
    print(f"\n{nm}: typical per-stock contribution (partially pooled) = {m['mu'].summary()['mean']*100:.2f}% a year, "
          f"P(>0) = {m['P(mu>0)']:.3f}; heterogeneity tau = {m['tau'].summary()['median']*100:.2f}%")
    display(m["table"].sort_values("estimate", ascending=False).head(10).round(4))
AUD["POOL"] = POOL
