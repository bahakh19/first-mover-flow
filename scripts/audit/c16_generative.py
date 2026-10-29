# Section 17.6 — Objection 8: "One five-year history has one or two drawdowns; your Calmar ratio is a single draw."
# Generative stress testing. We learn the JOINT daily dynamics of (FMF-N, equal-weight market, 12-1 momentum book) from
# the development and never-seen samples (each standardised by its own mean and volatility), then simulate thousands of
# five-year histories. Four generators are compared on the same fidelity battery:
#   block bootstrap   (Politis-Romano)                 -- re-orders history
#   FHS               (GJR-GARCH-t + joint residuals)  -- re-uses history's shock vectors
#   Diffusion-FHS     (GJR-GARCH-t + conditional DDPM innovations given the volatility regime)
#   TimeGAN           (Yoon et al. 2019, 24-day windows, stitched)
# Drift is NOT taken from the generator: paths are zero-mean in standardised units and the strategy's drift is drawn
# from the Bayesian posterior-predictive Sharpe ratio for a NEW universe (Section 17.4). Path risk and parameter
# uncertainty therefore enter the Calmar distribution together.
from fmflow.generative.windows import make_windows
from fmflow.generative.baselines import block_bootstrap_paths, fhs_paths
from fmflow.generative.diffusion_fhs import DiffusionFHS
from fmflow.generative.timegan import TimeGAN
from fmflow.generative import fidelity as GF, stress as GS

def _frame(o, w):
    b = comparable_books(o, include_allocations=False)
    return pd.DataFrame({"FMF": o.net_returns, "MKT": o.returns.mean(axis=1), "MOM": b["12-1 momentum (Jegadeesh-Titman)"]}).loc[w].dropna()
GEN_FRAMES = [_frame(DEV_OUT, W_AUD["FULL"]), _frame(OOS_OUT, W_OOS["OOS"])]
_std = pd.concat([(f - f.mean()) / f.std() for f in GEN_FRAMES])
NP, HZ = 1000, 1260
REAL64 = np.stack([_std.to_numpy()[i:i + 64] for i in range(0, len(_std) - 64)])

dfhs = DiffusionFHS(["FMF", "MKT", "MOM"], T=100, hidden=128, n_blocks=3, seed=0).fit(GEN_FRAMES, steps=4000)
tg_windows = make_windows(GEN_FRAMES, length=24)
tgan = TimeGAN(24, 3, hidden=24, seed=0).fit(tg_windows.X, steps=(1500, 1500, 2000))
rng_ = np.random.default_rng(0)
PATHS = {"block bootstrap": block_bootstrap_paths(_std, NP, HZ, block=21, seed=1),
         "FHS (GJR-GARCH-t)": fhs_paths(_std, NP, HZ, seed=1),
         "Diffusion-FHS": dfhs.simulate(NP, HZ, seed=1, units="standardised", steps=50),
         "TimeGAN": (lambda G: (G - G.mean((0, 1))) / G.std((0, 1)))(
             __import__("fmflow.generative.windows", fromlist=["stitch"]).stitch(tgan.sample(4000, seed=1), NP, HZ, rng_))}
FID = {}
for nm, Pp in PATHS.items():
    W64 = Pp[:, : (HZ // 64) * 64].reshape(-1, 64, 3)
    sf = GF.stylised_facts(W64, ["FMF", "MKT", "MOM"])
    FID[nm] = {"FMF drift (sd/yr)": float(Pp[:, :, 0].mean() * 252), "FMF vol (sd units)": float(Pp[:, :, 0].std()),
               **sf[["FMF ex.kurt", "MKT ex.kurt", "MOM ex.kurt", "MKT acf|r| 1-10", "corr FMF-MKT", "corr FMF-MOM",
                     "tail dep FMF|MKT (5%)", "tail dep FMF|MOM (5%)"]].to_dict(),
               "discriminative score (0 = indistinguishable)": GF.discriminative_score(REAL64, W64[:3000]),
               "memorisation ratio (1 = none)": GF.memorisation_ratio(REAL64[:1100], REAL64[1250:], W64[:3000])}
sf = GF.stylised_facts(REAL64, ["FMF", "MKT", "MOM"])
FID["REAL"] = {**{k: sf[k] for k in FID["Diffusion-FHS"] if k in sf.index}, "FMF drift (sd/yr)": 0.0, "FMF vol (sd units)": 1.0}
FID = pd.DataFrame(FID).T
display(FID.round(3))

# Posterior-predictive drift: the Sharpe ratio of the strategy in a NEW universe (Section 17.4 meta-analysis)
_sr_new = META["sharpe"]["theta_new"].draws
_sd_d = np.mean([f["FMF"].std() for f in GEN_FRAMES])
PATH_STATS, PATH_DRAWS = {}, {}
for nm, Pp in PATHS.items():
    for drift_nm, sr in (("posterior-predictive SR", rng_.choice(_sr_new, NP)), ("in-sample SR 1.1", np.full(NP, 1.1)),
                         ("zero drift", np.zeros(NP))):
        r = Pp[:, :, 0] * _sd_d + (sr * _sd_d / np.sqrt(252))[:, None]
        st = GS.path_stats(r[:, :, None])
        PATH_DRAWS[(nm, drift_nm)] = st[["Calmar", "MaxDD", "Sharpe"]].to_numpy()
        PATH_STATS[(nm, drift_nm)] = {"median Calmar": st["Calmar"].median(), "P(Calmar > 0.5)": (st["Calmar"] > 0.5).mean(),
                                      "median MaxDD": st["MaxDD"].median(), "5% MaxDD": st["MaxDD"].quantile(0.05),
                                      "1% MaxDD": st["MaxDD"].quantile(0.01), "P(5y return < 0)": (st["total return"] < 0).mean()}
PATH_STATS = pd.DataFrame(PATH_STATS).T
display(PATH_STATS.round(3))

# Conditional stress: 63-session windows in which the simulated MARKET falls >= 25%, or MOMENTUM has its worst 2%
COND = {}
for nm in ("block bootstrap", "FHS (GJR-GARCH-t)", "Diffusion-FHS"):
    Pp = PATHS[nm]
    W63 = Pp[:, : (HZ // 63) * 63].reshape(-1, 63, 3)
    sd = np.array([np.mean([f[c].std() for f in GEN_FRAMES]) for c in ("FMF", "MKT", "MOM")])
    cum = np.cumprod(1 + W63 * sd, axis=1)[:, -1] - 1
    mdd_m = (np.cumprod(1 + W63[:, :, 1] * sd[1], 1) / np.maximum.accumulate(np.cumprod(1 + W63[:, :, 1] * sd[1], 1), 1) - 1).min(1)
    crash = mdd_m <= -0.25
    momcrash = cum[:, 2] <= np.quantile(cum[:, 2], 0.02)
    COND[nm] = {"P(market crash window)": crash.mean(), "FMF mean 3m return | market crash": cum[crash, 0].mean() if crash.any() else np.nan,
                "FMF 5% 3m return | market crash": np.quantile(cum[crash, 0], 0.05) if crash.sum() > 20 else np.nan,
                "FMF mean 3m return | momentum crash": cum[momcrash, 0].mean(),
                "FMF 5% 3m return | momentum crash": np.quantile(cum[momcrash, 0], 0.05),
                "FMF 5% 3m return | all windows": np.quantile(cum[:, 0], 0.05)}
COND = pd.DataFrame(COND).T
display(COND.round(3))
AUD["GEN"] = {"FID": FID, "PATH_STATS": PATH_STATS, "PATH_DRAWS": PATH_DRAWS, "COND": COND, "dfhs_history": dfhs.model.history,
              "tgan_history": tgan.history}
