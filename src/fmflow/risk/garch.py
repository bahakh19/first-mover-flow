"""Walk-forward GJR-GARCH(1,1) with Student-t innovations (Glosten, Jagannathan and Runkle, 1993).

The value at session t is the forecast of the daily volatility of r_{t+1} using returns <= t. Parameters are
re-estimated every `refit` sessions on the expanding sample once `min_obs` observations exist; before that (or if an
estimate fails / is non-stationary) the expanding historical standard deviation is used. Univariate: the forecast for
one stock never depends on other stocks.

Two estimators (`DataConfig.garch_method`):

* ``"slsqp"`` (default; every number in the paper): quasi-maximum likelihood with `arch`'s SLSQP optimiser. A fit
  is accepted if it converged and is stationary; otherwise the previous estimate is kept.
* ``"grid"``: deterministic grid maximum likelihood with variance targeting (Engle and Mezrich, 1996). mu is the
  sample mean, omega is implied by the sample variance, and (alpha, gamma, beta, nu) maximise the Student-t
  likelihood over a fixed grid, evaluated in one vectorised pass. No iterative optimiser, so the estimate is the same
  on every machine.

Why two? On a few hundred daily observations the GJR-GARCH-t likelihood is nearly flat along a ridge and often has
two local maxima with almost equal likelihood (for one stock of the open universe: beta = 0.83 and beta = 0.91, a
log-likelihood gap of 0.0002). A gradient optimiser then stops at one mode or the other depending on floating-point
details of the machine (BLAS kernel, SIMD instruction set), so a fresh SLSQP re-estimation is reproducible on one
machine but not across machines: on the open universe the out-of-sample Sharpe ratio moves by about 0.02. For exact
reproduction of the published numbers, use the shipped forecasts (`tests/fixtures/open_sig_gjr.csv.gz`) as `sig_vol`;
for machine-independent fresh estimates, use ``garch_method="grid"``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import gammaln

from fmflow.config import DataConfig

# grid for the deterministic estimator (persistence alpha + gamma * P(eps < 0) + beta is capped at 0.995)
GRID_ALPHA = (0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.14, 0.18)
GRID_GAMMA = (0.0, 0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25)
GRID_BETA = (0.50, 0.60, 0.70, 0.75, 0.80, 0.84, 0.87, 0.90, 0.92, 0.94, 0.96, 0.975, 0.985)
GRID_NU = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 30.0)
_MAX_PERSISTENCE = 0.995


def gjr_variance_path(x: np.ndarray, mu: float, omega: float, alpha: float, gamma: float, beta: float,
                      sigma2_0: float) -> np.ndarray:
    """GJR-GARCH(1,1) conditional variance recursion for given parameters."""
    eps = x - mu
    s2 = np.empty(len(x) + 1)
    s2[0] = sigma2_0
    for t in range(len(x)):
        s2[t + 1] = omega + (alpha + gamma * (eps[t] < 0)) * eps[t] ** 2 + beta * s2[t]
    return s2


def _grid():
    a, g, b = np.meshgrid(GRID_ALPHA, GRID_GAMMA, GRID_BETA, indexing="ij")
    a, g, b = a.ravel(), g.ravel(), b.ravel()
    keep = (a + g > 0)                                  # alpha = gamma = 0 is a constant variance: excluded (ties)
    return a[keep], g[keep], b[keep]


def grid_fit(x: np.ndarray) -> tuple[float, float, float, float, float, float] | None:
    """Deterministic GJR-GARCH(1,1)-t estimate with variance targeting: (mu, omega, alpha, gamma, beta, nu).

    The log-likelihood of every grid point is computed in one vectorised recursion and rounded to 1e-6 before the
    arg-max, so floating-point noise (~1e-12) cannot change the selected point; ties go to the first grid point."""
    mu = float(np.mean(x))
    eps = x - mu
    s2bar = float(np.mean(eps ** 2))
    if not np.isfinite(s2bar) or s2bar <= 0:
        return None
    a, g, b = _grid()
    neg = float(np.mean(eps < 0))
    pers = a + g * neg + b
    ok = pers < _MAX_PERSISTENCE
    a, g, b, pers = a[ok], g[ok], b[ok], pers[ok]
    omega = s2bar * (1 - pers)
    s2 = np.empty((len(eps), len(a)))
    s2[0] = s2bar
    e2 = eps ** 2
    lev = (eps < 0).astype(float)
    for t in range(len(eps) - 1):
        s2[t + 1] = omega + (a + g * lev[t]) * e2[t] + b * s2[t]
    z2 = e2[:, None] / s2
    logs2 = np.log(s2).sum(0)
    best, best_ll = None, -np.inf
    for nu in GRID_NU:
        c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
        ll = len(eps) * c - 0.5 * logs2 - (nu + 1) / 2 * np.log1p(z2 / (nu - 2)).sum(0)
        ll = np.round(ll, 6)
        i = int(np.argmax(ll))
        if ll[i] > best_ll:
            best_ll, best = ll[i], (mu, float(omega[i]), float(a[i]), float(g[i]), float(b[i]), nu)
    return best


def fit_quietly(model, **kw):
    """`model.fit` for an `arch` model without console noise.

    `arch` re-enables its own ConvergenceWarning inside `fit()` (overriding the caller's filters), so a plain
    `warnings.catch_warnings()` is not enough: `show_warning=False` is needed too. The optimiser status stays available
    as `result.convergence_flag` (0 = converged)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp="off", show_warning=False, **kw)


_fit = fit_quietly


def garch_forecast_panel(ret: pd.DataFrame, cfg: DataConfig = DataConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward one-step GJR-GARCH(1,1)-t volatility forecasts for every column (and the fit log).

    The estimator is `cfg.garch_method` (see the module docstring). Convergence diagnostics are recorded in the log
    (`optimizer_status`: 0 = converged, -2 = exception; `new_estimate`: False when the previous estimate was kept),
    never printed."""
    if cfg.garch_method not in ("slsqp", "grid"):
        raise ValueError(f"garch_method must be 'slsqp' or 'grid', not {cfg.garch_method!r}")
    if cfg.garch_method == "slsqp":
        from arch import arch_model  # optional heavy dependency, imported lazily
    out = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    log = []
    for c in ret.columns:
        s = ret[c].dropna()
        x = s.values * 100.0
        n = len(x)
        fc = np.full(n, np.nan)
        exp_sd = pd.Series(x).expanding(min_periods=20).std().values
        fc[:min(cfg.garch_min_obs, n)] = exp_sd[:min(cfg.garch_min_obs, n)]
        params = None
        for st in range(cfg.garch_min_obs - 1, n, cfg.garch_refit):
            train = x[: st + 1]
            status, accepted = 0, False
            try:
                if cfg.garch_method == "slsqp":
                    res = _fit(arch_model(train, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t"))
                    p = res.params
                    status = int(res.convergence_flag)
                    if status == 0 and p["alpha[1]"] + p["gamma[1]"] / 2 + p["beta[1]"] < 1:
                        params, accepted = (p["mu"], p["omega"], p["alpha[1]"], p["gamma[1]"], p["beta[1]"]), True
                else:
                    est = grid_fit(train)
                    if est is not None:
                        params, accepted = est[:5], True
            except Exception:
                status = -2
            end = min(st + cfg.garch_refit, n)
            if params is None:
                fc[st:end] = exp_sd[st:end]
                continue
            s2 = gjr_variance_path(x[:end], *params, sigma2_0=train.var())
            fc[st:end] = np.sqrt(s2[st + 1:end + 1])
            log.append({"stock": c, "fit_date": s.index[st], "mu": params[0], "omega": params[1], "alpha": params[2],
                        "gamma": params[3], "beta": params[4], "optimizer_status": status, "new_estimate": accepted})
        out.loc[s.index, c] = fc / 100.0
    return out, pd.DataFrame(log)
