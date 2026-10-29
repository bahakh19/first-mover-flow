"""Bayesian inference for small samples.

With ten stocks and a few years of data, frequentist p-values answer a narrow question ("how surprising is this under
exactly zero?") and discard everything else that is known. The tools below answer the questions a PM or an allocator
actually asks, with explicit priors that can be argued about:

bayesian_sharpe        posterior of the annualised Sharpe ratio: nonparametric Bayesian bootstrap of weekly returns
                       (skew and fat tails enter through the data) with a sceptical prior centred on zero.
random_effects_meta    Bayesian random-effects meta-analysis (normal-normal, exact marginalisation over the mean and a
                       grid over the between-sample heterogeneity tau with a half-Cauchy prior). Combines evidence from
                       development, holdout and a new universe WITHOUT assuming they share one true value, and gives the
                       predictive distribution of the effect in the NEXT universe.
shrink_normal_means    empirical-Bayes (James-Stein type) shrinkage of a family of t-statistics: the multiple-testing
                       answer to "the best of 18 candidates looked good" -- shrink it towards the family.
bayes_factor_normal    Bayes factor for H1: theta > 0 against H0: theta = 0 under a normal approximation.

All samplers are plain numpy/scipy (no probabilistic-programming dependency), seeded and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Posterior:
    """Posterior draws of a scalar with HDI and threshold-probability summaries."""
    draws: np.ndarray

    def summary(self, thresholds=(0.0, 0.5, 1.0), hdi: float = 0.9) -> dict:
        """Mean, median, sd, highest-density interval (`hdi` mass) and P(draw > x) for each threshold x."""
        d = np.sort(self.draws)
        n = len(d)
        w = int(np.floor(hdi * n))
        i = int(np.argmin(d[w:] - d[: n - w]))
        out = {"mean": float(d.mean()), "median": float(np.median(d)), "sd": float(d.std(ddof=1)),
               f"hdi{int(hdi * 100)}_low": float(d[i]), f"hdi{int(hdi * 100)}_high": float(d[i + w])}
        out.update({f"P(>{x:g})": float((d > x).mean()) for x in thresholds})
        return out


def _weekly(r: pd.Series | np.ndarray, k: int = 5) -> np.ndarray:
    x = np.asarray(pd.Series(r).dropna(), float)
    n = len(x) // k
    return x[: n * k].reshape(n, k).sum(1)


def bayesian_sharpe(r, prior_sr_sd: float | None = 1.0, prior_sr_mean: float = 0.0, aggregate: int = 5,
                    periods: int = 252, n_draws: int = 20000, seed: int = 0, method: str = "bayesian_bootstrap",
                    burn: int = 5000) -> Posterior:
    """Posterior of the annualised Sharpe ratio (mean / sd of returns, the quantity investors are paid on).

    method = "bayesian_bootstrap" (default): Rubin's (1981) Bayesian bootstrap on non-overlapping `aggregate`-period
        returns -- a nonparametric posterior that makes no normality or symmetry assumption (skew and fat tails enter
        through the data) -- combined with the prior by importance re-weighting. Prior on the annual Sharpe ratio:
        N(prior_sr_mean, prior_sr_sd^2), or flat if prior_sr_sd is None. A sceptical prior (mean 0, sd 1) says most
        strategies have |SR| < 2; N(0, 0.5) is a 'published-anomalies-decay' prior.
    method = "student_t": parametric Student-t likelihood with adaptive Metropolis. Its location parameter is a ROBUST
        centre, not the mean, so for skewed P&L it estimates a different (more conservative) quantity; provided for
        comparison only.
    """
    y = _weekly(r, aggregate)
    ann = np.sqrt(periods / aggregate)
    rng = np.random.default_rng(seed)
    if method == "bayesian_bootstrap":
        n = len(y)
        w = rng.dirichlet(np.ones(n), size=n_draws)                   # posterior over the data-generating distribution
        m = w @ y
        v = w @ (y ** 2) - m ** 2
        sr = m / np.sqrt(np.maximum(v * n / (n - 1), 1e-30)) * ann
        if prior_sr_sd is None:
            return Posterior(sr)
        lw = stats.norm.logpdf(sr, prior_sr_mean, prior_sr_sd)
        p = np.exp(lw - lw.max())
        p /= p.sum()
        idx = rng.choice(n_draws, size=n_draws, p=p)                 # sampling-importance-resampling
        post = Posterior(sr[idx])
        post.ess = float(1 / (p ** 2).sum())  # type: ignore[attr-defined]
        return post
    if method != "student_t":
        raise ValueError("method must be 'bayesian_bootstrap' or 'student_t'")

    def sd_of(s, nu):
        return s * np.sqrt(nu / (nu - 2))

    def logpost(th):
        mu, ls, lnu = th
        s, nu = np.exp(ls), 2.1 + np.exp(lnu)
        lp = stats.t.logpdf(y, nu, loc=mu, scale=s).sum()
        lp += -(nu - 2.1) / 30.0 + lnu                    # Exp(1/30) on nu - 2.1, Jacobian of log transform
        if prior_sr_sd is not None:
            lp += stats.norm.logpdf(mu / sd_of(s, nu) * ann, prior_sr_mean, prior_sr_sd)
        return lp

    th = np.array([y.mean(), np.log(y.std(ddof=1) * 0.8), np.log(5.0)])
    lp = logpost(th)
    cov = np.diag([(y.std() / np.sqrt(len(y))) ** 2, 0.05 ** 2, 0.3 ** 2])
    chain = np.empty((n_draws, 3))
    for i in range(n_draws):
        if 500 < i < burn and i % 500 == 0:              # adapt during burn-in only (keeps the chain valid)
            cov = np.cov(chain[max(0, i - 2000):i].T) * 2.38 ** 2 / 3 + 1e-12 * np.eye(3)
        prop = rng.multivariate_normal(th, cov)
        lpp = logpost(prop)
        if np.log(rng.random()) < lpp - lp:
            th, lp = prop, lpp
        chain[i] = th
    c = chain[burn:]
    s, nu = np.exp(c[:, 1]), 2.1 + np.exp(c[:, 2])
    return Posterior(c[:, 0] / sd_of(s, nu) * ann)


def random_effects_meta(estimates, ses, labels=None, tau_scale: float | None = None, prior_mu_sd: float | None = None,
                        n_draws: int = 20000, seed: int = 0) -> dict:
    """Bayesian random-effects meta-analysis: y_j ~ N(theta_j, se_j^2), theta_j ~ N(mu, tau^2).

    Priors: mu flat (or N(0, prior_mu_sd^2)); tau ~ half-Cauchy(tau_scale) (default scale = median se). The posterior of
    (mu, tau) is computed exactly on a fine tau grid with mu integrated analytically. Returns posterior draws of mu,
    tau, the shrunken per-sample effects theta_j, and the predictive effect in a NEW sample, theta_new ~ N(mu, tau^2).
    """
    y, s = np.asarray(estimates, float), np.asarray(ses, float)
    labels = list(labels) if labels is not None else [f"sample_{j}" for j in range(len(y))]
    tau_scale = float(np.median(s)) if tau_scale is None else tau_scale
    grid = np.linspace(0, 6 * max(tau_scale, y.std() + s.max()), 2000)
    logp = np.empty_like(grid)
    mus, vs = np.empty_like(grid), np.empty_like(grid)
    for g, tau in enumerate(grid):
        w = 1 / (s ** 2 + tau ** 2)
        prec = w.sum() + (0 if prior_mu_sd is None else 1 / prior_mu_sd ** 2)
        mu_hat = (w * y).sum() / prec
        mus[g], vs[g] = mu_hat, 1 / prec
        logp[g] = (stats.halfcauchy.logpdf(tau, scale=tau_scale) + 0.5 * np.log(1 / prec) + 0.5 * np.log(w).sum()
                   - 0.5 * (w * (y - mu_hat) ** 2).sum() - (0 if prior_mu_sd is None else 0.5 * mu_hat ** 2 / prior_mu_sd ** 2))
    p = np.exp(logp - logp.max())
    p /= p.sum()
    rng = np.random.default_rng(seed)
    gi = rng.choice(len(grid), size=n_draws, p=p)
    tau = grid[gi]
    mu = rng.normal(mus[gi], np.sqrt(vs[gi]))
    theta = np.empty((n_draws, len(y)))
    for j in range(len(y)):                               # conditional posterior of each sample's true effect
        v = 1 / (1 / s[j] ** 2 + 1 / np.maximum(tau, 1e-12) ** 2)
        m = v * (y[j] / s[j] ** 2 + mu / np.maximum(tau, 1e-12) ** 2)
        theta[:, j] = np.where(tau > 1e-10, rng.normal(m, np.sqrt(v)), mu)
    new = rng.normal(mu, tau)
    table = pd.DataFrame({"estimate": y, "se": s, "shrunk_mean": theta.mean(0), "P(theta>0)": (theta > 0).mean(0)},
                         index=labels)
    return {"mu": Posterior(mu), "tau": Posterior(tau), "theta_new": Posterior(new), "table": table,
            "P(mu>0)": float((mu > 0).mean()), "P(theta_new>0)": float((new > 0).mean())}


def shrink_normal_means(t: pd.Series) -> pd.DataFrame:
    """Empirical-Bayes shrinkage of z/t-statistics t_k ~ N(theta_k, 1), theta_k ~ N(m, A).

    m and A are estimated from the family (A = max(0, var(t) - 1), positive-part James-Stein). The posterior mean
    B*m + (1-B)*t_k with B = 1/(1+A) is what a statistic is 'worth' once the search over the family is accounted for.
    """
    t = pd.Series(t).dropna().astype(float)
    m = float(t.mean())
    A = max(0.0, float(t.var(ddof=1)) - 1.0)
    B = 1.0 / (1.0 + A)
    post_mean = B * m + (1 - B) * t
    post_sd = np.sqrt(1 - B) if A > 0 else 0.0
    p_pos = stats.norm.sf(0, loc=post_mean, scale=post_sd) if post_sd > 0 else (post_mean > 0).astype(float)
    return pd.DataFrame({"t": t, "shrunk_t": post_mean, "P(theta>0)": p_pos, "shrinkage_B": B}).sort_values("t", ascending=False)


def bayes_factor_normal(estimate: float, se: float, prior_sd: float, one_sided: bool = True) -> float:
    """BF10 for H1: theta ~ N(0, prior_sd^2) (truncated to theta > 0 if one_sided) vs H0: theta = 0."""
    m0 = stats.norm.pdf(estimate, 0, se)
    m1 = stats.norm.pdf(estimate, 0, np.sqrt(se ** 2 + prior_sd ** 2))
    if not one_sided:
        return float(m1 / m0)
    v = 1 / (1 / se ** 2 + 1 / prior_sd ** 2)
    mpost = v * estimate / se ** 2
    return float(2 * m1 * stats.norm.cdf(mpost / np.sqrt(v)) / m0)
