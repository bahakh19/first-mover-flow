"""Diffusion-FHS: GJR-GARCH volatility dynamics with innovations drawn from a CONDITIONAL diffusion model.

Filtered historical simulation (FHS) re-uses the exact shock vectors of history. With five years of data that is a few
hundred distinct 'bad days', and the joint shocks of the strategy and the market in a crash are whatever the one crash
in the sample happened to be. Window-level generative models, on the other hand, are starved: 1,900 overlapping
64-day windows contain only ~30 independent ones, and a network trained on them memorises (see `fidelity`).

Diffusion-FHS splits the problem where the data are rich enough for each part:
  * dynamics  : per-channel GJR-GARCH(1,1)-t on the pooled standardised samples (volatility clustering, leverage),
                with variance targeting and persistence capped at 0.99 so that a near-integrated fit on a short sample
                cannot make simulated volatility drift away from the observed level;
  * shocks    : a conditional DDPM learns the JOINT law of the daily innovation vector z_t (C dimensions, ~1,900
                observations) given the volatility regime c_t = standardised log sigma_t of all channels. It can
                therefore learn that correlations and tails change with the regime, and it can produce shock
                combinations that history contains rarely, instead of only re-playing them;
  * marginals : diffusion models with Gaussian noise under-represent heavy tails when data are scarce. Each innovation
                is therefore mapped to normal scores before training (the Gaussian quantile transform used for tabular
                diffusion, e.g. TabDDPM) and mapped back with the empirical quantile function, extended beyond the
                sample by Student-t tails with the GARCH-estimated degrees of freedom. The diffusion model learns the
                regime-dependent DEPENDENCE (a conditional copula); the marginal tails come from the data.
Simulation runs the GARCH recursion forward and draws each day's shock vector from the diffusion model conditioned on
that day's simulated volatility state. The whole path loop is one compiled JAX scan.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fmflow.generative.ddpm import DDPM, _jax


@dataclass
class DiffusionFHS:
    """GJR-GARCH dynamics with shocks from a conditional diffusion model (see module docstring)."""
    channels: list
    T: int = 100
    hidden: int = 128
    n_blocks: int = 3
    seed: int = 0
    max_persistence: float = 0.99
    garch: dict = field(default_factory=dict, repr=False)
    model: DDPM = field(default=None, repr=False)

    def fit(self, samples: list[pd.DataFrame], steps: int = 4000, val_share: float = 0.15, verbose: int = 0) -> DiffusionFHS:
        """Fit the per-channel GJR-GARCH dynamics and the conditional diffusion model of the innovations on one or
        several (T x C) return samples (each standardised by its own mean and volatility)."""
        from arch import arch_model

        from fmflow.risk.garch import fit_quietly
        std = [(s[self.channels].dropna() - s[self.channels].mean()) / s[self.channels].std() for s in samples]
        self.scale_ = np.mean([s[self.channels].std().to_numpy() for s in samples], 0)
        self.mean_ = np.mean([s[self.channels].mean().to_numpy() for s in samples], 0)
        pooled = pd.concat(std, ignore_index=True)
        Z, LS = {}, {}
        for c in self.channels:
            f = fit_quietly(arch_model(pooled[c] * 10, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t"))
            p = f.params
            mean_c = float(pooled[c].mean())          # sample mean (0 after standardisation), NOT the robust t-location
            al, ga, be = float(p["alpha[1]"]), float(p["gamma[1]"]), float(p["beta[1]"])
            pers = al + ga / 2 + be
            if pers > self.max_persistence:          # near-integrated GARCH on short samples: shrink towards stationarity
                k = self.max_persistence / pers
                al, ga, be = al * k, ga * k, be * k
                pers = self.max_persistence
            # variance targeting (Engle and Mezrich, 1996): the long-run variance equals the SAMPLE variance
            omega = (1 - pers) * float(pooled[c].var())
            self.garch[c] = {"mu": mean_c, "omega": omega, "alpha": al, "gamma": ga, "beta": be, "nu": p["nu"],
                             "persistence": pers}
            sig = f.conditional_volatility.to_numpy() / 10
            Z[c] = (pooled[c].to_numpy() - mean_c) / sig
            LS[c] = np.log(sig)
        Zm = np.column_stack([Z[c] for c in self.channels])
        Lm = np.column_stack([LS[c] for c in self.channels])
        Um, self.u_grid_, self.z_grid_ = self._normal_scores(Zm)
        self.lm_, self.ls_ = Lm.mean(0), Lm.std(0) + 1e-8
        Cm = (Lm - self.lm_) / self.ls_
        self.z_max_ = np.abs(Zm).max(0)
        # chronological validation split inside each sample
        tr, va, ofs = [], [], 0
        for s in std:
            n = len(s)
            cut = int(n * (1 - val_share))
            tr.append(np.arange(ofs, ofs + cut))
            va.append(np.arange(ofs + cut, ofs + n))
            ofs += n
        tr, va = np.concatenate(tr), np.concatenate(va)
        C = len(self.channels)
        self.model = DDPM(1, C, T=self.T, hidden=self.hidden, n_blocks=self.n_blocks, seed=self.seed, whiten=False, cond_dim=C)
        self.model.fit(Um[tr, None, :].astype(np.float32), steps=steps, cond=Cm[tr], X_val=Um[va, None, :].astype(np.float32),
                       cond_val=Cm[va], verbose=verbose)
        self.Z_, self.C_ = Zm, Cm
        return self

    def _normal_scores(self, Z: np.ndarray, grid: int = 4001, u_max: float = 6.0):
        """Normal scores of each column and a monotone (u -> z) map: empirical quantiles inside the sample range,
        standardised Student-t quantiles (nu from the GARCH fit), joined continuously, outside it."""
        from scipy import stats
        n, C = Z.shape
        U = np.empty_like(Z)
        ug = np.linspace(-u_max, u_max, grid)
        zg = np.empty((grid, C))
        for j, c in enumerate(self.channels):
            r = stats.rankdata(Z[:, j])
            U[:, j] = stats.norm.ppf((r - 0.5) / n)
            zs = np.sort(Z[:, j])
            p = stats.norm.cdf(ug)
            emp = np.interp(p, (np.arange(1, n + 1) - 0.5) / n, zs)
            nu = float(self.garch[c]["nu"])
            tq = stats.t.ppf(p, nu) * np.sqrt((nu - 2) / nu)
            lo, hi = 0.5 / n, 1 - 0.5 / n
            t_lo, t_hi = stats.t.ppf(lo, nu) * np.sqrt((nu - 2) / nu), stats.t.ppf(hi, nu) * np.sqrt((nu - 2) / nu)
            zg[:, j] = np.where(p < lo, zs[0] + (tq - t_lo), np.where(p > hi, zs[-1] + (tq - t_hi), emp))
        return U, ug.astype(np.float32), zg.astype(np.float32)

    def _to_z(self, u):
        """Map normal scores back to innovations (vectorised over columns, JAX-friendly)."""
        jax, jnp = _jax()
        ug, zg = jnp.asarray(self.u_grid_), jnp.asarray(self.z_grid_)
        return jnp.stack([jnp.interp(u[:, j], ug, zg[:, j]) for j in range(len(self.channels))], axis=1)

    def _sampler(self, steps: int):
        """Strided ancestral sampler (DDIM with eta = 1; Song, Meng and Ermon, 2021) over `steps` of the T training
        noise levels: a 4x shorter chain than the full T-step DDPM with indistinguishable draws for this model."""
        jax, jnp = _jax()
        m, p = self.model, self.model.params
        ts = np.unique(np.linspace(0, m.T - 1, steps).round().astype(int))[::-1]
        ab = m.abar[ts]
        ab_next = np.r_[m.abar[ts[1:]], 1.0]
        tsj, abj, abn = jnp.asarray(ts), jnp.asarray(ab, jnp.float32), jnp.asarray(ab_next, jnp.float32)
        C = len(self.channels)

        def draw(c, key):
            x = jax.random.normal(key, (c.shape[0], C))

            def body(i, carry):
                x, key = carry
                key, k1 = jax.random.split(key)
                a_t, a_n = abj[i], abn[i]
                eps = m._apply(p, x, jnp.full((x.shape[0],), tsj[i]), c)
                x0 = jnp.clip((x - jnp.sqrt(1 - a_t) * eps) / jnp.sqrt(a_t), -15, 15)
                sig = jnp.sqrt((1 - a_n) / (1 - a_t)) * jnp.sqrt(jnp.maximum(1 - a_t / a_n, 0.0))
                dirn = jnp.sqrt(jnp.maximum(1 - a_n - sig ** 2, 0.0)) * eps
                return jnp.sqrt(a_n) * x0 + dirn + sig * jax.random.normal(k1, x.shape), key
            x, _ = jax.lax.fori_loop(0, len(ts), body, (x, jax.random.split(key)[0]))
            return x
        return draw

    def sample_shocks(self, cond: np.ndarray, seed: int = 0, steps: int = 25) -> np.ndarray:
        """Draw one innovation vector per row of `cond` (standardised log-vol states)."""
        jax, jnp = _jax()
        draw = jax.jit(self._sampler(steps))
        u = draw(jnp.asarray(cond, jnp.float32), jax.random.PRNGKey(seed))
        return np.asarray(self._to_z(u))

    def simulate(self, n_paths: int, horizon: int, seed: int = 0, units: str = "returns",
                 extrapolation: float = 0.5, steps: int = 50, moment_match: bool = True) -> np.ndarray:
        """(n_paths, horizon, C) simulated daily returns (original pooled units, or standardised).

        moment_match: each simulated day, the cross-path shock vectors are affinely re-centred to zero and re-scaled to the
        empirical innovation standard deviation of each channel (moment matching, a classic Monte Carlo
        variance-reduction device; Boyle, Broadie and Glasserman, 1997). A diffusion sampler trained on ~1,900 points
        has small first- and second-moment biases (a few hundredths of a standard deviation) that would otherwise
        compound into a large spurious DRIFT over a five-year path. The scaled shocks sigma_t * z_t are then re-centred
        across paths as well, because a learned dependence of the shock mean on the volatility state would otherwise
        turn into drift. Matching keeps everything the model learned about shape, tails and regime-dependent
        dependence. Needs n_paths >= ~100."""
        jax, jnp = _jax()
        C = len(self.channels)
        g = {k: jnp.asarray([self.garch[c][k] for c in self.channels], jnp.float32) for k in ("mu", "omega", "alpha", "gamma", "beta")}
        pers = g["alpha"] + g["gamma"] / 2 + g["beta"]
        s2_0 = g["omega"] / jnp.maximum(1 - pers, 1e-3)
        lm, ls = jnp.asarray(self.lm_, jnp.float32), jnp.asarray(self.ls_, jnp.float32)
        lo, hi = jnp.asarray(self.C_.min(0) - extrapolation), jnp.asarray(self.C_.max(0) + extrapolation)
        zmax = jnp.asarray(1.25 * self.z_max_, jnp.float32)
        zmu, zsd = jnp.zeros(C, jnp.float32), jnp.asarray(self.Z_.std(0), jnp.float32)   # zero-mean shocks: drift is set explicitly
        sampler = self._sampler(steps)

        def draw(c, key):
            return jnp.clip(self._to_z(sampler(c, key)), -zmax, zmax)

        @jax.jit
        def run(key):
            def day(carry, _):
                s2, key = carry
                key, k1 = jax.random.split(key)
                c = jnp.clip((0.5 * jnp.log(s2) - lm) / ls, lo, hi)
                z = draw(c, k1)
                if moment_match:
                    z = (z - z.mean(0)) / (z.std(0) + 1e-8) * zsd + zmu
                e = jnp.sqrt(s2) * z
                if moment_match:
                    e = e - e.mean(0)      # exact zero drift: the shocks' regime dependence cannot leak into a mean
                r = g["mu"] + e
                s2n = g["omega"] + (g["alpha"] + g["gamma"] * (e < 0)) * e ** 2 + g["beta"] * s2
                s2n = jnp.exp(2 * (jnp.clip((0.5 * jnp.log(s2n) - lm) / ls, lo, hi) * ls + lm))   # support constraint
                return (s2n, key), r
            init = (jnp.broadcast_to(s2_0, (n_paths, C)), key)
            _, rs = jax.lax.scan(day, init, None, length=horizon)
            return jnp.swapaxes(rs, 0, 1)
        out = np.asarray(run(jax.random.PRNGKey(seed)))
        return out * self.scale_ + self.mean_ if units == "returns" else out
