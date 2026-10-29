"""Denoising diffusion probabilistic model for multichannel return windows (JAX, CPU-friendly).

Forward process (Ho, Jain and Abbeel, 2020) with the cosine schedule of Nichol and Dhariwal (2021):
    x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,   eps ~ N(0, I),   t = 1..T.
A residual MLP eps_theta(x_t, t) with sinusoidal time embedding is trained to predict eps (the simplified loss). Sampling
runs the ancestral reverse chain. Windows are flattened (L x C); for L = 64 and C = 3 the input has 192 dimensions,
small enough for an MLP and for training in a few minutes on a laptop CPU.

Whitening. Return windows contain very persistent components (a volatility path is almost constant within a window),
so the data covariance is extremely ill-conditioned and an MLP denoiser mis-estimates the variance along the dominant
directions. The model is therefore trained in PCA-whitened coordinates (eigenvalues floored), where the target
distribution is close to isotropic, and draws are mapped back. This is what makes the generated volatility levels
match the observed range instead of overshooting it.

Conditional scenarios (training-free): to ask "what does the strategy do when the market follows THIS path?", the
market coordinates are known. At every reverse step the predicted clean window x0 is computed, its known coordinates are
replaced by the scenario, and the reverse step continues from the corrected x0 -- an x0-replacement variant of the
inpainting idea of RePaint (Lugmayr et al., 2022). The strategy channel is thus sampled from the model's conditional
distribution given the scenario, with no re-training. Draws far outside the training data inherit the model's
extrapolation error, which the fidelity diagnostics quantify.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _jax():
    from fmflow.generative import jax_status
    ok, msg = jax_status()
    if not ok:
        raise RuntimeError(f"fmflow.generative needs a working JAX for diffusion models and TimeGAN. {msg}")
    import jax
    import jax.numpy as jnp
    return jax, jnp


def cosine_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """Cosine noise schedule (Nichol and Dhariwal, 2021): beta_1..beta_T."""
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    abar = f / f[0]
    beta = np.clip(1 - abar[1:] / abar[:-1], 1e-5, 0.999)
    return beta


@dataclass
class DDPM:
    """Denoising diffusion probabilistic model with a residual-MLP denoiser, optional whitening and conditioning."""
    length: int
    channels: int
    T: int = 200
    hidden: int = 256
    n_blocks: int = 2
    temb_dim: int = 64
    seed: int = 0
    whiten: bool = True
    cond_dim: int = 0                                   # > 0: conditional model eps(x_t, t, c)
    params: dict = field(default=None, repr=False)
    history: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.mu_ = np.zeros(self.length * self.channels, np.float32)
        self.A_ = None                                     # data = A @ white + mu
        self.D = self.length * self.channels
        self.beta = cosine_schedule(self.T).astype(np.float32)
        self.alpha = 1 - self.beta
        self.abar = np.cumprod(self.alpha).astype(np.float32)

    # ---------------------------------------------------------------------------------------------------------- model
    def _init(self, key):
        jax, jnp = _jax()
        def dense(k, i, o, scale=1.0):
            return {"w": jax.random.normal(k, (i, o)) * scale * np.sqrt(2.0 / i), "b": jnp.zeros(o)}
        ks = jax.random.split(key, 4 + 3 * self.n_blocks)
        p = {"inp": dense(ks[0], self.D + self.cond_dim, self.hidden), "t1": dense(ks[1], self.temb_dim, self.hidden),
             "out": dense(ks[2], self.hidden, self.D, 0.0), "blocks": []}
        for b in range(self.n_blocks):
            p["blocks"].append({"d1": dense(ks[3 + 3 * b], self.hidden, self.hidden),
                                "d2": dense(ks[4 + 3 * b], self.hidden, self.hidden, 0.1),
                                "t": dense(ks[5 + 3 * b], self.hidden, self.hidden)})
        return p

    def _temb(self, t):
        jax, jnp = _jax()
        half = self.temb_dim // 2
        freqs = jnp.exp(-np.log(10000.0) * jnp.arange(half) / half)
        a = t[:, None].astype(jnp.float32) * freqs[None]
        return jnp.concatenate([jnp.sin(a), jnp.cos(a)], axis=1)

    def _apply(self, p, x, t, c=None):
        jax, jnp = _jax()
        act = jax.nn.silu
        te = act(self._temb(t) @ p["t1"]["w"] + p["t1"]["b"])
        xin = x if c is None else jnp.concatenate([x, c], axis=1)
        h = xin @ p["inp"]["w"] + p["inp"]["b"] + te
        for b in p["blocks"]:
            z = act(h) @ b["d1"]["w"] + b["d1"]["b"] + te @ b["t"]["w"] + b["t"]["b"]
            h = h + act(z) @ b["d2"]["w"] + b["d2"]["b"]
        return act(h) @ p["out"]["w"] + p["out"]["b"]

    # ------------------------------------------------------------------------------------------------------- training
    def fit(self, X: np.ndarray, steps: int = 4000, batch: int = 256, lr: float = 1e-3, ema: float = 0.999,
            X_val: np.ndarray | None = None, eval_every: int = 200, verbose: int = 0,
            cond: np.ndarray | None = None, cond_val: np.ndarray | None = None) -> DDPM:
        """X: (n, L, C) standardised windows. Sampling uses an exponential moving average of the weights (decay `ema`).

        Small data memorise easily: a denoiser that drives the training loss BELOW what is achievable on unseen data has
        learned the training windows, and its samples collapse towards them (under-dispersed draws). Pass `X_val`
        (windows from a later, non-overlapping period) to keep the EMA weights with the lowest validation denoising
        loss (early stopping); the loss curves are stored in `history` as (step, train, validation)."""
        jax, jnp = _jax()
        Xf = X.reshape(len(X), -1).astype(np.float64)
        if self.whiten:
            self.mu_ = Xf.mean(0)
            ev, U = np.linalg.eigh(np.cov(Xf.T))
            ev = np.maximum(ev, ev.max() * 1e-6)
            self.A_ = (U * np.sqrt(ev)).astype(np.float32)
            self.Ainv_ = ((U / np.sqrt(ev)).T).astype(np.float32)
            Xf = (Xf - self.mu_) @ self.Ainv_.T
        Xf = jnp.asarray(Xf, jnp.float32)
        Cf = jnp.asarray(cond, jnp.float32) if cond is not None else jnp.zeros((Xf.shape[0], 0), jnp.float32)
        Cv = jnp.asarray(cond_val, jnp.float32) if cond_val is not None else None
        if X_val is not None:
            Vf = X_val.reshape(len(X_val), -1).astype(np.float64)
            Vf = jnp.asarray((Vf - self.mu_) @ self.Ainv_.T if self.whiten else Vf, jnp.float32)
            vkey = jax.random.PRNGKey(self.seed + 99)
            vt = jax.random.randint(vkey, (8, len(Vf)), 0, self.T)
            veps = jax.random.normal(jax.random.split(vkey)[0], (8, len(Vf), self.D))
        key = jax.random.PRNGKey(self.seed)
        key, k0 = jax.random.split(key)
        p = self._init(k0)
        abar = jnp.asarray(self.abar)
        m = jax.tree_util.tree_map(jnp.zeros_like, p)
        v = jax.tree_util.tree_map(jnp.zeros_like, p)
        pe = p

        def loss_fn(p, x0, t, eps, c=None):
            ab = abar[t][:, None]
            xt = jnp.sqrt(ab) * x0 + jnp.sqrt(1 - ab) * eps
            return jnp.mean((self._apply(p, xt, t, c if self.cond_dim else None) - eps) ** 2)

        @jax.jit
        def step(p, m, v, pe, i, key):
            k1, k2, k3 = jax.random.split(key, 3)
            idx = jax.random.randint(k1, (batch,), 0, Xf.shape[0])
            t = jax.random.randint(k2, (batch,), 0, self.T)
            eps = jax.random.normal(k3, (batch, self.D))
            l, g = jax.value_and_grad(loss_fn)(p, Xf[idx], t, eps, Cf[idx])
            lr_t = lr * jnp.minimum(1.0, (i + 1) / 200) * 0.5 * (1 + jnp.cos(jnp.pi * i / steps))   # warm-up + cosine
            m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
            v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
            c1, c2 = 1 - 0.9 ** (i + 1), 1 - 0.999 ** (i + 1)
            p = jax.tree_util.tree_map(lambda w, a, b: w - lr_t * (a / c1) / (jnp.sqrt(b / c2) + 1e-8), p, m, v)
            d = jnp.minimum(ema, (1 + i) / (10 + i))
            pe = jax.tree_util.tree_map(lambda a, b: d * a + (1 - d) * b, pe, p)
            return p, m, v, pe, l

        if X_val is not None:
            Cvv = Cv if Cv is not None else jnp.zeros((Vf.shape[0], 0), jnp.float32)
            vloss = jax.jit(lambda q: jnp.mean(jnp.stack([loss_fn(q, Vf, vt[j], veps[j], Cvv) for j in range(8)])))
        best, best_p = np.inf, pe
        for i in range(steps):
            key, sk = jax.random.split(key)
            p, m, v, pe, l = step(p, m, v, pe, i, sk)
            if i % eval_every == 0 or i == steps - 1:
                vl = float(vloss(pe)) if X_val is not None else np.nan
                self.history.append((i, float(l), vl))
                if X_val is not None and vl < best:
                    best, best_p, self.best_step = vl, pe, i
                if verbose:
                    print(f"step {i:5d}  train {float(l):.4f}  validation {vl:.4f}")
        self.params = best_p if X_val is not None else pe
        return self

    # ------------------------------------------------------------------------------------------------------- sampling
    def _to_data(self, w):
        return w if self.A_ is None else w @ self.A_.T + self.mu_

    def sample(self, n: int, seed: int = 1, known: np.ndarray | None = None, known_mask: np.ndarray | None = None,
               clip: float = 12.0, batch: int = 2000, variance: str = "beta") -> np.ndarray:
        """Draw n windows (n, L, C). With `known` (L, C) and boolean `known_mask` (L, C), sample conditionally: the
        coordinates where the mask is True follow the scenario. The predicted clean window is clipped to +/- `clip`
        (data units) for numerical safety. `variance` is the reverse-step variance: "beta" (sigma_t^2 = beta_t, the
        upper bound of Ho et al., better calibrated when the denoiser is imperfect, as with a few thousand windows)
        or "posterior" (beta-tilde_t, the lower bound)."""
        if self.params is None:
            raise RuntimeError("fit the model first")
        outs = []
        for b0 in range(0, n, batch):
            outs.append(self._sample(min(batch, n - b0), seed + b0, known, known_mask, clip, variance))
        return np.concatenate(outs)

    def _sample(self, n, seed, known, known_mask, clip, variance):
        jax, jnp = _jax()
        p = self.params
        beta, alpha, abar = (jnp.asarray(a) for a in (self.beta, self.alpha, self.abar))
        A = None if self.A_ is None else jnp.asarray(self.A_)
        Ai = None if self.A_ is None else jnp.asarray(self.Ainv_)
        mu = jnp.asarray(self.mu_, jnp.float32)
        cond = known is not None
        kx = jnp.asarray(np.asarray(known, np.float32).reshape(-1)) if cond else None
        km = jnp.asarray(np.asarray(known_mask, bool).reshape(-1)) if cond else None
        key = jax.random.PRNGKey(seed)
        key, k0 = jax.random.split(key)
        w = jax.random.normal(k0, (n, self.D))

        @jax.jit
        def reverse(w, t, key):
            tt = jnp.full((w.shape[0],), t)
            eps = self._apply(p, w, tt)
            ab_prev = jnp.where(t > 0, abar[t - 1], 1.0)
            w0 = (w - jnp.sqrt(1 - abar[t]) * eps) / jnp.sqrt(abar[t])
            x0 = w0 if A is None else w0 @ A.T + mu
            x0 = jnp.clip(x0, -clip, clip)
            if cond:
                x0 = jnp.where(km[None], kx[None], x0)
            w0 = x0 if A is None else (x0 - mu) @ Ai.T
            mean = (beta[t] * jnp.sqrt(ab_prev) / (1 - abar[t])) * w0 + ((1 - ab_prev) * jnp.sqrt(alpha[t]) / (1 - abar[t])) * w
            var = jnp.where(t > 0, beta[t] if variance == "beta" else beta[t] * (1 - ab_prev) / (1 - abar[t]), 0.0)
            return mean + jnp.sqrt(var) * jax.random.normal(key, w.shape)

        for t in range(self.T - 1, -1, -1):
            key, sk = jax.random.split(key)
            w = reverse(w, t, sk)
        x = np.asarray(self._to_data(np.asarray(w)))
        out = x.reshape(n, self.length, self.channels)
        if cond:
            out = np.where(np.asarray(known_mask, bool)[None], np.asarray(known)[None], out)
        return out
