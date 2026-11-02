"""TimeGAN (Yoon, Jarrett and van der Schaar, NeurIPS 2019), compact JAX implementation.

Five GRU networks share a latent space H:
    embedder   X -> H          recovery   H -> X~       (autoencoder, reconstruction loss L_R)
    generator  Z -> E^         supervisor H -> H_next   (one-step latent dynamics, supervised loss L_S)
    discriminator  H -> logits (real latent sequences vs synthetic ones)
Training follows the paper's three phases: (1) embedding/recovery on L_R; (2) supervisor on L_S, which teaches the
generator the stepwise conditional dynamics that a purely adversarial loss learns poorly; (3) joint training where the
generator minimises the adversarial losses on S(E^) and E^, sqrt(L_S) and a first/second-moment matching loss, the
embedder minimises L_R + 0.1 L_S, and the discriminator is updated only while its loss exceeds 0.15.
Data are min-max scaled per channel to [0, 1] (as in the paper); `sample` returns the original scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fmflow.generative.ddpm import _jax


@dataclass
class TimeGAN:
    """TimeGAN with GRU embedder, recovery, generator, supervisor and discriminator (see module docstring)."""
    length: int
    channels: int
    hidden: int = 24
    z_dim: int | None = None
    gamma: float = 1.0
    seed: int = 0
    params: dict = field(default=None, repr=False)
    history: list = field(default_factory=list, repr=False)

    # -------------------------------------------------------------------------------------------------------- networks
    def _gru_init(self, key, i, h):
        jax, jnp = _jax()
        k = jax.random.split(key, 2)
        s = 1 / np.sqrt(h)
        return {"W": jax.random.uniform(k[0], (i, 3 * h), minval=-s, maxval=s),
                "U": jax.random.uniform(k[1], (h, 3 * h), minval=-s, maxval=s), "b": jnp.zeros(3 * h)}

    def _dense_init(self, key, i, o):
        jax, jnp = _jax()
        return {"w": jax.random.normal(key, (i, o)) * np.sqrt(1.0 / i), "b": jnp.zeros(o)}

    def _gru(self, p, x):                                    # x: (B, L, i) -> (B, L, h)
        jax, jnp = _jax()
        h_dim = p["U"].shape[0]

        def cell(h, xt):
            gx = xt @ p["W"] + p["b"]
            gh = h @ p["U"]
            r = jax.nn.sigmoid(gx[:, :h_dim] + gh[:, :h_dim])
            z = jax.nn.sigmoid(gx[:, h_dim:2 * h_dim] + gh[:, h_dim:2 * h_dim])
            n = jnp.tanh(gx[:, 2 * h_dim:] + r * gh[:, 2 * h_dim:])
            h = (1 - z) * n + z * h
            return h, h
        h0 = jnp.zeros((x.shape[0], h_dim))
        _, hs = jax.lax.scan(cell, h0, jnp.swapaxes(x, 0, 1))
        return jnp.swapaxes(hs, 0, 1)

    def _net(self, p, x, out_act):
        jax, jnp = _jax()
        h = self._gru(p["gru"], x)
        y = h @ p["d"]["w"] + p["d"]["b"]
        return jax.nn.sigmoid(y) if out_act == "sigmoid" else y

    def _init(self, key):
        jax, _ = _jax()
        H, C, Z = self.hidden, self.channels, self.z_dim or self.channels
        ks = jax.random.split(key, 10)
        def mk(i, o, k1, k2):
            return {"gru": self._gru_init(k1, i, H), "d": self._dense_init(k2, H, o)}
        return {"emb": mk(C, H, ks[0], ks[1]), "rec": mk(H, C, ks[2], ks[3]), "gen": mk(Z, H, ks[4], ks[5]),
                "sup": mk(H, H, ks[6], ks[7]), "dis": mk(H, 1, ks[8], ks[9])}

    # -------------------------------------------------------------------------------------------------------- training
    def fit(self, X: np.ndarray, steps: tuple = (1500, 1500, 2500), batch: int = 128, lr: float = 1e-3,
            verbose: int = 0) -> TimeGAN:
        """Train on (n, L, C) windows: embedding, supervised and joint adversarial phases (`steps` iterations each)."""
        jax, jnp = _jax()
        self.lo, self.hi = X.min((0, 1)), X.max((0, 1))
        Xs = jnp.asarray((X - self.lo) / (self.hi - self.lo + 1e-8), jnp.float32)
        key = jax.random.PRNGKey(self.seed)
        key, k0 = jax.random.split(key)
        P = self._init(k0)
        Zd = self.z_dim or self.channels
        L = self.length

        def bce(logits, y):
            return jnp.mean(jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits))))

        def forward(P, x, z):
            H = self._net(P["emb"], x, "sigmoid")
            Xt = self._net(P["rec"], H, "sigmoid")
            E = self._net(P["gen"], z, "sigmoid")
            Hh = self._net(P["sup"], E, "sigmoid")
            Hs = self._net(P["sup"], H, "sigmoid")
            Xh = self._net(P["rec"], Hh, "sigmoid")
            return H, Xt, E, Hh, Hs, Xh

        def l_rec(P, x, z):
            H, Xt, *_ = forward(P, x, z)
            return jnp.mean((x - Xt) ** 2)

        def l_sup(P, x, z):
            H, _, _, _, Hs, _ = forward(P, x, z)
            return jnp.mean((H[:, 1:] - Hs[:, :-1]) ** 2)

        def l_gen(P, x, z):
            H, _, E, Hh, Hs, Xh = forward(P, x, z)
            yf = self._net(P["dis"], Hh, "linear")
            ye = self._net(P["dis"], E, "linear")
            lu = bce(yf, 1.0) + self.gamma * bce(ye, 1.0)
            ls = jnp.mean((H[:, 1:] - Hs[:, :-1]) ** 2)
            v1 = jnp.mean(jnp.abs(jnp.sqrt(Xh.var(0) + 1e-6) - jnp.sqrt(x.var(0) + 1e-6)))
            v2 = jnp.mean(jnp.abs(Xh.mean(0) - x.mean(0)))
            return lu + 100 * jnp.sqrt(ls) + 100 * (v1 + v2)

        def l_emb(P, x, z):
            return 10 * jnp.sqrt(l_rec(P, x, z)) + 0.1 * l_sup(P, x, z)

        def l_dis(P, x, z):
            H, _, E, Hh, *_ = forward(P, x, z)
            return (bce(self._net(P["dis"], H, "linear"), 1.0) + bce(self._net(P["dis"], Hh, "linear"), 0.0)
                    + self.gamma * bce(self._net(P["dis"], E, "linear"), 0.0))

        def make_step(loss, groups):
            @jax.jit
            def s(P, st_m, st_v, i, x, z):
                l, g = jax.value_and_grad(loss)(P, x, z)
                m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, st_m, g)
                v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, st_v, g)
                newP = dict(P)
                for k in groups:
                    newP[k] = jax.tree_util.tree_map(lambda w, a, b: w - lr * (a / (1 - 0.9 ** i)) / (jnp.sqrt(b / (1 - 0.999 ** i)) + 1e-8),
                                                     P[k], m[k], v[k])
                return newP, m, v, l
            return s

        st_rec, st_sup = make_step(l_rec, ["emb", "rec"]), make_step(l_sup, ["sup", "gen"])
        st_gen, st_emb, st_dis = make_step(l_gen, ["gen", "sup"]), make_step(l_emb, ["emb", "rec"]), make_step(l_dis, ["dis"])
        def zeros():
            return jax.tree_util.tree_map(jnp.zeros_like, P)
        opt = {n: [zeros(), zeros(), 0] for n in ("rec", "sup", "gen", "emb", "dis")}
        dis_loss = jax.jit(l_dis)

        def batch_xz(key):
            k1, k2 = jax.random.split(key)
            idx = jax.random.randint(k1, (batch,), 0, Xs.shape[0])
            return Xs[idx], jax.random.uniform(k2, (batch, L, Zd))

        def run(name, fn, P, key):
            key, sk = jax.random.split(key)
            x, z = batch_xz(sk)
            m, v, i = opt[name]
            P, m, v, l = fn(P, m, v, i + 1, x, z)
            opt[name] = [m, v, i + 1]
            return P, key, float(l), x, z

        for it in range(steps[0]):
            P, key, l, *_ = run("rec", st_rec, P, key)
            if it % 500 == 0:
                self.history.append(("embedding", it, l))
        for it in range(steps[1]):
            P, key, l, *_ = run("sup", st_sup, P, key)
            if it % 500 == 0:
                self.history.append(("supervised", it, l))
        for it in range(steps[2]):
            for _ in range(2):
                P, key, lg, *_ = run("gen", st_gen, P, key)
                P, key, le, *_ = run("emb", st_emb, P, key)
            key, sk = jax.random.split(key)
            x, z = batch_xz(sk)
            if float(dis_loss(P, x, z)) > 0.15:
                P, key, *_ = run("dis", st_dis, P, key)
            if it % 250 == 0:
                self.history.append(("joint", it, lg))
                if verbose:
                    print(f"joint {it:5d}  G {lg:.3f}  E {le:.3f}")
        self.params = P
        return self

    def sample(self, n: int, seed: int = 1) -> np.ndarray:
        """Draw n synthetic (L, C) windows."""
        jax, jnp = _jax()
        z = jax.random.uniform(jax.random.PRNGKey(seed), (n, self.length, self.z_dim or self.channels))
        P = self.params
        E = self._net(P["gen"], z, "sigmoid")
        Xh = self._net(P["rec"], self._net(P["sup"], E, "sigmoid"), "sigmoid")
        return np.asarray(Xh) * (self.hi - self.lo + 1e-8) + self.lo
