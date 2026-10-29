"""Training windows for generative models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowData:
    """Standardised training windows and the statistics needed to map draws back to returns."""
    X: np.ndarray            # (n_windows, L, C) standardised
    mean: np.ndarray         # (C,) per-channel mean used for standardisation
    std: np.ndarray          # (C,)
    channels: list
    length: int

    def to_returns(self, Z: np.ndarray) -> np.ndarray:
        """Map standardised windows back to returns with the pooled mean and standard deviation."""
        return Z * self.std + self.mean


def make_windows(samples: list[pd.DataFrame] | pd.DataFrame, length: int = 64, stride: int = 1,
                 channels: list | None = None) -> WindowData:
    """Overlapping windows from one or several (T x C) daily-return frames with identical columns.

    Each sample is standardised with its OWN mean and standard deviation (so pooling a 10-stock and a 30-stock universe
    pools their dynamics, not their scales); the pooled mean/std returned are the averages, used to map draws back to
    returns. Windows never straddle two samples."""
    samples = [samples] if isinstance(samples, pd.DataFrame) else list(samples)
    channels = list(channels or samples[0].columns)
    Xs, means, stds = [], [], []
    for s in samples:
        a = s[channels].dropna().to_numpy(float)
        m, sd = a.mean(0), a.std(0, ddof=1)
        z = (a - m) / sd
        idx = np.arange(0, len(z) - length + 1, stride)
        if len(idx) == 0:                               # shorter than one window: contributes nothing
            continue
        Xs.append(np.stack([z[i:i + length] for i in idx]))
        means.append(m)
        stds.append(sd)
    if not Xs:
        raise ValueError(f"no sample has the {length} complete observations needed for one window")
    return WindowData(np.concatenate(Xs).astype(np.float32), np.mean(means, 0), np.mean(stds, 0), channels, length)


def stitch(windows: np.ndarray, n_paths: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
    """Concatenate randomly drawn generated windows into (n_paths, horizon, C) paths."""
    L = windows.shape[1]
    k = int(np.ceil(horizon / L))
    pick = rng.integers(0, len(windows), size=(n_paths, k))
    return windows[pick].reshape(n_paths, k * L, -1)[:, :horizon]


class VolScaled:
    """Two-scale representation for generative models: each window is split into its per-channel realised-volatility
    level (log scale, standardised across windows) and the within-window returns divided by that level. The two are
    stacked as (L + 1, C): row 0 holds the standardised log-volatilities, rows 1..L the vol-normalised returns.

    Heavy tails in daily returns are mostly a scale mixture -- calm and turbulent months -- which a network fed with raw
    windows tends to average away. Modelling the scale explicitly lets a generator reproduce the mixture (and hence the
    kurtosis and the drawdown tail) while learning the within-window shape separately."""

    def __init__(self, X: np.ndarray):
        lv = np.log(X.std(1) + 1e-8)                       # (n, C)
        self.m, self.s = lv.mean(0), lv.std(0) + 1e-8

    def encode(self, X: np.ndarray) -> np.ndarray:
        """(n, L, C) windows -> (n, L + 1, C): standardised log-volatility row, then vol-normalised returns."""
        sd = X.std(1, keepdims=True) + 1e-8
        lv = (np.log(sd[:, 0]) - self.m) / self.s
        return np.concatenate([lv[:, None, :], X / sd], axis=1).astype(np.float32)

    def decode(self, Y: np.ndarray) -> np.ndarray:
        """Inverse of `encode`: rescale the within-window returns by the decoded volatility level."""
        sd = np.exp(Y[:, 0] * self.s + self.m)[:, None, :]
        Z = Y[:, 1:]
        Z = (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-8) + Z.mean(1, keepdims=True)
        return Z * sd


class GarchScaled:
    """Econometric-deep hybrid representation: r_t = sigma_t * z_t per channel, with sigma_t the in-sample GJR-GARCH(1,1)-t
    conditional volatility. Each day is described by (z_t, standardised log sigma_t) for every channel, so a window of
    length L becomes an (L, 2C) array. The generator then learns the persistent volatility path (clustering) and the
    innovations jointly -- including their cross-channel dependence and any nonlinearity the GARCH model misses -- and the
    product sigma_t * z_t restores fat tails and volatility clustering by construction."""

    def __init__(self, samples: list[pd.DataFrame], channels: list | None = None):
        from arch import arch_model

        from fmflow.risk.garch import fit_quietly
        self.channels = list(channels or samples[0].columns)
        self.frames, self.scales = [], []
        for smp in samples:
            a = smp[self.channels].dropna()
            sd = a.std()
            zs, ls = {}, {}
            for c in self.channels:
                f = fit_quietly(arch_model(a[c] / sd[c] * 10, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t"))
                sig = f.conditional_volatility / 10
                zs[c] = (a[c] / sd[c] - (a[c] / sd[c]).mean()) / sig
                ls[c] = np.log(sig)
            self.frames.append(pd.concat({**{f"z_{c}": zs[c] for c in self.channels},
                                          **{f"ls_{c}": ls[c] for c in self.channels}}, axis=1).dropna())
            self.scales.append(sd)
        allf = pd.concat(self.frames)
        allls = allf[[f"ls_{c}" for c in self.channels]]
        self.lm, self.ls = allls.mean().to_numpy(), allls.std().to_numpy() + 1e-8
        zs_ = ((allls - self.lm) / self.ls).to_numpy()
        self.ls_lo, self.ls_hi = zs_.min(0), zs_.max(0)
        self.z_max = np.abs(allf[[f"z_{c}" for c in self.channels]].to_numpy()).max(0)
        self.sd = np.mean([s.to_numpy() for s in self.scales], 0)

    def windows(self, length: int = 64, stride: int = 1) -> np.ndarray:
        """(n, length, 2C) training windows of innovations and standardised log-volatilities."""
        C = len(self.channels)
        out = []
        for f in self.frames:
            a = f.to_numpy(float).copy()
            a[:, C:] = (a[:, C:] - self.lm) / self.ls
            idx = np.arange(0, len(a) - length + 1, stride)
            out.append(np.stack([a[i:i + length] for i in idx]))
        return np.concatenate(out).astype(np.float32)

    def decode(self, Y: np.ndarray, units: str = "returns", extrapolation: float = 0.5) -> np.ndarray:
        """(n, L, 2C) -> (n, L, C) returns in the pooled original units (or standardised units if units='standardised').

        Support constraint: generated log-volatility is kept within the observed range widened by `extrapolation`
        standard deviations (0.5 by default: volatility up to ~exp(0.5 * sd) beyond the historical extreme), and
        innovations within 1.25 x the largest observed |z|. Without it, the exponential map turns rare denoising
        errors into absurd volatilities; with it, the generator can still produce paths worse than history."""
        C = len(self.channels)
        lsz = np.clip(Y[:, :, C:], self.ls_lo - extrapolation, self.ls_hi + extrapolation)
        sig = np.exp(lsz * self.ls + self.lm)
        z = np.clip(Y[:, :, :C], -1.25 * self.z_max, 1.25 * self.z_max)
        r = z * sig
        return r * self.sd if units == "returns" else r

    def real_windows(self, length: int = 64, stride: int = 1, units: str = "returns") -> np.ndarray:
        """The historical windows decoded like generated ones (the reference for fidelity tests)."""
        return self.decode(self.windows(length, stride), units)
