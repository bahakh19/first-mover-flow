"""Is a synthetic return generator any good? Fidelity, usefulness and memorisation diagnostics.

All functions take arrays of windows (n, L, C) in the SAME units (returns or standardised returns).

stylised_facts         per-channel mean, volatility, skew, excess kurtosis, lag-1 autocorrelation of returns and the
                       mean autocorrelation of |r| over lags 1-10 (volatility clustering), cross-correlations, lower-tail
                       dependence (5% quantile co-exceedance) between channel 0 and each other channel.
drawdown_profile       distribution of the within-window maximum drawdown of a channel (path risk).
discriminative_score   |accuracy - 0.5| of a classifier trained to tell real from synthetic windows (0 = indistinguishable;
                       Yoon et al. 2019), evaluated on held-out windows with a gradient-boosting classifier on summary
                       features plus the raw window.
predictive_score       train-on-synthetic, test-on-real (TSTR): MAE of a one-step-ahead ridge forecaster of channel 0
                       trained on synthetic windows and evaluated on real ones, relative to train-on-real.
memorisation_ratio     median distance from each synthetic window to its nearest TRAINING window divided by the median
                       distance from held-out real windows to their nearest training window. << 1 means the generator
                       copies its training data (a real risk with ~1,900 days of data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _acf(x: np.ndarray, k: int) -> float:
    x = x - x.mean(1, keepdims=True)
    num = (x[:, k:] * x[:, :-k]).sum(1)
    den = (x * x).sum(1)
    return float(np.nanmean(num / np.where(den > 0, den, np.nan)))


def stylised_facts(W: np.ndarray, names: list | None = None) -> pd.Series:
    """Moments, autocorrelations, cross-correlations and tail dependence of a set of windows."""
    n, L, C = W.shape
    names = names or [f"c{j}" for j in range(C)]
    flat = W.reshape(-1, C)
    out = {}
    for j, nm in enumerate(names):
        x = flat[:, j]
        out[f"{nm} mean x1e4"] = x.mean() * 1e4
        out[f"{nm} vol"] = x.std()
        out[f"{nm} skew"] = stats.skew(x)
        out[f"{nm} ex.kurt"] = stats.kurtosis(x)
        out[f"{nm} acf1"] = _acf(W[:, :, j], 1)
        out[f"{nm} acf|r| 1-10"] = np.mean([_acf(np.abs(W[:, :, j]), k) for k in range(1, 11)])
    for j in range(1, C):
        out[f"corr {names[0]}-{names[j]}"] = np.corrcoef(flat[:, 0], flat[:, j])[0, 1]
        q0, qj = np.quantile(flat[:, 0], 0.05), np.quantile(flat[:, j], 0.05)
        out[f"tail dep {names[0]}|{names[j]} (5%)"] = float(((flat[:, 0] <= q0) & (flat[:, j] <= qj)).mean() / 0.05)
    return pd.Series(out)


def max_drawdowns(W: np.ndarray, channel: int = 0) -> np.ndarray:
    """Within-window maximum drawdown of a channel for every window."""
    eq = np.cumprod(1 + W[:, :, channel], axis=1)
    return (eq / np.maximum.accumulate(eq, axis=1) - 1).min(1)


def drawdown_profile(W: np.ndarray, channel: int = 0) -> pd.Series:
    """Mean, 5% and 1% quantiles of the within-window maximum drawdown of a channel."""
    d = max_drawdowns(W, channel)
    return pd.Series({"mean maxDD": d.mean(), "5% maxDD": np.quantile(d, 0.05), "1% maxDD": np.quantile(d, 0.01)})


def _features(W: np.ndarray) -> np.ndarray:
    f = [W.mean(1), W.std(1), stats.skew(W, axis=1), stats.kurtosis(W, axis=1), np.abs(W).max(1),
         np.cumsum(W, 1)[:, -1], np.array([[_acf(W[i:i + 1, :, j], 1) for j in range(W.shape[2])] for i in range(len(W))])]
    if W.shape[2] > 1:
        f.append(np.array([[np.corrcoef(w[:, 0], w[:, j])[0, 1] for j in range(1, W.shape[2])] for w in W]))
    return np.nan_to_num(np.column_stack(f + [W.reshape(len(W), -1)]))


def discriminative_score(real: np.ndarray, synth: np.ndarray, seed: int = 0, test_share: float = 0.3) -> float:
    """|accuracy - 0.5| of a classifier separating real from synthetic windows.

    Real windows usually overlap (stride 1), so a random train/test split would let the classifier recognise a test
    window from its one-day-shifted twin in the training set. The real windows are therefore split CHRONOLOGICALLY with
    a gap of one window length; synthetic windows are split at random."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    L = real.shape[1]
    n_te = int(len(real) * test_share)
    r_tr, r_te = real[: len(real) - n_te - L], real[len(real) - n_te:]
    s = synth[rng.permutation(len(synth))]
    if len(s) < len(r_tr) + len(r_te):                    # balance classes when fewer synthetic windows are available
        k = len(s) / (len(r_tr) + len(r_te))
        r_tr = r_tr[rng.choice(len(r_tr), max(1, int(k * len(r_tr))), replace=False)]
        r_te = r_te[rng.choice(len(r_te), max(1, int(k * len(r_te))), replace=False)]
    s_tr, s_te = s[: len(r_tr)], s[len(r_tr): len(r_tr) + len(r_te)]
    Xtr = np.vstack([_features(r_tr), _features(s_tr)])
    ytr = np.r_[np.ones(len(r_tr)), np.zeros(len(s_tr))]
    Xte = np.vstack([_features(r_te), _features(s_te)])
    yte = np.r_[np.ones(len(r_te)), np.zeros(len(s_te))]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xtr, ytr)
    return float(abs((clf.predict(Xte) == yte).mean() - 0.5))


def predictive_score(real_train: np.ndarray, real_test: np.ndarray, synth: np.ndarray, lags: int = 5) -> dict:
    """Train-on-synthetic, test-on-real MAE of a one-step |return| forecaster, relative to train-on-real."""
    from sklearn.linear_model import Ridge

    def xy(W):
        X, y = [], []
        for t in range(lags, W.shape[1]):
            X.append(np.abs(W[:, t - lags:t, :]).reshape(len(W), -1))
            y.append(np.abs(W[:, t, 0]))
        return np.vstack(X), np.concatenate(y)
    Xr, yr = xy(real_train)
    Xs, ys = xy(synth)
    Xt, yt = xy(real_test)
    mae_trtr = float(np.abs(Ridge(1.0).fit(Xr, yr).predict(Xt) - yt).mean())
    mae_tstr = float(np.abs(Ridge(1.0).fit(Xs, ys).predict(Xt) - yt).mean())
    return {"MAE train-real": mae_trtr, "MAE train-synthetic": mae_tstr, "ratio": mae_tstr / mae_trtr}


def memorisation_ratio(train: np.ndarray, holdout: np.ndarray, synth: np.ndarray, max_n: int = 2000, seed: int = 0) -> float:
    """Median nearest-training-window distance of synthetic vs held-out real windows (<< 1 = copying)."""
    rng = np.random.default_rng(seed)
    T = train.reshape(len(train), -1)
    def nn(A):
        A = A.reshape(len(A), -1)
        A = A[rng.choice(len(A), min(len(A), max_n), replace=False)]
        d = ((A ** 2).sum(1)[:, None] + (T ** 2).sum(1)[None] - 2 * A @ T.T)
        return np.sqrt(np.maximum(d.min(1), 0))
    return float(np.median(nn(synth)) / np.median(nn(holdout)))
