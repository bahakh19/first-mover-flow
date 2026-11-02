"""Data-snooping-robust inference for families of strategies (all bootstrap-based, dependence-preserving).

Every function takes a (T x K) matrix `d` of per-period performance differentials: d[t, k] is the return of candidate k
minus the return of the benchmark (pass raw returns to test against zero). Inference uses the stationary bootstrap
(Politis and Romano, 1994) so serial dependence and the cross-correlation between candidates are preserved.

reality_check       White (2000): H0 "no candidate beats the benchmark", non-studentised, conservative.
spa_test            Hansen (2005): studentised, with the consistent recentring that discards poor candidates, so adding
                    irrelevant strategies to the family does not inflate the p-value.
romano_wolf         Romano and Wolf (2005) StepM with step-down adjusted p-values: WHICH candidates beat the benchmark,
                    controlling the family-wise error rate under arbitrary dependence.
sharpe_diff_test    circular-block-bootstrap test of a Sharpe-ratio difference (Ledoit and Wolf, 2008, basic bootstrap
                    version), robust to fat tails and serial correlation, unlike Jobson-Korkie/Memmel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fmflow.analytics.inference import stationary_bootstrap_indices


def _as_matrix(d) -> tuple[np.ndarray, list]:
    if isinstance(d, pd.Series):
        d = d.to_frame()
    if isinstance(d, pd.DataFrame):
        d = d.dropna()
        return d.to_numpy(float), list(d.columns)
    a = np.asarray(d, float)
    a = a[:, None] if a.ndim == 1 else a
    return a[np.isfinite(a).all(1)], list(range(a.shape[1]))


def _boot_means(X: np.ndarray, n_boot: int, block: int, seed: int) -> np.ndarray:
    idx = stationary_bootstrap_indices(len(X), n_boot, block, seed)
    return np.stack([X[i].mean(0) for i in idx])          # (B, K)


def reality_check(d, n_boot: int = 2000, block: int = 21, seed: int = 0) -> dict:
    """White's (2000) Reality Check p-value for the best of K candidates against the benchmark."""
    X, _ = _as_matrix(d)
    T = len(X)
    m = X.mean(0)
    stat = np.sqrt(T) * m.max()
    bm = _boot_means(X, n_boot, block, seed)
    null = np.sqrt(T) * (bm - m).max(1)
    return {"statistic": float(stat), "p_value": float((1 + (null >= stat).sum()) / (1 + n_boot)), "best": int(m.argmax())}


def spa_test(d, n_boot: int = 2000, block: int = 21, seed: int = 0) -> dict:
    """Hansen's SPA test. Returns p-values for the consistent (c), lower (l) and upper (u) recentrings."""
    X, names = _as_matrix(d)
    T = len(X)
    m = X.mean(0)
    bm = _boot_means(X, n_boot, block, seed)
    omega = np.sqrt(T) * bm.std(0, ddof=1)
    omega = np.where(omega > 0, omega, np.nan)
    tstat = np.sqrt(T) * m / omega
    stat = max(np.nanmax(tstat), 0.0)
    thr = -np.sqrt(2 * np.log(np.log(T)))
    out = {}
    for tag, g in (("l", np.maximum(m, 0)), ("c", np.where(tstat >= thr, m, 0.0)), ("u", m)):
        z = np.sqrt(T) * (bm - g) / omega
        null = np.maximum(np.nanmax(z, axis=1), 0.0)
        out[f"p_{tag}"] = float((1 + (null >= stat).sum()) / (1 + n_boot))
    best = int(np.nanargmax(tstat))
    return {"statistic": float(stat), "p_value": out["p_c"], **out, "best": names[best], "n_candidates": X.shape[1]}


def romano_wolf(d, n_boot: int = 2000, block: int = 21, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """StepM with studentised statistics. H0_k: E[d_k] <= 0. Returns t-stats, raw and FWER-adjusted p-values."""
    X, names = _as_matrix(d)
    T, K = X.shape
    m = X.mean(0)
    bm = _boot_means(X, n_boot, block, seed)
    se = bm.std(0, ddof=1)
    t = m / se
    tb = (bm - m) / se                                  # bootstrap null distribution of the studentised statistics
    order = np.argsort(-t)
    adj = np.empty(K)
    for j, k in enumerate(order):
        rest = order[j:]
        adj[k] = (1 + (tb[:, rest].max(1) >= t[k]).sum()) / (1 + n_boot)
    adj_sorted = np.maximum.accumulate(adj[order])      # monotone step-down
    adj[order] = adj_sorted
    raw = np.array([(1 + (tb[:, k] >= t[k]).sum()) / (1 + n_boot) for k in range(K)])
    return pd.DataFrame({"mean": m, "t_boot": t, "p_raw": raw, "p_romano_wolf": adj, "reject": adj <= alpha},
                        index=names).sort_values("t_boot", ascending=False)


def _sharpe(x: np.ndarray, periods: int = 252) -> np.ndarray:
    return x.mean(0) / x.std(0, ddof=1) * np.sqrt(periods)


def sharpe_diff_test(r1: pd.Series, r2: pd.Series, n_boot: int = 2000, block: int = 21, seed: int = 0,
                     periods: int = 252) -> dict:
    """Two-sided test of SR(r1) = SR(r2) by circular block bootstrap of the PAIRS (keeps their correlation)."""
    a, b = r1.align(r2, join="inner")
    X = pd.concat([a, b], axis=1).dropna().to_numpy(float)
    T = len(X)
    obs = _sharpe(X[:, 0], periods) - _sharpe(X[:, 1], periods)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:T] % T
        Y = X[idx]
        diffs[i] = _sharpe(Y[:, 0], periods) - _sharpe(Y[:, 1], periods)
    se = diffs.std(ddof=1)
    p = float((1 + (np.abs(diffs - obs) >= abs(obs)).sum()) / (1 + n_boot))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"sharpe_diff": float(obs), "se": float(se), "ci95": (float(2 * obs - hi), float(2 * obs - lo)), "p_value": p}
