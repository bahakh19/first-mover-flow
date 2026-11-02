"""Statistical inference for small, overlapping, heavily searched samples.

rank_ic            daily Spearman cross-sectional IC between a signal and forward log returns
ic_phase_test      phase-averaged NON-overlapping t-test for an IC series with h-session overlapping labels (calibrated
                   size ~4-5%; the Newey-West t with h lags over-rejects at ~9% on ~730 daily obs, h = 21)
holm               Holm (1979) step-down adjusted p-values
psr / deflated_sharpe / expected_max_sr / min_track_record   Bailey & Lopez de Prado (2012, 2014)
memmel_test        Jobson-Korkie test of a Sharpe-ratio difference with Memmel's (2003) correction
stationary_bootstrap_indices   Politis & Romano (1994)
cscv_pbo           probability of backtest overfitting by combinatorially symmetric cross-validation (Bailey et al. 2017)
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def forward_log_return(px: pd.DataFrame, h: int) -> pd.DataFrame:
    """LABEL (forward-looking by design): log return from close t to close t+h. Never use as a feature."""
    return np.log(px.shift(-h) / px)


def rank_ic(feature: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 6) -> pd.Series:
    """Daily cross-sectional Spearman correlation between a feature and forward returns (NaN below `min_names`)."""
    ok = feature.notna() & fwd.notna()
    fr, yr = feature.where(ok).rank(axis=1), fwd.where(ok).rank(axis=1)
    fc, yc = fr.sub(fr.mean(axis=1), axis=0), yr.sub(yr.mean(axis=1), axis=0)
    ic = (fc * yc).sum(axis=1) / np.sqrt((fc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1))
    return ic.where(ok.sum(axis=1) >= min_names)


def ic_phase_test(ic: pd.Series, h: int = 21) -> dict:
    """One-sided test of a mean IC whose h-period labels overlap: average of the h non-overlapping sub-sample standard errors."""
    z = ic.dropna().values
    if len(z) < 6 * h:
        return {"IC": float(np.mean(z)) if len(z) else np.nan, "t": np.nan, "df": np.nan, "p": np.nan}
    ses, ns = [], []
    for o in range(h):
        s_ = z[o::h]
        if len(s_) > 5:
            ses.append(s_.std(ddof=1) / np.sqrt(len(s_)))
            ns.append(len(s_))
    se, df_ = float(np.mean(ses)), int(np.median(ns)) - 1
    t_ = z.mean() / se
    return {"IC": float(z.mean()), "t": float(t_), "df": df_, "p": float(stats.t.sf(t_, df_))}


def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = p.dropna().sort_values()
    m = len(p)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(p.values)]).clip(max=1.0)
    return pd.Series(adj, index=p.index)


def psr(r, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe ratio P(true SR > sr_star); sr_star in per-period (daily) units."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or not r.std(ddof=1) > 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    den = np.sqrt(max(1e-12, 1 - stats.skew(r) * sr + (stats.kurtosis(r, fisher=False) - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(len(r) - 1) / den))


def expected_max_sr(sr_std: float, n_eff: float) -> float:
    """Expected maximum of n_eff independent null Sharpe ratios with cross-trial std sr_std (same units)."""
    if n_eff <= 1:
        return 0.0
    # The asymptotic formula turns negative for n_eff close to 1; the expected maximum of ~one null trial is 0.
    return float(max(0.0, sr_std * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_eff)
                                    + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_eff * np.e)))))


def deflated_sharpe(r, trials: pd.DataFrame | None = None, n_eff: float | None = None, sr_std: float | None = None) -> dict:
    """DSR of return series `r` given the matrix of all trial return series (columns) or (n_eff, sr_std)."""
    if trials is None and (n_eff is None or sr_std is None):
        raise ValueError("deflated_sharpe needs either `trials` (one column per trial) or both `n_eff` and `sr_std`")
    if trials is not None:
        C = trials.corr().values
        M = trials.shape[1]
        rho = (C.sum() - M) / (M * (M - 1)) if M > 1 else 0.0
        n_eff = rho + (1 - rho) * M
        sr_std = float(trials.apply(lambda c: c.mean() / c.std(ddof=1)).std(ddof=1))
    sr0 = expected_max_sr(sr_std, n_eff)
    return {"n_eff": float(n_eff), "SR0_daily": sr0, "PSR": psr(r), "DSR": psr(r, sr0)}


def min_track_record(r, sr_star: float = 0.0, alpha: float = 0.05, periods: int = 252) -> float:
    """Minimum track record length (years) for the observed Sharpe ratio to exceed `sr_star` at level `alpha`."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    sr = r.mean() / r.std(ddof=1)
    if sr <= sr_star:
        return float("inf")
    n = 1 + (1 - stats.skew(r) * sr + (stats.kurtosis(r, fisher=False) - 1) / 4 * sr ** 2) * \
        (stats.norm.ppf(1 - alpha) / (sr - sr_star)) ** 2
    return float(n / periods)


def memmel_test(r1: pd.Series, r2: pd.Series) -> tuple[float, float]:
    """Jobson-Korkie test of equal Sharpe ratios with Memmel's (2003) correction; returns (z, two-sided p)."""
    a, b = r1.dropna().align(r2.dropna(), join="inner")
    T = len(a)
    s1, s2 = a.mean() / a.std(), b.mean() / b.std()
    rho = np.corrcoef(a, b)[0, 1]
    v = (2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / T
    z = (s1 - s2) / np.sqrt(v)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def stationary_bootstrap_indices(n: int, n_boot: int = 1000, block: int = 21, seed: int = 42) -> np.ndarray:
    """Index matrix (n_boot x n) of Politis-Romano stationary-bootstrap resamples with mean block `block`."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, n), int)
    for b in range(n_boot):
        i = 0
        while i < n:
            s_ = rng.integers(0, n)
            L = min(rng.geometric(1 / block), n - i)
            out[b, i:i + L] = (s_ + np.arange(L)) % n
            i += L
    return out


def bootstrap_calmar(r: np.ndarray, idx: np.ndarray, periods: int = 252) -> np.ndarray:
    """Calmar ratio of each bootstrap resample (rows of `idx`) of the return series `r`."""
    R = np.asarray(r)[idx]
    eq = np.cumprod(1 + R, axis=1)
    cagr = eq[:, -1] ** (periods / R.shape[1]) - 1
    mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
    return cagr / np.abs(mdd)


def cscv_pbo(trials: pd.DataFrame, n_blocks: int = 16) -> dict:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation (Bailey et al., 2017).

    `trials` holds one return series per specification (columns); `n_blocks` must be even."""
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even number >= 2")
    if trials.shape[1] < 2:
        raise ValueError("cscv_pbo needs at least two trials (columns)")
    X = trials.dropna().to_numpy()
    if len(X) < n_blocks:
        raise ValueError(f"cscv_pbo needs at least n_blocks = {n_blocks} complete rows, got {len(X)}")
    blocks = np.array_split(np.arange(len(X)), n_blocks)
    mu = np.array([X[b].mean(0) for b in blocks])
    sq = np.array([(X[b] ** 2).sum(0) for b in blocks])
    n = np.array([len(b) for b in blocks])

    def sharpe(idx):
        m = (mu[idx] * n[idx, None]).sum(0) / n[idx].sum()
        v = sq[idx].sum(0) / n[idx].sum() - m ** 2
        return np.nan_to_num(m / np.sqrt(np.maximum(v, 1e-18)))

    logits, oos = [], []
    for c in combinations(range(n_blocks), n_blocks // 2):
        i_, o_ = np.array(c), np.setdiff1d(np.arange(n_blocks), c)
        si, so = sharpe(i_), sharpe(o_)
        b = int(np.argmax(si))
        w = min(max((so < so[b]).sum() / (len(so) - 1), 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
        oos.append(so[b] * np.sqrt(252))
    return {"PBO": float((np.array(logits) <= 0).mean()), "median_oos_sharpe_of_is_winner": float(np.median(oos)),
            "n_splits": len(logits)}


def fama_macbeth(feature: pd.DataFrame, controls: dict, fwd: pd.DataFrame, step: int = 5, min_names: int = 6,
                 lags: int | None = None, horizon: int = 21) -> pd.DataFrame:
    """Fama-MacBeth regressions of forward returns on the cross-sectional RANK of the feature and of each control
    (ranks centred to mean zero each date), every `step` sessions. Returns mean slopes with Newey-West t-statistics.

    `horizon` is the length (sessions) of the forward-return labels in `fwd`; by default the Newey-West lag is
    horizon / step, which absorbs the overlap of consecutive labels. Pass `lags` to override."""
    from fmflow.signals.composite import centred_rank
    X = {"feature": centred_rank(feature, 2), **{k: centred_rank(v, 2) for k, v in controls.items()}}
    dates = fwd.index[::step]
    rows = []
    for d in dates:
        y = fwd.loc[d]
        df = pd.DataFrame({k: v.loc[d] for k, v in X.items()}).assign(y=y).dropna()
        if len(df) < max(min_names, len(X) + 2):
            continue
        A = np.column_stack([np.ones(len(df)), df[list(X)].to_numpy()])
        b = np.linalg.lstsq(A, df["y"].to_numpy(), rcond=None)[0]
        rows.append(pd.Series(b[1:], index=list(X), name=d))
    B = pd.DataFrame(rows)
    if B.empty:
        return pd.DataFrame(columns=["slope", "t_NW", "n_dates"])
    L = lags if lags is not None else max(1, int(round(horizon / step)))
    out = {}
    for c in B.columns:
        x = B[c].to_numpy() - B[c].mean()
        T = len(x)
        v = (x @ x) / T + 2 * sum((1 - k / (L + 1)) * (x[k:] @ x[:-k]) / T for k in range(1, min(L, T - 1) + 1))
        out[c] = {"slope": float(B[c].mean()), "t_NW": float(B[c].mean() / np.sqrt(max(v, 1e-30) / T)), "n_dates": T}
    return pd.DataFrame(out).T
