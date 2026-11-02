"""Placebo nulls passed through the IDENTICAL construction.

iid          : random N(0,1) scores every session on the real NaN mask
permutation  : stock-label derangements; each stock's real signal history is assigned to another stock's returns
               (preserves the signal's persistence and cross-sectional structure)
time_shift   : circular shift of the whole signal panel by >= `min_shift` sessions (also preserves each stock's
               habitual rank: the hardest null; asks whether TIMING adds value beyond stock identity)
p-value = (1 + #{null >= observed}) / (1 + draws).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd


def placebo_test(score: pd.DataFrame, book_returns: Callable[[pd.DataFrame], pd.Series],
                 windows: Mapping[str, slice], metric: Callable[[pd.Series], float], n_iid: int = 300,
                 n_perm: int = 500, shift_step: int = 4, min_shift: int = 126, seed: int = 42) -> pd.DataFrame:
    """Observed statistic vs IID, stock-permutation and circular time-shift null scores passed through `book_returns`."""
    if score.shape[1] < 2:
        raise ValueError("placebo_test needs at least two stocks (the permutation null swaps stock labels)")
    rng = np.random.default_rng(seed)

    def stats_of(sc: pd.DataFrame) -> dict:
        r = book_returns(sc)
        return {wn: metric(r.loc[w]) for wn, w in windows.items()}

    obs = stats_of(score)
    nulls: dict[str, list] = {"iid": [], "permutation": [], "time_shift": []}
    for _ in range(n_iid):
        nulls["iid"].append(stats_of(pd.DataFrame(rng.normal(size=score.shape), index=score.index,
                                                  columns=score.columns).where(score.notna())))
    ident = np.arange(score.shape[1])
    for _ in range(n_perm):
        perm = rng.permutation(ident)
        while (perm == ident).any():
            perm = rng.permutation(ident)
        nulls["permutation"].append(stats_of(pd.DataFrame(score.to_numpy()[:, perm], index=score.index, columns=score.columns)))
    for L in range(min_shift, len(score) - min_shift, shift_step):
        nulls["time_shift"].append(stats_of(pd.DataFrame(np.roll(score.to_numpy(), L, axis=0), index=score.index,
                                                         columns=score.columns)))
    rows = {}
    for wn in windows:
        for nn, draws in nulls.items():
            a = np.array([d[wn] for d in draws], float)
            rows[(wn, nn)] = {"draws": len(a), "observed": obs[wn], "null_median": np.nanmedian(a),
                              "null_95": np.nanquantile(a, 0.95), "p": (1 + np.nansum(a >= obs[wn])) / (1 + len(a))}
    return pd.DataFrame(rows).T
