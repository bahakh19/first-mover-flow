"""The validation battery of the paper, as reusable functions that work on any dataset in the data contract.

signal_ic_table   rank IC of every signal (and the leader / herder parts) with the phase-averaged non-overlapping test
book_from_score   re-run ANY score panel through the strategy's exact construction (used by placebos and comparables)
strategy_placebo  IID, stock-label-permutation and circular time-shift nulls through the identical construction
leave_one_out     rebuild the whole pipeline (cleaning, snapshot, signals, ranks, book) without each stock in turn
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from fmflow.analytics.inference import forward_log_return, ic_phase_test, rank_ic
from fmflow.analytics.metrics import perf_stats
from fmflow.analytics.placebo import placebo_test
from fmflow.backtest.engine import backtest
from fmflow.config import StrategyConfig
from fmflow.data.cleaning import natural_sort_key, to_calendar_dates
from fmflow.data.schema import normalise_columns
from fmflow.pipeline import StrategyOutput, construct_book, run_strategy


def signal_ic_table(out: StrategyOutput, h: int = 21, window: slice | None = None, min_names: int = 6) -> pd.DataFrame:
    """IC_h of each composite input, the composite and the LMH parts. Labels must END inside `window` (no leakage
    across a split): the last h sessions of the window are dropped."""
    fwd = forward_log_return(out.prices, h)
    idx = out.prices.index
    if window is not None:
        lo = idx[0] if window.start is None else pd.Timestamp(window.start)
        hi = idx[-1] if window.stop is None else pd.Timestamp(window.stop)
        w = idx[(idx >= lo) & (idx <= hi)]
        idx = w[:-h] if len(w) > h else w[:0]
    panels = {**{n: out.signals[n] for n in out.config.signals}, "composite": out.score,
              "leaders_only": out.signals.get("leaders_only"), "herders_only": out.signals.get("herders_only")}
    rows = {}
    for name, f in panels.items():
        if f is None:
            continue
        ic = rank_ic(f.loc[idx], fwd.loc[idx], min_names).dropna()
        t = ic_phase_test(ic, h)
        rows[name] = {"IC": t["IC"], "t_phase": t["t"], "p (one-sided)": t["p"], "hit rate": float((ic > 0).mean()) if len(ic) else np.nan,
                      "coverage": float(f.loc[idx].notna().mean().mean()), "days": len(ic)}
    return pd.DataFrame(rows).T


def book_from_score(score: pd.DataFrame, out: StrategyOutput, net_exposure: float | None = None) -> pd.Series:
    """Net returns of `score` traded with exactly the construction of `out.config` (legs, allocator, neutralisation,
    position cap, tranches, volatility target, costs, borrow and execution delay)."""
    cfg = out.config
    _, weights = construct_book(score, out.returns, cfg, out.sig_vol, net_exposure)
    return backtest(weights, out.returns, cfg.cost_bps, cfg.borrow_bps, cfg.execution_lag).net


def _sharpe(r: pd.Series) -> float:
    return float(perf_stats(r)["Sharpe"])


def strategy_placebo(out: StrategyOutput, windows: Mapping[str, slice], n_iid: int = 300, n_perm: int = 500,
                     shift_step: int = 4, min_shift: int = 126, metric: Callable[[pd.Series], float] = _sharpe,
                     seed: int = 42) -> pd.DataFrame:
    """Placebo p-values of the strategy's composite under three nulls (see fmflow.analytics.placebo)."""
    return placebo_test(out.score, lambda sc: book_from_score(sc, out), windows, metric, n_iid=n_iid, n_perm=n_perm,
                        shift_step=shift_step, min_shift=min_shift, seed=seed)


def leave_one_out(prices_raw: pd.DataFrame, ratings_raw: pd.DataFrame, cfg: StrategyConfig, windows: Mapping[str, slice],
                  sig_vol: pd.DataFrame | None = None) -> pd.DataFrame:
    """Sharpe / Calmar / max drawdown of the full pipeline rebuilt on the universe without each stock.
    Pass `sig_vol` (univariate GARCH forecasts, unaffected by dropping a stock) to avoid re-estimating it."""
    p, r = normalise_columns(prices_raw), normalise_columns(ratings_raw)
    p_ids, r_ids = p["stock"].astype(str).str.strip(), r["stock"].astype(str).str.strip()
    stocks = sorted(p_ids.unique(), key=natural_sort_key)

    def stats_of(net: pd.Series) -> dict:
        row = {}
        for wn, w in windows.items():
            s = perf_stats(net.loc[w])
            row.update({f"Sharpe {wn}": s["Sharpe"], f"Calmar {wn}": s["Calmar"], f"MaxDD {wn}": s["Max drawdown"]})
        return row
    rows = {"all stocks": stats_of(run_strategy(p, r, cfg, sig_vol=sig_vol).net_returns)}
    for s in stocks:
        sv = sig_vol.drop(columns=s, errors="ignore") if sig_vol is not None else None
        o = run_strategy(p[p_ids != s], r[r_ids != s], cfg, sig_vol=sv)
        rows[f"without {s}"] = stats_of(o.net_returns)
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------------------------------------------------
# Multiverse / specification-curve analysis
# ---------------------------------------------------------------------------------------------------------------------
def _apply(cfg: StrategyConfig, params: dict) -> StrategyConfig:
    """Set StrategyConfig fields, 'data.<field>' DataConfig fields, and the pseudo-field 'legs' (int k or float fraction)."""
    from dataclasses import replace
    top, data = {}, {}
    for k, v in params.items():
        if k == "legs":
            top.update({"k": int(v), "leg_fraction": None} if float(v).is_integer() and v >= 1 else {"k": None, "leg_fraction": float(v)})
        elif k.startswith("data."):
            data[k[5:]] = v
        elif k in ("herd_window",):
            top.update({"herd_lo": int(v[0]), "herd_hi": int(v[1])})
        else:
            top[k] = tuple(v) if k == "signals" else v
    out = replace(cfg, **top)
    return replace(out, data=replace(out.data, **data)) if data else out


def _register_in_worker(builders: dict) -> None:
    """Process-pool initializer: make user-registered signals known in a worker started by spawn / forkserver."""
    from fmflow.signals.registry import register_signal
    for name, (fn, family) in builders.items():
        register_signal(name, fn, family=family, overwrite=True)


def _importable(fn) -> bool:
    """True if a worker process can import `fn` by reference (not a lambda, closure or notebook / __main__ function)."""
    return getattr(fn, "__module__", "__main__") != "__main__" and "<" not in getattr(fn, "__qualname__", "<")


def _one_run(args):
    prices, ratings, cfg, sig_vol, windows = args
    try:
        net = run_strategy(prices, ratings, cfg, sig_vol=sig_vol).net_returns
    except Exception as e:  # a configuration may be infeasible on a small universe (too few names): record it
        return None, str(e)
    row = {}
    for wn, w in windows.items():
        s = perf_stats(net.loc[w])
        row.update({f"Sharpe {wn}": s.get("Sharpe", np.nan), f"Calmar {wn}": s.get("Calmar", np.nan),
                    f"MaxDD {wn}": s.get("Max drawdown", np.nan)})
    return net, row


def multiverse(prices: pd.DataFrame, ratings: pd.DataFrame, base: StrategyConfig, grid: Mapping[str, list],
               windows: Mapping[str, slice], sig_vol: pd.DataFrame | None = None, mode: str = "oat",
               n_random: int = 200, seed: int = 0, n_jobs: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the strategy over a family of reasonable specifications (Steegen et al. 2016; Simonsohn et al. 2020).

    mode = "oat"    one-at-a-time: vary each parameter around `base` (sensitivity curves);
           "full"   the full factorial grid;
           "random" `n_random` joint draws from the grid (plus the base).
    Parameters: any StrategyConfig field, 'data.<DataConfig field>', 'legs' (k or leg fraction), 'herd_window'
    ((lo, hi) pairs). Returns (table of specifications and statistics, matrix of daily net returns), the latter being the
    input to spa_test / cscv_pbo / deflated_sharpe over the whole family.
    """
    import itertools
    keys = list(grid)
    specs: list[dict] = [{}]
    if mode == "oat":
        specs += [{k: v} for k in keys for v in grid[k]]
    elif mode == "full":
        specs += [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))]
    elif mode == "random":
        rng = np.random.default_rng(seed)
        specs += [{k: grid[k][rng.integers(len(grid[k]))] for k in keys} for _ in range(n_random)]
    else:
        raise ValueError("mode must be 'oat', 'full' or 'random'")
    uniq, seen = [], set()
    for s in specs:
        key = repr(sorted((k, repr(v)) for k, v in s.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    jobs = [(prices, ratings, _apply(base, s), sig_vol, windows) for s in uniq]
    if n_jobs > 1:
        import multiprocessing
        import warnings
        from concurrent.futures import ProcessPoolExecutor

        from fmflow.signals.registry import custom_signals
        builders = custom_signals([n for _, _, c, _, _ in jobs for n in c.signals])
        forked = multiprocessing.get_start_method() == "fork"           # workers inherit the registry
        if builders and not forked and not all(_importable(fn) for fn, _ in builders.values()):
            # spawn / forkserver (macOS, Windows, Python >= 3.14): a worker cannot see signals registered in a
            # notebook or a script's main block, so every specification would fail
            warnings.warn(f"custom signals {sorted(builders)} are defined in __main__ or as lambdas, which worker processes "
                          "cannot import: running the multiverse sequentially (define them in a module to parallelise)",
                          UserWarning, stacklevel=2)
            n_jobs = 1
    if n_jobs > 1:
        init = None if forked or not builders else _register_in_worker
        with ProcessPoolExecutor(n_jobs, initializer=init, initargs=(builders,) if init else ()) as ex:
            results = list(ex.map(_one_run, jobs))
    else:
        results = [_one_run(j) for j in jobs]
    failed = [(s, r) for s, (net, r) in zip(uniq, results, strict=True) if net is None]
    if len(failed) == len(uniq):
        raise RuntimeError(f"every specification failed; the first error: {failed[0][1]}")
    if failed:
        import warnings
        warnings.warn(f"{len(failed)} of {len(uniq)} specifications failed (see the 'error' column); "
                      f"the first error: {failed[0][1]}", UserWarning, stacklevel=2)
    rows, nets = [], {}
    for s, (net, row) in zip(uniq, results, strict=True):
        label = "base" if not s else ", ".join(f"{k}={v}" for k, v in s.items())
        if net is None:
            rows.append({"spec": label, "error": row, **{k: repr(v) for k, v in s.items()}})
            continue
        nets[label] = net
        rows.append({"spec": label, **{k: repr(v) for k, v in s.items()}, **row, "is_base": not s})
    return pd.DataFrame(rows).set_index("spec"), pd.DataFrame(nets)


# ---------------------------------------------------------------------------------------------------------------------
# Causality audit for any configuration, including user-registered signals
# ---------------------------------------------------------------------------------------------------------------------
def truncation_check(prices_raw: pd.DataFrame, ratings_raw: pd.DataFrame, cfg: StrategyConfig, cut_dates,
                     sig_vol: pd.DataFrame | None = None, extra: dict | None = None, atol: float = 1e-10) -> pd.DataFrame:
    """Recompute every signal, the composite and the weights on raw data truncated at each cut date T and compare with
    the full-sample values up to T. Any difference is look-ahead. Ratings are truncated by PUBLICATION date (what a
    live system would have received); `extra` user panels are truncated by index."""
    p, r = normalise_columns(prices_raw), normalise_columns(ratings_raw)
    p_dates, r_dates = to_calendar_dates(p["date"]).to_numpy(), to_calendar_dates(r["date"]).to_numpy()
    full = run_strategy(p, r, cfg, sig_vol=sig_vol, extra=extra)
    rows = []
    for T in cut_dates:
        T = pd.Timestamp(T)
        pt, rt = p[p_dates <= T], r[r_dates <= T]
        ex = {k: (v.loc[:T] if hasattr(v, "loc") else v) for k, v in (extra or {}).items()}
        cut = run_strategy(pt, rt, cfg, sig_vol=None if sig_vol is None else sig_vol.loc[:T], extra=ex)
        panels = {**{f"signal:{k}": (cut.signals[k], full.signals[k]) for k in cut.signals},
                  "composite": (cut.score, full.score), "weights": (cut.weights, full.weights)}
        for name, (a, b) in panels.items():
            a2, b2 = a.align(b.loc[:T], join="inner")
            diff = (a2.fillna(0) - b2.fillna(0)).abs().to_numpy()
            nan_mismatch = int((a2.isna() != b2.isna()).to_numpy().sum())
            rows.append({"T": T.date(), "object": name, "max_abs_diff": float(diff.max()) if diff.size else 0.0,
                         "nan_mismatches": nan_mismatch, "causal": bool((diff.max() if diff.size else 0) <= atol and nan_mismatch == 0)})
    return pd.DataFrame(rows)
