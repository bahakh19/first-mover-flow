"""End-to-end First-Mover Flow pipeline: raw tables -> clean panels -> signals -> composite -> weights -> P&L.

    out = run_strategy(prices_raw, ratings_raw, StrategyConfig.fmf_n())

Every step is causal; `tests/test_causality.py` verifies that weights computed on data truncated at T equal the
full-sample weights up to T.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fmflow.backtest.engine import BacktestResult, backtest
from fmflow.config import StrategyConfig
from fmflow.data.cleaning import AuditLog, clean_prices, clean_ratings
from fmflow.data.schema import validate_prices, validate_ratings
from fmflow.data.snapshot import TPSnapshot, build_tp_snapshot
from fmflow.portfolio.construction import (
    beta_neutral_budgets,
    cap_weights,
    rolling_betas,
    sector_demean,
    target_weights,
    tranche_weights,
    vol_target_weights,
)
from fmflow.risk.garch import garch_forecast_panel
from fmflow.signals.composite import composite_score
from fmflow.signals.registry import FMF_SIGNALS, SignalContext, build_signals


@dataclass(repr=False)
class StrategyOutput:
    """Everything a run produces: clean data, snapshot, signals, score, risk forecasts, weights and P&L."""
    config: StrategyConfig
    prices: pd.DataFrame                 # clean session x stock panel
    returns: pd.DataFrame
    ratings: pd.DataFrame                # clean, knowledge-date-stamped ratings
    snapshot: TPSnapshot
    signals: dict                        # name -> panel (plus diagnostics leaders_only / herders_only)
    score: pd.DataFrame                  # composite
    sig_vol: pd.DataFrame | None
    target: pd.DataFrame
    weights: pd.DataFrame
    result: BacktestResult
    audit: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def net_returns(self) -> pd.Series:
        """Daily net returns of the held book (after costs, borrow and execution delay)."""
        return self.result.net

    def summary(self, windows: dict | None = None) -> pd.DataFrame:
        """Headline statistics (CAGR, vol, Sharpe, max drawdown, Calmar, beta, alpha, PSR, turnover) over named windows;
        by default the sample after a one-year warm-up and its two halves."""
        from fmflow.report import default_windows, strategy_report
        return strategy_report(self, windows or default_windows(self.prices.index))

    def __repr__(self) -> str:                      # the default dataclass repr would print every DataFrame
        from fmflow.report import default_windows
        span = f"{self.prices.index[0]:%Y-%m-%d} to {self.prices.index[-1]:%Y-%m-%d}" if len(self.prices) else "empty"
        sr_txt = ""
        if len(self.prices) > 1:
            w = default_windows(self.prices.index)["evaluation"]
            r = self.net_returns.loc[w].dropna()
            if len(r) > 1 and r.std() > 0:
                sr_txt = f", net Sharpe {float(r.mean() / r.std() * 252 ** 0.5):.2f} after a one-year warm-up"
        n = len(self.prices)
        return (f"StrategyOutput({self.config.name!r}: {self.prices.shape[1]} stocks, {n} session{'s' * (n != 1)} ({span}), "
                f"{len(self.ratings):,} accepted ratings{sr_txt}; see .summary(), .weights, .net_returns, .signals, .audit)")


def prepare_data(prices_raw: pd.DataFrame, ratings_raw: pd.DataFrame, cfg: StrategyConfig):
    """Validate and clean both tables and build the point-in-time snapshot (returns px, ratings, snapshot, log).

    The ratings are validated against the prices, so targets on another scale than the prices (the most common
    ingestion error) raise a DataContractError instead of being silently removed by the plausibility filters."""
    log = AuditLog()
    prices_raw = validate_prices(prices_raw)
    px = clean_prices(prices_raw, cfg.data, log)
    rt = clean_ratings(validate_ratings(ratings_raw, prices_raw), px, cfg.data, log)
    if rt.empty:
        warnings.warn("no rating survived cleaning (see the audit log): every analyst signal is NaN and the book is flat",
                      UserWarning, stacklevel=3)
    snap = build_tp_snapshot(rt, px.index, px.columns, cfg.data)
    return px, rt, snap, log


def build_signal_panels(px: pd.DataFrame, snap: TPSnapshot, cfg: StrategyConfig, extra: dict | None = None) -> dict:
    """The four FMF signals, the leader / herder diagnostics, and any other signal named in cfg.signals."""
    names = list(dict.fromkeys([*FMF_SIGNALS, "leaders_only", "herders_only", *cfg.signals]))
    return build_signals(names, SignalContext(px=px, snap=snap, cfg=cfg, extra=dict(extra or {})))


def construct_book(score: pd.DataFrame, ret: pd.DataFrame, cfg: StrategyConfig, sig_vol: pd.DataFrame | None = None,
                   net_exposure: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score -> (daily target book, held book): legs, allocator, beta neutralisation, position cap, tranches and
    volatility target exactly as configured. `net_exposure` overrides `cfg.net_exposure` (e.g. for a beta budget)."""
    c = cfg.net_exposure if net_exposure is None else net_exposure
    target = target_weights(score, cfg.k, cfg.leg_fraction, cfg.k_min, cfg.allocator, sig_vol, ret, c)
    if cfg.neutralize == "beta":
        target = beta_neutral_budgets(target, rolling_betas(ret, cfg.beta_window))
    if cfg.max_weight is not None:
        target = cap_weights(target, cfg.max_weight)
    weights = tranche_weights(target, cfg.tranche_period)
    if cfg.vol_target is not None:
        weights, _ = vol_target_weights(weights, ret, cfg.vol_target, cfg.vol_lookback, cfg.max_leverage)
    weights.index.name, weights.columns.name = "date", "stock"
    return target, weights


def run_strategy(prices_raw: pd.DataFrame | str | Path, ratings_raw: pd.DataFrame | str | Path,
                 cfg: StrategyConfig | None = None, sig_vol: pd.DataFrame | None = None, sectors: dict | None = None,
                 extra: dict | None = None, cache_dir: str | Path | None = None) -> StrategyOutput:
    """Run the full pipeline from the two raw tables (DataFrames, or paths to CSV / Excel / Parquet files).

    cfg        StrategyConfig (default: FMF-N, `StrategyConfig.fmf_n()`);

    sig_vol    precomputed GJR-GARCH forecasts (skips the slow univariate GARCH step);
    sectors    {stock: sector} -- demeans the composite within sectors before ranking (sector-neutral selection);
    extra      user data passed to custom signals through SignalContext.extra (e.g. short interest, fundamentals);
    cache_dir  directory for the on-disk GARCH cache (keyed by the returns and the GARCH settings).
    """
    cfg = StrategyConfig() if cfg is None else cfg
    cfg.validate()
    if isinstance(prices_raw, (str, Path)) or isinstance(ratings_raw, (str, Path)):
        from fmflow.data.io import read_table
        if isinstance(prices_raw, (str, Path)):
            prices_raw = read_table(prices_raw)
        if isinstance(ratings_raw, (str, Path)):
            ratings_raw = read_table(ratings_raw, dtype=str)
    px, rt, snap, log = prepare_data(prices_raw, ratings_raw, cfg)
    ret = px.pct_change(fill_method=None)
    panels = build_signal_panels(px, snap, cfg, extra)
    score = composite_score(panels, cfg.signals, index=px.index, min_names=cfg.rank_min_names)
    if sectors:
        score = sector_demean(score, sectors)
    if cfg.allocator == "IV-GJR" and sig_vol is None:
        if cache_dir is not None:
            from fmflow.cache import cached_garch
            sig_vol = cached_garch(ret, cfg.data, cache_dir)
        else:
            sig_vol, _ = garch_forecast_panel(ret, cfg.data)
    if sig_vol is not None:
        sig_vol = sig_vol.reindex(index=px.index, columns=px.columns)
    target, weights = construct_book(score, ret, cfg, sig_vol)
    result = backtest(weights, ret, cfg.cost_bps, cfg.borrow_bps, cfg.execution_lag)
    return StrategyOutput(config=cfg, prices=px, returns=ret, ratings=rt, snapshot=snap, signals=panels, score=score,
                          sig_vol=sig_vol, target=target, weights=weights, result=result, audit=log.to_frame())
