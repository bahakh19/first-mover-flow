"""QIS index calculation: turn a strategy into a rules-based index level series with fees, replication costs,
optional volatility targeting and an excess-return / total-return distinction, plus a factsheet.

    rules = IndexRules(name="FMF-N Excess Return Index", base_date="2017-01-03", fee_bps=50, vol_target=0.10)
    idx   = compute_index(out, rules)                 # DataFrame: level, daily return, leverage, costs, fees
    fs    = index_factsheet(idx["level"])             # calendar-year and full-period statistics
    md    = rulebook_markdown(rules, out.config)      # human-readable rulebook stub for documentation

Conventions. The strategy is self-financing (long and short legs fund each other), so its natural index is an EXCESS-
RETURN index. A TOTAL-RETURN version adds a cash rate on the index level. The leverage applied on day t is decided from
information up to t-1 (lagged realised volatility), and every cost is deducted on the day it is incurred.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from fmflow.analytics.metrics import perf_stats


@dataclass
class IndexRules:
    """Parameters of a rules-based index built on a strategy (fees, costs, volatility target, cash rate)."""
    name: str = "FMF-N Excess Return Index"
    base_date: str | None = None          # first index date (level = base_level); default: first date with a position
    base_level: float = 100.0
    fee_bps: float = 0.0                  # annual index / management fee, accrued daily on the level
    replication_cost_bps: float = 5.0     # per unit of traded notional (replaces the strategy's own cost assumption)
    borrow_bps: float = 0.0               # annual fee on short notional
    vol_target: float | None = None       # annualised; None = unlevered strategy
    vol_lookback: int = 63                # realised-volatility window for the leverage decision
    max_leverage: float = 1.5
    rebalance_band: float = 0.0           # leverage is changed only if it moves by more than this (in absolute terms)
    cash_rate: pd.Series | None = None    # annual rate (e.g. 0.03); provides the total-return index


def compute_index(out, rules: IndexRules) -> pd.DataFrame:
    """Index level series (excess return, optional total return) with leverage, costs and fees."""
    from fmflow.backtest.engine import backtest
    res = backtest(out.weights, out.returns, rules.replication_cost_bps, rules.borrow_bps, out.config.execution_lag)
    r = res.net.copy()
    start = pd.Timestamp(rules.base_date) if rules.base_date else out.weights.index[(out.weights.abs().sum(axis=1) > 0).argmax()]
    r = r.loc[start:]
    if rules.vol_target is not None:
        rv = r.rolling(rules.vol_lookback, min_periods=max(20, rules.vol_lookback // 2)).std() * np.sqrt(252)
        lev_raw = (rules.vol_target / rv).clip(upper=rules.max_leverage).shift(1).fillna(1.0)
        lev = lev_raw.copy()
        if rules.rebalance_band > 0:
            cur = lev_raw.iloc[0]
            vals = []
            for v in lev_raw:
                if abs(v - cur) > rules.rebalance_band:
                    cur = v
                vals.append(cur)
            lev = pd.Series(vals, index=lev_raw.index)
        lev_cost = lev.diff().abs().fillna(0.0) * res.weights.loc[start:].abs().sum(axis=1) * rules.replication_cost_bps * 1e-4
    else:
        lev = pd.Series(1.0, index=r.index)
        lev_cost = pd.Series(0.0, index=r.index)
    fee = rules.fee_bps * 1e-4 / 252
    daily = lev * r - lev_cost - fee
    daily.iloc[0] = 0.0
    out_df = pd.DataFrame({"daily_return": daily, "strategy_net": r, "leverage": lev, "leverage_cost": lev_cost,
                           "fee": fee, "level": rules.base_level * (1 + daily).cumprod()})
    if rules.cash_rate is not None:
        cr = rules.cash_rate.reindex(out_df.index).ffill().fillna(0.0) / 252
        out_df["level_total_return"] = rules.base_level * (1 + daily + cr.shift(1).fillna(0.0)).cumprod()
    out_df.attrs["rules"] = {k: v for k, v in asdict(rules).items() if k != "cash_rate"}
    return out_df


def index_factsheet(level: pd.Series) -> pd.DataFrame:
    """Calendar-year and full-period return, volatility, Sharpe, drawdown (and Calmar) of an index level."""
    r = level.pct_change().dropna()
    rows = {}
    for y, ry in r.groupby(r.index.year):
        s = perf_stats(ry)
        rows[str(y)] = {"return": float((1 + ry).prod() - 1), "vol": s["Ann. vol"], "Sharpe": s["Sharpe"], "max drawdown": s["Max drawdown"]}
    s = perf_stats(r)
    rows["full period"] = {"return": s["CAGR"], "vol": s["Ann. vol"], "Sharpe": s["Sharpe"], "max drawdown": s["Max drawdown"],
                           "Calmar": s["Calmar"]}
    return pd.DataFrame(rows).T


def rulebook_markdown(rules: IndexRules, cfg) -> str:
    """A rulebook stub generated from the exact parameters (to be completed with legal / calculation-agent text)."""
    sig = ", ".join(cfg.signals)
    legs = f"{cfg.k} names per leg" if cfg.k else f"{cfg.leg_fraction:.0%} of the names with a valid score per leg"
    vt = (f"Volatility target {rules.vol_target:.0%} using {rules.vol_lookback}-day realised volatility, leverage capped at "
          f"{rules.max_leverage:.2f}x and decided with a one-day lag." if rules.vol_target else "No volatility targeting.")
    return f"""# {rules.name} -- Index Rulebook (generated)

1. **Universe.** All stocks in the price file with a valid composite score on the calculation date.
2. **Signals.** {sig}. Each is ranked cross-sectionally; the composite is the equal-weight mean of available ranks.
   Analyst data are used from the session after publication; targets expire after {cfg.data.tp_expiry_days} sessions.
3. **Selection.** Long the highest-ranked and short the lowest-ranked {legs}.
4. **Weighting.** {"Inverse GJR-GARCH-t volatility" if cfg.allocator == "IV-GJR" else "Equal"} weights within each leg;
   net exposure {cfg.net_exposure:+.2f}; {"zero ex-ante beta legs" if cfg.neutralize == "beta" else "dollar-neutral legs"}.
5. **Rebalancing.** Daily, in {cfg.tranche_period} overlapping tranches (1/{cfg.tranche_period} of the book is re-targeted each day).
6. **Costs.** Replication cost {rules.replication_cost_bps:g} bp per unit traded; short borrow {rules.borrow_bps:g} bp p.a.;
   index fee {rules.fee_bps:g} bp p.a. accrued daily.
7. **Risk control.** {vt}
8. **Index level.** Base {rules.base_level:g} on {rules.base_date or 'the first calculation date'}; excess-return
   construction (self-financing long/short).
"""
