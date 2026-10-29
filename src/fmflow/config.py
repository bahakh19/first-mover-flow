"""Configuration objects. Every tunable number of the method lives here, nowhere else.

`DataConfig` governs ingestion and point-in-time construction; `StrategyConfig` governs signals, portfolio construction
and costs. Both are frozen dataclasses (hashable, immutable) and round-trip to JSON so that a specification can be
frozen to disk *before* it is evaluated, which is how out-of-sample tests are run in this project.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Ingestion and point-in-time rules (identical for every dataset; never tuned on returns)."""

    annualization: int = 252                   # sessions per year (recorded in frozen configs; metrics use 252)
    min_calendar_coverage: float = 0.5          # a date is a trading session if >= 50% of the active stocks print
    calendar_active_days: int = 30              # a stock is active at t if it printed in the 30 calendar days up to t
    price_ffill_limit: int = 5                  # bounded forward fill of isolated gaps; never backward
    knowledge_lag: int = 1                      # a rating becomes usable this many sessions after publication
    tp_expiry_days: int = 252                   # an analyst target expires after 252 sessions
    tp_ratio_bounds: tuple[float, float] = (0.2, 5.0)        # admissible target / price at publication
    revision_excess_bounds: tuple[float, float] = (1 / 3, 3.0)  # admissible (target change)/(price change)
    event_min_ratings: int = 5                  # >= 5 same-day ratings on a stock = information event
    min_analysts: int = 3                       # minimum active analysts for a consensus statistic
    garch_min_obs: int = 250                    # observations before the first GJR-GARCH estimation
    garch_refit: int = 21                       # re-estimation frequency (sessions)
    garch_method: str = "slsqp"                 # "slsqp" (paper) or "grid" (deterministic, machine-independent)

    def validate(self) -> None:
        """Raise ValueError if a data rule is out of range. knowledge_lag >= 1 is what keeps the pipeline causal."""
        if self.knowledge_lag < 1:
            raise ValueError("knowledge_lag must be >= 1: a rating is only usable from the session after its publication "
                             "(0 would trade on information published during or after the same session)")
        if not 0 < self.min_calendar_coverage <= 1:
            raise ValueError("min_calendar_coverage must lie in (0, 1]")
        for name in ("calendar_active_days", "tp_expiry_days", "event_min_ratings", "min_analysts", "garch_min_obs",
                     "garch_refit"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.price_ffill_limit < 0:
            raise ValueError("price_ffill_limit must be >= 0")
        for name in ("tp_ratio_bounds", "revision_excess_bounds"):
            lo, hi = getattr(self, name)
            if not 0 < lo < hi:
                raise ValueError(f"{name} must satisfy 0 < low < high")
        if self.garch_method not in ("slsqp", "grid"):
            raise ValueError("garch_method must be 'slsqp' (paper) or 'grid' (deterministic)")


@dataclass(frozen=True)
class StrategyConfig:
    """Signal, construction and cost parameters of a First-Mover Flow book."""

    name: str = "FMF-N"
    signals: tuple[str, ...] = ("breadth_63", "event_breadth_63", "breadth_resid_mom", "leader_minus_herd")
    # leader / herder classification (leader_minus_herd)
    leader_gap: int = 10                        # a revision after >= 10 silent sessions on the stock is a leader
    herd_lo: int = 3                            # herders arrive 3..15 sessions after the latest event cluster
    herd_hi: int = 15
    lmh_window: int = 126                       # trailing window (two earnings cycles)
    lmh_min_n: int = 3
    breadth_window: int = 63
    breadth_min_n: int = 3
    event_breadth_min_n: int = 2
    rank_min_names: int = 5                     # cross-sectional ranks need >= 5 valid names
    # portfolio construction
    k: int | None = 4                        # names per leg (fixed); ignored if leg_fraction is set
    leg_fraction: float | None = None        # alternative: k_t = max(k_min, floor(leg_fraction * N_t))
    k_min: int = 2
    tranche_period: int = 42                    # staggered overlapping tranches (sessions)
    allocator: str = "IV-GJR"                   # "EW" or "IV-GJR" (inverse GJR-GARCH-t volatility within each leg)
    net_exposure: float = 0.0                   # constant net market exposure c (beta budget); 0 = dollar-neutral
    neutralize: str = "dollar"                  # "dollar" (legs 0.5 / 0.5 +/- c/2) or "beta" (zero ex-ante beta legs)
    beta_window: int = 252                      # rolling window for the betas used by neutralize="beta"
    max_weight: float | None = None          # cap on |w_i| in the target book (None = no cap)
    vol_target: float | None = None          # annualised ex-ante volatility target of the held book (None = off)
    vol_lookback: int = 126
    max_leverage: float = 2.0
    cost_bps: float = 5.0                       # one-way cost per unit of traded notional
    borrow_bps: float = 0.0                     # annual fee on short notional
    execution_lag: int = 0                      # sessions between signal close and execution (0 = trade at the close)
    data: DataConfig = field(default_factory=DataConfig)

    # ---- presets ---------------------------------------------------------------------------------------------------
    @classmethod
    def fmf_n(cls) -> StrategyConfig:
        """The frozen specification selected on 2017-2019 (FMF-N, market-neutral)."""
        return cls()

    @classmethod
    def fmf_b20(cls) -> StrategyConfig:
        """FMF-N plus an explicit, disclosed beta budget of 0.2 net long (FMF-B20)."""
        return cls(name="FMF-B20", net_exposure=0.2)

    @classmethod
    def fmf_n_scalable(cls) -> StrategyConfig:
        """FMF-N with legs as a fraction (40%) of the names with a valid score: the rule for universes other than 10."""
        return cls(name="FMF-N (leg fraction 0.4)", k=None, leg_fraction=0.4)

    @classmethod
    def fmf_beta_neutral(cls) -> StrategyConfig:
        """FMF-N with zero ex-ante BETA (rolling betas to the equal-weight market) instead of zero net dollars."""
        return cls(name="FMF-BN", neutralize="beta")

    @classmethod
    def fmf_vt10(cls) -> StrategyConfig:
        """FMF-N scaled to 10% annualised ex-ante volatility (QIS-style volatility-targeted sleeve)."""
        return cls(name="FMF-N VT10", vol_target=0.10, max_leverage=2.0)

    @classmethod
    def conservative_costs(cls) -> StrategyConfig:
        """FMF-N under pessimistic implementation: 10 bp costs, 100 bp annual borrow, one-session execution delay."""
        return cls(name="FMF-N (10bp, 100bp borrow, 1-day lag)", cost_bps=10.0, borrow_bps=100.0, execution_lag=1)

    @classmethod
    def round3(cls) -> StrategyConfig:
        """Earlier specification (equal weights, 63-session tranches), kept for comparison."""
        return cls(name="FMF-N (Round-3)", allocator="EW", tranche_period=63)

    # ---- (de)serialisation -----------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        """JSON-ready dictionary (tuples as lists), the inverse of `from_dict`."""
        d = asdict(self)
        d["signals"] = list(self.signals)
        d["data"]["tp_ratio_bounds"] = list(self.data.tp_ratio_bounds)
        d["data"]["revision_excess_bounds"] = list(self.data.revision_excess_bounds)
        return d

    def save(self, path: str | Path) -> None:
        """Write the configuration as JSON (UTF-8)."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> StrategyConfig:
        """Build from a dictionary; unknown fields raise ValueError, missing ones take their defaults."""
        d = dict(d)
        data = d.pop("data", {}) or {}
        for key in ("tp_ratio_bounds", "revision_excess_bounds"):
            if key in data:
                data[key] = tuple(data[key])
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown StrategyConfig fields: {sorted(unknown)}")
        unknown = set(data) - set(DataConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown DataConfig fields: {sorted(unknown)}")
        if "signals" in d:
            d["signals"] = tuple(d["signals"])
        return cls(**d, data=DataConfig(**data))

    @classmethod
    def load(cls, path: str | Path) -> StrategyConfig:
        """Read a configuration written by `save` (or by `fmflow run`)."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_(self, **kwargs) -> StrategyConfig:
        """Functional update: `cfg.with_(k=None, leg_fraction=0.4)`."""
        return replace(self, **kwargs)

    def validate(self) -> None:
        """Raise ValueError with an actionable message if any parameter is out of its admissible range."""
        self.data.validate()
        if self.allocator not in ("EW", "IV-GJR"):
            raise ValueError("allocator must be 'EW' or 'IV-GJR'")
        if (self.k is None) == (self.leg_fraction is None):
            raise ValueError("set exactly one of k or leg_fraction")
        if self.k is not None and self.k < 1:
            raise ValueError("k (names per leg) must be >= 1")
        if self.k_min < 1:
            raise ValueError("k_min must be >= 1")
        if not 1 <= self.herd_lo <= self.herd_hi:
            raise ValueError("herder window needs 1 <= herd_lo <= herd_hi")
        for name in ("leader_gap", "lmh_window", "lmh_min_n", "breadth_window", "breadth_min_n", "event_breadth_min_n",
                     "vol_lookback"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.beta_window < 2:
            raise ValueError("beta_window must be >= 2 (a beta needs at least two returns)")
        if self.rank_min_names < 2:
            raise ValueError("rank_min_names must be >= 2 (a cross-sectional rank needs at least two names)")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if self.leg_fraction is not None and not 0 < self.leg_fraction <= 0.5:
            raise ValueError("leg_fraction must lie in (0, 0.5]")
        if self.tranche_period < 1:
            raise ValueError("tranche_period must be >= 1")
        if not -1.0 <= self.net_exposure <= 1.0:
            raise ValueError("net_exposure must lie in [-1, 1] (leg budgets 0.5 +/- c/2 must stay non-negative)")
        if not self.signals:
            raise ValueError("signals must name at least one registered signal")
        from fmflow.signals.registry import available_signals
        bad = set(self.signals) - set(available_signals())
        if bad:
            raise ValueError(f"unknown signals {sorted(bad)}; registered: {available_signals()}")
        if self.neutralize not in ("dollar", "beta"):
            raise ValueError("neutralize must be 'dollar' or 'beta'")
        if self.neutralize == "beta" and self.net_exposure != 0:
            raise ValueError("a beta budget (net_exposure) is defined for dollar-neutral legs; use neutralize='dollar'")
        if self.max_weight is not None and not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must lie in (0, 1]")
        if self.vol_target is not None and self.vol_target <= 0:
            raise ValueError("vol_target must be positive")
        if self.execution_lag < 0 or self.borrow_bps < 0 or self.cost_bps < 0:
            raise ValueError("execution_lag, borrow_bps and cost_bps must be non-negative")

