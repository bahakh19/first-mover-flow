"""Signal registry: every signal the pipeline can rank is a named builder `fn(ctx) -> (sessions x stocks) panel`.

Built-in signals are registered below; research code adds its own without touching the package:

    from fmflow.signals.registry import register_signal

    @register_signal("short_interest_change")
    def _(ctx):                                   # ctx.px, ctx.ret, ctx.snap, ctx.cfg, ctx.extra (user data), ctx.cache
        return -ctx.extra["short_interest"].pct_change(21)

    cfg = StrategyConfig.fmf_n().with_(signals=("breadth_63", "leader_minus_herd", "short_interest_change"))

A builder must be CAUSAL (value at t uses data known at t); `fmflow.robustness.truncation_check` verifies it.
Builders are memoised per run in `ctx.cache`, so signals that share work (leader / herder parts) compute it once.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

SignalFn = Callable[["SignalContext"], pd.DataFrame]
_REGISTRY: dict[str, tuple[SignalFn, str]] = {}


@dataclass
class SignalContext:
    """Inputs available to a signal builder at run time (prices, snapshot, config, user data, memo cache)."""
    px: pd.DataFrame
    snap: object
    cfg: object
    extra: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)

    @property
    def ret(self) -> pd.DataFrame:
        """Simple daily returns of `px` (computed once per run)."""
        if "_ret" not in self.cache:
            self.cache["_ret"] = self.px.pct_change(fill_method=None)
        return self.cache["_ret"]

    def memo(self, key: str, fn: Callable[[], object]):
        """`fn()` computed once per run and cached under `key` (shared work between signals)."""
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]


def register_signal(name: str, fn: SignalFn | None = None, *, family: str = "custom", overwrite: bool = False):
    """Register `fn` under `name` (usable as a decorator)."""
    def deco(f: SignalFn) -> SignalFn:
        if name in _REGISTRY and not overwrite:
            raise KeyError(f"signal {name!r} already registered (pass overwrite=True to replace it)")
        _REGISTRY[name] = (f, family)
        return f
    return deco(fn) if fn is not None else deco


def unregister_signal(name: str) -> None:
    """Remove a signal from the registry (no error if absent)."""
    _REGISTRY.pop(name, None)


def available_signals(family: str | None = None) -> list[str]:
    """Names of registered signals, optionally of one family."""
    return sorted(k for k, (_, fam) in _REGISTRY.items() if family is None or fam == family)


def signal_family(name: str) -> str:
    """Family label of a registered signal."""
    return _REGISTRY[name][1]


def custom_signals(names) -> dict[str, tuple[SignalFn, str]]:
    """{name: (builder, family)} of the user-registered (non built-in) signals among `names`."""
    return {n: _REGISTRY[n] for n in dict.fromkeys(names) if n in _REGISTRY and n not in BUILTIN_SIGNALS}


def build_signals(names, ctx: SignalContext) -> dict[str, pd.DataFrame]:
    """Build the named signal panels on the context's calendar and universe."""
    out = {}
    for n in names:
        if n not in _REGISTRY:
            raise KeyError(f"unknown signal {n!r}; registered: {available_signals()}")
        out[n] = _REGISTRY[n][0](ctx).reindex(index=ctx.px.index, columns=ctx.px.columns)
    return out


# ---------------------------------------------------------------------------------------------------- built-in signals
def _register_builtins() -> None:
    from fmflow.signals.analyst import analyst_signals
    from fmflow.signals.breadth import breadth, breadth_resid_mom, event_breadth
    from fmflow.signals.leader_herd import leader_minus_herd
    from fmflow.signals.price import price_signals

    def _breadth(ctx):
        return ctx.memo("breadth", lambda: breadth(ctx.snap, ctx.cfg.breadth_window, ctx.cfg.breadth_min_n))

    def _lmh_parts(ctx):
        c = ctx.cfg
        return ctx.memo("lmh", lambda: leader_minus_herd(ctx.snap, ctx.px.index, list(ctx.px.columns), c.leader_gap, c.herd_lo,
                                                         c.herd_hi, c.lmh_window, c.lmh_min_n, return_parts=True))

    register_signal("breadth_63", _breadth, family="revision-flow", overwrite=True)
    register_signal("event_breadth_63", lambda ctx: event_breadth(ctx.snap, ctx.px.index, list(ctx.px.columns),
                                                                   ctx.cfg.breadth_window, ctx.cfg.event_breadth_min_n),
                    family="revision-flow", overwrite=True)
    register_signal("breadth_resid_mom", lambda ctx: breadth_resid_mom(_breadth(ctx), ctx.px, ctx.cfg.rank_min_names),
                    family="revision-flow", overwrite=True)
    for part in ("leader_minus_herd", "leaders_only", "herders_only"):
        register_signal(part, (lambda p: lambda ctx: _lmh_parts(ctx)[p])(part), family="revision-flow", overwrite=True)
    for nm in ("mom_12_1", "mom_6_1", "rev_1m", "low_vol_60", "high_52w", "low_beta_252", "resid_mom_12_1"):
        register_signal(nm, (lambda n: lambda ctx: ctx.memo("price", lambda: price_signals(ctx.px))[n])(nm),
                        family="price", overwrite=True)
    for nm in ("tp_upside", "cons_change_63", "low_dispersion"):
        register_signal(nm, (lambda n: lambda ctx: ctx.memo("analyst", lambda: analyst_signals(ctx.px, ctx.snap))[n])(nm),
                        family="analyst", overwrite=True)


_register_builtins()
BUILTIN_SIGNALS = frozenset(_REGISTRY)
FMF_SIGNALS = ("breadth_63", "event_breadth_63", "breadth_resid_mom", "leader_minus_herd")
