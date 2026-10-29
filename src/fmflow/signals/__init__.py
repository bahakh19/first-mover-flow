"""Signals. Every function maps point-in-time inputs at session t to a (session x stock) panel, higher = more attractive."""
from fmflow.signals.breadth import breadth, breadth_resid_mom, event_breadth
from fmflow.signals.composite import centred_rank, composite_score, cs_rank
from fmflow.signals.leader_herd import leader_minus_herd
from fmflow.signals.price import price_signals

__all__ = ["breadth", "event_breadth", "breadth_resid_mom", "leader_minus_herd", "cs_rank", "centred_rank",
           "composite_score", "price_signals"]
