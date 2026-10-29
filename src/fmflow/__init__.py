"""fmflow — First-Mover Flow: zero-beta stock selection from the flow and timing of analyst revisions.

Public API (see README.md):

    from fmflow import StrategyConfig, DataConfig, run_strategy, load_prices, load_ratings
    out = run_strategy(load_prices("prices.csv"), load_ratings("ratings.csv"), StrategyConfig.fmf_n())
    out.weights      # sessions x stocks, daily target book (dollar-neutral)
"""
from fmflow.config import DataConfig, StrategyConfig
from fmflow.data.io import load_prices, load_ratings
from fmflow.pipeline import StrategyOutput, build_signal_panels, run_strategy

__all__ = ["DataConfig", "StrategyConfig", "load_prices", "load_ratings", "run_strategy", "build_signal_panels",
           "StrategyOutput"]
__version__ = "2.2.4"
