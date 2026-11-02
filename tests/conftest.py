"""Shared fixtures. Synthetic data keeps the default test run fast (< 1 min) and dependency-free (no GARCH)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fmflow import StrategyConfig
from fmflow.synthetic import SyntheticConfig, generate

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def synth():
    return generate(SyntheticConfig(n_sessions=900, seed=11))


@pytest.fixture(scope="session")
def ew_cfg():
    return StrategyConfig.fmf_n().with_(allocator="EW")


@pytest.fixture(scope="session")
def synth_out(synth, ew_cfg):
    from fmflow import run_strategy
    return run_strategy(*synth, ew_cfg)


def toy_prices(n_stocks: int = 8, n_days: int = 400, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    px = np.cumprod(1 + rng.normal(0.0004, 0.015, (n_days, n_stocks)), axis=0)
    return pd.DataFrame({"date": np.repeat(dates.strftime("%Y-%m-%d"), n_stocks),
                         "stock": np.tile([f"s{i}" for i in range(n_stocks)], n_days), "price": px.ravel()})
