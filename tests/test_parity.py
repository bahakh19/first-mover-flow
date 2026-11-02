"""Parity with the research notebook (notebooks/Research_notebook_FMF.ipynb).

The reference weights were produced by the notebook's own implementation. The package must reproduce them exactly:
  * open universe (shipped in data/open): always runs; uses the notebook's GJR-GARCH forecasts as a fixture;
  * development universe: runs only if data/private/MarketData.csv and Ratings.csv are present.

Re-estimating GARCH from scratch with the default SLSQP optimiser is reproducible on one machine but NOT bit-for-bit
across machines: on short samples the likelihood has near-equal local maxima and the optimiser's stopping point
depends on floating-point details (BLAS kernel, AVX/FMA). The fresh-estimation test therefore checks economic, not
bitwise, agreement. The deterministic grid estimator (garch_method="grid") is checked bit-for-bit on every machine.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fmflow import StrategyConfig, run_strategy

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures"
OPEN = ROOT / "data" / "open"
PRIVATE = ROOT / "data" / "private"


def _ref(name):
    return pd.read_csv(FIX / name, index_col=0, parse_dates=True)


def _open():
    return pd.read_csv(OPEN / "MarketData_Normalized.csv"), pd.read_csv(OPEN / "Ratings_Normalized.csv", dtype=str)


@pytest.mark.skipif(not (OPEN / "MarketData_Normalized.csv").exists(), reason="open dataset not present")
@pytest.mark.parametrize("cfg_kw, ref", [(dict(k=None, leg_fraction=0.4), "open_reference_weights_fmf_n_leg040.csv.gz"),
                                         (dict(), "open_reference_weights_fmf_n_k4.csv.gz")])
def test_open_universe_parity(cfg_kw, ref):
    sig = _ref("open_sig_gjr.csv.gz")
    out = run_strategy(*_open(), StrategyConfig.fmf_n().with_(**cfg_kw), sig_vol=sig)
    a, b = out.weights.align(_ref(ref), join="outer")
    assert float((a - b).abs().max().max()) < 1e-9


def _sharpe(r):
    r = r.loc["2024-01-05":"2026-08-05"]
    return float(r.mean() / r.std() * 252 ** 0.5)


@pytest.mark.slow
@pytest.mark.skipif(not (OPEN / "MarketData_Normalized.csv").exists(), reason="open dataset not present")
def test_open_universe_fresh_slsqp_garch_agrees_economically():
    """Fresh SLSQP estimation: identical on the reference machine, within optimiser noise elsewhere.

    Observed spread across CPU instruction sets: max |weight difference| 0.023, out-of-sample Sharpe 1.02 to 1.05."""
    cfg = StrategyConfig.fmf_n().with_(k=None, leg_fraction=0.4)
    out = run_strategy(*_open(), cfg)
    ref = run_strategy(*_open(), cfg, sig_vol=_ref("open_sig_gjr.csv.gz"))
    a, b = out.weights.align(_ref("open_reference_weights_fmf_n_leg040.csv.gz"), join="outer")
    a, b = a.fillna(0.0), b.fillna(0.0)
    assert float((a - b).abs().max().max()) < 0.05
    assert np.corrcoef(a.to_numpy().ravel(), b.to_numpy().ravel())[0, 1] > 0.999
    assert abs(_sharpe(out.net_returns) - _sharpe(ref.net_returns)) < 0.05


@pytest.mark.slow
@pytest.mark.skipif(not (OPEN / "MarketData_Normalized.csv").exists(), reason="open dataset not present")
def test_grid_garch_is_bitwise_reproducible_on_any_machine():
    """The deterministic estimator must reproduce the shipped forecasts exactly, whatever the CPU or library build."""
    from fmflow.config import DataConfig
    from fmflow.pipeline import prepare_data
    from fmflow.risk.garch import garch_forecast_panel
    cfg = StrategyConfig.fmf_n()
    px, _, _, _ = prepare_data(*_open(), cfg)
    sig, _ = garch_forecast_panel(px.pct_change(fill_method=None), DataConfig(garch_method="grid"))
    ref = _ref("open_sig_gjr_grid.csv.gz")
    assert float((sig - ref).abs().max().max()) < 1e-12


@pytest.mark.slow
@pytest.mark.skipif(not (PRIVATE / "MarketData.csv").exists(), reason="development dataset not distributed")
def test_development_parity():
    """Fresh SLSQP GARCH on the development data: same caveat as the open-universe fresh test above.

    The notebook's development GARCH forecasts are not shipped (they are derived from the proprietary prices), so this
    test re-estimates GARCH and checks economic agreement. On a second machine (Python 3.13, NumPy 2.5.4, SciPy 1.18.1,
    arch 8.0.0): max |weight difference| 0.019, weight correlation > 0.9999, identical long/short membership on every
    date, gross Sharpe 2017-21 1.152 against 1.156. Bitwise parity holds only on the reference machine."""
    md, rt = pd.read_csv(PRIVATE / "MarketData.csv"), pd.read_csv(PRIVATE / "Ratings.csv", dtype=str)
    out = run_strategy(md, rt, StrategyConfig.fmf_n())
    a, b = out.weights.align(_ref("development_reference_weights_fmf_n.csv.gz"), join="outer")
    a, b = a.fillna(0.0), b.fillna(0.0)
    assert float((a - b).abs().max().max()) < 0.05
    assert np.corrcoef(a.to_numpy().ravel(), b.to_numpy().ravel())[0, 1] > 0.999
    assert (np.sign(a.round(12)) == np.sign(b.round(12))).all().all()   # same names long and short every day
    r = out.returns.reindex(a.index).fillna(0.0)

    def gross_sharpe(w):
        x = (w.shift(1) * r).sum(axis=1).loc["2017-01-01":"2021-12-31"]
        return float(x.mean() / x.std() * 252 ** 0.5)

    assert abs(gross_sharpe(a) - gross_sharpe(b)) < 0.05
