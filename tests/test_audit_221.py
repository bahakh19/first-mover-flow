"""Regression tests for the defects fixed in 2.2.1 (one test per defect, named after it)."""
from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from fmflow import StrategyConfig, run_strategy
from fmflow.analytics.inference import cscv_pbo, fama_macbeth, forward_log_return, psr
from fmflow.analytics.metrics import perf_stats
from fmflow.analytics.placebo import placebo_test
from fmflow.backtest.engine import backtest
from fmflow.cli import main
from fmflow.data.cleaning import clean_prices, to_calendar_dates
from fmflow.data.io import read_table
from fmflow.data.schema import DataContractError, validate_prices, validate_ratings
from fmflow.pipeline import construct_book
from fmflow.portfolio.construction import cap_weights, ledoit_wolf_cov, target_weights
from fmflow.production import latest_book, trade_list
from fmflow.robustness import book_from_score

from conftest import ROOT, toy_prices


# ---------------------------------------------------------------------------------------------------------- dates
def test_calendar_dates_accept_times_offsets_and_mixed_formats():
    d = to_calendar_dates(["2023-01-04", "2023-01-04 16:00", "2023-01-04T08:30:00+05:00", "2023-01-05T23:30:00-04:00",
                           "garbage", None])
    assert d.iloc[:4].tolist() == [pd.Timestamp("2023-01-04")] * 3 + [pd.Timestamp("2023-01-05")]
    assert d.iloc[4:].isna().all()
    assert to_calendar_dates([]).empty
    assert to_calendar_dates(["01/04/2023", "01/05/2023"]).tolist() == [pd.Timestamp("2023-01-04"), pd.Timestamp("2023-01-05")]


def test_timestamped_inputs_give_the_same_book(synth, ew_cfg, synth_out):
    p, r = synth
    p2 = p.assign(date=p["date"] + " 16:00:00")
    r2 = r.assign(date=r["date"].astype(str) + " 08:30:00")       # published before the open: still usable only at t+1
    o = run_strategy(p2, r2, ew_cfg)
    assert float((o.weights - synth_out.weights).abs().max().max()) == 0.0


def test_integer_stock_ids_match_text_ids(synth, ew_cfg, synth_out):
    p, r = synth
    p2 = p.assign(stock=p["stock"].str.replace("stock_", "").astype(int))
    r2 = r.assign(stock=r["stock"].str.replace("stock_", ""))
    validate_ratings(r2, validate_prices(p2))
    o = run_strategy(p2, r2, ew_cfg)
    assert np.allclose(o.net_returns.to_numpy(), synth_out.net_returns.to_numpy(), equal_nan=True)


# ------------------------------------------------------------------------------------------------------- cleaning
def test_missing_stock_ids_are_dropped_not_turned_into_a_nan_column():
    p = toy_prices(8, 50)
    p.loc[:4, "stock"] = np.nan
    px = clean_prices(p)
    assert "nan" not in map(str, px.columns) and px.shape[1] == 8


def test_no_session_left_raises_an_actionable_error():
    p = toy_prices(9, 60)
    day = pd.factorize(p["date"])[0]
    group = p["stock"].str[1:].astype(int) % 3
    p = p[day % 3 == group]                     # each date only a third of the (active) stocks print: no session
    with pytest.raises(ValueError, match="min_calendar_coverage"):
        clean_prices(p)


def test_run_strategy_applies_the_scale_check(synth, ew_cfg):
    p, r = synth
    with pytest.raises(DataContractError, match="same scale"):
        run_strategy(p, r.assign(estimate_value=pd.to_numeric(r["estimate_value"]) * 150), ew_cfg)


def test_no_accepted_rating_warns(synth, ew_cfg):
    p, r = synth
    with pytest.warns(UserWarning, match="no rating survived cleaning"):
        o = run_strategy(p, r.assign(date="2099-01-01"), ew_cfg)
    assert float(o.weights.abs().sum().sum()) == 0.0


# --------------------------------------------------------------------------------------------------- construction
def test_infeasible_cap_is_never_exceeded():
    t = pd.DataFrame([[0.5 / 3] * 3 + [-0.5 / 3] * 3 + [0.0, 0.0]])
    w = cap_weights(t, 0.1)
    assert float(w.abs().max().max()) <= 0.1 + 1e-12
    assert np.isclose(w.clip(lower=0).sum(axis=1).iloc[0], 0.3)


def test_feasible_cap_keeps_leg_budgets_with_many_names():
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(rng.pareto(1.2, (5, 200)) + 0.01)
    long_ = raw.div(raw.sum(axis=1), axis=0) * 0.5
    w = cap_weights(pd.concat([long_, -long_.set_axis(range(200, 400), axis=1)], axis=1), 0.004)
    assert float(w.abs().max().max()) <= 0.004 + 1e-12
    assert np.allclose(w.clip(lower=0).sum(axis=1), 0.5) and np.allclose(w.clip(upper=0).sum(axis=1), -0.5)


def test_ledoit_wolf_vectorised_matches_the_definition():
    x = np.random.default_rng(1).normal(size=(126, 9))
    xc = x - x.mean(0)
    n, p = xc.shape
    S = xc.T @ xc / n
    mu = np.trace(S) / p
    d2 = ((S - mu * np.eye(p)) ** 2).sum() / p
    b2 = sum(((np.outer(r, r) - S) ** 2).sum() for r in xc) / (n ** 2 * p)
    sh = min(b2, d2) / d2
    assert np.allclose(ledoit_wolf_cov(x), sh * mu * np.eye(p) + (1 - sh) * S, atol=1e-15)


def test_iv_gjr_without_volatility_estimates_keeps_the_book_balanced():
    idx = pd.bdate_range("2021-01-01", periods=5)
    score = pd.DataFrame(np.tile(np.arange(8.0), (5, 1)), index=idx, columns=list("abcdefgh"))
    sig = pd.DataFrame(0.01, index=idx, columns=score.columns)
    sig[["g", "h"]] = np.nan                                         # the long leg has no forecast at all
    ret = pd.DataFrame(np.nan, index=idx, columns=score.columns)     # and no return history to fall back on
    t = target_weights(score, k=2, allocator="IV-GJR", sig_vol=sig, ret=ret)
    assert np.allclose(t.sum(axis=1), 0.0) and np.allclose(t.abs().sum(axis=1), 1.0)


# ------------------------------------------------------------------------------------------------------- backtest
def test_opening_trade_is_charged_and_nan_weights_are_flat():
    idx = pd.bdate_range("2021-01-01", periods=3)
    w = pd.DataFrame({"a": [0.5, 0.5, np.nan], "b": [-0.5, -0.5, -0.5]}, index=idx)
    r = pd.DataFrame(0.0, index=idx, columns=["a", "b"])
    res = backtest(w, r, cost_bps=10.0)
    assert np.isclose(res.traded.iloc[0], 1.0) and np.isclose(res.costs.iloc[0], 1e-3)
    assert np.isclose(res.traded.iloc[2], 0.5)                       # NaN = flat: the 0.5 long is closed


def test_sortino_uses_the_downside_deviation():
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01] * 60)
    dd = np.sqrt((np.minimum(r, 0) ** 2).mean() * 252)
    assert np.isclose(perf_stats(r)["Sortino"], r.mean() * 252 / dd)
    assert perf_stats(pd.Series([0.1, -1.0, 0.1]))["CAGR"] == -1.0  # a wiped-out book, not NaN


def test_book_from_score_uses_the_full_construction(synth, ew_cfg):
    cfg = ew_cfg.with_(borrow_bps=100.0, execution_lag=1, max_weight=0.2)
    o = run_strategy(*synth, cfg)
    assert np.allclose(book_from_score(o.score, o).to_numpy(), o.net_returns.to_numpy(), equal_nan=True)
    _, w = construct_book(o.score, o.returns, cfg, o.sig_vol)
    assert float((w - o.weights).abs().max().max()) == 0.0


# ------------------------------------------------------------------------------------------------ config / output
def test_config_rejects_an_empty_signal_list():
    with pytest.raises(ValueError, match="at least one"):
        StrategyConfig.fmf_n().with_(signals=()).validate()


def test_repr_and_summary(synth_out):
    text = repr(synth_out)
    assert "sessions" in text and "see .summary()" in text and len(text) < 400
    assert {"CAGR", "Sharpe", "turnover"} <= set(synth_out.summary().columns)


def test_production_guards(synth_out):
    with pytest.raises(ValueError, match="no session on or before"):
        latest_book(synth_out, "1990-01-01")
    w = pd.Series({"a": 0.5, "b": -0.5})
    with pytest.raises(ValueError, match="nav"):
        trade_list(w, None, pd.Series({"a": 10.0, "b": 20.0}), nav=0)
    with pytest.raises(ValueError, match="lot_size"):
        trade_list(w, None, pd.Series({"a": 10.0, "b": 20.0}), nav=1e6, lot_size=0)


# ------------------------------------------------------------------------------------------------------ inference
def test_inference_guards():
    assert np.isnan(psr(np.zeros(50))) and np.isnan(psr([0.01, 0.02]))
    trials = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 3)))
    with pytest.raises(ValueError, match="even"):
        cscv_pbo(trials, 5)
    with pytest.raises(ValueError, match="two trials"):
        cscv_pbo(trials[[0]], 4)
    with pytest.raises(ValueError, match="two stocks"):
        placebo_test(pd.DataFrame({"a": [1.0, 2.0]}), lambda s: pd.Series(0.0, index=s.index), {"all": slice(None)}, float)


def test_fama_macbeth_lag_follows_the_label_horizon(synth_out):
    fwd = forward_log_return(synth_out.prices, 63)
    f = synth_out.signals["breadth_63"]
    ctrl = {"mom": synth_out.prices.pct_change(126, fill_method=None)}
    a = fama_macbeth(f, ctrl, fwd, step=5, horizon=63)
    b = fama_macbeth(f, ctrl, fwd, step=5, lags=13)
    c = fama_macbeth(f, ctrl, fwd, step=5, lags=4)
    assert np.allclose(a["t_NW"], b["t_NW"]) and not np.allclose(a["t_NW"], c["t_NW"])


# ------------------------------------------------------------------------------------------------------------- io
def test_legacy_windows_encoding(tmp_path):
    f = tmp_path / "ansi.csv"
    f.write_bytes("date;stock;price\n2023-01-04;Soci\xe9t\xe9;1,25\n".encode("cp1252"))
    t = read_table(f)
    assert t["stock"].iloc[0] == "Société" and t["price"].iloc[0] == 1.25


# ------------------------------------------------------------------------------------------------------------ CLI
def test_cli_version_help_and_errors(capsys, tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and "fmflow " in capsys.readouterr().out
    assert main([]) == 0 and "commands" in capsys.readouterr().out
    assert main(["run", "--prices", str(tmp_path / "missing.csv"), "--ratings", "x.csv"]) == 2
    err = capsys.readouterr().err
    assert "error:" in err and "file not found" in err and "Traceback" not in err


def test_cli_validate_reports_cleaning_and_refuses_unusable_data(tmp_path, capsys):
    d = tmp_path / "syn"
    main(["synth", "--preset", "development", "--seed", "5", "--out", str(d)])
    assert main(["validate", "--prices", str(d / "prices.csv"), "--ratings", str(d / "ratings.csv")]) == 0
    assert "Cleaning:" in capsys.readouterr().out
    r = pd.read_csv(d / "ratings.csv").assign(date="2099-01-01")
    r.to_csv(tmp_path / "late.csv", index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                               # the CLI reports it itself, no stray warning
        rc = main(["validate", "--prices", str(d / "prices.csv"), "--ratings", str(tmp_path / "late.csv")])
    assert rc == 1 and "Not usable" in capsys.readouterr().out


def test_cli_leg_fraction_is_reflected_in_the_name(tmp_path):
    d = tmp_path / "syn"
    main(["synth", "--preset", "development", "--seed", "5", "--out", str(d)])
    out = tmp_path / "res"
    assert main(["run", "--prices", str(d / "prices.csv"), "--ratings", str(d / "ratings.csv"), "--allocator", "EW",
                 "--leg-fraction", "0.4", "--out", str(out)]) == 0
    cfg = StrategyConfig.load(out / "config_used.json")
    assert cfg.leg_fraction == 0.4 and cfg.k is None and "leg fraction 0.4" in cfg.name and "EW" in cfg.name


def test_python_dash_m_fmflow():
    rc = subprocess.run([sys.executable, "-m", "fmflow", "--version"], capture_output=True, text=True, cwd=ROOT)
    assert rc.returncode == 0 and rc.stdout.startswith("fmflow ")


def test_version_matches_pyproject():
    import re

    import fmflow
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == fmflow.__version__


# ------------------------------------------------------------------------------------------------ data utilities
def test_normalize_keeps_a_missing_analyst_missing():
    from fmflow.data.normalize import normalize_universe
    p = pd.DataFrame({"date": ["2023-01-03", "2023-01-04"] * 2, "stock": ["A", "A", "B", "B"], "price": [10, 11, 20, 21]})
    r = pd.DataFrame({"date": ["2023-01-04"] * 3, "estimate_value": [12, 25, 13], "stock": ["A", "B", "A"],
                      "analyst": ["x", None, "y"]})
    u = normalize_universe(p, r)
    assert u.ratings["analyst"].isna().sum() == 1 and set(u.analyst_map.index) == {"x", "y"}


def test_unreadable_garch_cache_entry_is_recomputed(tmp_path, monkeypatch):
    import fmflow.risk.garch as G
    from fmflow.cache import _key, cached_garch
    from fmflow.config import DataConfig
    ret = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (30, 2)), columns=["a", "b"])
    fake = pd.DataFrame(0.01, index=ret.index, columns=ret.columns)
    monkeypatch.setattr(G, "garch_forecast_panel", lambda r, c: (fake, None))
    (tmp_path / f"garch_{_key(ret, DataConfig())}.pkl").write_bytes(b"not a pickle")
    assert cached_garch(ret, DataConfig(), tmp_path).equals(fake)


def test_doctor_outside_the_repository_says_so(tmp_path, capsys):
    main(["doctor", "--root", str(tmp_path)])
    assert "run from the repository root" in capsys.readouterr().out
