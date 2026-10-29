"""Regression tests for the defects fixed in 2.2.2 (one test per defect, named after it)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from fmflow import run_strategy
from fmflow.cli import main
from fmflow.data.cleaning import to_float
from fmflow.data.io import read_table
from fmflow.robustness import _importable, multiverse


# ------------------------------------------------------------------------------------------------- input dtypes
def test_to_float_turns_every_numeric_dtype_into_plain_float64():
    for values in (pd.Series(["1.5", "x"]), pd.Series([1.5, None], dtype="Float64"),
                   pd.Series([1, None], dtype="Int64"), pd.Series(["1.5", None], dtype="string")):
        out = to_float(values)
        assert out.dtype == np.float64 and out.isna().sum() == 1


def test_nullable_and_pyarrow_dtypes_give_the_same_book(synth, ew_cfg, synth_out):
    p, r = synth
    o = run_strategy(p.convert_dtypes(), r.convert_dtypes(), ew_cfg)
    assert float((o.weights - synth_out.weights).abs().max().max()) == 0.0
    pytest.importorskip("pyarrow")
    o2 = run_strategy(p.convert_dtypes(dtype_backend="pyarrow"), r.convert_dtypes(dtype_backend="pyarrow"), ew_cfg)
    assert float((o2.weights - synth_out.weights).abs().max().max()) == 0.0


def test_decimal_comma_files_run_end_to_end(tmp_path, synth, ew_cfg, synth_out):
    p, r = synth
    p.to_csv(tmp_path / "p.csv", sep=";", decimal=",", index=False)
    r.assign(estimate_value=r["estimate_value"].astype(str).str.replace(".", ",")).to_csv(tmp_path / "r.csv", sep=";", index=False)
    rp, rr = read_table(tmp_path / "p.csv"), read_table(tmp_path / "r.csv", dtype=str)
    assert rp["price"].dtype == np.float64 and rr["estimate_value"].dtype == object
    o = run_strategy(tmp_path / "p.csv", tmp_path / "r.csv", ew_cfg)
    assert np.allclose(o.weights.to_numpy(), synth_out.weights.to_numpy())


def test_missing_excel_or_parquet_engine_explains_the_extra(tmp_path, monkeypatch):
    f = tmp_path / "prices.parquet"
    f.write_bytes(b"PAR1")

    def no_engine(*a, **k):
        raise ImportError("Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.")
    monkeypatch.setattr(pd, "read_parquet", no_engine)
    with pytest.raises(ImportError, match=r"fmflow\[io\]"):
        read_table(f)


# ----------------------------------------------------------------------------------------------- multiverse
def test_multiverse_reports_failures_instead_of_hiding_them(synth, ew_cfg):
    w = {"all": slice(None)}
    with pytest.raises(RuntimeError, match="every specification failed"):
        multiverse(*synth, ew_cfg.with_(net_exposure=2.0), {"tranche_period": [21]}, w, mode="oat")
    with pytest.warns(UserWarning, match="1 of 2 specifications failed"):
        tab, nets = multiverse(*synth, ew_cfg, {"legs": [0.9]}, w, mode="oat")    # leg fraction > 0.5 is invalid
    assert "error" in tab.columns and nets.shape[1] == 1


def test_custom_signals_defined_in_main_are_detected():
    def local(ctx):
        return ctx.px
    assert not _importable(lambda ctx: ctx.px) and not _importable(local)
    assert _importable(to_float)


# ----------------------------------------------------------------------------------------------------- doctor
def test_doctor_marks_missing_optional_packages_as_opt(capsys, monkeypatch):
    import importlib
    real = importlib.import_module

    def fake(name, *a, **k):
        if name in ("sklearn", "jupyterlab"):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real(name, *a, **k)
    monkeypatch.setattr(importlib, "import_module", fake)
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0 and "[ opt] scikit-learn" in out and 'pip install "fmflow[gen]"' in out and "[  ok] fmflow" in out


# --------------------------------------------------------------------------------------------------- generative
def test_windows_need_one_complete_window():
    from fmflow.generative.windows import make_windows
    short = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 2)), columns=["a", "b"])
    with pytest.raises(ValueError, match="complete observations"):
        make_windows(short, length=64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        long = pd.DataFrame(np.random.default_rng(1).normal(size=(100, 2)), columns=["a", "b"])
        assert make_windows([short, long], length=64).X.shape == (37, 64, 2)

