import pandas as pd

from fmflow.cli import main


def test_synth_run_compare(tmp_path):
    d = tmp_path / "syn"
    assert main(["synth", "--preset", "development", "--seed", "5", "--out", str(d)]) == 0
    assert main(["validate", "--prices", str(d / "prices.csv"), "--ratings", str(d / "ratings.csv")]) == 0
    out = tmp_path / "res"
    assert main(["run", "--prices", str(d / "prices.csv"), "--ratings", str(d / "ratings.csv"), "--allocator", "EW", "--out", str(out)]) == 0
    w = pd.read_csv(out / "weights.csv", index_col=0)
    assert w.shape[1] == 10 and (out / "summary.csv").exists() and (out / "config_used.json").exists()


def test_evaluate_and_register(tmp_path):
    d = tmp_path / "syn"
    main(["synth", "--preset", "development", "--seed", "6", "--out", str(d)])
    out = tmp_path / "eval"
    assert main(["evaluate", "--prices", str(d / "prices.csv"), "--ratings", str(d / "ratings.csv"), "--allocator", "EW",
                 "--out", str(out)]) == 0
    for f in ("performance.csv", "signal_ic21.csv", "comparables.csv", "config_used.json"):
        assert (out / f).exists()
    assert len(pd.read_csv(out / "comparables.csv")) == 17          # FMF + 13 signals + 3 allocations
    reg = tmp_path / "prereg.json"
    assert main(["register", "--config", str(out / "config_used.json"), "--title", "test", "--out", str(reg)]) == 0
    from fmflow.prereg import verify
    assert verify(reg)["frozen"]["allocator"] == "EW"


def test_new_commands_on_synthetic(tmp_path):
    d = tmp_path / "syn"
    main(["synth", "--preset", "development", "--seed", "8", "--out", str(d)])
    common = ["--prices", str(d / "prices.csv"), "--ratings", str(d / "ratings.csv"), "--allocator", "EW",
              "--cache-dir", str(tmp_path / "cache")]
    assert main(["book", *common]) == 0
    assert main(["index", *common, "--fee-bps", "50", "--vol-target", "0.1", "--out", str(tmp_path / "idx")]) == 0
    assert (tmp_path / "idx" / "RULEBOOK.md").exists()
    assert main(["stress", *common, "--generator", "bootstrap", "--paths", "50", "--horizon", "252"]) == 0
    assert main(["multiverse", *common, "--n-random", "3"]) == 0


def test_doctor_reports_every_component(capsys):
    rc = main(["doctor"])
    out = capsys.readouterr().out
    for name in ("numpy", "pandas", "scipy", "statsmodels", "arch (GJR-GARCH fit)", "jax", "data/open/MarketData_Normalized.csv"):
        assert name in out
    assert rc in (0, 1) and ("All required components work." in out) == (rc == 0)


def test_stress_diffusion_explains_when_jax_is_unusable(tmp_path, monkeypatch, capsys):
    import fmflow.generative as G
    d = tmp_path / "syn"
    main(["synth", "--preset", "development", "--seed", "8", "--out", str(d)])
    monkeypatch.setattr(G, "jax_status", lambda: (False, "simulated: CPU without AVX"))
    rc = main(["stress", "--prices", str(d / "prices.csv"), "--ratings", str(d / "ratings.csv"), "--allocator", "EW",
               "--generator", "diffusion", "--paths", "5", "--horizon", "50"])
    assert rc == 2 and "simulated: CPU without AVX" in capsys.readouterr().out
