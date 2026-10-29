import pandas as pd

from fmflow.cache import cached_garch
from fmflow.robustness import multiverse


def test_multiverse_oat_and_random(synth, ew_cfg):
    idx = pd.to_datetime(sorted(synth[0]["date"].unique()))
    w = {"eval": slice(idx[300], idx[-1])}
    tab, nets = multiverse(*synth, ew_cfg, {"tranche_period": [21, 63], "data.tp_expiry_days": [126]}, w, mode="oat")
    assert len(tab) == 4 and tab["is_base"].sum() == 1 and nets.shape[1] == 4
    tab2, _ = multiverse(*synth, ew_cfg, {"tranche_period": [21, 42, 63], "legs": [3, 0.4]}, w, mode="random", n_random=5)
    assert 2 <= len(tab2) <= 6


def test_garch_cache_roundtrip(tmp_path, synth_out):
    import pytest
    pytest.importorskip("arch")
    ret = synth_out.returns.iloc[:300, :3]
    from fmflow.config import DataConfig
    cfg = DataConfig(garch_min_obs=250, garch_refit=50)
    a = cached_garch(ret, cfg, tmp_path)
    b = cached_garch(ret, cfg, tmp_path)
    pd.testing.assert_frame_equal(a, b)
    assert len(list(tmp_path.glob("garch_*.pkl"))) == 1
