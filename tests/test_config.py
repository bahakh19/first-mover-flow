import pytest

from fmflow import StrategyConfig


def test_presets_and_roundtrip(tmp_path):
    for cfg in (StrategyConfig.fmf_n(), StrategyConfig.fmf_b20(), StrategyConfig.round3(),
                StrategyConfig.fmf_n().with_(k=None, leg_fraction=0.4)):
        cfg.validate()
        p = tmp_path / "c.json"
        cfg.save(p)
        assert StrategyConfig.load(p) == cfg


def test_frozen_specification_values():
    c = StrategyConfig.fmf_n()
    assert (c.k, c.tranche_period, c.allocator, c.net_exposure, c.cost_bps) == (4, 42, "IV-GJR", 0.0, 5.0)
    assert c.signals == ("breadth_63", "event_breadth_63", "breadth_resid_mom", "leader_minus_herd")
    assert (c.leader_gap, c.herd_lo, c.herd_hi, c.lmh_window) == (10, 3, 15, 126)


@pytest.mark.parametrize("kw", [dict(allocator="MV"), dict(k=None), dict(k=4, leg_fraction=0.4), dict(leg_fraction=0.7, k=None),
                                dict(tranche_period=0), dict(net_exposure=1.5), dict(signals=("nope",))])
def test_validation_rejects_bad_configs(kw):
    with pytest.raises(ValueError):
        StrategyConfig.fmf_n().with_(**kw).validate()


def test_unknown_field_rejected():
    d = StrategyConfig.fmf_n().to_dict()
    d["foo"] = 1
    with pytest.raises(ValueError):
        StrategyConfig.from_dict(d)
