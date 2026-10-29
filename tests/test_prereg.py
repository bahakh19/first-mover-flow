import json

import pytest

from fmflow import StrategyConfig
from fmflow.prereg import PreregistrationError, fingerprint, register, verify

from conftest import ROOT


def test_shipped_open_universe_registration_is_intact():
    doc = verify(ROOT / "configs" / "oos_open_universe_preregistration.json")
    assert doc["sha256_16"] == "0b2b0792fd357d61"
    assert doc["frozen_from_development"]["tranche_period"] == StrategyConfig.fmf_n().tranche_period
    assert doc["frozen_from_development"]["signals"] == list(StrategyConfig.fmf_n().signals)


def test_register_verify_and_tamper_detection(tmp_path):
    p = tmp_path / "test.json"
    doc = register({"title": "t", "frozen": StrategyConfig.fmf_n().to_dict()}, p)
    assert verify(p)["sha256_16"] == doc["sha256_16"] == fingerprint(doc)
    with pytest.raises(FileExistsError):
        register({"title": "t2"}, p)
    d = json.loads(p.read_text(encoding="utf-8"))
    d["frozen"]["k"] = 3                                    # a silent post-hoc change ...
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    with pytest.raises(PreregistrationError):               # ... is detected
        verify(p)


def test_fingerprint_ignores_stamp_fields_only():
    a = {"x": 1, "sha256_16": "abc", "registered_utc": "now"}
    assert fingerprint(a) == fingerprint({"x": 1}) != fingerprint({"x": 2})
