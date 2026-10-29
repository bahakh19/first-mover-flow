import numpy as np

from fmflow.config import DataConfig

from helpers import make


def test_classification_and_expiry():
    px, rt, snap = make([(10, "s1", "a", 1.10), (30, "s1", "a", 1.20), (50, "s1", "a", 1.20), (70, "s1", "a", 1.00),
                         (400 - 20, "s1", "b", 1.1), (10, "s2", "c", 1.1), (300, "s2", "c", 1.3)])
    ev = snap.events.set_index(["stock", "analyst", "k_pos"])["kind"]
    d = px.index
    assert ev[("s1", "a", 11)] == "initiation" and ev[("s1", "a", 31)] == "up"
    assert ev[("s1", "a", 51)] == "reiteration" and ev[("s1", "a", 71)] == "down"
    assert ev[("s2", "c", 301)] == "initiation"          # previous target (290 sessions old) had expired: re-initiation
    E = DataConfig().tp_expiry_days
    tp = snap.tp[("s2", "c")]
    assert np.isnan(tp.iloc[10]) and tp.iloc[11] == 1.1   # known the session after publication
    assert tp.iloc[11 + E - 1] == 1.1 and np.isnan(tp.iloc[11 + E])  # expires after 252 sessions
    assert snap.n_up.loc[d[31], "s1"] == 1 and snap.n_down.loc[d[71], "s1"] == 1
