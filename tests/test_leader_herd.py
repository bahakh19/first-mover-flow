import numpy as np
import pytest

from fmflow.signals.leader_herd import leader_minus_herd

from helpers import make


def scenario():
    rows = []
    rows += [(10, "s1", f"a{j}", 1.10) for j in range(6)]            # prior targets (initiations)
    rows += [(60, "s1", f"a{j}", 1.20) for j in range(5)]            # EVENT DAY: 5 same-day up revisions -> leaders
    rows += [(66, "s1", "a5", 1.25)]                                 # up, 6 sessions after the cluster -> HERDER
    rows += [(100, "s1", "a0", 1.10)]                                # down after 34 silent sessions -> LEADER
    rows += [(102, "s1", "a1", 1.30)]                                # up, 2 sessions later, not after a cluster -> neither
    return make(rows)


def test_classification_and_value():
    px, rt, snap = scenario()
    parts = leader_minus_herd(snap, px.index, px.columns, return_parts=True)
    d = px.index
    lmh, lead, herd = parts["leader_minus_herd"]["s1"], parts["leaders_only"]["s1"], parts["herders_only"]["s1"]
    # at session 70: 5 leaders (+1 each) and 1 herder (+1): (5 - 1) / 6
    assert lmh.loc[d[70]] == pytest.approx(4 / 6)
    # at session 110: leaders 5 up + 1 down = +4 (6 leaders), herder +1 -> (4 - 1) / 7
    assert lmh.loc[d[110]] == pytest.approx(3 / 7)
    assert lead.loc[d[110]] == pytest.approx(4 / 6)
    assert np.isnan(herd.loc[d[110]])                                 # only one herder: below min count of 2
    assert np.isnan(lmh.loc[d[60]])                                    # cluster published at 60 is known only at 61


def test_no_revisions_gives_nan():
    px, rt, snap = make([(10, "s1", "a", 1.1)])
    assert leader_minus_herd(snap, px.index, px.columns).isna().all().all()
