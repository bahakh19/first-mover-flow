import numpy as np
import pandas as pd

from fmflow.robustness import book_from_score, leave_one_out, signal_ic_table, strategy_placebo
from fmflow.synthetic import SyntheticConfig, generate


def test_book_from_score_reproduces_the_strategy(synth_out):
    r = book_from_score(synth_out.score, synth_out)
    assert np.allclose(r.fillna(0).values, synth_out.net_returns.fillna(0).values, atol=1e-14)


def test_signal_ic_table_structure_and_split_hygiene(synth_out):
    idx = synth_out.prices.index
    w = slice(idx[300], idx[600])
    t = signal_ic_table(synth_out, h=21, window=w)
    assert {"composite", "leader_minus_herd"} <= set(t.index)
    assert t["IC"].between(-1, 1).all() and (t["days"] <= 301 - 21).all()   # labels end inside the window


def test_placebo_returns_valid_p_values(synth_out):
    idx = synth_out.prices.index
    tab = strategy_placebo(synth_out, {"eval": slice(idx[252], idx[-1])}, n_iid=5, n_perm=5, shift_step=200)
    assert set(tab.index.get_level_values(1)) == {"iid", "permutation", "time_shift"}
    assert ((tab["p"] > 0) & (tab["p"] <= 1)).all()


def test_leave_one_out_rows(ew_cfg):
    p, r = generate(SyntheticConfig(n_stocks=7, n_sessions=500, seed=5))
    idx = pd.to_datetime(sorted(p["date"].unique()))
    tab = leave_one_out(p, r, ew_cfg, {"eval": slice(idx[252], idx[-1])})
    assert len(tab) == 8 and tab.index[0] == "all stocks" and "without stock_7" in tab.index
