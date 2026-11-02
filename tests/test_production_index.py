
from fmflow.index import IndexRules, compute_index, index_factsheet, rulebook_markdown
from fmflow.production import latest_book, risk_report, signal_monitor, trade_list


def test_latest_book_and_trades(synth_out):
    b = latest_book(synth_out)
    assert set(b["leg"]) <= {"long", "short", "flat"} and abs(b["held_weight"].sum()) < 1e-9
    px = synth_out.prices.iloc[-1] * 50
    tl = trade_list(b["held_weight"], None, px, nav=1e6, lot_size=10)
    assert (tl["target_shares"] % 10 == 0).all()
    again = trade_list(b["held_weight"], tl["target_shares"], px, nav=1e6, lot_size=10)
    assert again.empty                                            # already at target: nothing to trade


def test_risk_report_and_monitor(synth_out):
    r = risk_report(synth_out)
    assert abs(r["net"]) < 1e-9 and r["ex-ante vol (ann.)"] > 0
    m = signal_monitor(synth_out)
    assert m.shape[1] == 2 and m.dropna().shape[0] > 0


def test_index_levels_fees_and_vol_target(synth_out):
    a = compute_index(synth_out, IndexRules(fee_bps=0))
    b = compute_index(synth_out, IndexRules(fee_bps=100))
    assert a["level"].iloc[0] == 100 and b["level"].iloc[-1] < a["level"].iloc[-1]
    vt = compute_index(synth_out, IndexRules(vol_target=0.05, max_leverage=3))
    assert (vt["leverage"] <= 3 + 1e-12).all()
    assert "full period" in index_factsheet(a["level"]).index
    assert "Index Rulebook" in rulebook_markdown(IndexRules(), synth_out.config)
