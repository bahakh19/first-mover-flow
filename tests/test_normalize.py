import pandas as pd

from fmflow.data.normalize import compare_tables, normalize_universe


def _raw():
    prices = pd.DataFrame({"date": ["2023-01-04", "2023-01-04", "2023-01-05", "2023-01-05"],
                           "stock": ["MSFT", "AAPL", "MSFT", "AAPL"], "price": [200.0, 126.36, 210.0, 125.02]})
    ratings = pd.DataFrame({"date": ["2023-01-04", "2023-01-04", "2023-01-05", "2023-01-05"],
                            "estimate_value": [210.0, 175.0, 260.0, 999.0], "stock": ["AAPL", "AAPL", "MSFT", "XXXX"],
                            "analyst": ["Wedbush", "Tigress Financial", "BMO Capital", "Nobody"]})
    return prices, ratings


def test_alphabetical_ids_and_first_price_scaling():
    u = normalize_universe(*_raw())
    assert u.stock_map.to_dict() == {"AAPL": "stock_1", "MSFT": "stock_2"}
    assert u.analyst_map.to_dict() == {"BMO Capital": "analyst_1", "Tigress Financial": "analyst_2", "Wedbush": "analyst_3"}
    p = u.prices.set_index(["date", "stock"])["price"]
    assert p[("2023-01-04", "stock_1")] == 1.0 and p[("2023-01-05", "stock_1")] == 0.9894       # 125.02 / 126.36
    r = u.ratings
    assert len(r) == 3                                                    # the rating on an unpriced stock is dropped
    assert r.loc[r.analyst == "analyst_2", "estimate_value"].item() == 1.3849                   # 175 / 126.36 (the shipped value)
    assert r.loc[r.analyst == "analyst_3", "estimate_value"].item() == 1.6619                   # 210 / 126.36
    assert list(u.prices.columns) == ["date", "stock", "price"] and list(r.columns) == ["date", "estimate_value", "stock", "analyst"]


def test_normalisation_leaves_every_ratio_unchanged():
    u = normalize_universe(*_raw(), decimals=None)
    p = u.prices.pivot(index="date", columns="stock", values="price")
    assert abs(p["stock_2"].iloc[1] / p["stock_2"].iloc[0] - 210 / 200) < 1e-12


def test_compare_tables():
    u = normalize_universe(*_raw())
    same = compare_tables(u.prices, u.prices.iloc[::-1], ["date", "stock"], "price")
    assert same["only_in_a"] == same["only_in_b"] == 0 and same["max_abs_diff"] == 0 and same["share_within_tol"] == 1
