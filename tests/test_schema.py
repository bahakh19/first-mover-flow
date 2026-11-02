import pandas as pd
import pytest

from fmflow.data.schema import DataContractError, validate_prices, validate_ratings

from conftest import toy_prices


def test_valid_tables_pass():
    p = toy_prices()
    r = pd.DataFrame({"date": ["2020-03-02"], "estimate_value": [1.2], "stock": ["s1"], "analyst": ["a1"]})
    assert validate_prices(p).shape == p.shape
    validate_ratings(r, p)


def test_missing_columns_and_case_insensitivity():
    p = toy_prices().rename(columns={"price": "PRICE "})
    validate_prices(p)                                     # header case / whitespace are normalised
    with pytest.raises(DataContractError, match="missing columns"):
        validate_prices(toy_prices().drop(columns="price"))
    with pytest.raises(DataContractError, match="missing columns"):
        validate_ratings(pd.DataFrame({"date": [], "stock": [], "analyst": []}))


def test_too_few_stocks_and_bad_values():
    with pytest.raises(DataContractError, match="at least 6"):
        validate_prices(toy_prices(n_stocks=4))
    p = toy_prices()
    p.loc[: len(p) // 10, "price"] = -1
    with pytest.raises(DataContractError, match="non-positive"):
        validate_prices(p)


def test_scale_mismatch_detected():
    p = toy_prices()
    r = pd.DataFrame({"date": ["2020-03-02"] * 3, "estimate_value": [150.0] * 3, "stock": ["s1", "s2", "s3"], "analyst": ["a"] * 3})
    with pytest.raises(DataContractError, match="same scale"):
        validate_ratings(r, p)
