"""The loaders accept the CSV dialects real users export (Excel, European locales, compressed files)."""
import gzip

import pandas as pd
import pytest

from fmflow.data.io import load_prices, load_ratings, read_table, sniff_csv

ROWS = [("2023-01-04", "s1", 1.0), ("2023-01-05", "s1", 0.9894), ("2023-01-04", "s2", 1.0), ("2023-01-05", "s2", 1.0123)]


def _write(path, text, encoding="utf-8"):
    path.write_bytes(text.encode(encoding))
    return path


def test_comma_csv_with_byte_order_mark(tmp_path):
    p = _write(tmp_path / "p.csv", "﻿date,stock,price\n" + "\n".join(f"{d},{s},{v}" for d, s, v in ROWS))
    df = read_table(p)
    assert list(df.columns) == ["date", "stock", "price"] and df["price"].iloc[1] == 0.9894


def test_european_semicolon_decimal_comma(tmp_path):
    p = _write(tmp_path / "p.csv", "date;stock;price\n" + "\n".join(f"{d};{s};{str(v).replace('.', ',')}" for d, s, v in ROWS))
    assert sniff_csv(p) == {"sep": ";", "decimal": ","}
    df = read_table(p)
    assert df["price"].tolist() == [1.0, 0.9894, 1.0, 1.0123]
    r = _write(tmp_path / "r.csv", "date;estimate_value;stock;analyst\n2023-01-04;1,6619;s1;a1\n")
    assert read_table(r, dtype=str)["estimate_value"].iloc[0] == "1.6619"


def test_tab_separated_and_gzip(tmp_path):
    text = "date\tstock\tprice\n" + "\n".join(f"{d}\t{s}\t{v}" for d, s, v in ROWS)
    p = tmp_path / "p.tsv.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write(text)
    assert sniff_csv(p)["sep"] == "\t" and len(read_table(p)) == 4


def test_excel_if_available(tmp_path):
    pytest.importorskip("openpyxl")
    p = tmp_path / "p.xlsx"
    pd.DataFrame(ROWS, columns=["date", "stock", "price"]).to_excel(p, index=False)
    assert read_table(p)["price"].tolist() == [1.0, 0.9894, 1.0, 1.0123]


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="file not found"):
        read_table(tmp_path / "nope.csv")


def test_shipped_files_load_identically_through_the_new_reader():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "data" / "open"
    if not (root / "MarketData_Normalized.csv").exists():
        pytest.skip("open dataset not present")
    a = load_prices(root / "MarketData_Normalized.csv")
    b = pd.read_csv(root / "MarketData_Normalized.csv")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
    assert len(load_ratings(root / "Ratings_Normalized.csv", a)) == 2373
