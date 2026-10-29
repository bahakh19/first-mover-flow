"""Loaders that apply the data contract.

Accepted inputs, detected automatically:

* CSV in any common dialect: comma, semicolon or tab separated; UTF-8 with or without a byte-order mark (Excel's
  "CSV UTF-8") or legacy Windows-1252; European exports with a semicolon separator and a decimal comma ("1,2345");
  gzip / zip / bz2 / xz compressed (`.csv.gz` ...);
* Parquet (`.parquet`, `.pq`; needs pyarrow or fastparquet);
* Excel (`.xlsx`, `.xls`; needs openpyxl / xlrd), first sheet.
"""
from __future__ import annotations

import bz2
import gzip
import lzma
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from fmflow.data.cleaning import to_float
from fmflow.data.schema import validate_prices, validate_ratings

_VALUE_COLUMNS = ("price", "estimate_value")


def _head(path: Path, n: int = 8192) -> str:
    """First bytes of a (possibly compressed) text file, decoded leniently, for dialect sniffing."""
    name = path.name.lower()
    try:
        if name.endswith(".gz"):
            with gzip.open(path, "rb") as f:
                raw = f.read(n)
        elif name.endswith(".bz2"):
            with bz2.open(path, "rb") as f:
                raw = f.read(n)
        elif name.endswith(".xz"):
            with lzma.open(path, "rb") as f:
                raw = f.read(n)
        elif name.endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                raw = z.open(z.namelist()[0]).read(n)
        else:
            with path.open("rb") as f:
                raw = f.read(n)
    except (OSError, zipfile.BadZipFile, IndexError):
        return ""
    return raw.decode("utf-8", errors="replace").lstrip("\ufeff")


def sniff_csv(path: str | Path) -> dict:
    """Separator and decimal mark of a CSV file: {"sep": ",", ";" or "\\t", "decimal": "." or ","}."""
    lines = [ln for ln in _head(Path(path)).splitlines() if ln.strip()][:20]
    if not lines:
        return {"sep": ",", "decimal": "."}
    header = lines[0]
    sep = max((",", ";", "\t"), key=lambda s: (header.count(s), s == ","))
    if header.count(sep) == 0:
        sep = ","
    body = "\n".join(lines[1:])
    decimal = "," if sep in (";", "\t") and any(ch.isdigit() for ch in body) and \
        any("," in field and field.replace(",", "", 1).replace("-", "", 1).strip().isdigit()
            for ln in lines[1:] for field in ln.split(sep)) else "."
    return {"sep": sep, "decimal": decimal}


def read_table(path: str | Path, **kw) -> pd.DataFrame:
    """Read a CSV, Parquet or Excel file into a DataFrame (dialect detected, see the module docstring)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path}: file not found (working directory: {Path.cwd()})")
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq", ".xlsx", ".xlsm", ".xls"):
        try:
            df = pd.read_parquet(path) if suffix in (".parquet", ".pq") else pd.read_excel(path, **kw)
        except ImportError as e:                      # optional engines: pyarrow / openpyxl / xlrd
            raise ImportError(f"{path.name}: reading {suffix} files needs an optional package. Install the 'io' extra "
                              f'(pip install "fmflow[io]", or uv sync --extra io) or save the table as CSV. '
                              f"Details: {str(e).splitlines()[0]}") from e
    else:
        d = sniff_csv(path)
        try:
            df = pd.read_csv(path, sep=d["sep"], encoding="utf-8-sig", **kw)
        except UnicodeDecodeError:                    # legacy Windows / Excel "ANSI" export
            df = pd.read_csv(path, sep=d["sep"], encoding="cp1252", encoding_errors="replace", **kw)
        if d["decimal"] == ",":                       # "1,2345" -> "1.2345" in the numeric columns only
            for c in df.columns:
                if str(c).strip().lower() in _VALUE_COLUMNS:
                    txt = df[c].astype("string").str.replace(",", ".", regex=False)
                    # plain dtypes, as for any other CSV: text stays object (NaN when missing), numbers become float64
                    df[c] = txt.astype(object).where(txt.notna(), np.nan) if kw.get("dtype") is str else to_float(txt)
    return df


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load a long price table (date, stock, price) and validate it."""
    return validate_prices(read_table(path))


def load_ratings(path: str | Path, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load a long ratings table (date, estimate_value, stock, analyst); values are kept as text until cleaning."""
    return validate_ratings(read_table(path, dtype=str), prices)


_read = read_table      # backwards compatibility
