# data/

| Folder | Contents | In git |
|---|---|---|
| `open/` | The 30-stock dataset of the pre-registered out-of-sample test (2023–2026), built from public sources (Kaggle target prices, Yahoo Finance closes) by `notebooks/Data_Gathering.ipynb`: `MarketData_Normalized.csv` (date, stock, price) and `Ratings_Normalized.csv` (date, estimate_value, stock, analyst). Prices and targets are divided by each stock's first price; stocks and brokers are numbered alphabetically. Frozen: never overwrite. | yes |
| `raw/`, `rebuilt/` | Written by `Data_Gathering.ipynb`: raw downloads with tickers and dollar prices, and the rebuilt normalised files. | no (git-ignored) |
| `private/` | Put the original development files `MarketData.csv` and `Ratings.csv` here to reproduce Sections 5–8 and 10–11 and run the development parity test. **Proprietary, not distributed.** | no (git-ignored) |
| `synthetic/` | Output of `fmflow synth` / `python tasks.py synth`. | no |

Provenance, coverage and licences of both datasets: [`../DATA_SOURCES.md`](../DATA_SOURCES.md). The input contract, cleaning rules and vendor mappings: [`../DATA.md`](../DATA.md).
