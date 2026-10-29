# Data sources

This project uses two datasets. They have the same format (DATA.md §1) but different origins, licences and roles.

| | Development dataset | Open dataset |
|---|---|---|
| Role | every research choice (2017–2019) and the holdout (2020–2021) | the pre-registered out-of-sample test (2024–2026) |
| Origin | **proprietary**, supplied to the author | **public**: Kaggle (target prices) and Yahoo Finance (prices) |
| In this repository | no (`data/private/` is git-ignored) | yes: `data/open/MarketData_Normalized.csv`, `data/open/Ratings_Normalized.csv` |
| Rebuild from scratch | not possible without the original files | `notebooks/Data_Gathering.ipynb` |
| Stocks | 10 US large caps | 30 US large caps (including the same 10) |
| Period | 2016-01-04 to 2021-12-31 (1,511 sessions after cleaning) | 2023-01-04 to 2026-08-05 (899 sessions) |
| Target prices (raw / accepted) | 5,987 / 5,929 | 2,373 / 2,363 |
| Rating sources | 304 individual analysts | 51 brokers (50 after cleaning) |

---

## 1. Development dataset (proprietary, not distributed)

### 1.1 What it is

Two long-format CSV files supplied to the author as proprietary data. They may not be redistributed, so they are not in this repository.

| File | Columns | Rows | Content |
|---|---|---:|---|
| `MarketData.csv` | `date, stock, price` | 15,660 | daily closing prices of 10 stocks, 2016-01-01 to 2021-12-31 (1,566 calendar dates, of which 55 non-trading dates are removed by the cleaning) |
| `Ratings.csv` | `date, estimate_value, stock, analyst` | 5,987 | sell-side 12-month target prices with their **publication date**, 2016-01-04 to 2021-12-27 |

The data arrive **anonymised** by the provider: stocks are `stock_1` to `stock_10`, analysts `analyst_1` to `analyst_304`. Each stock's prices are normalised to about 1 at the start of the sample, and target prices are on the same scale, so the target / price ratio is preserved. Some values arrive as text, a few are unparseable or missing, and 47 ratings have no analyst identifier. The cleaning rules handle all of these and log every row they drop (Appendix A of the paper, `out.audit`).

| Cleaning step | Rule | Rows |
|---|---|---:|
| prices | dates on which fewer than 50% of the active stocks print are not sessions | 55 dates |
| R1 | unparseable or non-positive target | 9 |
| R2 | missing analyst identifier | 47 |
| R3 | target / price outside (0.2, 5.0) | 2 |
| R4 | target change inconsistent with the price change | 0 |
| accepted | | **5,929** (3,168 up revisions, 1,020 down, 1,260 reiterations, 481 initiations) |

**Price basis:** The prices are split-adjusted closes **without dividends**: the reconciliation matches the unadjusted close exactly, and the dividend-adjusted close less well. For example, stock_5 (JPMorgan) grows 2.49× in the data, the ratio of its public closing prices on 2016-01-04 (63.62) and 2021-12-31 (158.35), with dividends reinvested the multiple would be close to 3×. Strategy returns are therefore price returns. Dividends are ignored in both datasets, which for a dollar-neutral book biases returns only through the dividend-yield spread between the long and short legs.

### 1.2 The ten stocks

The anonymised stocks were identified from public prices in the research notebook (Section 0, ticker reconciliation): for each of the ten, the daily returns in the data have a correlation of 1.0000 (R² 1.0000) with the returns of one public stock's split-adjusted, dividend-unadjusted closing price over 1,510 sessions. The identification is used **only** to interpret results economically (sectors, the 2020 crash, the 2021 growth sell-off), no rule anywhere in the code refers to a specific stock.

| Id | Company | Total return 2016–21 | Annual vol. | Analysts | Ratings | Fresh revisions | Event clusters | Median active analysts (2017–19) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `stock_1` | Apple (AAPL) | 6.74× | 29.4% | 72 | 775 | 588 | 25 | 30 |
| `stock_2` | Boeing (BA) | 1.43× | 44.1% | 31 | 430 | 325 | 5 | 15 |
| `stock_3` | Chegg (CHGG) | 4.92× | 52.6% | 17 | 240 | 162 | 17 | 8 |
| `stock_4` | Intel (INTC) | 1.52× | 33.0% | 50 | 639 | 401 | 29 | 25 |
| `stock_5` | JPMorgan Chase (JPM) | 2.49× | 29.2% | 29 | 463 | 355 | 2 | 18 |
| `stock_6` | Microsoft (MSFT) | 6.14× | 26.8% | 58 | 670 | 467 | 24 | 25 |
| `stock_7` | Nike (NKE) | 2.71× | 28.2% | 44 | 620 | 418 | 27 | 24 |
| `stock_8` | NVIDIA (NVDA) | 36.35× | 47.1% | 54 | 801 | 616 | 30 | 27 |
| `stock_9` | Palo Alto Networks (PANW) | 3.25× | 36.9% | 54 | 810 | 528 | 30 | 30 |
| `stock_10` | Qualcomm (QCOM) | 3.65× | 36.4% | 36 | 481 | 328 | 25 | 17 |

*Fresh revisions: target changes by an analyst whose previous target is still active. Event clusters: (stock, day) pairs with at least five ratings. Median active: analysts with a target at most 252 sessions old.*

What makes this sample hard: ten names only, dominated by technology and semiconductors, one extreme winner (NVDA, 36×), two names with thin coverage (CHGG, BA), and a single regime break (2020). Section 4 of the paper discusses these features.

### 1.3 Reproducing the development results

Place the two original files in `data/private/` under their original names. The parity test (`tests/test_parity.py::test_development_parity`), `python tasks.py development`, `python tasks.py audit` and the research notebook then run, without the files they skip cleanly.

---

## 2. Open dataset (public sources, shipped)

### 2.1 Sources

| Source | Used for | Licence / terms |
|---|---|---|
| Kaggle: [`yijian2427/us-market-analyst-insights-with-full-stock-symbols`](https://www.kaggle.com/datasets/yijian2427/us-market-analyst-insights-with-full-stock-symbols) | target prices: `date`, `ticker`, `firm`, `price_target` (the file has 53,776 rows and 17 columns, including the rating, rating action and a text summary) | MIT, as stated on the dataset page |
| Yahoo Finance via the `yfinance` package | daily closing prices (`Close`, `auto_adjust=False`: split-adjusted, not dividend-adjusted) | Yahoo's terms of use, the repository ships only normalised, derived prices |

### 2.2 Construction (`notebooks/Data_Gathering.ipynb`)

1. Keep the Kaggle ratings for 30 tickers: the ten development names plus twenty large caps (GOOGL, AMZN, META, TSLA, AMD, AVGO, CRM, NFLX, ORCL, CSCO, TXN, ADBE, IBM, GS, MS, UNH, V, MA, HD, COST).
2. Drop rows with a missing or non-positive target, normalise broker names (whitespace), and keep the last target per (date, stock, broker). This leaves 2,373 target prices from 51 brokers, 2023-01-04 to 2026-08-06.
3. Download closing prices for the same tickers from the first to the last rating date (Yahoo's end date is exclusive, so prices stop on 2026-08-05): 26,970 rows, 899 sessions × 30 stocks.
4. Anonymise and normalise with `fmflow.data.normalize_universe`:
   * stocks numbered in alphabetical order of ticker (`AAPL` → `stock_1`, …, `V` → `stock_30`)
   * brokers numbered in alphabetical order of name (Python string order) → `analyst_1` … `analyst_51`
   * each stock's prices **and** targets divided by its first closing price (2023-01-04), rounded to 4 decimals.

Two checks of the procedure against public values: Apple closed at 126.36 on 2023-01-04 and Tigress Financial's target that day was 210, giving the shipped 1.6619 (= 210 / 126.36), BMO Capital's 267 target on Microsoft on 2023-01-23, against Microsoft's 229.10 close on 2023-01-04, gives the shipped 1.1654.

**Frozen files:** The pre-registered test uses the files exactly as shipped:

```
0db3f19a456b71f085fef29f48003e66abf5c0ca1c9dfcff96bcc76175725609  data/open/MarketData_Normalized.csv
97f4a8a7f4496877de4263cfd5d7f376cfcf040572745b0570534d7caa821a42  data/open/Ratings_Normalized.csv
```

Kaggle datasets and Yahoo prices are revised over time, so a rebuild can differ in a few rows. The notebook writes rebuilt files to `data/rebuilt/` (git-ignored), compares them with the frozen ones row by row, and never overwrites `data/open/`.

### 2.3 Stock map and coverage

The map follows from the published ticker list and the alphabetical rule, so it is given here for transparency.

| Id | Ticker | Development name | Ratings | Brokers | Price 2026-08 / 2023-01 | Price 2026-08 / 2024-01 | Annual vol. |
|---|---|:---:|---:|---:|---:|---:|---:|
| `stock_1` | AAPL | yes | 136 | 14 | 2.46 | 1.72 | 26.0% |
| `stock_2` | ADBE |  | 55 | 8 | 0.76 | 0.46 | 35.4% |
| `stock_3` | AMD |  | 82 | 17 | 7.46 | 3.48 | 57.0% |
| `stock_4` | AMZN |  | 192 | 27 | 3.20 | 1.88 | 32.9% |
| `stock_5` | AVGO |  | 67 | 14 | 7.47 | 3.99 | 47.4% |
| `stock_6` | BA | yes | 19 | 6 | 1.18 | 0.96 | 33.7% |
| `stock_7` | CHGG | yes | 13 | 3 | 0.04 | 0.10 | 93.8% |
| `stock_8` | COST |  | 105 | 13 | 2.06 | 1.44 | 20.1% |
| `stock_9` | CRM |  | 146 | 26 | 1.38 | 0.77 | 35.4% |
| `stock_10` | CSCO |  | 22 | 6 | 2.56 | 2.43 | 24.9% |
| `stock_11` | GOOGL |  | 132 | 21 | 4.11 | 2.67 | 31.0% |
| `stock_12` | GS |  | 16 | 6 | 3.05 | 2.74 | 28.8% |
| `stock_13` | HD |  | 67 | 12 | 1.10 | 1.03 | 22.8% |
| `stock_14` | IBM |  | 40 | 5 | 1.65 | 1.48 | 32.5% |
| `stock_15` | INTC | yes | 63 | 14 | 3.65 | 2.16 | 60.1% |
| `stock_16` | JPM | yes | 17 | 5 | 2.63 | 2.09 | 23.3% |
| `stock_17` | MA |  | 41 | 13 | 1.61 | 1.36 | 19.9% |
| `stock_18` | META |  | 142 | 24 | 4.62 | 1.67 | 39.2% |
| `stock_19` | MS |  | 13 | 4 | 2.52 | 2.34 | 28.3% |
| `stock_20` | MSFT | yes | 138 | 21 | 2.13 | 1.33 | 26.4% |
| `stock_21` | NFLX |  | 113 | 13 | 2.40 | 1.57 | 34.5% |
| `stock_22` | NKE | yes | 82 | 13 | 0.35 | 0.42 | 34.7% |
| `stock_23` | NVDA | yes | 174 | 18 | 14.86 | 4.47 | 48.5% |
| `stock_24` | ORCL |  | 94 | 19 | 1.71 | 1.41 | 47.1% |
| `stock_25` | PANW | yes | 117 | 27 | 5.24 | 2.56 | 40.9% |
| `stock_26` | QCOM | yes | 42 | 9 | 1.41 | 1.15 | 41.5% |
| `stock_27` | TSLA |  | 134 | 18 | 2.83 | 1.35 | 58.1% |
| `stock_28` | TXN |  | 44 | 10 | 1.64 | 1.68 | 35.0% |
| `stock_29` | UNH |  | 34 | 11 | 0.82 | 0.77 | 35.0% |
| `stock_30` | V |  | 33 | 10 | 1.73 | 1.42 | 19.5% |

*Ratings and brokers before cleaning. Two events confirm the map: stock_7 (Chegg) fell 48% on 2023-05-02, and stock_24 (Oracle) rose 36% on 2025-09-10.*

---

## 3. How the two datasets relate, and what that implies

**The periods never overlap, ten of the names do:** The out-of-sample window (2024-01-05 to 2026-08-05) starts two years after the development data end. Twenty of the thirty names were never used in development, but the ten development names are included. "Never-seen" in this repository and in the paper refers to the **period**, for the names it applies to two thirds of the universe.

**Does the overlap drive the result?** It does not. The table below re-runs the frozen specification (leg fraction 0.4, same GARCH forecasts) on each part of the universe separately. This split was not pre-registered: it was run after the overlap was documented.

| Universe (2024-01-05 to 2026-08-05) | Net Sharpe | CAGR | Max drawdown | Calmar | LMH IC21 (one-sided *p*) |
|---|---:|---:|---:|---:|---:|
| all 30 names (pre-registered) | 1.05 | 10.5% | −8.0% | 1.32 | 0.096 (0.042) |
| the 20 names never used in development | 0.80 | 9.0% | −9.2% | 0.98 | 0.105 (0.058) |
| the 10 development names only | 0.31 | 2.9% | −11.0% | 0.26 | 0.022 (0.45) |

The leader-minus-herd signal keeps its information coefficient on the twenty new names. The ten development names alone are too few for this construction: with sparse broker coverage, a valid five-name cross-section exists on only part of the window. → `python scripts/overlap_analysis.py`.

**Hindsight in the choice of names:** The twenty expansion names are today's large, heavily covered companies. A list chosen in 2026 knows which companies grew large, so the universe tilts towards winners. A dollar-neutral book is exposed to this only through the **dispersion** between its long and short legs, not the level of returns. The causal decomposition (paper §10.1) shows that the static tilt towards particular stocks loses money out of sample (Sharpe −0.49) while the timing component earns 1.33.

**Different rating sources:** The development data are analyst-level (304 analysts, median 47 per stock). The open data are broker-level (51 brokers, median 13 per stock). A "leader" in the open data is the first **broker** to revise after a quiet period. That the signal transfers across this change of granularity is part of the out-of-sample evidence.

---

## 4. Known limitations

* **Price returns, not total returns:** Both datasets use closes without dividends (§1.1).
* **Rounding of the open prices:** Four decimals on a normalised price quantise the returns of a stock that falls far below its starting level. For Chegg (stock_7), whose normalised price falls to 0.0177, the rounding error reaches 0.3% of the price. `normalize_universe(decimals=None)` keeps full precision for your own rebuilds.
* **Publication dates are calendar dates:** Neither source gives a time of day, so a rating becomes usable on the session **after** publication (`knowledge_lag = 1`). The paper shows that a two-session lag leaves the results unchanged.
* **Coverage of the open data is thin for a few names** (13 to 22 ratings for BA, CHGG, JPM, GS, MS, CSCO over 3.5 years). Signals are `NaN` where coverage is insufficient, and those names drop out of the cross-section on those days.
* **Third-party data may contain errors:** The cleaning rules R1–R6 catch the common ones (DATA.md §2). Two ratings in the open data fail R3 (implausible target / price) and eight are published after the last price.

## 5. Licences and redistribution

* The **code** is MIT-licensed (LICENSE).
* The **open dataset** combines MIT-licensed Kaggle data with prices derived from Yahoo Finance. It is shipped normalised (no price levels, no names in the files) for research reproducibility. If you redistribute it, check Yahoo's current terms.
* The **development dataset** is proprietary and is not distributed. Nothing in this repository allows it to be reconstructed: the reference weights in `tests/fixtures/` are strategy outputs, not data.
