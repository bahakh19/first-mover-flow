# Data outline

*Where the two datasets come from, how well they are covered, how they overlap and how they are licensed is all in [DATA_SOURCES.md](DATA_SOURCES.md). In short, the development data are proprietary, and the open data come from Kaggle and Yahoo Finance.*

All `fmflow` needs is **two long-format tables**. It does not ask for a sector map, fundamentals or index weights. This page gives the input contract and the cleaning rules applied on loading. It then covers the point-in-time conventions, the mapping of common vendor feeds, and how we calibrated the synthetic generator.

## 1. Input contract

### `prices`: one row per (session, stock)

| Column | Type | Required | Description |
|---|---|---|---|
| `date` | date, ISO `yyyy-mm-dd` preferred | yes | trading session (close). A time of day or a UTC offset is accepted and ignored: only the local calendar day is kept |
| `stock` | string | yes | stock identifier, stable over time (ticker, PERMNO, ISIN or an anonymised id) |
| `price` | float > 0 | yes | **total-return or split-adjusted close**, on any scale that is constant per stock (e.g. normalised to 1 on the first day) |

```csv
date,stock,price
2023-01-04,stock_1,1.0
2023-01-04,stock_2,1.0
2023-01-05,stock_1,0.9893
```

### `ratings`: one row per published target price

| Column | Type | Required | Description |
|---|---|---|---|
| `date` | date, ISO preferred | yes | **publication** date of the target (not the effective or entry date). A time of day is ignored: a target published at 08:30 on day D is still usable only from the session after D |
| `stock` | string | yes | must match a `stock` in `prices` |
| `analyst` | string | yes | stable analyst (or broker-analyst) identifier |
| `estimate_value` | float > 0 | yes | 12-month target price, **on the same scale and adjustment basis as `price`** |

```csv
date,estimate_value,stock,analyst
2023-01-04,1.6619,stock_1,analyst_46
2023-01-23,1.1654,stock_20,analyst_2
```

Column names can be in any case, and whitespace around them doesn't matter. Values may come in as text. Anything that can't be parsed is logged and dropped rather than quietly coerced. Accepted formats are CSV with any delimiter, decimal commas, a UTF-8 BOM or gzip compression, as well as Parquet and Excel. Load them with `fmflow.load_prices` and `fmflow.load_ratings`, or just pass the file path to `run_strategy`.

### Structural checks (`fmflow validate`, `fmflow.data.schema`)

If any of the following checks fails, the package raises `DataContractError` with a message saying what to do about it.

* a required column is missing or a table is empty
* more than 1% of price dates or 5% of rating dates are unparseable
* more than 1% of prices are non-positive
* fewer than 6 stocks are present (a long/short cross-section needs at least 2 × `k_min` + 2 names)
* no rated stock appears in the price table
* **scale check:** the median ratio of target to the price at publication lies outside (0.3, 3.5). The check exists because the commonest loading mistake is to set raw-dollar targets against normalised or adjusted prices

### Minimum history and coverage

| Requirement | Why |
|---|---|
| ≥ 252 sessions before the first evaluated day | target expiry (252), LMH window (126), 12-1 momentum, GARCH needs 250 observations |
| ≥ 6 stocks with a valid score on a session | ranks need ≥ 5 names, legs need 2 × k names |
| ≥ 3 analysts revising a stock within 63 sessions | breadth is undefined below `breadth_min_n = 3` |
| some multi-analyst days (≥ 5 same-day ratings) | information events define leaders and herder windows, without them LMH falls back to the quiet-period leader rule |

§6 reports the coverage of the two datasets used here. If your coverage falls below these levels the pipeline still runs, and the signals affected are `NaN` and simply drop out of the composite on those days.

## 2. Ingestion and cleaning (`fmflow.data.cleaning`)

We fixed every one of these rules before looking at a single return. Each rule writes a row to the audit log, which is `out.audit` in Python and `cleaning_log.csv` when you use `fmflow run`, and their parameters all live in `DataConfig`.

### Prices

| Step | Rule | Parameter |
|---|---|---|
| parse | unparseable dates and missing stock identifiers dropped, non-positive prices set to NaN, duplicate (date, stock) keeps the last row | – |
| calendar | a date is a session if ≥ 50% of the stocks *active* at that date print on it, where active means printed in the previous 30 calendar days (in the file's first 30 days: anywhere in that first window). The exchange calendar is inferred from the data, so no holiday file is needed, and stocks that list later never change past sessions | `min_calendar_coverage = 0.5`, `calendar_active_days = 30` |
| stale rows | dates on which **every** price is unchanged are dropped (vendor fill-forward days) | – |
| gaps | isolated missing prices are forward-filled up to 5 sessions, **never** backward-filled | `price_ffill_limit = 5` |

### Ratings

| Check | Rule | Parameter |
|---|---|---|
| R1 | unparseable or non-positive target, or unparseable date: drop | – |
| R2 | missing analyst identifier: drop | – |
| – | stock absent from the price file: drop, duplicate (date, stock, analyst) keeps the last | – |
| R6 | **knowledge date** = first session strictly after publication. Ratings published after the last session are dropped | `knowledge_lag = 1` |
| – | reference price = last close on or before publication (`merge_asof`, backward) | – |
| R3 | implausible level: target / price outside (0.2, 5.0) is dropped (fat-finger values, unadjusted splits) | `tp_ratio_bounds` |
| R4 | implausible revision: (target / previous target) ÷ (price / previous reference price) outside (1/3, 3) is dropped (a split applied to one series only) | `revision_excess_bounds` |

**Derived fields**, one row for each accepted target: `knowledge_date, pub_date, stock, analyst, tp, ref_price, tp_ratio, prev_tp, revision = log(tp / prev_tp), is_initiation, is_reiteration, n_same_day, event_day (n_same_day ≥ 5)`.

## 3. Point-in-time conventions (`fmflow.data.snapshot`)

A target is usable from its **knowledge date**, never from its publication date; cleaning checks `knowledge_date > pub_date` on every row and stops with an error otherwise. An analyst's active target on a stock is their latest target with knowledge date ≤ t, and it **expires** after 252 sessions without an update (`tp_expiry_days`). Each rating is classified against the analyst's own history as an **initiation** (first target, or first after an expiry), a **reiteration** (unchanged) or a **fresh up** / **fresh down** revision, and only fresh revisions enter breadth and LMH.

Signals at t use prices up to the close of t and ratings with knowledge date ≤ t. Weights built at the close of t earn the return from t to t+1: `net(t+1) = Σ w(t)·r(t+1) − cost·Σ|w(t) − w(t−1)|`. `tests/test_causality.py` recomputes signals and weights on data truncated at T and requires exact equality with the full-sample run up to T, and a centred-window negative control must fail the same test.

## 4. Mapping vendor data

| Source | `prices.price` | `ratings.date` | `ratings.estimate_value` | `ratings.analyst` |
|---|---|---|---|---|
| I/B/E/S detail (WRDS `ibes.ptgdet`) | CRSP `prc` adjusted by `cfacpr`, or a total-return index | `anndats` (announcement date) | `value` adjusted to the same basis as the price (I/B/E/S adjustment factors) | `amaskcd` (or `estimid` + `amaskcd`) |
| FactSet Estimates (broker detail) | FactSet adjusted price | estimate date | target price, split-adjusted | broker + analyst id |
| Bloomberg (`BEst` target price, analyst level via `ANR`) | `PX_LAST` adjusted (`CshAdjNormal`, `CshAdjAbnormal`, `CapChg`) | recommendation date | target price | firm + analyst name |
| Refinitiv / LSEG StarMine detail | adjusted close | estimate date | target price | analyst code |

A few practical points.

* **Put both series on the same adjustment basis.** When targets are in raw dollars and prices are adjusted (or vice versa), every split shows up as a false revision. R3 and R4 will catch most of them, but it is better to fix the source than to lean on the filters
* **Use publication dates rather than "as-of" snapshot dates.** A consensus snapshot that repeats stale targets month after month inflates the reiteration count and wipes out the leader/herder timing
* **Normalising is fine.** If you divide each stock's prices *and* targets by its first price, as `data/open` does, no signal changes and the levels are anonymised
* **Delisted names** may stay in the tables. After delisting their prices become NaN and they drop out of the cross-section

## 5. Shipped datasets

| | `data/open/` (open data, never-seen period) | development (proprietary, not distributed) |
|---|---|---|
| Stocks | 30 (anonymised `stock_1`…`stock_30`) | 10 US large caps |
| Sessions | 899, 2023-01-04 to 2026-08-05 | 1,511, 2016-01-04 to 2021-12-31 |
| Ratings (raw / accepted) | 2,373 / 2,363 | 5,987 / 5,929 |
| Analysts | 51 | 304 |
| Cleaning | 8 published after the last session (R6), 2 implausible target/price (R3) | see the paper, Appendix A |
| Role | pre-registered out-of-sample test (`configs/oos_open_universe_preregistration.json`) | all model choices (2017–2019), holdout 2020–2021 |

Reproducing the development results requires the original `MarketData.csv` and `Ratings.csv` (same contract) in `data/private/`. Git ignores that folder, and when it's empty the tests and scripts that depend on it are skipped without failing.

## 6. Synthetic data that matches the real data (`fmflow.synthetic`)

Our generator does not resample the data; it is a **structural** model. Thus you can turn the economic mechanisms on and off and check that the pipeline finds alpha when, and only when, it is there.

* **Returns** come from a market factor with GJR-GARCH(1,1)-t volatility, sector factors, persistent stock-level drifts and jumps on earnings days. There is also a latent fundamental state *z*: an AR(1) process with a quarterly half-life, shocked at each earnings date
* **Analysts** are drawn from one shared pool, so a single analyst may cover several stocks. Most of their publishing happens around earnings, which produces clusters of five or more ratings on the same day, with the odd rating in between. Targets chase the price with partial adjustment plus an optimism that differs from one analyst to the next. **First movers** get to see *z* early, **herders** copy the direction of the latest cluster 3–15 sessions afterwards, and cuts tend to stick
* **Presets.**
  * `development_like` mimics the development data
  * `open_universe_like` mimics the open universe
  * `null` turns both channels of predictability off
  * `inject_errors=True` throws in fat-finger targets, missing ids, text values and duplicates so the cleaning gets a workout

Real and synthetic moments are computed by one and the same function (`fmflow describe`).

| Moment | Development data | Synthetic `development_like` | Open data | Synthetic `open_universe_like` |
|---|---:|---:|---:|---:|
| stocks | 10 | 10 | 30 | 30 |
| sessions | 1,511 | 1,512 | 899 | 899 |
| avg pairwise return correlation | 0.40 | 0.39 | 0.26 | 0.25 |
| annual volatility, min – max | 0.27 – 0.53 | 0.29 – 0.54 | 0.20 – 0.94 | 0.24 – 0.78 |
| accepted ratings | 5,929 | 6,364 | 2,363 | 2,446 |
| analysts | 304 | 232 | 50 | 49 |
| analysts per stock (median) | 47 | 42 | 13 | 12 |
| stocks per analyst (median) | 1 | 1 | 6 | 9 |
| median days between an analyst's updates | 65 | 86 | 77 | 98 |
| share up / down / reiteration / initiation | 0.53 / 0.17 / 0.21 / 0.08 | 0.44 / 0.24 / 0.23 / 0.09 | 0.35 / 0.14 / 0.31 / 0.20 | 0.32 / 0.13 / 0.33 / 0.22 |
| median target / price | 1.17 | 1.18 | 1.19 | 1.23 |
| event clusters (≥ 5 same-day ratings) | 214 | 219 | 109 | 130 |

**Does the pipeline only find alpha when there is some?** Run `python scripts/synthetic_power_study.py` to see (the slow test in `tests/test_synthetic.py` covers it too). Across 40 seeds the mean net Sharpe is 0.05 (*t* = 0.8) in the `null` world, and 0.79 (*t* = 8.7) once first movers are informed. Had the strategy shown a Sharpe well above zero in the null world, it would have been fitting noise or seeing the future.

```bash
fmflow synth --preset open --seed 3 --out data/synthetic/open3/
fmflow describe --prices data/synthetic/open3/prices.csv --ratings data/synthetic/open3/ratings.csv
```
