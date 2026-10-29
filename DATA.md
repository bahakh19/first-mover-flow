# Data outline

*Where the two datasets come from (proprietary development data; Kaggle and Yahoo Finance for the open data), their coverage, the overlap between them and their licences: see [DATA_SOURCES.md](DATA_SOURCES.md).*

`fmflow` needs **two long-format tables**. Nothing else is required: no sector map, no fundamentals, no index weights. This file defines the contract, the cleaning rules applied on ingestion, the point-in-time conventions, how to map common vendor feeds, and how the synthetic generator was calibrated.

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

Column names are case-insensitive and surrounding whitespace is ignored. Values may arrive as text: unparseable values are logged and dropped, never coerced silently. CSV (any delimiter, decimal comma, UTF-8 BOM, gzip), Parquet and Excel are accepted (`fmflow.load_prices`, `fmflow.load_ratings`, or a file path passed straight to `run_strategy`).

### Structural checks (`fmflow validate`, `fmflow.data.schema`)

`DataContractError` is raised, with an actionable message, if any of these fails:

* a required column is missing or a table is empty
* more than 1% of price dates or 5% of rating dates are unparseable
* more than 1% of prices are non-positive
* fewer than 6 stocks are present (a long/short cross-section needs at least 2 × `k_min` + 2 names)
* no rated stock appears in the price table
* **scale check:** the median ratio of target to the price at publication lies outside (0.3, 3.5). This catches the most common ingestion error, raw-dollar targets set against normalised or adjusted prices

### Minimum history and coverage

| Requirement | Why |
|---|---|
| ≥ 252 sessions before the first evaluated day | target expiry (252), LMH window (126), 12-1 momentum, GARCH needs 250 observations |
| ≥ 6 stocks with a valid score on a session | ranks need ≥ 5 names; legs need 2 × k names |
| ≥ 3 analysts revising a stock within 63 sessions | breadth is undefined below `breadth_min_n = 3` |
| some multi-analyst days (≥ 5 same-day ratings) | information events define leaders and herder windows; without them LMH falls back to the quiet-period leader rule |

Coverage observed in the two real datasets is given in §6. Coverage below these levels does not break the pipeline: the affected signals are `NaN` and drop out of the composite on those days.

## 2. Ingestion and cleaning (`fmflow.data.cleaning`)

Every rule was fixed before any return was examined. Each one writes a row to the audit log (`out.audit`, or `cleaning_log.csv` from `fmflow run`). All parameters live in `DataConfig`.

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
| – | stock absent from the price file: drop; duplicate (date, stock, analyst) keeps the last | – |
| R6 | **knowledge date** = first session strictly after publication. Ratings published after the last session are dropped | `knowledge_lag = 1` |
| – | reference price = last close on or before publication (`merge_asof`, backward) | – |
| R3 | implausible level: target / price outside (0.2, 5.0) is dropped (fat-finger values, unadjusted splits) | `tp_ratio_bounds` |
| R4 | implausible revision: (target / previous target) ÷ (price / previous reference price) outside (1/3, 3) is dropped (a split applied to one series only) | `revision_excess_bounds` |

**Derived fields** (one row per accepted target): `knowledge_date, pub_date, stock, analyst, tp, ref_price, tp_ratio, prev_tp, revision = log(tp / prev_tp), is_initiation, is_reiteration, n_same_day, event_day (n_same_day ≥ 5)`.

## 3. Point-in-time conventions (`fmflow.data.snapshot`)

* A target is **usable from its knowledge date**, never on its publication date. The cleaning step checks `knowledge_date > pub_date` for every row and raises an error otherwise.
* An analyst's active target on a stock is their latest target with knowledge date ≤ t. It **expires** after 252 sessions without an update (`tp_expiry_days`).
* Each rating is classified by its own history: **initiation** (first target, or first after expiry), **reiteration** (unchanged), **fresh up** or **fresh down** revision. Only fresh revisions enter breadth and LMH.
* Signals at t use prices up to the close of t and ratings with knowledge date ≤ t. The weights built at the close of t earn the return from t to t+1: `net(t+1) = Σ w(t)·r(t+1) − cost·Σ|w(t) − w(t−1)|`.
* `tests/test_causality.py` recomputes signals and weights on data truncated at T and requires exact equality with the full-sample run up to T. A centred-window negative control must fail the same test.

## 4. Mapping vendor data

| Source | `prices.price` | `ratings.date` | `ratings.estimate_value` | `ratings.analyst` |
|---|---|---|---|---|
| I/B/E/S detail (WRDS `ibes.ptgdet`) | CRSP `prc` adjusted by `cfacpr`, or a total-return index | `anndats` (announcement date) | `value` adjusted to the same basis as the price (I/B/E/S adjustment factors) | `amaskcd` (or `estimid` + `amaskcd`) |
| FactSet Estimates (broker detail) | FactSet adjusted price | estimate date | target price, split-adjusted | broker + analyst id |
| Bloomberg (`BEst` target price, analyst level via `ANR`) | `PX_LAST` adjusted (`CshAdjNormal`, `CshAdjAbnormal`, `CapChg`) | recommendation date | target price | firm + analyst name |
| Refinitiv / LSEG StarMine detail | adjusted close | estimate date | target price | analyst code |

Practical notes:

* **Adjust both series identically:** If targets are raw dollars and prices are adjusted (or the reverse), splits create false revisions. R3 and R4 catch most of these, but fix the source rather than relying on the filters
* **Use publication dates, not "as-of" snapshot dates:** Consensus snapshots that repeat stale targets every month inflate reiterations and destroy the leader/herder timing
* **Normalising is allowed:** Dividing each stock's prices *and* targets by the stock's first price, as in `data/open`, keeps every signal unchanged and anonymises the levels
* **Delisted names:** can stay in the tables. Their prices become NaN after delisting and they drop out of the cross-section

## 5. Shipped datasets

| | `data/open/` (open data, never-seen period) | development (proprietary, not distributed) |
|---|---|---|
| Stocks | 30 (anonymised `stock_1`…`stock_30`) | 10 US large caps |
| Sessions | 899, 2023-01-04 to 2026-08-05 | 1,511, 2016-01-04 to 2021-12-31 |
| Ratings (raw / accepted) | 2,373 / 2,363 | 5,987 / 5,929 |
| Analysts | 51 | 304 |
| Cleaning | 8 published after the last session (R6), 2 implausible target/price (R3) | see the paper, Appendix A |
| Role | pre-registered out-of-sample test (`configs/oos_open_universe_preregistration.json`) | all model choices (2017–2019), holdout 2020–2021 |

To reproduce the development results, place the original `MarketData.csv` and `Ratings.csv` (same contract) in `data/private/`. That folder is git-ignored, and the tests and scripts that need it skip cleanly when it is empty.

## 6. Synthetic data that matches the real data (`fmflow.synthetic`)

The generator is a **structural** model rather than a resampler. That lets you switch the economics on and off and check that the pipeline finds alpha only when it exists.

* **Returns:** Market factor with GJR-GARCH(1,1)-t volatility, plus sector factors, persistent stock drifts, a latent fundamental state *z* (AR(1), quarterly half-life, shocked at earnings), and earnings-day jumps
* **Analysts:** Each stock is covered by analysts drawn from a shared pool (multi-stock coverage). They publish around earnings, which creates clusters of five or more same-day ratings, and occasionally in between. Targets chase price with partial adjustment and an analyst-specific optimism. **First movers** see *z* early; **herders** copy the latest cluster's direction 3–15 sessions later, cuts are sticky
* **Presets:**
  * `development_like` reproduces the development data
  * `open_universe_like` reproduces the open universe
  * `null` switches off both predictability channels
  * `inject_errors=True` adds fat-finger targets, missing ids, text values and duplicates, to exercise the cleaning

Moments are computed with the same function on real and synthetic data (`fmflow describe`):

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

**Does the pipeline find alpha only when it exists?** Run `python scripts/synthetic_power_study.py` (and see the slow test in `tests/test_synthetic.py`). Over 40 seeds, the mean net Sharpe is 0.05 (*t* = 0.8) in the `null` world and 0.79 (*t* = 8.7) when first movers are informed. A strategy that showed a Sharpe in the null world would be fitting noise or leaking the future.

```bash
fmflow synth --preset open --seed 3 --out data/synthetic/open3/
fmflow describe --prices data/synthetic/open3/prices.csv --ratings data/synthetic/open3/ratings.csv
```
