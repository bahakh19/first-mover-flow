# Results

Every number below is produced by the research notebook (`notebooks/Research_notebook_FMF.ipynb`) and reproduced by the package (`scripts/`). Returns are daily and net of 5 bp per unit traded unless stated as gross. Alphas are annualised from weekly regressions with HAC(4) errors. All model choices were made on the development window (2017–2019).

## 1. Development and holdout: 10 US large caps, 2017–2021

| Book | Window | CAGR | Vol | Sharpe | Max DD | Calmar | Beta | Alpha (t) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FMF-N (net 5 bp) | Development 2017–19 | 7.7% | 7.9% | 0.98 | −9.7% | 0.79 | −0.01 | 8.2% (1.82) |
| FMF-N at 10 bp | Development 2017–19 | 7.5% | 7.9% | 0.95 | −9.7% | 0.77 | −0.01 | 8.0% (1.77) |
| Gen-1 RTB, full strategy with regime tilt (gross) | Development 2017–19 | 16.6% | 10.2% | 1.56 | −9.4% | 1.77 | 0.20 | 10.2% (2.46) |
| Gen-1 RTB, selection book only (gross) | Development 2017–19 | 5.0% | 8.4% | 0.63 | −13.9% | 0.36 | −0.07 | 7.5% (1.78) |
| 12-1 momentum, same construction | Development 2017–19 | 4.8% | 8.0% | 0.63 | −11.5% | 0.42 | 0.07 | 2.9% (0.51) |
| Equal-weight market | Development 2017–19 | 32.7% | 18.9% | 1.59 | −25.1% | 1.31 | 1.00 | – |
| FMF-N (net 5 bp) | Holdout 2020–21 | 16.0% | 11.7% | 1.33 | −14.1% | 1.14 | −0.14 | 21.1% (2.19) |
| FMF-N at 10 bp | Holdout 2020–21 | 15.7% | 11.7% | 1.31 | −14.1% | 1.11 | −0.14 | 20.8% (2.16) |
| Gen-1 RTB, full strategy with regime tilt (gross) | Holdout 2020–21 | 22.5% | 14.7% | 1.45 | −13.6% | 1.65 | 0.03 | 20.9% (1.99) |
| Gen-1 RTB, selection book only (gross) | Holdout 2020–21 | 13.2% | 13.3% | 1.00 | −15.5% | 0.85 | −0.11 | 17.5% (2.07) |
| 12-1 momentum, same construction | Holdout 2020–21 | 5.2% | 14.2% | 0.43 | −21.8% | 0.24 | −0.17 | 12.4% (1.15) |
| Equal-weight market | Holdout 2020–21 | 34.1% | 32.1% | 1.08 | −38.1% | 0.89 | 1.00 | – |
| FMF-N (net 5 bp) | Full 2017–21 | 11.0% | 9.6% | 1.13 | −14.1% | 0.78 | −0.02 | 11.9% (2.32) |
| FMF-N at 10 bp | Full 2017–21 | 10.7% | 9.6% | 1.11 | −14.1% | 0.76 | −0.02 | 11.7% (2.27) |
| Gen-1 RTB, full strategy with regime tilt (gross) | Full 2017–21 | 19.0% | 12.2% | 1.48 | −13.6% | 1.39 | 0.11 | 15.0% (2.48) |
| Gen-1 RTB, selection book only (gross) | Full 2017–21 | 8.3% | 10.6% | 0.80 | −15.5% | 0.53 | −0.07 | 10.9% (2.35) |
| 12-1 momentum, same construction | Full 2017–21 | 5.0% | 10.9% | 0.50 | −21.8% | 0.23 | −0.02 | 6.4% (1.00) |
| Equal-weight market | Full 2017–21 | 33.3% | 25.0% | 1.28 | −38.1% | 0.87 | 1.00 | – |

*Gen-1 RTB is the first-generation strategy. Its Calmar came from timed net exposure: its selection book alone (row 4) has a development Sharpe of 0.63. FMF-N is the market-neutral replacement. FMF-B20 adds a constant 0.2 beta budget: Calmar 1.82 (development), 2.17 (holdout), 1.66 (full), with a maximum drawdown of −10.8%.*

**Signal evidence** (development, rank IC at 21 sessions, Fama–MacBeth controls: 12-1 and 6-1 momentum, 1-month reversal, 60-day low volatility):

| Signal | IC5 | IC21 | IC63 | Hit rate | Newey–West *t* | Phase *t* | *p* (one-sided) | Fama–MacBeth *t* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| breadth_63 | 0.036 | 0.072 | 0.064 | 61% | 1.50 | 1.16 | 0.127 | 0.21 |
| event_breadth_63 | 0.033 | 0.085 | 0.082 | 58% | 1.73 | 1.26 | 0.108 | 0.15 |
| breadth_resid_mom | 0.045 | 0.083 | 0.109 | 56% | 2.03 | 1.45 | 0.078 | 0.36 |
| **leader_minus_herd** | 0.049 | **0.095** | 0.104 | 60% | 1.81 | 1.47 | 0.075 | **2.46** |
| composite | 0.053 | 0.109 | 0.114 | 62% | 2.19 | 1.74 | 0.045 | 0.60 |
| leaders only | 0.039 | 0.066 | 0.051 | 57% | 1.28 | 1.03 | 0.155 | 0.85 |
| herders only | −0.123 | −0.084 | −0.148 | 39% | −0.78 | – | – | −0.10 |

*Newey–West t with 21 lags over-rejects with overlapping 21-session labels (about 9% at a nominal 5%), the phase-averaged test (`fmflow.analytics.inference.ic_phase_test`) has the nominal size and gives the p-values. Phase t and p are what `scripts/reproduce_development.py` prints (`signal_ic21_dev.csv`).*

**Placebos** (Sharpe, the identical construction is applied to each counterfeit score):

| Window | IID scores | Stock-label permutation | Circular time-shift |
|---|---:|---:|---:|
| Development 2017–19 | p = 0.033 | p = 0.022 | p = 0.146 |
| Holdout 2020–21 | p = 0.010 | p = 0.016 | p = 0.025 |
| Full 2017–21 | p = 0.003 | p = 0.002 | p = 0.076 |

**Price of the search.**

* The 30-book construction grid has a probability of backtest overfitting of 0.945 for its *ranking*. The in-sample winner's median out-of-sample Sharpe is 0.66, so the *level* persists.
* The deflated Sharpe ratio is 0.94 against the grid, 0.38 against the 71-configuration research log, and 0.18 against all selection.
* The development Sharpe alone is therefore not conclusive. Sections 3–5 carry the evidence.

## 2. Leave-one-out: the full pipeline rebuilt without each stock

| Universe | Sharpe DEV | Sharpe holdout | Sharpe full | Calmar full | Alpha t full |
|---|---:|---:|---:|---:|---:|
| FMF-N, all 10 stocks | 0.98 | 1.33 | 1.13 | 0.78 | 2.32 |
| without stock_1 | 1.14 | 1.14 | 1.12 | 0.75 | 2.39 |
| without stock_2 | 0.58 | 1.10 | 0.81 | 0.62 | 1.37 |
| without stock_3 | 0.78 | 0.91 | 0.83 | 0.49 | 2.03 |
| without stock_4 | 0.90 | 0.72 | 0.80 | 0.46 | 1.72 |
| without stock_5 | 1.03 | 1.53 | 1.25 | 1.05 | 2.73 |
| without stock_6 | 0.87 | 1.34 | 1.08 | 0.77 | 1.98 |
| without stock_7 | 1.03 | 1.32 | 1.15 | 0.80 | 2.42 |
| without stock_8 | 0.80 | 1.27 | 1.02 | 0.73 | 1.99 |
| without stock_9 | 0.86 | 1.47 | 1.14 | 0.84 | 2.33 |
| without stock_10 | 0.98 | 1.15 | 1.04 | 0.65 | 2.13 |

## 3. Against the institutional toolkit (in sample)

| Signal (same construction, net 5 bp) | Sharpe DEV | Sharpe holdout | Sharpe full | Calmar full | Max DD full | Corr. with FMF-N |
|---|---:|---:|---:|---:|---:|---:|
| **FMF-N (this paper)** | 0.98 | 1.33 | 1.13 | 0.78 | −14.1% | 1.00 |
| Revision breadth (classic) | 0.57 | 1.28 | 0.92 | 0.58 | −15.5% | 0.95 |
| Momentum + revisions blend | 0.78 | 1.01 | 0.87 | 0.49 | −18.5% | 0.90 |
| Multi-factor: mom + low vol + reversal | 0.88 | 0.80 | 0.81 | 0.42 | −14.3% | 0.44 |
| Low volatility (60d) | 0.37 | 0.80 | 0.56 | 0.37 | −13.6% | 0.04 |
| 12-1 momentum (Jegadeesh-Titman) | 0.63 | 0.43 | 0.50 | 0.23 | −21.8% | 0.71 |
| Target-price upside (consensus) | 0.47 | 0.29 | 0.38 | 0.24 | −12.1% | −0.38 |
| 52-week-high proximity | −0.36 | 1.06 | 0.37 | 0.17 | −18.2% | 0.73 |
| Target-price momentum (63d) | −0.36 | 0.68 | 0.19 | 0.07 | −20.4% | 0.71 |
| Residual momentum (Blitz et al.) | 0.42 | −0.00 | 0.17 | 0.05 | −24.1% | 0.64 |
| 6-1 momentum | −0.51 | 0.55 | 0.06 | 0.00 | −24.8% | 0.67 |
| Betting-against-beta (low beta) | −0.28 | −0.00 | −0.15 | −0.07 | −32.2% | 0.10 |
| Short-term reversal (1m) | 0.25 | −0.69 | −0.24 | −0.09 | −18.7% | −0.51 |
| Low analyst dispersion | −0.36 | −0.74 | −0.53 | −0.20 | −27.7% | −0.15 |

* A 50/50 blend with FMF-N raises the Sharpe of 16 of 18 books and the Calmar of 16 of 18. The two exceptions are the books that already carry market beta (RTB and FMF-B20).
* **Spanning:** FMF-N's alpha after the market and the 11 comparables that are not built from its own components is 6.1% a year (*t* = 2.51).
* **Crisis episodes:**
  * COVID crash (Feb–Mar 2020): FMF-N **+12.0%**, 12-1 momentum +12.2%, market −35.5%, betting-against-beta −15.2%, trend following −18.9%.
  * Vaccine rotation (Nov 2020): FMF-N −5.8%, 12-1 momentum −8.2%.
  * Growth sell-off (Feb–Mar 2021): FMF-N −7.1%, 12-1 momentum −10.5%, trend following −12.1%.

## 4. Out of sample: 30 stocks over a never-seen period, 2024-01-05 to 2026-08-05 (pre-registered)

The period starts two years after the development data end. Twenty of the thirty names were never used in development, ten are the development names (DATA_SOURCES.md §3).

Registration `configs/oos_open_universe_preregistration.json`, fingerprint `0b2b0792fd357d61`, written before any return was computed.

| Book | Sharpe | CAGR | Vol | Max DD | Calmar | Beta | Alpha (t) | Sharpe H1 / H2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **FMF-N primary** (leg fraction 0.4) | **1.05** | 10.5% | 10.0% | −8.0% | **1.32** | −0.02 | 11.4% (2.04) | 1.44 / 0.88 |
| FMF-N secondary (*k* = 4) | 0.70 | 7.3% | 10.9% | −10.2% | 0.72 | −0.04 | 8.8% (1.48) | 0.94 / 0.64 |
| FMF-B20 (*exploratory, not pre-registered*) | 1.49 | 15.9% | 10.3% | −6.5% | 2.43 | 0.16 | 11.8% (2.09) | 1.73 / 1.44 |
| Equal-weight market | 1.14 | 23.6% | 20.4% | −25.3% | 0.94 | 1.00 | – | 0.28 / 2.35 |

| Signal (IC21, labels inside the window) | IC | Phase *t* | *p* (one-sided) | Coverage |
|---|---:|---:|---:|---:|
| breadth_63 | 0.051 | 0.98 | 0.168 | 47% |
| event_breadth_63 | −0.041 | −0.49 | 0.685 | 26% |
| breadth_resid_mom | 0.020 | 0.39 | 0.351 | 46% |
| **leader_minus_herd** | **0.096** | 1.80 | **0.042** | 53% |
| composite | 0.073 | 1.56 | 0.065 | 59% |

**Placebos:** IID p = 0.027, stock-label permutation p = 0.056, circular time-shift p = 0.226. The permutation value is the research notebook's, which draws unrestricted permutations here, so a few stocks keep their own signal in each draw. The package (`fmflow evaluate --placebo`, `scripts/reproduce_open_universe.py --placebo`) draws derangements, in which no stock keeps its own signal, as the development placebos above do. It gives p = 0.046, the other two p-values are identical.

**After the market and the four price-control books,** alpha is 7.3% a year (*t* = 1.79, R² = 0.47).

**Comparables, same construction:** FMF-N ranks 2nd of 14 cross-sectional books on both Sharpe and Calmar, target-price momentum is 1st (1.14 / 1.44). The market and CTA-style trend have higher Sharpe ratios in this bull window, but with betas of 1.00 and 0.43. A 50/50 blend with FMF-N raises the Sharpe and Calmar of all 16 comparables.

**Post hoc: is the result carried by the ten development names?** (`scripts/overlap_analysis.py`, same frozen specification and GARCH forecasts)

| Universe | Net Sharpe | CAGR | Max DD | Calmar | LMH IC21 (*p*) |
|---|---:|---:|---:|---:|---:|
| all 30 names (pre-registered) | 1.05 | 10.5% | −8.0% | 1.32 | 0.096 (0.042) |
| 20 names never used in development | 0.80 | 9.0% | −9.2% | 0.98 | 0.105 (0.058) |
| 10 development names only | 0.31 | 2.9% | −11.0% | 0.26 | 0.022 (0.45) |

**GARCH estimator:** With the deterministic grid estimator (`DataConfig(garch_method="grid")`, bit-identical on every machine) the out-of-sample Sharpe ratio is 1.09 (Calmar 1.39), and 1.12 over 2017–21 (INSTALL.md).

## 5. What did not work

* **Time-shift placebo:** It is not passed in sample (p = 0.076) or out of sample (p = 0.226). Part of the edge is *which* stocks the revision flow favours, not only *when*.
* **Event breadth does not transfer:** Clusters of five same-day ratings are rare with 13 analysts per stock.
* **The literal *k* = 4 rule is weaker in a 30-stock universe.** Breadth should scale with the universe, as pre-registered.
* **The research log:** More than fifty other hypotheses were pre-registered and rejected: fresh-vs-stale wedge, conviction shock, PEAD drift, bold revisions, peer spillovers, analyst-skill weighting and others (paper, Table 2).

## 6. Audit: the result under attack (paper Sections 10–11, `scripts/reproduce_audit.py`)

| Test | Development 2017–21 | Never-seen 2024–26 |
|---|---:|---:|
| Causal timing book: Sharpe (*t*) | 1.04 (2.41) | 1.33 (2.21) |
| Causal static tilt: Sharpe | 0.48 | −0.49 |
| One-day delay, 100 bp borrow, 10 bp costs: Sharpe | 1.06 | 0.99 |
| Harshest cell (two-day delay, 200 bp borrow, 20 bp): Sharpe | 0.96 | 0.89 |
| Breakeven cost per unit traded | 234 bp | 197 bp |
| Lo (2002) autocorrelation-adjusted Sharpe | 1.06 | 1.06 |
| Multiverse (~300 specifications): share with positive Sharpe | 99% | 97% |
| Multiverse median Sharpe / frozen specification's percentile | 0.80 / 98th | 0.72 / 83rd |
| Selection-free ensemble of all specifications: Sharpe | 0.92 | 0.86 |
| Hansen SPA over the specification family: *p* | 0.34 | 0.14 |
| Target expiry 63 / 126 sessions (frozen: 252): Sharpe | 0.94 / 1.12 | 0.88 / 1.15 |
| Reverse SPA (does any of 16 comparables beat FMF-N?): *p* | 0.86 | 0.86 |
| Marginal value of leader-minus-herd: ΔSharpe (*p*) | +0.29 (0.028) | +0.30 (0.25) |

**Pooled across the three non-overlapping periods** (development, holdout, never-seen period): random-effects LMH IC 0.105 (90% HDI 0.033 to 0.179), P(IC > 0 in the next universe) 0.96, Stouffer *p* 0.002, pooled Sharpe 1.10 with P(SR > 0 next universe) 0.96. James–Stein shrinkage over the 18 development candidates reduces LMH's *t* from 2.46 to 1.06: the development search alone is suggestive, the replication is the evidence.

**Generative path risk** (pooled samples, 1,000 five-year paths, posterior-predictive drift): Diffusion-FHS median Calmar 0.70, 5% maximum drawdown −34%, P(five-year loss) 14%, block bootstrap 0.91 / −26%, FHS 0.84 / −29%. TimeGAN fails the fidelity battery and is excluded. The binding stress is a momentum crash: the 5% quantile of the three-month return falls to between −14% and −17%, against about −8% unconditionally.

Objection-by-objection answers, with residual risk: [OBJECTIONS.md](OBJECTIONS.md).
