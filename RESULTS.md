# Results

All the numbers on this page come out of the research notebook, `notebooks/Research_notebook_FMF.ipynb`, and the scripts in `scripts/` reproduce them with the package. Returns are daily and net of 5 bp per unit traded, except where a row says gross. Alphas are annualised from weekly regressions with HAC(4) errors. Every modelling choice was made on the development window (2017–2019) and nowhere else.

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

*Gen-1 RTB was our first-generation strategy, and most of its Calmar came from timing its net exposure. Strip that away and the selection book (row 4) has a development Sharpe of just 0.63. FMF-N replaces it with a market-neutral book. FMF-B20 is FMF-N plus a constant 0.2 beta budget. Its Calmar is 1.82 in development, 2.17 in the holdout and 1.66 over the full period, with a maximum drawdown of −10.8%.*

**Signal evidence.** Development window, rank IC at 21 sessions. The Fama–MacBeth regressions control for 12-1 and 6-1 momentum, 1-month reversal and 60-day low volatility.

| Signal | IC5 | IC21 | IC63 | Hit rate | Newey–West *t* | Phase *t* | *p* (one-sided) | Fama–MacBeth *t* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| breadth_63 | 0.036 | 0.072 | 0.064 | 61% | 1.50 | 1.16 | 0.127 | 0.21 |
| event_breadth_63 | 0.033 | 0.085 | 0.082 | 58% | 1.73 | 1.26 | 0.108 | 0.15 |
| breadth_resid_mom | 0.045 | 0.083 | 0.109 | 56% | 2.03 | 1.45 | 0.078 | 0.36 |
| **leader_minus_herd** | 0.049 | **0.095** | 0.104 | 60% | 1.81 | 1.47 | 0.075 | **2.46** |
| composite | 0.053 | 0.109 | 0.114 | 62% | 2.19 | 1.74 | 0.045 | 0.60 |
| leaders only | 0.039 | 0.066 | 0.051 | 57% | 1.28 | 1.03 | 0.155 | 0.85 |
| herders only | −0.123 | −0.084 | −0.148 | 39% | −0.78 | – | – | −0.10 |

*With overlapping 21-session labels, a Newey–West t with 21 lags rejects too often (about 9% of the time at a nominal 5%). The phase-averaged test (`fmflow.analytics.inference.ic_phase_test`) holds its nominal size, so the p-values come from it. Phase t and p are exactly what `scripts/reproduce_development.py` prints (`signal_ic21_dev.csv`).*

**Placebos.** Each counterfeit score goes through the identical construction, and the Sharpe ratios are compared.

| Window | IID scores | Stock-label permutation | Circular time-shift |
|---|---:|---:|---:|
| Development 2017–19 | p = 0.033 | p = 0.022 | p = 0.146 |
| Holdout 2020–21 | p = 0.010 | p = 0.016 | p = 0.025 |
| Full 2017–21 | p = 0.003 | p = 0.002 | p = 0.076 |

**What the search cost us.** Over the 30-book construction grid, the probability of backtest overfitting is 0.945 for the *ranking* of books. The *level*, on the other hand, survives: the in-sample winner still has a median out-of-sample Sharpe of 0.66. Deflated, the Sharpe ratio is 0.94 against the grid, 0.38 against the 71-configuration research log and 0.18 against all the selection we did. On its own, then, the development Sharpe proves little; Sections 3–5 are where the evidence is.

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

Mixing any of these books 50/50 with FMF-N lifts the Sharpe of 16 of the 18 and the Calmar of 16 of the 18. The two that don't improve are the books already carrying market beta (RTB and FMF-B20). In a spanning regression on the market and on the 11 comparables not built from FMF-N's own components, FMF-N keeps an alpha of 6.1% a year (*t* = 2.51).

Three crises:
  * COVID crash (Feb–Mar 2020): FMF-N **+12.0%**, 12-1 momentum +12.2%, market −35.5%, betting-against-beta −15.2%, trend following −18.9%.
  * Vaccine rotation (Nov 2020): FMF-N −5.8%, 12-1 momentum −8.2%.
  * Growth sell-off (Feb–Mar 2021): FMF-N −7.1%, 12-1 momentum −10.5%, trend following −12.1%.

## 4. Out of sample: 30 stocks over a never-seen period, 2024-01-05 to 2026-08-05 (pre-registered)

This window opens two years after the development data stop. Of its thirty names, twenty were never touched during development and ten are the development names themselves, as DATA_SOURCES.md §3 explains. The rules were registered in `configs/oos_open_universe_preregistration.json`, whose fingerprint is `0b2b0792fd357d61`, and that file was written before a single return had been computed.

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

**Placebos.** IID p = 0.027, stock-label permutation p = 0.056, circular time-shift p = 0.226. The permutation figure is the one from the research notebook, which here draws unrestricted permutations, so in each draw a few stocks keep their own signal. The package draws derangements instead, as the development placebos above do, so no stock ever keeps its own signal. Run through `fmflow evaluate --placebo` or `scripts/reproduce_open_universe.py --placebo`, it gets p = 0.046. The other two p-values don't change.

Controlling for the market and the four price-control books leaves an alpha of 7.3% a year (*t* = 1.79, R² = 0.47).

Against the comparables (same construction), FMF-N comes 2nd of 14 cross-sectional books on Sharpe and on Calmar, and target-price momentum is 1st (1.14 / 1.44). In this bull window the market and CTA-style trend both post higher Sharpe ratios, though with betas of 1.00 and 0.43. Blending FMF-N 50/50 into each of the 16 comparables improves every one of them on both Sharpe and Calmar.

**Post hoc check: do the ten development names carry the result?** Same frozen specification, same GARCH forecasts, computed by `scripts/overlap_analysis.py`.

| Universe | Net Sharpe | CAGR | Max DD | Calmar | LMH IC21 (*p*) |
|---|---:|---:|---:|---:|---:|
| all 30 names (pre-registered) | 1.05 | 10.5% | −8.0% | 1.32 | 0.096 (0.042) |
| 20 names never used in development | 0.80 | 9.0% | −9.2% | 0.98 | 0.105 (0.058) |
| 10 development names only | 0.31 | 2.9% | −11.0% | 0.26 | 0.022 (0.45) |

Switching GARCH to the deterministic grid estimator, `DataConfig(garch_method="grid")`, which is bit-identical on every machine, gives an out-of-sample Sharpe ratio of 1.09 (Calmar 1.39) and 1.12 over 2017–21. INSTALL.md explains why the two estimators differ.

## 5. What did not work

The time-shift placebo fails, both in sample (p = 0.076) and out of sample (p = 0.226). Part of the edge lies in *which* stocks the revision flow favours, and only part in *when* it favours them.

Event breadth did not carry over to the open data. With 13 analysts per stock, clusters of five ratings on the same day hardly ever happen.

In a 30-stock universe the literal *k* = 4 rule is weaker; breadth ought to grow with the universe, which is what we pre-registered.

Finally, Table 2 of the paper lists more than fifty further hypotheses that were pre-registered and then rejected, among them a fresh-vs-stale wedge, a conviction shock, PEAD drift, bold revisions, peer spillovers and analyst-skill weighting.

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

**Pooling the three non-overlapping periods** (development, holdout and the never-seen period), the random-effects LMH IC is 0.105, with a 90% HDI of 0.033 to 0.179. P(IC > 0 in the next universe) is 0.96 and Stouffer's *p* is 0.002. The pooled Sharpe is 1.10, with P(SR > 0 next universe) 0.96. James–Stein shrinkage across the 18 development candidates takes LMH's *t* from 2.46 down to 1.06. The development search, in other words, only hints at the effect. The replication is what establishes it.

**Generative path risk.** We pool the samples, simulate 1,000 five-year paths and draw the drift from the posterior predictive. Diffusion-FHS gives a median Calmar of 0.70, a 5% maximum drawdown of −34% and a 14% chance of a five-year loss. The block bootstrap gives 0.91 / −26% and FHS 0.84 / −29%. TimeGAN fails the fidelity battery, so it is left out. What actually hurts is a momentum crash, in which the 5% quantile of the three-month return drops to somewhere between −14% and −17%, against about −8% unconditionally.

The answers to each objection, with the residual risk that remains, are in [OBJECTIONS.md](OBJECTIONS.md).
