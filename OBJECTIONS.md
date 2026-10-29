# Objections and answers

This file collects some objections that we have raised during the exploration of the datasets that we had and the theoretical limitations that we faced. Each entry gives the answer, the evidence (with the command or notebook that reproduces it) and the **residual risk** that remains after the test. Numbers refer to the development universe (10 US large caps, 2017–2021) and to the never-seen period (30 stocks, 2024-01-05 to 2026-08-05, pre-registered). The period never overlaps development, ten of the thirty names are the development names (DATA_SOURCES.md §3).

Verdicts:
- **Answered**: the test is designed to break the result, and the result survives.
- **Partly answered**: the evidence points the right way but is not conclusive.
- **Open**: the data cannot answer it.

---

## A. Data and look-ahead

**A1. "Target prices are timestamped by day, not by hour. You may be trading on ratings published after the close."** · *Answered*
Ratings are usable from the session **after** publication (`knowledge_date > pub_date` is checked for every row, and a violation stops the run). A two-session knowledge lag leaves the Sharpe ratio unchanged: 1.13 (development) and 1.06 (never-seen), against 1.13 and 1.05. Truncation tests recompute every signal and weight on data cut at T and require exact equality with the full run, a centred-window control fails the same test. → `tests/test_causality.py`, `robustness.truncation_check`, notebook 02.

**A2. "Vendors back-fill and revise analyst histories."** · *Open (data-dependent)*
No backtest can detect back-filling in its own input. Use point-in-time detail feeds (I/B/E/S detail history, FactSet or Bloomberg analyst-level data with publication dates), never consensus snapshots. The cleaning rules R3/R4 catch the commonest symptom, a split applied to one series only. → `DATA.md` §4.

**A3. "Prices and targets on different adjustment bases create fake revisions."** · *Answered*
The data contract rejects a median target/price ratio outside (0.3, 3.5). Rules R3 and R4 drop implausible levels and revisions inconsistent with the price move, and every drop is logged. → `data.schema`, `out.audit`.

## B. Selection and hindsight

**B1. "The ten stocks were chosen knowing how 2016–2021 turned out. The book is simply long the winners."** · *Answered*
Lo's (2008) decomposition splits the mean return into a passive tilt and active timing. The causal version builds two tradable books:
- a **static:** book holding each stock's expanding mean of past weights,
- a **timing:** book holding today's deviation from that mean.

The timing book carries the edge: Sharpe 1.04 (*t* 2.41) over 2017–21 and 1.33 (*t* 2.21) out of sample. The static tilt is insignificant in sample (Sharpe 0.48, *t* 1.08) and loses money out of sample (−0.49). → the research notebook, Section 17.1, notebook 05 §1, paper §10.1.

**B2. "The time-shift placebo is not passed, so stock identity explains the result."** · *Answered, with a caveat*
The circular time-shift keeps each stock's habitual rank and slow-moving structure, so its null distribution is very wide (95th percentile of the null Sharpe above 2). It is a low-power test, and B1 is the direct one. The placebo result is still reported as it came out (p = 0.076 full sample, 0.23 out of sample).

**B3. "One or two names carry the result."** · *Answered*
Rebuilding the full pipeline without each stock gives a Sharpe between 0.80 and 1.25 (development) and between 0.74 and 1.33 (never-seen). Bayesian partial pooling of per-stock contributions shrinks the largest contributor from 5.7% to 4.5% a year, out of sample the contributions are small and diffuse. → Section 15.2, Section 17.1, notebook 05 §2.

**B4. "How was the 30-stock universe chosen? Is it really unseen?"** · *Partly answered*
The universe is the ten development names plus twenty of today's most covered US large caps (the list is in `notebooks/Data_Gathering.ipynb`). Two consequences follow, and both are tested:
- **Ten names re-appear** (in a period two years after development ends). Re-running the frozen specification on the **twenty new names only** gives a Sharpe ratio of 0.80 (Calmar 0.98), and leader minus herd keeps its IC (0.105, one-sided *p* = 0.058). The ten development names alone earn 0.31, with too few covered names for a reliable cross-section. The result is not carried by the overlap. → `scripts/overlap_analysis.py` (post hoc, not pre-registered).
- **The list was chosen knowing which companies grew large:** A dollar-neutral book is exposed to this only through the spread between its long and short legs, and B1 shows that the static tilt towards particular stocks loses money out of sample (Sharpe −0.49). A universe defined by an independent rule (for example index membership on the first day) would be stronger evidence, `fmflow` runs on any such universe unchanged.

## C. Research process

**C1. "You searched until something worked."** · *Answered for the sign, partly for the level*
The research log is published (paper, Table 2), and the out-of-sample rules were hashed before evaluation (`configs/oos_open_universe_preregistration.json`). The full pipeline was re-run over about 300 specifications covering windows, thresholds, legs, tranches, allocator, signal set, target expiry and knowledge lag:
- **99%** of development specifications and **97%** of never-seen ones earn a positive Sharpe,
- the selection-free **ensemble** of all specifications earns 0.92 (2017–21, *t* 1.97) and 0.86 out of sample,
- the frozen specification sits at the 98th percentile in development (as selection implies) and the 83rd out of sample (which selection does not imply).

On any single short window, Hansen's SPA test over the family is not significant (development p = 0.34, never-seen p = 0.14, 2017–21 ensemble p = 0.10). The ranking of specifications does not persist (PBO 0.76). → Section 17.3, `fmflow multiverse`, notebook 05 §3.

**C2. "LMH was the best of 18 candidates. Of course its t-statistic is 2.5."** · *Answered by replication*
Empirical-Bayes shrinkage of the 18 t-statistics reduces LMH's from 2.46 to 1.06, so the development search alone is only suggestive. The evidence comes from the two samples that played no part in the search. LMH's IC is 0.129 in the 2020–21 holdout and 0.096 over the never-seen period (0.105 on its twenty new names), against 0.095 in development. → Section 17.4.

**C3. "Why not the quarterly (63-session) target expiry that matches the earnings cycle?"** · *Answered*
It was tested. A hard 63-session expiry **lowers** the Sharpe ratio (0.94 development, 0.88 never-seen) because it discards analysts who update semi-annually. A 126-session expiry is equivalent to 252 (1.12 / 1.15). Staleness is handled inside LMH, through its event and silence clocks, rather than by truncating the snapshot.

## D. Statistics

**D1. "Ten stocks and five years prove nothing."** · *Answered by pooling*
Power is low by construction. The minimum detectable mean IC is 0.138 for ten stocks over three years and 0.077 for thirty. That is why the three independent samples are combined formally:
- **Random-effects meta-analysis:** The pooled LMH IC is 0.105 (90% HDI 0.033 to 0.179). P(IC > 0 in the **next** universe) = 0.96. Stouffer p = 0.002, and the product of the Bayes factors is about 22.
- **Strategy Sharpe ratio:** Pooled 1.10, with P(SR > 0 next universe) = 0.96 and P(SR > 0.5) = 0.87.
- **Bayesian bootstrap under a sceptical N(0, 1) prior:** P(SR > 0) is 0.99 over 2017–21 and 0.93 on the never-seen window.

→ Section 17.4–17.5, `fmflow infer`, notebook 05 §5–6.

**D2. "Overlapping labels and smoothed P&L inflate t-statistics."** · *Answered*
IC tests average the standard errors of the h non-overlapping sub-samples. Lo's (2002) autocorrelation-adjusted Sharpe is 1.06 in both samples (lag-1 autocorrelation 0.04). Alphas use non-overlapping weekly returns with HAC errors.

**D3. "Beating the institutional toolkit is cherry-picking the comparison set."** · *Answered*
The reverse SPA test finds no comparable among 16 that significantly beats FMF-N (p = 0.86 in both samples). Romano–Wolf step-down shows FMF-N significantly beating two books (target-price momentum and 6–1 momentum) in sample after family-wise control, and none out of sample. The honest claim is "never beaten, better than most", not "significantly better than all".

**D4. "LMH adds nothing to classic revision breadth."** · *Answered in sample, consistent out of sample*
FMF-N was compared with the identical book without LMH, using a circular block bootstrap of paired returns. LMH adds +0.29 Sharpe in sample (95% CI 0.03 to 0.54, p = 0.028) and +0.30 out of sample (CI −0.21 to 0.80). Pooled, P(gain > 0) = 0.94.

## E. Implementation and capacity

**E1. "Trading at the signal close is optimistic."** · *Answered*
A one-session delay leaves the Sharpe at 1.13 (1.06 never-seen), two sessions, at 1.13. Overlapping tranches re-target 1/42 of the book per day, so a day's delay costs almost nothing.

**E2. "Shorts are not free."** · *Answered*
With a 100 bp annual borrow fee on all short notional, 10 bp costs and a one-day delay, the Sharpe is 1.06 (0.99 never-seen). The harshest cell (200 bp borrow, 20 bp costs, two-day delay) gives 0.96 (0.89). Hard-to-borrow names need a name-level fee, which you can apply by passing a higher `borrow_bps` or excluding them from the universe.

**E3. "Capacity?"** · *Partly answered*
No volume data are used. The breakeven cost is 234 bp per unit traded (197 never-seen) at a turnover of 2–3× a year, more than an order of magnitude above large-cap execution costs. With ADV data, apply a square-root impact cost through `cost_bps` per name, or use the `max_weight` cap.

**E4. "Dollar-neutral is not beta-neutral."** · *Answered*
The realised beta is −0.02. The down-market beta is −0.07 (t −1.24), and the mean return in the worst decile of market weeks is positive. If you need ex-ante beta neutrality, `StrategyConfig.fmf_beta_neutral()` solves the leg budgets with rolling betas. Market exposure is a constant, disclosed budget (FMF-B20), never a timed overlay.

## F. Risk

**F1. "One history has one drawdown, the Calmar ratio is a single draw."** · *Answered with a distribution*
Four generators are compared on a fidelity battery: block bootstrap, FHS, Diffusion-FHS and TimeGAN. Diffusion-FHS combines GJR-GARCH dynamics with variance targeting and conditional-diffusion shocks in normal-score space. The bootstrap matches the sample moments by construction but can only re-order history. Diffusion-FHS has the lowest discriminative score (0.014, 0 = a classifier cannot separate synthetic from real 64-day windows), a memorisation ratio of 0.96 (no copying) and a correlation with momentum of 0.62 against 0.66. Its tails are heavier than the sample's (strategy excess kurtosis 10.2 against 5.5, market 34 against 13), which errs on the side of caution, FHS errs further (market 63). TimeGAN fails: its draws are near-Gaussian and nearly collinear (correlation with momentum 0.99), a classifier separates them perfectly (0.50), and it is excluded from inference. The strategy's drift is drawn from the Bayesian posterior-predictive Sharpe ratio. Under Diffusion-FHS the median five-year Calmar ratio is 0.70 (bootstrap 0.91, FHS 0.84), P(Calmar > 0.5) = 59%, the median maximum drawdown is −15% and its 5% quantile −34%, the probability of losing money over five years is 14%. Quote these, not the single in-sample Calmar ratio. → Section 17.6, `fmflow stress`, notebook 06.

**F2. "What is the stress scenario?"** · *Answered*
Momentum crashes, not market crashes. In simulated three-month windows where the equal-weight market falls 25% or more, FMF-N's mean return is positive under every generator (bootstrap +6.4%, FHS +0.6%, Diffusion-FHS +10.9%). In the worst 2% of momentum windows, the 5% quantile of FMF-N's three-month return falls to −14.1% (bootstrap), −16.8% (FHS) and −14.6% (Diffusion-FHS), roughly twice the unconditional −8%. Three generators that build the tail in different ways agree on it. Practical responses are a momentum hedge or a cap on momentum exposure in high-volatility regimes, and disclosure of the scenario in an index rulebook.

**F3. "The development sample is a bull market."** · *Partly answered*
The holdout contains the COVID crash, in which FMF-N returned +12% while the market fell 35%, and the 2021 growth sell-off (−7%, against −10.5% for momentum). The never-seen window adds 2024–26. A long bear market is not in the data, generative paths with zero drift (F1) show the risk side.

**F4. "Factor exposures beyond the market and four price factors (size, value, sector)?"** · *Open with this data*
These cannot be measured with prices and targets alone. `run_strategy(..., sectors=...)` neutralises sectors if you have a map, and `analytics.metrics.factor_attribution` accepts any factor returns you supply.

## G. Engineering

**G1. "Can I trust that the package is the strategy in the paper?"** · *Answered*
With the shipped GARCH forecasts, parity tests reproduce the notebook's weights exactly on both datasets. A fresh GARCH re-estimation is exact on the reference machine but **not bit-for-bit across machines**: the short-sample GJR-GARCH-t likelihood has near-equal local maxima, and the optimiser's stopping point depends on the CPU's floating-point path (on another BLAS kernel the out-of-sample Sharpe ratio is 1.02 instead of 1.05). The deterministic estimator `DataConfig(garch_method="grid")` is bit-identical on every machine and gives 1.09 out of sample and 1.12 over 2017–21. The tests run in CI on Linux (pandas 2.1, 2.2, 3.x) and Windows. → INSTALL.md.

**G2. "Will it work on my data?"** · *Answered*
The data contract fails early with actionable messages. Notebook 09 runs a complete, honest research cycle (power → development → pre-registration → confirmation) on a synthetic universe, replace two lines with your files. A calibrated generator (`fmflow synth`) reproduces both real datasets' moments, and in a null world the pipeline finds nothing (mean Sharpe 0.05 over 40 seeds).
