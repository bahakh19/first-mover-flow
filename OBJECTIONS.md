# Objections and answers

While working through the datasets we kept running into objections, some raised by us and some forced on us by the theoretical limits of what the data can show. They are collected here. For each one we give our answer, the evidence behind it (with the command or notebook that reproduces it) and the **residual risk** left over once the test has been run. Unless stated otherwise, numbers refer to the development universe (10 US large caps, 2017–2021) and to the never-seen period (30 stocks, 2024-01-05 to 2026-08-05, pre-registered). That period never overlaps development, although ten of its thirty names are the development names (DATA_SOURCES.md §3).

Verdicts:
*Answered* means the test could have broken the result and did not. *Partly answered* means the evidence points the right way but does not settle it, and *Open* means this data cannot answer it.

---

## A. Data and look-ahead

**A1. "Target prices are timestamped by day, not by hour. You may be trading on ratings published after the close."** · *Answered*
Ratings are used from the session **after** publication; `knowledge_date > pub_date` is checked on every row, and a violation stops the run. With a two-session knowledge lag the Sharpe ratio is 1.13 (development) and 1.06 (never-seen), against 1.13 and 1.05 with one session. Truncation tests recompute every signal and weight on data cut at T and require exact equality with the full run; a centred-window control fails. Code: `tests/test_causality.py`, `robustness.truncation_check`, notebook 02.

**A2. "Vendors back-fill and revise analyst histories."** · *Open (data-dependent)*
No backtest can catch back-filling inside its own input data. The remedy lies upstream, in the choice of feed: point-in-time detail data (I/B/E/S detail history, or FactSet or Bloomberg analyst-level data carrying publication dates), and never consensus snapshots. Cleaning rules R3/R4 do catch the most frequent symptom, which is a split applied to one series but not the other (`DATA.md` §4).

**A3. "Prices and targets on different adjustment bases create fake revisions."** · *Answered*
If the median target/price ratio falls outside (0.3, 3.5), the data contract rejects the file outright. Beyond that, R3 and R4 remove implausible levels and revisions that the price move cannot explain, and each removal is written to the log (`data.schema`, `out.audit`).

## B. Selection and hindsight

**B1. "The ten stocks were chosen knowing how 2016–2021 turned out. The book is simply long the winners."** · *Answered*
Lo's (2008) decomposition separates the mean return into two parts, a passive tilt and active timing. We use a causal version of it. It is built from two books that could be traded:
- a **static** book, which holds each stock at the expanding mean of its past weights,
- a **timing** book, which holds today's deviation from that mean.

The timing book is the one that earns the money, with a Sharpe of 1.04 (*t* 2.41) over 2017–21 and 1.33 (*t* 2.21) out of sample. The static tilt, by contrast, is insignificant in sample (Sharpe 0.48, *t* 1.08) and loses money out of sample (−0.49). Details are in Section 17.1 of the research notebook, §1 of notebook 05 and §10.1 of the paper.

**B2. "The time-shift placebo is not passed, so stock identity explains the result."** · *Answered, with a caveat*
Rolling the panel in time preserves each stock's habitual rank and the slow parts of its structure. As a consequence the null distribution is very wide (its 95th percentile Sharpe sits above 2), which makes this a test with little power. B1 addresses the question directly. We still report the placebo exactly as it came out: p = 0.076 on the full sample and 0.23 out of sample.

**B3. "One or two names carry the result."** · *Answered*
We rebuilt the whole pipeline ten times, each time leaving one stock out, and the Sharpe ratio ranged from 0.80 to 1.25 in development and from 0.74 to 1.33 over the never-seen period. No rebuild broke it. Bayesian partial pooling of the per-stock contributions pulls the largest one down from 5.7% to 4.5% a year. Out of sample the contributions are small and spread across names (Section 15.2, Section 17.1, notebook 05 §2).

**B4. "How was the 30-stock universe chosen? Is it really unseen?"** · *Partly answered*
Ten development names plus twenty of today's most covered US large caps (list in `notebooks/Data_Gathering.ipynb`). Two consequences, both tested:
- **Ten names come back**, in a period that begins two years after development ends. Re-run on the **twenty new names only**, the frozen specification earns a Sharpe ratio of 0.80 (Calmar 0.98). Leader minus herd keeps its IC (0.105, one-sided *p* = 0.058). On their own, the ten development names make 0.31, but there are too few covered names among them to form a dependable cross-section. Thus the overlap is not what carries the result (`scripts/overlap_analysis.py`; post hoc, not pre-registered).
- **The list was drawn up knowing which companies grew large.** A dollar-neutral book feels this only through the spread between its long and short legs. And B1 has already shown that the static tilt towards particular stocks loses money out of sample (Sharpe −0.49). Stronger evidence would come from a universe defined by an independent rule, say index membership on the first day, and `fmflow` would run on such a universe without any change.

## C. Research process

**C1. "You searched until something worked."** · *Answered for the sign, partly for the level*
The research log is public (paper, Table 2), and we hashed the out-of-sample rules before evaluating anything (`configs/oos_open_universe_preregistration.json`). We also re-ran the full pipeline over roughly 300 specifications that vary the windows, thresholds, legs, tranches, allocator, signal set, target expiry and knowledge lag. Three things came out of that exercise.
- **99%** of the development specifications have a positive Sharpe, as do **97%** of the never-seen ones.
- An **ensemble** of every specification, which involves no selection at all, earns 0.92 (2017–21, *t* 1.97) and 0.86 out of sample.
- Our frozen specification lands at the 98th percentile in development, which is what selection would predict, and at the 83rd out of sample, which selection would not.

Hansen's SPA test over the whole family is not significant on any single short window (development p = 0.34, never-seen p = 0.14, 2017–21 ensemble p = 0.095). Nor does the ranking of specifications persist from one period to the next (PBO 0.76). See Section 17.3, `fmflow multiverse` and notebook 05 §3.

**C2. "LMH was the best of 18 candidates. Of course its t-statistic is 2.5."** · *Answered by replication*
Fair point. Shrinking the 18 t-statistics with empirical Bayes drops LMH's from 2.46 to 1.06. Thus the development search by itself only suggests an effect. The evidence comes from the two samples that had no part in the search. LMH's IC is 0.129 in the 2020–21 holdout and 0.096 over the never-seen period (0.105 on its twenty new names). In development it was 0.095 (Section 17.4).

**C3. "Why not the quarterly (63-session) target expiry that matches the earnings cycle?"** · *Answered*
We tried it. A hard 63-session expiry **lowers** the Sharpe ratio, to 0.94 in development and 0.88 over the never-seen period, mostly because it throws away the analysts who update only twice a year. Expiring at 126 sessions works about as well as 252 (1.12 / 1.15). Thus staleness is dealt with inside LMH, through its event and silence clocks, instead of by cutting the snapshot short.

## D. Statistics

**D1. "Ten stocks and five years prove nothing."** · *Answered by pooling*
Power is low by construction. With ten stocks over three years the minimum detectable mean IC is 0.138, and with thirty it is 0.077. Thus we combine the three independent samples formally.
- A **random-effects meta-analysis** puts the pooled LMH IC at 0.105 (90% HDI 0.033 to 0.179) and P(IC > 0 in the **next** universe) at 0.96. Stouffer's p is 0.002, and multiplying the Bayes factors together gives roughly 22.
- For the **strategy Sharpe ratio** the pooled value is 1.10, with P(SR > 0 next universe) = 0.96 and P(SR > 0.5) = 0.86.
- A **Bayesian bootstrap with a sceptical N(0, 1) prior** gives P(SR > 0) of 0.99 over 2017–21 and 0.93 on the never-seen window.

The calculations are in Sections 17.4–17.5, `fmflow infer` and notebook 05 §5–6.

**D2. "Overlapping labels and smoothed P&L inflate t-statistics."** · *Answered*
Our IC tests average the standard errors across the h non-overlapping sub-samples. Lo's (2002) autocorrelation-adjusted Sharpe comes to 1.06 in both samples (lag-1 autocorrelation 0.04), and the alphas are estimated on non-overlapping weekly returns with HAC errors.

**D3. "Beating the institutional toolkit is cherry-picking the comparison set."** · *Answered*
Run in reverse, the SPA test finds that none of the 16 comparables significantly beats FMF-N (p = 0.86 in both samples). The Romano–Wolf step-down procedure, after family-wise control, has FMF-N significantly ahead of two books in sample (target-price momentum and 6–1 momentum) and of none out of sample. So what we can fairly claim is "never beaten, better than most"; "significantly better than all" would overstate it.

**D4. "LMH adds nothing to classic revision breadth."** · *Answered in sample, consistent out of sample*
We compared FMF-N against the same book with LMH removed, using a circular block bootstrap of the paired returns. In sample LMH adds +0.28 Sharpe (95% CI 0.03 to 0.54, p = 0.028); out of sample it adds +0.30 (CI −0.21 to 0.80). Pooling the two, P(gain > 0) = 0.94.

## E. Implementation and capacity

**E1. "Trading at the signal close is optimistic."** · *Answered*
With a one-session delay the Sharpe stays at 1.13 (1.06 never-seen), and with two sessions it is still 1.13. The overlapping tranches re-target only 1/42 of the book each day. So waiting a day costs next to nothing.

**E2. "Shorts are not free."** · *Answered*
With a 100 bp annual borrow fee on all short notional, 10 bp costs and a one-day delay, the Sharpe is 1.06 (0.99 never-seen). In the harshest cell we ran (200 bp borrow, 20 bp costs, two-day delay) it is 0.96 (0.89). Hard-to-borrow names need a fee set name by name. You can approximate that by passing a higher `borrow_bps`, or simply leave those names out of the universe.

**E3. "Capacity?"** · *Partly answered*
We use no volume data at all. We can only report a breakeven cost of 234 bp per unit traded (197 never-seen) at a turnover of 2–3× a year, more than ten times typical large-cap execution costs. If you have ADV data, put a square-root impact cost into `cost_bps` per name, or use the `max_weight` cap.

**E4. "Dollar-neutral is not beta-neutral."** · *Answered*
Realised beta is −0.02. The down-market beta is −0.07 (t −1.24), and in the worst decile of market weeks the book's mean return is positive. For ex-ante beta neutrality, `StrategyConfig.fmf_beta_neutral()` solves for the leg budgets using rolling betas. Any market exposure enters as a constant budget that is disclosed up front (FMF-B20) and is never a timed overlay.

## F. Risk

**F1. "One history has one drawdown, the Calmar ratio is a single draw."** · *Answered with a distribution*
Four generators (block bootstrap, FHS, Diffusion-FHS, TimeGAN) go through one fidelity battery. Diffusion-FHS = GJR-GARCH dynamics with variance targeting + shocks from a conditional diffusion model in normal-score space. The bootstrap matches sample moments by construction but only re-orders history. Diffusion-FHS has a discriminative score of 0.014 (lowest; 0 = a classifier cannot separate synthetic from real 64-day windows), memorisation ratio 0.96 (no copying), correlation with momentum 0.62 vs 0.66. Tails heavier than the sample (strategy excess kurtosis 10.2 vs 5.5, market 34 vs 13), i.e. conservative; FHS more so (market 63). TimeGAN fails (near-Gaussian, nearly collinear draws, correlation with momentum 0.99; classifier score 0.50) and is excluded. Drift: Bayesian posterior-predictive Sharpe ratio. Diffusion-FHS results: median five-year Calmar 0.70 (bootstrap 0.91, FHS 0.84), P(Calmar > 0.5) = 59%, median maximum drawdown −15%, 5% quantile −34%, probability of a five-year loss 14%. Use these instead of the single in-sample Calmar ratio (Section 17.6, `fmflow stress`, notebook 06).

**F2. "What is the stress scenario?"** · *Answered*
A momentum crash, as it turns out, rather than a market crash. Across simulated three-month windows in which the equal-weight market drops by 25% or more, FMF-N's mean return is positive under every generator (bootstrap +6.4%, FHS +0.6%, Diffusion-FHS +10.9%). Momentum is a different story. In the worst 2% of momentum windows, the 5% quantile of FMF-N's three-month return falls to −14.1% (bootstrap), −16.8% (FHS) and −14.6% (Diffusion-FHS), about twice the unconditional −8%. Three generators that build their tails in quite different ways agree on this. In practice the answer is a momentum hedge, or a cap on momentum exposure when volatility is high, together with disclosure of the scenario in any index rulebook.

**F3. "The development sample is a bull market."** · *Partly answered*
The holdout does contain the COVID crash, during which FMF-N made +12% while the market lost 35%. It also contains the 2021 growth sell-off (−7%, against −10.5% for momentum). The never-seen window adds 2024–26. A long bear market, though, is simply absent from the data. The zero-drift generative paths in F1 are our view of the downside.

**F4. "Factor exposures beyond the market and four price factors (size, value, sector)?"** · *Open with this data*
Prices and targets alone can't measure these. If you have a sector map, `run_strategy(..., sectors=...)` neutralises sectors, and `analytics.metrics.factor_attribution` will take any factor returns you give it.

## G. Engineering

**G1. "Can I trust that the package is the strategy in the paper?"** · *Answered*
Given the shipped GARCH forecasts, the parity tests reproduce the notebook's weights exactly on both datasets. Re-estimating GARCH from scratch is exact on the reference machine but **not bit-for-bit across machines**. The reason is that the GJR-GARCH-t likelihood on a short sample has local maxima of nearly equal height. Where the optimiser stops then depends on the CPU's floating-point path (with another BLAS kernel the out-of-sample Sharpe ratio is 1.02 rather than 1.05). The deterministic estimator, `DataConfig(garch_method="grid")`, is bit-identical everywhere and gives 1.09 out of sample and 1.12 over 2017–21. CI runs the tests on Linux (pandas 2.1, 2.2, 3.x) and on Windows; INSTALL.md has the details.

**G2. "Will it work on my data?"** · *Answered*
The data contract fails early, and its messages say what to fix. Notebook 09 walks through a complete, honest research cycle (power → development → pre-registration → confirmation) on a synthetic universe. Swap two lines for your own files and it runs on yours. There is also a calibrated generator (`fmflow synth`) that reproduces the moments of both datasets, and in a null world the pipeline finds nothing (mean Sharpe 0.05 over 40 seeds).
