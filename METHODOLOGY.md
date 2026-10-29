# Methodology

This file gives the exact definition of every object the package computes, in the order the pipeline computes it. It also covers the validation protocol. The notation follows the paper. *t* is a session, *i* a stock, *a* an analyst, *r* a rating. Every quantity at *t* uses prices up to the close of *t* and ratings whose knowledge date is on or before *t*.

## 1. Point-in-time analyst state

The **knowledge date** of a rating published on day *p* is the first session strictly after *p*. The active target $TP_{a,i,t}$ is analyst *a*'s latest target on stock *i* with knowledge date $\le t$ and age $< 252$ sessions. Each rating is classified as one of:

| Kind | Condition |
|---|---|
| initiation | no previous target from *a* on *i*, or the previous one had expired |
| reiteration | $TP_r = TP_{r^-}$ |
| fresh up / down | $\mathrm{sign}\log(TP_r/TP_{r^-}) = \pm 1$ |

$U_{i,t}$ and $D_{i,t}$ count fresh up and down revisions arriving at *t*. An **information event** is a (stock, publication day) pair with $\ge 5$ ratings.

## 2. Signals

**Revision breadth** (63 sessions, at least 3 revisions):
$$B_{i,t}=\frac{\sum_{s=t-62}^{t}(U_{i,s}-D_{i,s})}{\sum_{s=t-62}^{t}(U_{i,s}+D_{i,s})}$$

**Event breadth** is the same ratio restricted to revisions published on information-event days (at least 2 such revisions).

**Breadth orthogonal to momentum** is the residual of a daily cross-sectional regression of the rank of $B$ on the rank of 6-1 momentum $\log P_{t-21}-\log P_{t-126}$.

**Leader minus herd (LMH):** For each fresh revision *r* on stock *i* with direction $d_r=\pm1$:

* *r* is a **leader** if it is published on an information-event day, or if it is the first revision of its day and comes $\ge 10$ sessions after the previous fresh revision on *i*
* *r* is a **herder** if it is not a leader, arrives 3–15 sessions after the latest information event on *i*, and has the sign of that event's net revision.

$$\mathrm{LMH}_{i,t}=\frac{\sum_{r\in\mathcal W_{i,t}} d_r\mathbf{1}^{\text{lead}}_r-\sum_{r\in\mathcal W_{i,t}} d_r\mathbf{1}^{\text{herd}}_r}{\sum_{r\in\mathcal W_{i,t}}(\mathbf{1}^{\text{lead}}_r+\mathbf{1}^{\text{herd}}_r)},\qquad \mathcal W_{i,t}=\text{revisions with knowledge date in }(t-126,t]$$

LMH is defined when the denominator is at least 3. `leaders_only` and `herders_only` are diagnostics, not composite inputs.

**Composite:** Each signal is converted to a cross-sectional percentile rank minus 0.5 (a day needs at least 5 valid names). The score is the equal-weight mean of the available ranks. No weights are estimated.

*Why these choices.* Using the sign rather than the size of a revision is robust to the heavy tails of target changes. The 126-session window covers two earnings cycles. The leader gap (10), the herd window (3–15) and the event size (5) were fixed from the analyst-behaviour literature and a single pre-registered round on 2017–2019. They were never tuned on the holdout or the open universe.

## 3. Portfolio construction

**Legs:** On each session with $N_t$ valid scores, the long leg $L_t$ is the top $k_t$ names and the short leg $S_t$ is the bottom $k_t$, where $k_t=4$ (fixed) or $k_t=\max(2,\lfloor 0.4N_t\rfloor)$ (leg fraction). A day needs $N_t\ge 2k_t$, otherwise the target is flat.

**Allocation within each leg:** Weights are proportional to $1/\hat\sigma_{i,t}$, the one-step-ahead GJR-GARCH(1,1)-t volatility. The model is estimated walk-forward per stock: first fit after 250 returns, refit every 21 sessions. Before the first fit, and whenever no fit has been accepted yet, the forecast is the expanding standard deviation of past returns. A fit is accepted only if the optimiser converged and the process is stationary, otherwise the previous estimate is kept. Any forecast still missing is filled with the 60-day realised volatility. `EW` sets equal weights.

**GARCH estimator and reproducibility:** The default estimator (`garch_method="slsqp"`, used for every published number) is `arch`'s quasi-maximum likelihood with the SLSQP optimiser. On a few hundred observations the GJR-GARCH-t likelihood is nearly flat along a ridge and often has two local maxima of almost equal height, so the optimiser's stopping point, and hence the forecast, can differ between machines (INSTALL.md). `garch_method="grid"` maximises the same likelihood over a fixed grid of $(\alpha, \gamma, \beta, \nu)$ with $\mu$ set to the sample mean and $\omega$ implied by variance targeting (Engle and Mezrich, 1996), with persistence capped at 0.995. The log-likelihood of every grid point is computed in one vectorised recursion and rounded to $10^{-6}$ before the arg-max, so the estimate is identical on every machine. The two estimators give the same conclusions: net Sharpe 1.13 vs 1.12 over 2017–21 and 1.05 vs 1.09 out of sample.

**Leg budgets and beta budget:** With a constant net exposure *c*:

$$w^{\text{target}}_{i,t}=\Big(\tfrac12+\tfrac c2\Big)\frac{1/\hat\sigma_{i,t}}{\sum_{j\in L_t}1/\hat\sigma_{j,t}}\mathbf{1}_{i\in L_t}-\Big(\tfrac12-\tfrac c2\Big)\frac{1/\hat\sigma_{i,t}}{\sum_{j\in S_t}1/\hat\sigma_{j,t}}\mathbf{1}_{i\in S_t}$$

FMF-N uses *c* = 0 (dollar-neutral); FMF-B20 uses *c* = 0.2. *c* is a disclosed constant. It is never timed, because timing net exposure is what manufactured the first-generation strategy's Calmar ratio.

**Tranches:** The held book is the average of the last *f* = 42 daily targets:

$$w_{i,t}=\frac1f\sum_{s=0}^{f-1}w^{\text{target}}_{i,t-s}$$

This equals holding 42 overlapping sub-portfolios, one opened each session (Jegadeesh–Titman). It matches the holding period to the signal's information horizon (IC rises from 0.026 at 1 day to 0.109 at 21 days) and removes rebalancing-date luck: with a single weekly rebalance day, the same signal gives a Sharpe between 0.34 and 0.71 depending on which weekday is chosen.

**Accounting**

$$\text{gross}_{t+1}=\sum_i w_{i,t}\,r_{i,t+1},\qquad \text{cost}_{t+1}=\kappa\sum_i|w_{i,t+1}-w_{i,t}|,\qquad \kappa=5\text{ bp}$$

## 4. Evaluation statistics

| Statistic | Definition (module) |
|---|---|
| Sharpe, CAGR, max drawdown, Calmar = CAGR / \|MaxDD\| | daily net returns, 252 periods (`analytics.metrics.perf_stats`) |
| CAPM beta and alpha | OLS on non-overlapping weekly (5-session) sums, HAC(4) errors (`metrics.capm`) |
| Factor attribution | the same regression on the market and four price books (12-1 and 6-1 momentum, 1-month reversal, low volatility) built with the identical construction (`metrics.factor_attribution`) |
| Rank IC$_h$ | daily Spearman correlation of the signal with the *h*-session forward log return (`inference.rank_ic`) |
| Phase-averaged IC test | averages the standard errors of the *h* non-overlapping sub-samples (offsets 0…h−1) and uses a one-sided *t* test with their median degrees of freedom. Calibrated to the nominal size under the null (`inference.ic_phase_test`) |
| Holm | step-down family-wise correction across signals (`inference.holm`) |
| PSR / DSR | probabilistic and deflated Sharpe ratio (Bailey & López de Prado). The deflation benchmark $SR_0$ is the expected maximum Sharpe over $N_{\text{eff}}$ independent trials, floored at 0 (`inference.psr`, `deflated_sharpe`) |
| PBO | combinatorially symmetric cross-validation over the construction grid (`inference.cscv_pbo`) |
| Bootstrap Calmar | stationary bootstrap (mean block 21) used to select a construction by its *median* Calmar rather than its point estimate (`inference.bootstrap_calmar`) |
| Memmel test | equality of two Sharpe ratios (`inference.memmel_test`) |

## 5. Placebos through the identical construction (`robustness.strategy_placebo`)

Each null replaces the composite score with a counterfeit and passes it through **exactly** the same legs, allocator, tranches and costs. The p-value is:

$$\frac{1 + \sum \mathbf{1}_{\{\text{null} \ge \text{observed}\}}}{1 + \text{draws}}$$

| Null | Counterfeit score | What it preserves | Question it answers |
|---|---|---|---|
| IID | N(0, 1) on the real missing-value mask | coverage | is the result luck? |
| Stock-label permutation | each stock's real score history assigned to another stock (derangement) | persistence and cross-sectional structure of the signal | does the signal belong to *these* stocks? |
| Circular time-shift | the whole panel rolled by $\ge 126$ sessions | each stock's habitual rank | does the *timing* add value beyond stock identity? (the hardest null) |

## 6. Comparables, blends and spanning (`benchmarks`)

The 13 cross-sectional comparables are:

* **Price signals:** 12-1 and 6-1 momentum, 1-month reversal, 60-day low volatility, betting-against-beta (low 252-day beta), residual momentum (Blitz–Huij–Martens), 52-week-high proximity.
* **Analyst signals:** consensus target-price upside, 63-day consensus target momentum, low dispersion, classic revision breadth.
* **Blends:** momentum + low vol + reversal; momentum + revisions.

Each is traded with FMF's construction. The three allocations are equal-weight long-only, inverse-volatility long-only, and 12-1 time-series momentum (CTA-style), which are traded natively.

* **Blend test:** 50/50 mix of each comparable's returns with FMF-N.
* **Spanning test:** regress FMF-N weekly returns on the market and all comparables, then again excluding the comparables that are built from FMF's own components.

## 7. Leave-one-out (`robustness.leave_one_out`)

For each stock *j* the **entire** pipeline is rebuilt on the universe without *j*: cleaning, calendar, snapshot, signals, ranks, legs, weights and P&L. The univariate GARCH forecasts do not depend on the other stocks and are reused.

## 8. Protocol

1. **Development:** (2017–2019): every signal, parameter and construction choice was made here, through pre-registered rounds whose failures are logged (paper, Table 2).
2. **Holdout** (2020–2021): evaluated once the specification was frozen.
3. **Never-seen period:** (2023–2026, 30 stocks: twenty names never used in development plus the ten development names, DATA_SOURCES.md §3) The transfer rules were written down first: leg fraction 0.4 as primary, *k* = 4 as secondary, the evaluation window, and the tests. The file was hashed (`configs/oos_open_universe_preregistration.json`, fingerprint `0b2b0792fd357d61`) before any return was computed. `fmflow.prereg.verify` checks the fingerprint in every evaluation script.
4. **Synthetic control**: the pipeline must find no alpha in a null world and must find it when first movers are informed (`scripts/synthetic_power_study.py`).

To repeat the study on your own universe: freeze a configuration with `fmflow register`, then run `fmflow evaluate --placebo --loo` and report what comes out.

## 9. The audit layer: small-sample and snooping-robust inference

| Question | Tool (module) | Definition |
|---|---|---|
| Identity or timing? | `analytics.attribution.lo_decomposition`, `static_timing_books` | Lo (2008): $E[\sum_i w_{i,t-1}r_{i,t}]=\sum_i\mathrm{Cov}(w,r)+\sum_i E[w]E[r]$. Causal version: static book $\bar w_{i,t}$ (expanding mean of past weights) and timing book $w_{i,t}-\bar w_{i,t}$; both tradable, summing to the strategy |
| Does any specification really work? | `analytics.snooping.spa_test`, `reality_check` | Hansen (2005) SPA with consistent recentring; White (2000) Reality Check; stationary bootstrap of the $T\times K$ matrix of performance differentials |
| Which comparables does it beat? | `analytics.snooping.romano_wolf` | Romano–Wolf (2005) StepM, studentised, step-down adjusted p-values (FWER) |
| Is a Sharpe difference real? | `analytics.snooping.sharpe_diff_test` | Circular block bootstrap of paired returns (Ledoit–Wolf 2008, basic bootstrap) |
| How much of the result is the specification? | `robustness.multiverse` | One-at-a-time, full-factorial or random joint draws over any `StrategyConfig` / `DataConfig` field; returns the family's daily returns for SPA, PBO and DSR, and the selection-free ensemble |
| What is the Sharpe ratio, given that most strategies fail? | `analytics.bayes.bayesian_sharpe` | Bayesian bootstrap (Rubin 1981) of weekly returns, importance-reweighted by a prior $SR\sim N(m,s^2)$ on the annual Sharpe ratio; flat, $N(0,1)$ and $N(0,0.5)$ priors reported |
| What do three samples say together, and about the next one? | `analytics.bayes.random_effects_meta` | $y_j\sim N(\theta_j,s_j^2)$, $\theta_j\sim N(\mu,\tau^2)$, $\tau\sim$ half-Cauchy; $\mu$ integrated analytically, $\tau$ on a grid; predictive $\theta_{new}\sim N(\mu,\tau^2)$ |
| What is the best of a search worth? | `analytics.bayes.shrink_normal_means` | Positive-part James–Stein shrinkage of the family's t-statistics towards their mean |
| Evidence for vs against an effect | `analytics.bayes.bayes_factor_normal` | $BF_{10}$ for $\theta>0$ with an $N(0,g^2)$ prior against $\theta=0$ |
| What could the sample detect? | `analytics.power.mde_ic`, `ic_power`, `simulate_power` | MDE $=(z_{1-\alpha}+z_{1-\beta})/\sqrt{(N-1)M}$; Monte Carlo power of the full pipeline on the calibrated generator |
| Is the Sharpe inflated by smoothing? | `analytics.metrics.lo_adjusted_sharpe` | $\eta(q)=q/\sqrt{q+2\sum_{k\le L}(q-k)\rho_k}$ (Lo 2002) |
| Crash beta? | `analytics.metrics.conditional_betas` | Weekly regression on $\max(m,0)$ and $\min(m,0)$ with HAC errors |
| Implementation? | `backtest.engine.cost_sensitivity`, `breakeven_cost_bps` | Grid over costs, borrow and execution delay; breakeven = mean(gross − borrow) / mean(traded) |
| Causality of a new signal? | `robustness.truncation_check` | Pipeline on data truncated at T must equal the full run up to T for every signal, the composite and the weights |

## 10. Generative stress testing (`fmflow.generative`, optional `[gen]` extra)

The question is what the *distribution* of a five-year Calmar ratio or maximum drawdown looks like when one history contains only one or two drawdowns.

* **Data:** Joint daily returns of the strategy and its risk factors (market, momentum). Each sample is standardised by its own mean and volatility so that samples can be pooled.
* **Generators:**
  * `baselines.block_bootstrap_paths`: stationary bootstrap
  * `baselines.fhs_paths`: filtered historical simulation. GJR-GARCH-t is fitted per channel with variance targeting and persistence capped at 0.99, and the empirical residual vectors are resampled jointly with zero mean
  * `diffusion_fhs.DiffusionFHS`: the same dynamics, with each day's shock vector drawn from a **conditional DDPM** $p(z_t\mid\text{log-vol state}_t)$
    * Shocks are first mapped to normal scores (rank-based Gaussian quantile transform, as in TabDDPM). The model learns their regime-dependent dependence, a conditional copula. Draws are mapped back through the empirical quantile function, with Student-t tails beyond the sample
    * The denoiser is a residual MLP with a cosine schedule (T = 100), an EMA of the weights, and early stopping on a chronological validation split
    * Sampling is a strided DDIM-type chain ($\eta = 1$)
    * Each day the shocks are moment-matched across paths, to zero mean and the empirical standard deviation
    * Volatility states are kept within the observed range widened by 0.5 standard deviations
  * `timegan.TimeGAN`: Yoon et al. (2019), with embedding, supervised and joint phases. `ddpm.DDPM`: a window-level diffusion model with PCA whitening and training-free conditional sampling
* **Fidelity:** (`fidelity`)
  * Stylised facts: moments, ACF of $|r|$, cross-correlations, tail dependence
  * A discriminative score. Real windows are split chronologically with a gap, because overlapping windows would otherwise leak
  * A train-on-synthetic / test-on-real predictive score
  * A memorisation ratio. On small data, a training loss below what held-out data achieve is memorisation, and it shows up as under-dispersed draws
* **Drift is an input:** Paths are zero-drift. The strategy's drift is set explicitly: zero, the in-sample Sharpe, or draws from the Bayesian posterior-predictive Sharpe. Path risk and parameter uncertainty then combine in `stress.path_stats`
* **Conditional stress:** Strategy statistics on the simulated windows where the market or momentum crashes (`stress.conditional_on`)
