# Methodology

Below you will find the exact definition of each object the package computes, taken in the order in which the pipeline computes them, followed by the validation protocol. We keep the paper's notation, where *t* is a session, *i* a stock, *a* an analyst and *r* a rating. Any quantity dated *t* draws on prices up to the close of *t* and on ratings whose knowledge date is *t* or earlier.

## 1. Point-in-time analyst state

A rating published on day *p* has as its **knowledge date** the first session strictly after *p*. Analyst *a*'s active target on stock *i*, $TP_{a,i,t}$, is that analyst's most recent target with knowledge date $\le t$ and age $< 252$ sessions. Every rating falls into one of the following classes.

| Kind | Condition |
|---|---|
| initiation | no previous target from *a* on *i*, or the previous one had expired |
| reiteration | $TP_r = TP_{r^-}$ |
| fresh up / down | $\mathrm{sign}\log(TP_r/TP_{r^-}) = \pm 1$ |

We write $U_{i,t}$ and $D_{i,t}$ for the number of fresh up and down revisions arriving at *t*. A (stock, publication day) pair with $\ge 5$ ratings counts as an **information event**.

## 2. Signals

**Revision breadth** is computed over 63 sessions and needs at least 3 revisions:
$$B_{i,t}=\frac{\sum_{s=t-62}^{t}(U_{i,s}-D_{i,s})}{\sum_{s=t-62}^{t}(U_{i,s}+D_{i,s})}$$

**Event breadth** is the same ratio, but counting only revisions published on information-event days (with a minimum of 2 of them).

**Breadth orthogonal to momentum** is what is left over when, each day, the rank of $B$ is regressed across stocks on the rank of 6-1 momentum $\log P_{t-21}-\log P_{t-126}$.

**Leader minus herd (LMH).** Take a fresh revision *r* on stock *i*, with direction $d_r=\pm1$.

* We call *r* a **leader** when it is published on an information-event day, or when it is the day's first revision and arrives $\ge 10$ sessions after the previous fresh revision on *i*.
* We call *r* a **herder** when it is not a leader, lands 3–15 sessions after the most recent information event on *i*, and points the same way as that event's net revision.

$$\mathrm{LMH}_{i,t}=\frac{\sum_{r\in\mathcal W_{i,t}} d_r\mathbf{1}^{\text{lead}}_r-\sum_{r\in\mathcal W_{i,t}} d_r\mathbf{1}^{\text{herd}}_r}{\sum_{r\in\mathcal W_{i,t}}(\mathbf{1}^{\text{lead}}_r+\mathbf{1}^{\text{herd}}_r)},\qquad \mathcal W_{i,t}=\text{revisions with knowledge date in }(t-126,t]$$

LMH is only defined once the denominator reaches 3. The `leaders_only` and `herders_only` series are there for diagnosis and do not feed the composite.

**Composite.** Each signal becomes a cross-sectional percentile rank minus 0.5, which requires at least 5 valid names on the day. The score is then the plain average of whichever ranks are available, with no estimated weights.

*Some reasons.* Target changes have heavy tails. Using the sign of a revision instead of its size sidesteps them. A 126-session window spans two earnings cycles. The leader gap (10), the herd window (3–15) and the event size (5) were set from the literature on analyst behaviour and one pre-registered round on 2017–2019. None of them was ever tuned on the holdout or on the open universe.

## 3. Portfolio construction

**Legs.** On a session with $N_t$ valid scores, the long leg $L_t$ holds the top $k_t$ names and the short leg $S_t$ the bottom $k_t$, with either $k_t=4$ (fixed) or $k_t=\max(2,\lfloor 0.4N_t\rfloor)$ (leg fraction). If $N_t\ge 2k_t$ does not hold, the target for that day is flat.

**Allocation within each leg.** Weights are proportional to $1/\hat\sigma_{i,t}$, the one-step-ahead GJR-GARCH(1,1)-t volatility. Estimation is walk-forward per stock: first fit after 250 returns, refit every 21 sessions. Before the first accepted fit the forecast is the expanding standard deviation of past returns. A fit is accepted if the optimiser converged and the process is stationary; otherwise the previous estimate is kept. Remaining gaps get the 60-day realised volatility. `EW` gives equal weights.

**GARCH estimator and reproducibility.** By default (`garch_method="slsqp"`, which produced every published number) the package uses `arch`'s quasi-maximum likelihood with the SLSQP optimiser. With only a few hundred observations, the GJR-GARCH-t likelihood is almost flat along a ridge and frequently has two local maxima of nearly the same height. Where the optimiser stops, and therefore the forecast, can then vary from one machine to another (INSTALL.md). The alternative, `garch_method="grid"`, maximises the same likelihood over a fixed grid of $(\alpha, \gamma, \beta, \nu)$. Here $\mu$ is set to the sample mean, $\omega$ follows from variance targeting (Engle and Mezrich, 1996) and persistence is capped at 0.995. Because the log-likelihood at every grid point comes out of a single vectorised recursion and is rounded to $10^{-6}$ before taking the arg-max, every machine arrives at the same estimate. Both estimators lead to the same conclusions (net Sharpe 1.13 vs 1.12 over 2017–21, and 1.05 vs 1.09 out of sample).

**Leg budgets and beta budget.** For a constant net exposure *c*,

$$w^{\text{target}}_{i,t}=\Big(\tfrac12+\tfrac c2\Big)\frac{1/\hat\sigma_{i,t}}{\sum_{j\in L_t}1/\hat\sigma_{j,t}}\mathbf{1}_{i\in L_t}-\Big(\tfrac12-\tfrac c2\Big)\frac{1/\hat\sigma_{i,t}}{\sum_{j\in S_t}1/\hat\sigma_{j,t}}\mathbf{1}_{i\in S_t}$$

FMF-N sets *c* = 0 (dollar-neutral) and FMF-B20 sets *c* = 0.2. Either way *c* is a constant that we disclose, and we never time it, since timing net exposure is precisely how the first-generation strategy got its Calmar ratio.

**Tranches.** The book actually held is the average of the last *f* = 42 daily targets,

$$w_{i,t}=\frac1f\sum_{s=0}^{f-1}w^{\text{target}}_{i,t-s}$$

which is the same thing as holding 42 overlapping sub-portfolios and opening a new one each session (Jegadeesh–Titman). This lines the holding period up with the signal's information horizon (the IC climbs from 0.026 at 1 day to 0.109 at 21 days). It also takes the luck out of the rebalancing date. Rebalance the same signal once a week instead and the Sharpe lands anywhere between 0.34 and 0.71, depending on the weekday you happen to pick.

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

In every null the composite score is swapped for a counterfeit, which then goes through **exactly** the same legs, allocator, tranches and costs. The p-value is

$$\frac{1 + \sum \mathbf{1}_{\{\text{null} \ge \text{observed}\}}}{1 + \text{draws}}$$

| Null | Counterfeit score | What it preserves | Question it answers |
|---|---|---|---|
| IID | N(0, 1) on the real missing-value mask | coverage | is the result luck? |
| Stock-label permutation | each stock's real score history assigned to another stock (derangement) | persistence and cross-sectional structure of the signal | does the signal belong to *these* stocks? |
| Circular time-shift | the whole panel rolled by $\ge 126$ sessions | each stock's habitual rank | does the *timing* add value beyond stock identity? (the hardest null) |

## 6. Comparables, blends and spanning (`benchmarks`)

There are 13 cross-sectional comparables. Seven are price signals (12-1 and 6-1 momentum, 1-month reversal, 60-day low volatility, betting-against-beta in the form of low 252-day beta, residual momentum after Blitz–Huij–Martens, and proximity to the 52-week high). Four are analyst signals (consensus target-price upside, 63-day momentum in the consensus target, low dispersion, and classic revision breadth). The last two are blends, one of momentum + low vol + reversal and one of momentum + revisions.

All of them are traded with FMF's construction. The three allocations (equal-weight long-only, inverse-volatility long-only, and CTA-style 12-1 time-series momentum) are traded in their own native way.

For the blend test, each comparable's returns are mixed 50/50 with FMF-N's. For the spanning test, FMF-N's weekly returns are regressed on the market and every comparable, and then a second time leaving out the comparables built from FMF's own components.

## 7. Leave-one-out (`robustness.leave_one_out`)

Leaving out stock *j* means rebuilding the **entire** pipeline on the remaining universe, from cleaning and the calendar through the snapshot, signals, ranks, legs and weights to the P&L. Only the univariate GARCH forecasts are reused, since they do not depend on the other stocks.

## 8. Protocol

1. **Development** (2017–2019). All signal, parameter and construction choices were made in this window, in pre-registered rounds; the ones that failed are logged (paper, Table 2).
2. **Holdout** (2020–2021). Looked at only after the specification had been frozen.
3. **Never-seen period** (2023–2026, 30 stocks, of which twenty were never used in development and ten are the development names; DATA_SOURCES.md §3). Before anything else we wrote down the transfer rules: leg fraction 0.4 as the primary test and *k* = 4 as the secondary, plus the evaluation window and the tests themselves. That file (`configs/oos_open_universe_preregistration.json`, fingerprint `0b2b0792fd357d61`) was hashed before any return had been computed, and every evaluation script checks the fingerprint through `fmflow.prereg.verify`.
4. **Synthetic control.** In a null world the pipeline has to find no alpha, and when first movers are informed it has to find some (`scripts/synthetic_power_study.py`).

If you want to repeat the study on your own universe, freeze a configuration with `fmflow register`, run `fmflow evaluate --placebo --loo`, and report whatever it gives you.

## 9. The audit layer: small-sample and snooping-robust inference

| Question | Tool (module) | Definition |
|---|---|---|
| Identity or timing? | `analytics.attribution.lo_decomposition`, `static_timing_books` | Lo (2008): $E[\sum_i w_{i,t-1}r_{i,t}]=\sum_i\mathrm{Cov}(w,r)+\sum_i E[w]E[r]$. Causal version: static book $\bar w_{i,t}$ (expanding mean of past weights) and timing book $w_{i,t}-\bar w_{i,t}$, both tradable, summing to the strategy |
| Does any specification really work? | `analytics.snooping.spa_test`, `reality_check` | Hansen (2005) SPA with consistent recentring, White (2000) Reality Check, stationary bootstrap of the $T\times K$ matrix of performance differentials |
| Which comparables does it beat? | `analytics.snooping.romano_wolf` | Romano–Wolf (2005) StepM, studentised, step-down adjusted p-values (FWER) |
| Is a Sharpe difference real? | `analytics.snooping.sharpe_diff_test` | Circular block bootstrap of paired returns (Ledoit–Wolf 2008, basic bootstrap) |
| How much of the result is the specification? | `robustness.multiverse` | One-at-a-time, full-factorial or random joint draws over any `StrategyConfig` / `DataConfig` field, returns the family's daily returns for SPA, PBO and DSR, and the selection-free ensemble |
| What is the Sharpe ratio, given that most strategies fail? | `analytics.bayes.bayesian_sharpe` | Bayesian bootstrap (Rubin 1981) of weekly returns, importance-reweighted by a prior $SR\sim N(m,s^2)$ on the annual Sharpe ratio, flat, $N(0,1)$ and $N(0,0.5)$ priors reported |
| What do three samples say together, and about the next one? | `analytics.bayes.random_effects_meta` | $y_j\sim N(\theta_j,s_j^2)$, $\theta_j\sim N(\mu,\tau^2)$, $\tau\sim$ half-Cauchy, $\mu$ integrated analytically, $\tau$ on a grid, predictive $\theta_{new}\sim N(\mu,\tau^2)$ |
| What is the best of a search worth? | `analytics.bayes.shrink_normal_means` | Positive-part James–Stein shrinkage of the family's t-statistics towards their mean |
| Evidence for vs against an effect | `analytics.bayes.bayes_factor_normal` | $BF_{10}$ for $\theta>0$ with an $N(0,g^2)$ prior against $\theta=0$ |
| What could the sample detect? | `analytics.power.mde_ic`, `ic_power`, `simulate_power` | MDE $=(z_{1-\alpha}+z_{1-\beta})/\sqrt{(N-1)M}$, Monte Carlo power of the full pipeline on the calibrated generator |
| Is the Sharpe inflated by smoothing? | `analytics.metrics.lo_adjusted_sharpe` | $\eta(q)=q/\sqrt{q+2\sum_{k\le L}(q-k)\rho_k}$ (Lo 2002) |
| Crash beta? | `analytics.metrics.conditional_betas` | Weekly regression on $\max(m,0)$ and $\min(m,0)$ with HAC errors |
| Implementation? | `backtest.engine.cost_sensitivity`, `breakeven_cost_bps` | Grid over costs, borrow and execution delay, breakeven = mean(gross − borrow) / mean(traded) |
| Causality of a new signal? | `robustness.truncation_check` | Pipeline on data truncated at T must equal the full run up to T for every signal, the composite and the weights |

## 10. Generative stress testing (`fmflow.generative`, optional `[gen]` extra)

A single history holds one or two drawdowns at most. What we want to know is the *distribution* of a five-year Calmar ratio or maximum drawdown.

The input is the joint daily return series of the strategy and its risk factors (market and momentum). Each sample is standardised by its own mean and volatility, which lets us pool samples.

`baselines.block_bootstrap_paths` is a stationary bootstrap. `baselines.fhs_paths` does filtered historical simulation. It fits GJR-GARCH-t to each channel with variance targeting and persistence capped at 0.99, then resamples the empirical residual vectors jointly with zero mean.

`diffusion_fhs.DiffusionFHS` keeps the same GARCH dynamics, but each day's shock vector comes from a **conditional DDPM** $p(z_t\mid\text{log-vol state}_t)$. Shocks are mapped to normal scores first (a rank-based Gaussian quantile transform, as in TabDDPM). The model then learns how their dependence changes with the regime, which amounts to a conditional copula, and its draws go back through the empirical quantile function, with Student-t tails beyond the sample. The denoiser is a residual MLP with a cosine schedule (T = 100), an EMA of the weights and early stopping on a chronological validation split. Sampling runs a strided DDIM-type chain ($\eta = 1$). Each day the shocks are moment-matched across paths to zero mean and the empirical standard deviation, and volatility states are kept within the observed range widened by 0.5 standard deviations.

Two more models are there for comparison. `timegan.TimeGAN` follows Yoon et al. (2019), with embedding, supervised and joint phases, and `ddpm.DDPM` is a window-level diffusion model with PCA whitening and training-free conditional sampling.

Fidelity is checked in `fidelity` in four ways.
  * Stylised facts, namely moments, the ACF of $|r|$, cross-correlations and tail dependence.
  * A discriminative score; real windows are split in time order with a gap between the parts, since overlapping windows would leak otherwise.
  * A predictive score from training on synthetic data and testing on real data.
  * A memorisation ratio. With small data, a training loss lower than the held-out loss is memorisation, and you see it as draws that are not dispersed enough.

We set the drift. Generated paths have zero drift. The strategy's drift is then added explicitly, as zero, as the in-sample Sharpe, or as draws from the Bayesian posterior-predictive Sharpe. `stress.path_stats` combines path risk with parameter uncertainty. Finally, `stress.conditional_on` gives the strategy's statistics on just those simulated windows in which the market or momentum crashes.
