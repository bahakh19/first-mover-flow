# API reference

Generated from the docstrings of `fmflow` 2.2.4 by `scripts/make_api_docs.py`. Run `python tasks.py docs` after changing the code.

## `fmflow`

fmflow — First-Mover Flow: zero-beta stock selection from the flow and timing of analyst revisions.


## `fmflow.analytics.attribution`

Where does the P&L come from? Identity versus timing.

- **`def lo_decomposition(weights: pd.DataFrame, returns: pd.DataFrame, window: slice | None = None, periods: int = 252) -> dict`**
  Lo (2008) split of the mean annualised gross return into passive (average tilt) and active (timing) parts.
- **`def mean_with_hac(x: pd.Series, lags: int = 10, periods: int = 252) -> dict`**
  Annualised mean of a daily series with a Newey-West t-statistic.
- **`def static_timing_books(weights: pd.DataFrame, returns: pd.DataFrame, min_periods: int = 21) -> pd.DataFrame`**
  Daily returns of the causal static-tilt book, the timing book and the strategy (gross).
- **`def stock_contributions(weights: pd.DataFrame, returns: pd.DataFrame, window: slice | None = None, lags: int = 10, periods: int = 252) -> pd.DataFrame`**
  Per-stock annualised P&L contribution with Newey-West t-statistics, average weight and share of days held.

## `fmflow.analytics.bayes`

Bayesian inference for small samples.

- **`class Posterior(draws: np.ndarray) -> None`**
  Posterior draws of a scalar with HDI and threshold-probability summaries.
- **`def bayes_factor_normal(estimate: float, se: float, prior_sd: float, one_sided: bool = True) -> float`**
  BF10 for H1: theta ~ N(0, prior_sd^2) (truncated to theta > 0 if one_sided) vs H0: theta = 0.
- **`def bayesian_sharpe(r, prior_sr_sd: float | None = 1.0, prior_sr_mean: float = 0.0, aggregate: int = 5, periods: int = 252, n_draws: int = 20000, seed: int = 0, method: str = 'bayesian_bootstrap', burn: int = 5000) -> Posterior`**
  Posterior of the annualised Sharpe ratio (mean / sd of returns, the quantity investors are paid on).
- **`def random_effects_meta(estimates, ses, labels=None, tau_scale: float | None = None, prior_mu_sd: float | None = None, n_draws: int = 20000, seed: int = 0) -> dict`**
  Bayesian random-effects meta-analysis: y_j ~ N(theta_j, se_j^2), theta_j ~ N(mu, tau^2).
- **`def shrink_normal_means(t: pd.Series) -> pd.DataFrame`**
  Empirical-Bayes shrinkage of z/t-statistics t_k ~ N(theta_k, 1), theta_k ~ N(m, A).

## `fmflow.analytics.inference`

Statistical inference for small, overlapping, heavily searched samples.

- **`def bootstrap_calmar(r: np.ndarray, idx: np.ndarray, periods: int = 252) -> np.ndarray`**
  Calmar ratio of each bootstrap resample (rows of `idx`) of the return series `r`.
- **`def cscv_pbo(trials: pd.DataFrame, n_blocks: int = 16) -> dict`**
  Probability of backtest overfitting by combinatorially symmetric cross-validation (Bailey et al., 2017).
- **`def deflated_sharpe(r, trials: pd.DataFrame | None = None, n_eff: float | None = None, sr_std: float | None = None) -> dict`**
  DSR of return series `r` given the matrix of all trial return series (columns) or (n_eff, sr_std).
- **`def expected_max_sr(sr_std: float, n_eff: float) -> float`**
  Expected maximum of n_eff independent null Sharpe ratios with cross-trial std sr_std (same units).
- **`def fama_macbeth(feature: pd.DataFrame, controls: dict, fwd: pd.DataFrame, step: int = 5, min_names: int = 6, lags: int | None = None, horizon: int = 21) -> pd.DataFrame`**
  Fama-MacBeth regressions of forward returns on the cross-sectional RANK of the feature and of each control (ranks centred to mean zero each date), every `step` sessions. Returns mean slopes with Newey-West t-statistics.
- **`def forward_log_return(px: pd.DataFrame, h: int) -> pd.DataFrame`**
  LABEL (forward-looking by design): log return from close t to close t+h. Never use as a feature.
- **`def holm(p: pd.Series) -> pd.Series`**
  Holm step-down adjusted p-values (family-wise error control).
- **`def ic_phase_test(ic: pd.Series, h: int = 21) -> dict`**
  One-sided test of a mean IC whose h-period labels overlap: average of the h non-overlapping sub-sample standard errors.
- **`def memmel_test(r1: pd.Series, r2: pd.Series) -> tuple[float, float]`**
  Jobson-Korkie test of equal Sharpe ratios with Memmel's (2003) correction; returns (z, two-sided p).
- **`def min_track_record(r, sr_star: float = 0.0, alpha: float = 0.05, periods: int = 252) -> float`**
  Minimum track record length (years) for the observed Sharpe ratio to exceed `sr_star` at level `alpha`.
- **`def psr(r, sr_star: float = 0.0) -> float`**
  Probabilistic Sharpe ratio P(true SR > sr_star); sr_star in per-period (daily) units.
- **`def rank_ic(feature: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 6) -> pd.Series`**
  Daily cross-sectional Spearman correlation between a feature and forward returns (NaN below `min_names`).
- **`def stationary_bootstrap_indices(n: int, n_boot: int = 1000, block: int = 21, seed: int = 42) -> np.ndarray`**
  Index matrix (n_boot x n) of Politis-Romano stationary-bootstrap resamples with mean block `block`.

## `fmflow.analytics.metrics`

Performance and attribution metrics.

- **`def capm(strategy: pd.Series, market: pd.Series, hac_lags: int = 4) -> dict`**
  Weekly (non-overlapping 5-session sums) regression on the market with HAC standard errors.
- **`def conditional_betas(strategy: pd.Series, market: pd.Series, hac_lags: int = 4) -> dict`**
  Weekly up-market and down-market betas (Henriksson-Merton style split) and the downside correlation.
- **`def drawdown(returns: pd.Series) -> pd.Series`**
  Drawdown path (equity / running peak - 1) of a daily return series.
- **`def factor_attribution(strategy: pd.Series, factors: Mapping[str, pd.Series], hac_lags: int = 4) -> dict`**
  Weekly regression of the strategy on several factor-return series; returns alpha, t, betas, R2.
- **`def lo_adjusted_sharpe(returns: pd.Series, periods: int = 252, max_lag: int = 10) -> dict`**
  Annualised Sharpe ratio corrected for serial correlation (Lo, 2002).
- **`def perf_stats(returns: pd.Series, periods: int = 252) -> pd.Series`**
  CAGR, volatility, Sharpe, Sortino (downside deviation, target 0), max drawdown, Calmar (CAGR/|MaxDD|), tails, hit rate. NaN-only or single-observation inputs give an empty Series.
- **`def stress_table(books: Mapping[str, pd.Series], episodes: Mapping[str, tuple]) -> pd.DataFrame`**
  Cumulative return of each book over named episodes {name: (start, end)}.

## `fmflow.analytics.placebo`

Placebo nulls passed through the IDENTICAL construction.

- **`def placebo_test(score: pd.DataFrame, book_returns: Callable[[pd.DataFrame], pd.Series], windows: Mapping[str, slice], metric: Callable[[pd.Series], float], n_iid: int = 300, n_perm: int = 500, shift_step: int = 4, min_shift: int = 126, seed: int = 42) -> pd.DataFrame`**
  Observed statistic vs IID, stock-permutation and circular time-shift null scores passed through `book_returns`.

## `fmflow.analytics.power`

Statistical power in small cross-sections: how large an edge can this sample detect, and how long must one wait?

- **`def ic_power(true_ic: float, n_stocks: int, n_obs: int, alpha: float = 0.05) -> float`**
  Power of the one-sided IC test for a true mean IC with `n_stocks` names and `n_obs` independent dates.
- **`def mde_ic(n_stocks: int, n_obs: int, alpha: float = 0.05, power: float = 0.8) -> float`**
  Smallest true mean IC detectable with `power`: (z_a + z_b) / (sqrt(N - 1) * sqrt(M)), M independent dates.
- **`def simulate_power(n_stocks=(10, 30, 100), years=(3, 5, 10), strengths=(0.0, 0.2, 0.4), seeds: int = 10, allocator: str = 'EW', alpha: float = 0.05) -> pd.DataFrame`**
  Share of synthetic worlds where the one-sided Sharpe test (PSR > 1 - alpha) rejects. strength = 0 gives size.

## `fmflow.analytics.snooping`

Data-snooping-robust inference for families of strategies (all bootstrap-based, dependence-preserving).

- **`def reality_check(d, n_boot: int = 2000, block: int = 21, seed: int = 0) -> dict`**
  White's (2000) Reality Check p-value for the best of K candidates against the benchmark.
- **`def romano_wolf(d, n_boot: int = 2000, block: int = 21, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame`**
  StepM with studentised statistics. H0_k: E[d_k] <= 0. Returns t-stats, raw and FWER-adjusted p-values.
- **`def sharpe_diff_test(r1: pd.Series, r2: pd.Series, n_boot: int = 2000, block: int = 21, seed: int = 0, periods: int = 252) -> dict`**
  Two-sided test of SR(r1) = SR(r2) by circular block bootstrap of the PAIRS (keeps their correlation).
- **`def spa_test(d, n_boot: int = 2000, block: int = 21, seed: int = 0) -> dict`**
  Hansen's SPA test. Returns p-values for the consistent (c), lower (l) and upper (u) recentrings.

## `fmflow.backtest.engine`

Vectorised daily backtest with linear transaction costs, short-borrow fees and an optional execution delay.

- **`class BacktestResult(weights: pd.DataFrame, gross: pd.Series, costs: pd.Series, net: pd.Series, traded: pd.Series, borrow: pd.Series | None = None) -> None`**
  Held weights, gross / cost / borrow / net daily returns and traded notional of a backtest.
- **`def annual_turnover(traded: pd.Series, periods: int = 252) -> float`**
  One-way turnover per year (half the traded notional).
- **`def backtest(weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = 5.0, borrow_bps: float = 0.0, execution_lag: int = 0) -> BacktestResult`**
  Run the daily backtest: held = weights lagged by `execution_lag`; net = gross - costs - borrow.
- **`def breakeven_cost_bps(result: BacktestResult, start=None, end=None) -> float`**
  Cost per unit traded (bps) at which the mean net return would be zero:  mean(gross - borrow) / mean(traded).
- **`def cost_sensitivity(weights: pd.DataFrame, returns: pd.DataFrame, cost_grid=(0, 2.5, 5, 10, 20, 30, 50), borrow_grid=(0, 25, 50, 100), execution_lags=(0, 1), window: slice | None = None) -> pd.DataFrame`**
  Annualised net Sharpe over a grid of trading costs, borrow fees and execution delays.

## `fmflow.benchmarks`

Institutional comparables: the signals and allocations most widely used by equity and QIS desks, built only from prices and analyst targets so they can be compared with FMF-N on any dataset that satisfies the data contract.

- **`def comparable_books(out: StrategyOutput, include_allocations: bool = True) -> dict[str, pd.Series]`**
  Net daily returns of every comparable, same construction as `out.config`.
- **`def comparable_signals(px: pd.DataFrame, snap) -> dict[str, pd.DataFrame]`**
  The 13 institutional cross-sectional signals, oriented higher = more attractive.

## `fmflow.cache`

On-disk cache for the GJR-GARCH forecasts (the only slow step of the pipeline: ~30 s for 10 stocks x 6 years).

- **`def cached_garch(ret: pd.DataFrame, cfg: DataConfig = DataConfig(), cache_dir: str | Path = '.fmflow_cache') -> pd.DataFrame`**
  GJR-GARCH forecasts for `ret`, read from or written to a content-addressed on-disk cache.

## `fmflow.config`

Configuration objects. Every tunable number of the method lives here, nowhere else.

- **`class DataConfig(annualization: int = 252, min_calendar_coverage: float = 0.5, calendar_active_days: int = 30, price_ffill_limit: int = 5, knowledge_lag: int = 1, tp_expiry_days: int = 252, tp_ratio_bounds: tuple[float, float] = (0.2, 5.0), revision_excess_bounds: tuple[float, float] = (0.3333333333333333, 3.0), event_min_ratings: int = 5, min_analysts: int = 3, garch_min_obs: int = 250, garch_refit: int = 21, garch_method: str = 'slsqp') -> None`**
  Ingestion and point-in-time rules (identical for every dataset; never tuned on returns).
- **`class StrategyConfig(name: str = 'FMF-N', signals: tuple[str, ...] = ('breadth_63', 'event_breadth_63', 'breadth_resid_mom', 'leader_minus_herd'), leader_gap: int = 10, herd_lo: int = 3, herd_hi: int = 15, lmh_window: int = 126, lmh_min_n: int = 3, breadth_window: int = 63, breadth_min_n: int = 3, event_breadth_min_n: int = 2, rank_min_names: int = 5, k: int | None = 4, leg_fraction: float | None = None, k_min: int = 2, tranche_period: int = 42, allocator: str = 'IV-GJR', net_exposure: float = 0.0, neutralize: str = 'dollar', beta_window: int = 252, max_weight: float | None = None, vol_target: float | None = None, vol_lookback: int = 126, max_leverage: float = 2.0, cost_bps: float = 5.0, borrow_bps: float = 0.0, execution_lag: int = 0, data: DataConfig = <factory>) -> None`**
  Signal, construction and cost parameters of a First-Mover Flow book.

## `fmflow.data.cleaning`

Causal cleaning of the raw price and rating tables.

- **`class AuditLog() -> None`**
  Append-only log of cleaning actions (exported to the research report).
- **`def clean_prices(prices_raw: pd.DataFrame, cfg: DataConfig = DataConfig(), log: AuditLog | None = None) -> pd.DataFrame`**
  Long price table -> clean (session x stock) panel on the inferred trading calendar.
- **`def clean_ratings(ratings_raw: pd.DataFrame, px: pd.DataFrame, cfg: DataConfig = DataConfig(), log: AuditLog | None = None) -> pd.DataFrame`**
  Long ratings table -> one row per accepted target, stamped with its point-in-time knowledge date.
- **`def natural_sort_key(name: str)`**
  Sort 'stock_2' before 'stock_10'.
- **`def to_calendar_dates(values) -> pd.Series`**
  Parse dates and reduce them to timezone-naive calendar days.
- **`def to_float(values) -> pd.Series`**
  Numbers as plain float64, NaN for anything unparseable, whatever the input dtype (text, pandas' nullable Float64 / Int64 / string, pyarrow-backed columns): nullable dtypes would otherwise carry pd.NA into the pipeline.
- **`def to_long(panel: pd.DataFrame, value_name: str) -> pd.DataFrame`**
  Wide (session x stock) panel to a long (date, stock, value) table.

## `fmflow.data.io`

Loaders that apply the data contract.

- **`def load_prices(path: str | Path) -> pd.DataFrame`**
  Load a long price table (date, stock, price) and validate it.
- **`def load_ratings(path: str | Path, prices: pd.DataFrame | None = None) -> pd.DataFrame`**
  Load a long ratings table (date, estimate_value, stock, analyst); values are kept as text until cleaning.
- **`def read_table(path: str | Path, **kw) -> pd.DataFrame`**
  Read a CSV, Parquet or Excel file into a DataFrame (dialect detected, see the module docstring).
- **`def sniff_csv(path: str | Path) -> dict`**
  Separator and decimal mark of a CSV file: {"sep": ",", ";" or "\t", "decimal": "." or ","}.

## `fmflow.data.normalize`

Turn a raw (ticker, broker, dollar) universe into the anonymised, normalised format of `data/open/`.

- **`class NormalizedUniverse(prices: pd.DataFrame, ratings: pd.DataFrame, stock_map: pd.Series, analyst_map: pd.Series, base_price: pd.Series) -> None`**
  Normalised tables plus the private keys needed to map them back (never ship the keys).
- **`def compare_tables(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], value: str, tol: float = 0.00015) -> dict`**
  Row-set and value agreement of two long tables (order-insensitive). Used to check a rebuilt dataset against the shipped one: `tol` absorbs the last-digit differences of 4-decimal rounding.
- **`def normalize_universe(prices: pd.DataFrame, ratings: pd.DataFrame, decimals: int | None = 4, stock_prefix: str = 'stock_', analyst_prefix: str = 'analyst_') -> NormalizedUniverse`**
  Anonymise identifiers and divide each stock's prices and targets by its first price.

## `fmflow.data.schema`

The input data contract (see DATA.md for the full outline).

- **`class DataContractError(...)`**
  Raised when an input table does not satisfy the documented contract.
- **`def normalise_columns(df: pd.DataFrame) -> pd.DataFrame`**
  Copy of `df` with column names stripped and lower-cased (' Date ' -> 'date').
- **`def validate_prices(prices: pd.DataFrame) -> pd.DataFrame`**
  Check the price table; returns a column-normalised copy.
- **`def validate_ratings(ratings: pd.DataFrame, prices: pd.DataFrame | None = None) -> pd.DataFrame`**
  Check the ratings table (and, if prices are given, that targets are on the price scale).

## `fmflow.data.snapshot`

Point-in-time snapshot of analyst targets on the trading calendar.

- **`class TPSnapshot(tp: pd.DataFrame, age: pd.DataFrame, n_active: pd.DataFrame, cons_median: pd.DataFrame, cons_mean: pd.DataFrame, dispersion: pd.DataFrame, n_up: pd.DataFrame, n_down: pd.DataFrame, n_reit: pd.DataFrame, n_init: pd.DataFrame, rev_sum: pd.DataFrame, events: pd.DataFrame) -> None`**
  Point-in-time analyst state on the trading calendar (targets, ages, consensus, revision arrivals).
- **`def build_tp_snapshot(rt: pd.DataFrame, sessions: pd.DatetimeIndex, stocks: Sequence[str], cfg: DataConfig = DataConfig()) -> TPSnapshot`**
  Build the point-in-time snapshot from knowledge-date-stamped ratings (targets expire after tp_expiry_days).
- **`def classify_ratings(rt: pd.DataFrame, sessions: pd.DatetimeIndex, cfg: DataConfig = DataConfig()) -> pd.DataFrame`**
  Add session positions and the fresh-revision / reiteration / (re-)initiation classification.

## `fmflow.generative`

Generative scenario models for stress-testing a strategy's path risk (optional extra: pip install "fmflow[gen]").

- **`def jax_available() -> bool`**
  True if the JAX backend can run on this machine (see `jax_status`).

## `fmflow.generative.baselines`

Classical path generators: stationary block bootstrap and filtered historical simulation (FHS).

- **`def block_bootstrap_paths(returns: pd.DataFrame, n_paths: int, horizon: int, block: int = 21, seed: int = 0) -> np.ndarray`**
  Stationary bootstrap (Politis-Romano) of the joint daily return vectors: (n_paths, horizon, C).
- **`def fhs_paths(returns: pd.DataFrame, n_paths: int, horizon: int, seed: int = 0, max_persistence: float = 0.99) -> np.ndarray`**
  Filtered historical simulation (Barone-Adesi, Giannopoulos and Vosper, 1999).

## `fmflow.generative.ddpm`

Denoising diffusion probabilistic model for multichannel return windows (JAX, CPU-friendly).

- **`class DDPM(length: int, channels: int, T: int = 200, hidden: int = 256, n_blocks: int = 2, temb_dim: int = 64, seed: int = 0, whiten: bool = True, cond_dim: int = 0, params: dict = None, history: list = <factory>) -> None`**
  Denoising diffusion probabilistic model with a residual-MLP denoiser, optional whitening and conditioning.
- **`def cosine_schedule(T: int, s: float = 0.008) -> np.ndarray`**
  Cosine noise schedule (Nichol and Dhariwal, 2021): beta_1..beta_T.

## `fmflow.generative.diffusion_fhs`

Diffusion-FHS: GJR-GARCH volatility dynamics with innovations drawn from a CONDITIONAL diffusion model.

- **`class DiffusionFHS(channels: list, T: int = 100, hidden: int = 128, n_blocks: int = 3, seed: int = 0, max_persistence: float = 0.99, garch: dict = <factory>, model: DDPM = None) -> None`**
  GJR-GARCH dynamics with shocks from a conditional diffusion model (see module docstring).

## `fmflow.generative.fidelity`

Is a synthetic return generator any good? Fidelity, usefulness and memorisation diagnostics.

- **`def discriminative_score(real: np.ndarray, synth: np.ndarray, seed: int = 0, test_share: float = 0.3) -> float`**
  |accuracy - 0.5| of a classifier separating real from synthetic windows.
- **`def drawdown_profile(W: np.ndarray, channel: int = 0) -> pd.Series`**
  Mean, 5% and 1% quantiles of the within-window maximum drawdown of a channel.
- **`def max_drawdowns(W: np.ndarray, channel: int = 0) -> np.ndarray`**
  Within-window maximum drawdown of a channel for every window.
- **`def memorisation_ratio(train: np.ndarray, holdout: np.ndarray, synth: np.ndarray, max_n: int = 2000, seed: int = 0) -> float`**
  Median nearest-training-window distance of synthetic vs held-out real windows (<< 1 = copying).
- **`def predictive_score(real_train: np.ndarray, real_test: np.ndarray, synth: np.ndarray, lags: int = 5) -> dict`**
  Train-on-synthetic, test-on-real MAE of a one-step |return| forecaster, relative to train-on-real.
- **`def stylised_facts(W: np.ndarray, names: list | None = None) -> pd.Series`**
  Moments, autocorrelations, cross-correlations and tail dependence of a set of windows.

## `fmflow.generative.stress`

Path-level stress statistics from simulated return paths (n_paths, horizon, C), channel 0 = the strategy.

- **`def conditional_on(paths: np.ndarray, condition: np.ndarray, channel: int = 0) -> pd.DataFrame`**
  Strategy path statistics on the subset of simulated paths where `condition` (boolean, n_paths) holds.
- **`def market_drawdown(paths: np.ndarray, channel: int = 1) -> np.ndarray`**
  Maximum drawdown of a channel (default: the market) along each simulated path.
- **`def path_stats(paths: np.ndarray, channel: int = 0, periods: int = 252) -> pd.DataFrame`**
  CAGR, max drawdown, Calmar, Sharpe and total return of each simulated path.
- **`def summarise(stats_: pd.DataFrame, q=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame`**
  Quantiles of path statistics and the probability that each is negative.

## `fmflow.generative.timegan`

TimeGAN (Yoon, Jarrett and van der Schaar, NeurIPS 2019), compact JAX implementation.

- **`class TimeGAN(length: int, channels: int, hidden: int = 24, z_dim: int | None = None, gamma: float = 1.0, seed: int = 0, params: dict = None, history: list = <factory>) -> None`**
  TimeGAN with GRU embedder, recovery, generator, supervisor and discriminator (see module docstring).

## `fmflow.generative.windows`

Training windows for generative models.

- **`class GarchScaled(samples: list[pd.DataFrame], channels: list | None = None)`**
  Econometric-deep hybrid representation: r_t = sigma_t * z_t per channel, with sigma_t the in-sample GJR-GARCH(1,1)-t conditional volatility. Each day is described by (z_t, standardised log sigma_t) for every channel, so a window of length L becomes an (L, 2C) array. The generator then learns the persistent volatility path (clustering) and the innovations jointly -- including their cross-channel dependence and any nonlinearity the GARCH model misses -- and the product sigma_t * z_t restores fat tails and volatility clustering by construction.
- **`class VolScaled(X: np.ndarray)`**
  Two-scale representation for generative models: each window is split into its per-channel realised-volatility level (log scale, standardised across windows) and the within-window returns divided by that level. The two are stacked as (L + 1, C): row 0 holds the standardised log-volatilities, rows 1..L the vol-normalised returns.
- **`class WindowData(X: np.ndarray, mean: np.ndarray, std: np.ndarray, channels: list, length: int) -> None`**
  Standardised training windows and the statistics needed to map draws back to returns.
- **`def make_windows(samples: list[pd.DataFrame] | pd.DataFrame, length: int = 64, stride: int = 1, channels: list | None = None) -> WindowData`**
  Overlapping windows from one or several (T x C) daily-return frames with identical columns.
- **`def stitch(windows: np.ndarray, n_paths: int, horizon: int, rng: np.random.Generator) -> np.ndarray`**
  Concatenate randomly drawn generated windows into (n_paths, horizon, C) paths.

## `fmflow.index`

QIS index calculation: turn a strategy into a rules-based index level series with fees, replication costs, optional volatility targeting and an excess-return / total-return distinction, plus a factsheet.

- **`class IndexRules(name: str = 'FMF-N Excess Return Index', base_date: str | None = None, base_level: float = 100.0, fee_bps: float = 0.0, replication_cost_bps: float = 5.0, borrow_bps: float = 0.0, vol_target: float | None = None, vol_lookback: int = 63, max_leverage: float = 1.5, rebalance_band: float = 0.0, cash_rate: pd.Series | None = None) -> None`**
  Parameters of a rules-based index built on a strategy (fees, costs, volatility target, cash rate).
- **`def compute_index(out, rules: IndexRules) -> pd.DataFrame`**
  Index level series (excess return, optional total return) with leverage, costs and fees.
- **`def index_factsheet(level: pd.Series) -> pd.DataFrame`**
  Calendar-year and full-period return, volatility, Sharpe, drawdown (and Calmar) of an index level.
- **`def rulebook_markdown(rules: IndexRules, cfg) -> str`**
  A rulebook stub generated from the exact parameters (to be completed with legal / calculation-agent text).

## `fmflow.pipeline`

End-to-end First-Mover Flow pipeline: raw tables -> clean panels -> signals -> composite -> weights -> P&L.

- **`class StrategyOutput(config: StrategyConfig, prices: pd.DataFrame, returns: pd.DataFrame, ratings: pd.DataFrame, snapshot: TPSnapshot, signals: dict, score: pd.DataFrame, sig_vol: pd.DataFrame | None, target: pd.DataFrame, weights: pd.DataFrame, result: BacktestResult, audit: pd.DataFrame = <factory>) -> None`**
  Everything a run produces: clean data, snapshot, signals, score, risk forecasts, weights and P&L.
- **`def build_signal_panels(px: pd.DataFrame, snap: TPSnapshot, cfg: StrategyConfig, extra: dict | None = None) -> dict`**
  The four FMF signals, the leader / herder diagnostics, and any other signal named in cfg.signals.
- **`def construct_book(score: pd.DataFrame, ret: pd.DataFrame, cfg: StrategyConfig, sig_vol: pd.DataFrame | None = None, net_exposure: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]`**
  Score -> (daily target book, held book): legs, allocator, beta neutralisation, position cap, tranches and volatility target exactly as configured. `net_exposure` overrides `cfg.net_exposure` (e.g. for a beta budget).
- **`def prepare_data(prices_raw: pd.DataFrame, ratings_raw: pd.DataFrame, cfg: StrategyConfig)`**
  Validate and clean both tables and build the point-in-time snapshot (returns px, ratings, snapshot, log).
- **`def run_strategy(prices_raw: pd.DataFrame | str | Path, ratings_raw: pd.DataFrame | str | Path, cfg: StrategyConfig | None = None, sig_vol: pd.DataFrame | None = None, sectors: dict | None = None, extra: dict | None = None, cache_dir: str | Path | None = None) -> StrategyOutput`**
  Run the full pipeline from the two raw tables (DataFrames, or paths to CSV / Excel / Parquet files).

## `fmflow.plotting`

Matplotlib helpers (optional dependency: pip install fmflow[plot]).

- **`def equity_curves(books: Mapping[str, pd.Series], window: slice | None = None, title: str = '', path: str | None = None)`**
  Growth-of-one chart of several return series over a window (matplotlib).

## `fmflow.portfolio.construction`

Portfolio construction.

- **`def beta_neutral_budgets(target: pd.DataFrame, betas: pd.DataFrame, gross: float = 1.0, floor: float = 0.2) -> pd.DataFrame`**
  Rescale long and short legs so that b_L * beta_L = b_S * beta_S with b_L + b_S = gross (target book beta = 0).
- **`def cap_weights(target: pd.DataFrame, max_weight: float) -> pd.DataFrame`**
  Cap |w_i| at `max_weight`, redistributing the excess pro rata over the uncapped names of the same leg.
- **`def ex_ante_vol(weights: pd.DataFrame, ret: pd.DataFrame, lookback: int = 126, periods: int = 252, every: int = 1) -> pd.Series`**
  Annualised ex-ante volatility of the book held at t, from returns up to t (Ledoit-Wolf covariance).
- **`def ledoit_wolf_cov(x: np.ndarray) -> np.ndarray`**
  Ledoit-Wolf (2004) shrinkage of the sample covariance towards a scaled identity.
- **`def rolling_betas(ret: pd.DataFrame, window: int = 252, min_periods: int = 126) -> pd.DataFrame`**
  Causal beta of each stock to the equal-weight market over the trailing `window` sessions (known at t). `min_periods` is capped at `window`, so a short window works without adjusting it.
- **`def sector_demean(score: pd.DataFrame, sectors: dict) -> pd.DataFrame`**
  Subtract the within-sector cross-sectional mean of the score (names without a sector are left unchanged).
- **`def target_weights(score: pd.DataFrame, k: int | None = 4, leg_fraction: float | None = None, k_min: int = 2, allocator: str = 'EW', sig_vol: pd.DataFrame | None = None, ret: pd.DataFrame | None = None, net_exposure: float = 0.0) -> pd.DataFrame`**
  Daily target weights (sessions x stocks). Rows with too few valid names are flat (all zeros).
- **`def tranche_weights(target: pd.DataFrame, period: int) -> pd.DataFrame`**
  Average of the last `period` daily targets (staggered overlapping tranches).
- **`def vol_target_weights(weights: pd.DataFrame, ret: pd.DataFrame, target: float, lookback: int = 126, max_leverage: float = 2.0, every: int = 5) -> tuple[pd.DataFrame, pd.Series]`**
  Scale the held book to `target` annualised ex-ante volatility (scale known at t, capped at max_leverage).

## `fmflow.prereg`

Pre-registration: freeze a test specification to disk and fingerprint it BEFORE any result is computed.

- **`class PreregistrationError(...)`**
  The pre-registration file was edited after it was hashed.
- **`def fingerprint(doc: dict) -> str`**
  First 16 hex characters of the SHA-256 of the canonical JSON body (stamp fields excluded).
- **`def register(doc: dict, path: str | Path, overwrite: bool = False) -> dict`**
  Stamp `doc` with its fingerprint and the UTC time, and write it to `path` (refuses to overwrite by default).
- **`def verify(path: str | Path) -> dict`**
  Load a registration and check that its body still matches the stored fingerprint.

## `fmflow.production`

Daily production workflow for a portfolio manager: today's book, the trades to get there, and the risk and health checks to run before sending orders.

- **`def latest_book(out, date=None) -> pd.DataFrame`**
  Per-stock view of the book on `date` (default: the last session).
- **`def risk_report(out, date=None, lookback: int = 126) -> pd.Series`**
  Exposures, ex-ante volatility and beta, concentration and data freshness of the book on `date`.
- **`def signal_monitor(out, h: int = 21, window: int = 126, signals=('composite', 'leader_minus_herd')) -> pd.DataFrame`**
  Rolling mean of the REALISED rank IC (labels end h sessions before each date, so the series is known at t).
- **`def trade_list(target_weights: pd.Series, current_shares: pd.Series | None, prices: pd.Series, nav: float, lot_size: int = 1, min_trade_notional: float = 0.0) -> pd.DataFrame`**
  Orders that move the current holdings to `target_weights` x `nav`, rounded to lots.

## `fmflow.report`

Tabular reports used by the CLI, the scripts and the notebooks.

- **`def default_windows(index: pd.DatetimeIndex, warmup: int = 252) -> dict[str, slice]`**
  Whole sample after a warm-up, and its two halves (the warm-up shrinks to a quarter of short samples).
- **`def strategy_report(out, windows: Mapping[str, slice]) -> pd.DataFrame`**
  Headline statistics of a StrategyOutput over named windows.
- **`def summary_table(books: Mapping[str, pd.Series], windows: Mapping[str, slice], market: pd.Series, reference: pd.Series | None = None) -> pd.DataFrame`**
  Performance, CAPM and PSR statistics of several books over several windows.

## `fmflow.risk.garch`

Walk-forward GJR-GARCH(1,1) with Student-t innovations (Glosten, Jagannathan and Runkle, 1993).

- **`def fit_quietly(model, **kw)`**
  `model.fit` for an `arch` model without console noise.
- **`def garch_forecast_panel(ret: pd.DataFrame, cfg: DataConfig = DataConfig()) -> tuple[pd.DataFrame, pd.DataFrame]`**
  Walk-forward one-step GJR-GARCH(1,1)-t volatility forecasts for every column (and the fit log).
- **`def gjr_variance_path(x: np.ndarray, mu: float, omega: float, alpha: float, gamma: float, beta: float, sigma2_0: float) -> np.ndarray`**
  GJR-GARCH(1,1) conditional variance recursion for given parameters.
- **`def grid_fit(x: np.ndarray) -> tuple[float, float, float, float, float, float] | None`**
  Deterministic GJR-GARCH(1,1)-t estimate with variance targeting: (mu, omega, alpha, gamma, beta, nu).

## `fmflow.robustness`

The validation battery of the paper, as reusable functions that work on any dataset in the data contract.

- **`def book_from_score(score: pd.DataFrame, out: StrategyOutput, net_exposure: float | None = None) -> pd.Series`**
  Net returns of `score` traded with exactly the construction of `out.config` (legs, allocator, neutralisation, position cap, tranches, volatility target, costs, borrow and execution delay).
- **`def leave_one_out(prices_raw: pd.DataFrame, ratings_raw: pd.DataFrame, cfg: StrategyConfig, windows: Mapping[str, slice], sig_vol: pd.DataFrame | None = None) -> pd.DataFrame`**
  Sharpe / Calmar / max drawdown of the full pipeline rebuilt on the universe without each stock. Pass `sig_vol` (univariate GARCH forecasts, unaffected by dropping a stock) to avoid re-estimating it.
- **`def multiverse(prices: pd.DataFrame, ratings: pd.DataFrame, base: StrategyConfig, grid: Mapping[str, list], windows: Mapping[str, slice], sig_vol: pd.DataFrame | None = None, mode: str = 'oat', n_random: int = 200, seed: int = 0, n_jobs: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]`**
  Run the strategy over a family of reasonable specifications (Steegen et al. 2016; Simonsohn et al. 2020).
- **`def signal_ic_table(out: StrategyOutput, h: int = 21, window: slice | None = None, min_names: int = 6) -> pd.DataFrame`**
  IC_h of each composite input, the composite and the LMH parts. Labels must END inside `window` (no leakage across a split): the last h sessions of the window are dropped.
- **`def strategy_placebo(out: StrategyOutput, windows: Mapping[str, slice], n_iid: int = 300, n_perm: int = 500, shift_step: int = 4, min_shift: int = 126, metric: Callable[[pd.Series], float] = <function _sharpe at 0x7f69de2745e0>, seed: int = 42) -> pd.DataFrame`**
  Placebo p-values of the strategy's composite under three nulls (see fmflow.analytics.placebo).
- **`def truncation_check(prices_raw: pd.DataFrame, ratings_raw: pd.DataFrame, cfg: StrategyConfig, cut_dates, sig_vol: pd.DataFrame | None = None, extra: dict | None = None, atol: float = 1e-10) -> pd.DataFrame`**
  Recompute every signal, the composite and the weights on raw data truncated at each cut date T and compare with the full-sample values up to T. Any difference is look-ahead. Ratings are truncated by PUBLICATION date (what a live system would have received); `extra` user panels are truncated by index.

## `fmflow.signals.analyst`

Analyst-data signals in common institutional use (comparables for FMF-N).

- **`def analyst_signals(px: pd.DataFrame, snap: TPSnapshot) -> dict[str, pd.DataFrame]`**
  Institutional analyst-data comparables: consensus upside, target-price momentum, low dispersion.

## `fmflow.signals.breadth`

Revision-breadth family.

- **`def breadth(snap: TPSnapshot, window: int = 63, min_n: int = 3) -> pd.DataFrame`**
  Revision breadth: (up - down) / (up + down) fresh revisions over `window` sessions.
- **`def breadth_resid_mom(breadth_panel: pd.DataFrame, px: pd.DataFrame, min_names: int = 5) -> pd.DataFrame`**
  Breadth rank orthogonalised to 6-1 momentum rank cross-sectionally each day.
- **`def event_breadth(snap: TPSnapshot, sessions: pd.DatetimeIndex, stocks, window: int = 63, min_n: int = 2) -> pd.DataFrame`**
  Breadth computed only from revisions published on information-event days.
- **`def momentum_6_1(px: pd.DataFrame) -> pd.DataFrame`**
  Log price change from t-126 to t-21.

## `fmflow.signals.composite`

Cross-sectional ranks and the equal-weight rank composite (no estimated weights).

- **`def centred_rank(panel: pd.DataFrame, min_names: int = 5) -> pd.DataFrame`**
  Cross-sectional rank demeaned date by date (sums to exactly zero).
- **`def composite_score(panels: Mapping[str, pd.DataFrame], names: Sequence[str], index: pd.Index | None = None, min_names: int = 5) -> pd.DataFrame`**
  Equal-weight average of the available cross-sectional ranks of `names` (NaN-aware).
- **`def cs_rank(panel: pd.DataFrame, min_names: int = 5) -> pd.DataFrame`**
  Date-by-date percentile rank minus 0.5 (NaN on dates with fewer than `min_names` valid names).

## `fmflow.signals.leader_herd`

Leader-minus-herd (LMH): the arrival-time asymmetry of analyst revisions.

- **`def leader_minus_herd(snap: TPSnapshot, sessions: pd.DatetimeIndex, stocks, leader_gap: int = 10, herd_lo: int = 3, herd_hi: int = 15, window: int = 126, min_n: int = 3, return_parts: bool = False)`**
  Net direction of leader minus herder revisions over `window` sessions (see module docstring).

## `fmflow.signals.price`

Price-based signals: the four controls used in all attribution tests and the institutional comparables.

- **`def price_signals(px: pd.DataFrame) -> dict[str, pd.DataFrame]`**
  Price-based controls and comparables (momentum, reversal, low vol, 52-week high, low beta, residual momentum).

## `fmflow.signals.registry`

Signal registry: every signal the pipeline can rank is a named builder `fn(ctx) -> (sessions x stocks) panel`.

- **`class SignalContext(px: pd.DataFrame, snap: object, cfg: object, extra: dict = <factory>, cache: dict = <factory>) -> None`**
  Inputs available to a signal builder at run time (prices, snapshot, config, user data, memo cache).
- **`def available_signals(family: str | None = None) -> list[str]`**
  Names of registered signals, optionally of one family.
- **`def build_signals(names, ctx: SignalContext) -> dict[str, pd.DataFrame]`**
  Build the named signal panels on the context's calendar and universe.
- **`def custom_signals(names) -> dict[str, tuple[SignalFn, str]]`**
  {name: (builder, family)} of the user-registered (non built-in) signals among `names`.
- **`def register_signal(name: str, fn: SignalFn | None = None, *, family: str = 'custom', overwrite: bool = False)`**
  Register `fn` under `name` (usable as a decorator).
- **`def signal_family(name: str) -> str`**
  Family label of a registered signal.
- **`def unregister_signal(name: str) -> None`**
  Remove a signal from the registry (no error if absent).

## `fmflow.synthetic`

Synthetic market + sell-side data with the statistical properties of the real datasets.

- **`class SyntheticConfig(n_stocks: int = 10, n_sessions: int = 1512, start: str = '2016-01-04', n_sectors: int = 2, vol_range: tuple = (0.26, 0.53), market_vol: float = 0.2, market_drift: float = 0.25, sector_vol: float = 0.13, stock_drift_sd: float = 0.15, strength: float = 0.15, z_half_life: int = 63, earnings_every: int = 63, earnings_jump_mult: float = 4.0, analysts_per_stock: tuple = (17, 72), analyst_pool_ratio: float = 0.7, p_publish_at_earnings: float = 0.3, off_event_rate: float = 0.007692307692307693, p_reiterate: float = 0.05, adjust_speed: float = 0.6, p_sticky_down: float = 0.45, p_herd: float = 0.25, optimism_mu: float = np.float64(0.1570037488096647), optimism_sd: float = 0.08, info_weight: float = 0.06, target_noise: float = 0.035, p_late_start: float = 0.25, p_early_stop: float = 0.2, inject_errors: bool = False, seed: int = 7) -> None`**
  Parameters of the structural synthetic market (see module docstring); presets via classmethods.
- **`def describe_dataset(prices: pd.DataFrame, ratings: pd.DataFrame) -> pd.Series`**
  Moments used to calibrate the generator (computed identically on real and synthetic data).
- **`def generate(cfg: SyntheticConfig = SyntheticConfig()) -> tuple[pd.DataFrame, pd.DataFrame]`**
  Return (prices, ratings) long tables in the input contract of DATA.md.
