# Changelog

## 2.2.4 (2026-10-29): every Python, every Jupyter

No change to any computation; every published number is unchanged.

**Python 3.10 to 3.14**
* Tested here end to end on Python 3.10 (from the lock file) and on Python 3.14 (pandas 3.1, NumPy 2.5): fast and slow tests, the pre-registered demo (parity 5e-13) and every use-case notebook, plus the parallel multiverse under Python 3.14's new default start method (forkserver). On Python 3.10 with the oldest allowed version of every dependency (numpy 1.24, pandas 2.1.2, SciPy 1.10, statsmodels 0.14.5, arch 6.2, matplotlib 3.7), the fast tests and the demo, exactly as the CI job does
* Python 3.14 is now declared (classifiers, README badge, INSTALL.md) and added to the CI matrix

**Notebook setup cell** (`scripts/refresh_setup_cell.py`, now in the repository)
* Executed here in three bare environments: with git and pip, without git (it downloads GitHub's zip archive instead of failing), and in an environment created by uv, which has no pip (it installs with uv instead of failing)
* It prints what it is doing ("Downloading ...", "Installing ..."); git's progress no longer appears as red error output; a failed download explains the likely cause (a private repository) and how to point the cell at a fork or a local copy (`FMFLOW_REPO_URL`)

**Code quality**
* Every public class and method of the package has a docstring; two unused private members of the generative models were removed
* `tasks.py` has a shebang, so `./tasks.py demo` works on macOS and Linux
* The pip fallback of `setup.sh` tries Python 3.12 first, the version uv installs

## 2.2.3 (2026-10-29): look-ahead through the universe's composition

Found by fuzzing the pipeline with randomly messy universes (late listings, delistings, gaps, duplicates, zero prices, injected vendor errors, 6 to 24 stocks), running the truncation test on every signal, the composite and the weights under several constructions, and running every configuration field at the ends of its admissible range. Three defects made results at date t depend on stocks that only appear after t, and three admissible parameter values crashed. Neither real dataset contains a late listing, so every published number is unchanged: parity with the research notebook stays within 5e-13 on both datasets, and the development calendar keeps its 1,511 sessions.

**Fixes**
* **Trading calendar:** A date counted as a session if at least 50% of *all the stocks in the file* printed on it. Stocks listing later raised the denominator, so adding later data could remove past sessions. The denominator is now the stocks active at that date (printed in the previous 30 calendar days, `DataConfig.calendar_active_days`); in the file's first 30 days, the stocks printing anywhere in that window, so a stray first row (a holiday on which two stocks print) is still not a session
* **New configuration field:** `DataConfig.calendar_active_days` (default 30) is part of every serialised configuration, so a specification registered with 2.2.3 has a different fingerprint from the same specification registered earlier. Existing registration files still verify, and configuration files without the field load with the default
* **Volatility target:** The Ledoit-Wolf covariance of the ex-ante volatility included stocks that had not listed yet (as columns of zeros), which changed the shrinkage. It now uses the stocks that have traded by t. The same applies to `production.risk_report`
* **Time-series momentum benchmark:** Each position was 1/N of all the stocks in the file; it is now 1/N of the stocks with a signal at t
* **Parameter edge cases:** `price_ffill_limit = 0` and `tp_expiry_days = 1` passed validation but crashed (pandas rejects `ffill(limit=0)`); they now mean "no forward fill" and "no carry-forward". A `beta_window` shorter than 126 sessions crashed the beta-neutral construction; the minimum number of observations now adapts to the window, and `beta_window` must be at least 2

**Known limitation, deliberately not changed**
* Residuals of ranks in `breadth_resid_mom` can be mathematically tied, and the floating-point row sums give such ties last-bit (1e-17) differences that depend on where missing values sit, including the columns of stocks that list later. The next ranking can then order two exactly tied stocks differently. Rounding the residuals removes this, but it changes the development weights by up to 0.003, because the reference results contain such ties. Parity with the published results was kept; the limitation is documented in the code. It can only appear when a universe gains stocks, and it never involves information from the future

**Tools**
* `scripts/fuzz_causality.py`: the fuzzing harness that found these defects, for anyone extending the package (messy random universes, truncation test of every signal, the composite and the weights under three constructions, book invariants)

**Documentation**
* RESULTS.md: the development signal table labelled its Newey–West *t* statistics "Phase *t*". It now shows both, with the phase-averaged *p*-values, as in the paper's Table 3 (signal evidence)
* RESULTS.md: the out-of-sample permutation placebo p = 0.056 comes from the research notebook, which draws unrestricted permutations there. The package draws derangements, as the development placebos do, and gives 0.046. Both values are now stated
* DATA.md and DATA_SOURCES.md describe the calendar rule; OBJECTIONS.md no longer says the knowledge-date check is an assertion
* 6 regression tests (`tests/test_audit_223.py`): five fail on 2.2.2, the sixth keeps the old behaviour for a stray first date

## 2.2.2 (2026-10-28): second audit

Every published number is unchanged (parity 5e-13; demo, development and audit tables as before).

**Correctness**
* **Nullable and pyarrow column types:** A table read with pandas' nullable types (`convert_dtypes()`), pyarrow-backed columns or a Parquet file with nullable columns carried `pd.NA` into the pipeline and crashed in the rating classification. Numbers are now converted to plain floats on entry (`fmflow.data.cleaning.to_float`)
* **European CSV files (semicolon and decimal comma)** crashed for the same reason, because the decimal-comma conversion produced nullable columns. They now run end to end
* **Parallel multiverse with your own signals:** With `n_jobs > 1` on macOS and Windows (and on Linux from Python 3.14), worker processes start fresh and do not know signals registered in a notebook or a script's main block, so every specification failed and the result was an empty table. Signals defined in an importable module are now registered in each worker; signals defined in `__main__` or as lambdas make the multiverse run sequentially, with a warning
* **Multiverse failures are reported:** a warning names how many specifications failed and the first error; if all fail, an error is raised instead of returning an empty table
* **Missing Excel or Parquet engine:** the error names the `io` extra to install
* **Generative windows:** samples shorter than one window are skipped, with a clear error if none is long enough
* **Trade list:** the error for a holding without a price explains that every held or targeted name needs one

**Setup and command line**
* `setup.sh` and `setup.ps1` fall back to `python -m venv` and pip when `uv sync` fails (proxy, offline index), not only when uv cannot be installed. `setup.ps1` also runs under PowerShell 7 on macOS and Linux. Both end with plain activation instructions; when uv was just installed outside the PATH, its full path is printed. Both fallbacks were run here with uv forced offline, and `setup.ps1` was checked with PSScriptAnalyzer
* `fmflow doctor` shows the interpreter and the `fmflow` installation in use (with paths), checks the Excel / Parquet readers and JupyterLab, and labels a missing optional package `opt` with the extra that adds it (instead of an exception name). INSTALL.md explains the statuses

**Repository**
* Dependabot updates only the GitHub Actions: its pip and uv ecosystems raise the lower bounds in `pyproject.toml`, which would silently drop the oldest versions the package is tested against. CONTRIBUTING.md explains how to refresh `uv.lock`
* `docs/API.md` shows readable signatures (`cfg: DataConfig = DataConfig()` instead of the full repr of every default); every subpackage has a docstring
* Stale `make` commands in DATA_SOURCES.md and data/README.md replaced by `python tasks.py`; README architecture and design rules describe `pipeline.construct_book`; INSTALL.md lists the minimum version of every core dependency and a "`fmflow: command not found`" entry
* Tested additionally with pandas 3.1.0 / NumPy 2.5.4 (the versions pip installs today): fast tests with deprecation warnings as errors, and the pre-registered demo (Sharpe 1.046, parity 5e-13)
* 8 more regression tests (`tests/test_audit_222.py`)

## 2.2.1 (2026-10-28): deep audit

Every published number is unchanged: the parity with the research notebook stays at 5e-13 on the open data, and `python tasks.py development` and `python tasks.py audit` reproduce the paper's development, holdout, placebo, leave-one-out, multiverse, Bayesian, power and generative tables.

**Correctness**
* **Dates with a time or a UTC offset:** A rating stamped "2023-01-04 08:30" is now usable from the session after 4 January, like a plain date. Before, it could be dropped or counted on the wrong day, and a column mixing formats lost rows. Times are ignored and the local calendar day is kept
* **Integer stock identifiers:** Prices with numeric identifiers (PERMNO-style) and ratings read as text now match. Before, the scale check failed with a merge error
* **Scale check in the pipeline:** `run_strategy` validates the ratings against the prices, as `fmflow validate` always did. Targets in dollars against normalised prices now raise a `DataContractError` instead of being removed one by one by the plausibility filters
* **Missing stock identifiers** in the price file are dropped and logged. Before, they became a stock called "nan"
* **Position cap:** `max_weight` is now a hard limit. When it cannot be met (too few names for the leg budget), every name sits at the cap and the gross shrinks. Before, a final rescaling could push weights above the cap, and more than 20 capped names were not handled. The redistribution is exact (water-filling). In notebook 04 the "position cap 6%" variant had in fact exceeded the cap on every day (up to 25% in one name); with the cap enforced its legs (6 to 8 names) cannot spend the full budget, the gross averages 0.75 and the Sharpe ratio is 0.98 instead of 1.05. A note in the notebook explains it
* **Inverse-volatility legs:** A selected stock without any volatility estimate gets the day's median inverse volatility. Before, it dropped out of its leg silently, and a leg with no estimate at all left the book one-sided
* **Backtest:** Opening the book is charged like any other trade, and missing weights count as flat positions
* **Sortino ratio:** Uses the downside deviation over all sessions (Sortino and Price, 1994), not the standard deviation of the negative returns. A wiped-out book reports a CAGR of −100% instead of NaN
* **One construction everywhere:** `fmflow.pipeline.construct_book` builds the held book (legs, allocator, beta neutralisation, cap, tranches, volatility target). Placebos (`book_from_score`) and the institutional comparables now use it with the configured borrow fee and execution delay. Before, they ignored the borrow fee, the execution delay, the cap and the volatility target
* **Guards with clear messages:** empty signal list, `cscv_pbo` with an odd number of blocks or a single trial, `placebo_test` on one stock, `psr` on constant returns, `trade_list` with a non-positive NAV or lot size, `latest_book` before the first session, a price file with no valid session, `default_windows` on fewer than two sessions
* **Ledoit-Wolf covariance:** The shrinkage intensity is computed in closed form instead of a Python loop (identical result, faster ex-ante volatility)
* **GARCH cache:** An entry that cannot be read (written by another pandas version) is recomputed; a read-only or full disk no longer stops the run
* **CSV files saved by Excel on Windows (cp1252)** are read; `normalize_universe` keeps a missing analyst missing

**Command line**
* `fmflow --version`, `fmflow` without arguments prints the help, every command and option has a help text, and `python -m fmflow` works
* Errors in the input are reported in one line with exit code 2 (`--debug` or `FMFLOW_DEBUG=1` shows the traceback); warnings are one line; piping into `head` no longer prints a traceback
* `fmflow validate` also runs the cleaning rules and reports what they removed, the sample, thin coverage and the minimum-history requirements, and exits with 1 when the data cannot be used
* `--out` writes CSV files for every analysis command (`infer`, `book`, `stress`, `multiverse` and `compare` printed only). `--leg-fraction` and `--allocator` are recorded in the strategy name and in `config_used.json`
* `fmflow run` saves the equity curve without opening a window and works without a display
* `fmflow doctor` explains when it is not run from the repository root

**Packaging and CI**
* The minimum pandas version is 2.1.2 (2.1.0 and 2.1.1 warn on every `pct_change(fill_method=None)`). CI now installs the lowest version allowed for every direct dependency (`uv pip install --resolution lowest-direct`) on Python 3.10 and reproduces the out-of-sample result with it
* `py.typed` marker, SPDX licence metadata, ruff pinned to the same version in CI and pre-commit
* `python tasks.py clean` never touches the virtual environment
* 28 regression tests, one per fixed defect (`tests/test_audit_221.py`)

## 2.2.0 (2026-10-28): runs anywhere

The strategy and every published number are unchanged.

**Install and run on any system**
* `setup.sh` (macOS, Linux), `setup.ps1` and a double-clickable `setup.cmd` (Windows): one command creates `.venv` with everything, installing uv and Python if they are missing, registers the Jupyter kernel "Python (fmflow)" and runs `fmflow doctor`. Without access to astral.sh they fall back to `python -m venv` and pip
* `tasks.py`: one task runner for every system (doctor, demo, test, test-all, lint, lab, notebooks, docs, synth, development, audit, power, clean); the Makefile now delegates to it
* `uv.lock`: one resolution for Python 3.10 to 3.13 on Windows, macOS and Linux, all from pre-built wheels (JAX is skipped on Intel Macs, which have no JAX wheels; the newest argon2-cffi-bindings, which has none either, is avoided there)
* Every use-case notebook starts with a setup cell: in an installed environment it only finds the repository; on Google Colab, Binder or a bare Jupyter it downloads the repository and installs the package. `FMFLOW_REPO_URL` points it at a fork
* Zero-install routes: Colab links for every notebook, a Binder configuration, a GitHub Codespaces dev container, and a Dockerfile that serves JupyterLab
* New extras: `notebooks`, `io` (Excel, Parquet), `research` (the full research notebook), `data` (rebuilding the open dataset); `pytest-xdist` runs the tests in parallel
* `.gitattributes` for line endings (shell scripts LF, PowerShell and batch files CRLF), editor recommendations, issue and pull-request templates, Dependabot

**Robustness**
* Loaders detect the CSV dialect: comma, semicolon or tab separators, decimal commas (European Excel exports), byte-order marks, compressed files; Excel and Parquet are read directly; a missing file is reported with the working directory
* All text files are read and written as UTF-8 (Windows defaults to cp1252, which cannot encode characters such as "≥" in the generated index rulebook); the CLI never crashes on a character the console cannot display
* The GARCH cache is written atomically, so parallel workers and interrupted runs cannot leave a corrupted entry
* `fmflow doctor` distinguishes a missing optional package from a broken one and explains the Intel-Mac JAX case

**Research notebook**
* `notebooks/Research_notebook_FMF.ipynb` (the author's curated version) replaces `01_research_notebook.ipynb`. It finds the data in `data/private/` and explains clearly when the proprietary data are absent, installs only missing packages (so it also works in uv environments, which have no pip), skips the ticker reconciliation gracefully without internet access, and uses the "never-seen period" wording with the overlap note
* Executive summary corrected against the notebook's own outputs: stock_6 is Microsoft; the placebo p-values are per window (IID 0.003 and permutation 0.002 over 2017–21; 0.027 and 0.056 out of sample; the time-shift placebo passed only in the holdout); the cost stress keeps a Sharpe of 1.06 in development and 0.99 out of sample
* DATA_SOURCES.md cites the notebook's ticker reconciliation for all ten development stocks (daily-return correlation 1.0000 with the dividend-unadjusted public closes)
* The two reconciliation cells read the data from `data/private/` and skip gracefully without internet access

**Documents**
* The paper and the tear sheet carry the author's Overleaf edits (labels with colons, commas instead of semicolons, rewordings, contact e-mail) together with the 2.1.0 corrections; the tear sheet is `docs/Tearsheet.pdf`
* README: a "Get started" section with three routes (browser, one command, manual), Colab links, a repository map; INSTALL.md rewritten around the new routes, with macOS and Windows notes

## 2.1.0 (2026-10-28) — portability and data-provenance release

The strategy and every published number are unchanged. This release fixes installation and reproducibility problems reported on two machines (Linux with Anaconda on a CPU without AVX; Windows), and documents the data.

**Fixes**
* **statsmodels / SciPy:** statsmodels ≤ 0.14.4 imports a function removed in SciPy 1.16 (`_lazywhere`), which broke CAPM and factor regressions (5 failing tests). The package now requires `statsmodels>=0.14.5`.
* **GARCH warnings:** `arch` re-enables its `ConvergenceWarning` inside `fit()`, so warnings leaked to the console despite the caller's filter. Fits now use `show_warning=False`; the optimiser status of every fit is recorded in the fit log (`optimizer_status`, `new_estimate`).
* **JAX on CPUs without AVX:** jaxlib raises a `RuntimeError` at import on such CPUs. `fmflow.generative.jax_status()` detects it; the JAX tests skip with the reason; notebook 06 falls back to the bootstrap and FHS generators; `fmflow stress --generator diffusion` explains instead of crashing.
* **Fresh GARCH parity across machines:** The test required bit-identical weights after re-estimating GARCH from scratch, which only holds on the reference machine: the short-sample GJR-GARCH-t likelihood has near-equal local maxima, and SLSQP's stopping point depends on the CPU's floating-point path (reproduced here by switching the BLAS kernel: max weight difference 0.023, out-of-sample Sharpe 1.02 instead of 1.05). The test now checks economic agreement (weights correlation > 0.999, Sharpe within 0.05). Exact parity with the shipped forecasts is unchanged.
* **Notebook runner:** `make notebooks` continued after a failure and needed bash; `scripts/run_notebooks.py` runs on any OS, stops at the first failure, skips the research notebook, without the private data, and silences the Windows event-loop warning.

**Added**
* `DataConfig(garch_method="grid")`: a deterministic GJR-GARCH-t estimator (grid maximum likelihood with variance targeting). Bit-identical across Python 3.12/3.13, NumPy 2.2/2.5, BLAS kernels and CPUs with or without AVX; tested against a shipped fixture on every machine. Out-of-sample Sharpe 1.09 (1.05 with SLSQP), 2017–21 Sharpe 1.12 (1.13).
* `fmflow doctor`: one-line-per-component environment check with the fix for each known failure.
* `INSTALL.md`: installation (venv, conda, Windows) and every known issue with its cause and fix.
* `DATA_SOURCES.md`: provenance, coverage, cleaning and licences of the proprietary development data and the public open data, the stock map, and the overlap between the two universes.
* `notebooks/Data_Gathering.ipynb`: rebuilds the open dataset from the Kaggle dataset `yijian2427/us-market-analyst-insights-with-full-stock-symbols` and Yahoo Finance, normalises it, and compares it row by row with the shipped files. Credentials come from the environment, never from the notebook.
* `fmflow.data.normalize_universe` and `compare_tables`: the anonymisation and normalisation used for `data/open/`, as a tested function.
* `scripts/overlap_analysis.py`: the open universe contains the ten development names. On the twenty new names alone the frozen specification earns a Sharpe ratio of 0.80 and leader minus herd keeps its IC (0.105). Documentation now says "never-seen period" where it said "never-seen stocks".
* CI: a Windows job and an environment check.

## 2.0.0 (2026-10-26) — the adversarial audit release

The strategy is unchanged (FMF-N reproduces v1.0.0 exactly). Everything added serves to attack it, or to use it in production.

* **Audit layer** (`analytics.snooping`, `analytics.bayes`, `analytics.attribution`, `analytics.power`, `robustness.multiverse`):
  * White's Reality Check, Hansen's SPA and Romano–Wolf StepM
  * circular-block-bootstrap Sharpe-difference tests
  * Bayesian-bootstrap Sharpe posteriors under sceptical priors
  * random-effects meta-analysis with predictive distributions, empirical-Bayes shrinkage and Bayes factors
  * Lo (2008) passive/active decomposition and causal static/timing books
  * analytic and Monte Carlo power
  * multiverse / specification-curve runner (one-at-a-time, full or random) with ensembles
  * `truncation_check` for any configuration, including user signals
* **Generative stress testing** (`fmflow.generative`, extra `[gen]`):
  * stationary bootstrap and FHS, with variance targeting
  * Diffusion-FHS: GJR-GARCH dynamics plus conditional-DDPM shocks in normal-score space, with moment matching
  * a window DDPM with whitening and training-free conditioning
  * TimeGAN
  * a fidelity battery (stylised facts, a leak-free discriminative score, TSTR, memorisation) and path-risk statistics
* **Construction and implementation:** beta-neutral legs, position caps, Ledoit–Wolf volatility targeting, sector-neutral scores, short-borrow fees, execution delay, breakeven cost and cost-sensitivity grids, and a GARCH cache.
* **Extensibility:** a signal registry (`register_signal`), `extra` user data and new presets
* **Production and QIS:** `production` (book, trade list, risk report, signal monitor) and `index` (ER/TR index levels, fees, vol target, factsheet, generated rulebook)
* **CLI:** `infer`, `multiverse`, `book`, `index` and `stress`
* **Notebooks 02–09:** data, signals, construction and risk, inference, generative stress, QIS, PM workflow, bring-your-own-universe
* **Engineering:** Docker, a lock file, pre-commit, CI with coverage and notebook execution, a generated API reference, and more than 90 tests
* **Fixes found by the audit:**
  * Simulated GARCH paths used the robust t-location instead of the sample mean, which injected a spurious drift
  * The discriminative score leaked overlapping windows across the train/test split
  * The equal-weight market's self-alpha t-statistic was a numerical artefact and is now reported as undefined

## 1.0.0 (2026-10-26)

First public release.

* **Strategy:** FMF-N: four revision-flow signals (breadth, event breadth, breadth orthogonal to momentum, leader minus herd) in an equal-weight rank composite. Top/bottom 4 names (or a leg fraction of 0.4), inverse GJR-GARCH-t weights, 42 overlapping tranches, dollar-neutral, 5 bp costs. FMF-B20 adds a constant 0.2 beta budget
* **Data layer:** Input contract with actionable errors (including a target/price scale check), audited causal cleaning, point-in-time target snapshot.
* **Validation:** Phase-averaged IC tests, Holm, PSR/DSR, CSCV PBO, stationary bootstrap, placebos through the identical construction, leave-one-out, 13 institutional signals and 3 allocations, pre-registration fingerprints.
* **Synthetic generator:** Structural model calibrated to both real universes. Null, development-like and open-universe-like presets, plus error injection.
* **Parity:** Exact reproduction of the research notebook's weights on the development data (0.0) and within 5e-13 on the open universe (with the shipped GARCH forecasts, see 2.1.0 for fresh re-estimation).
* **Out-of-sample:** The pre-registered test on 30 stocks over the never-seen period 2024–2026 ships with its registration file (`0b2b0792fd357d61`).

### Research history (before the package)

* *Gen 1: Regime-Timed Breadth (RTB).* Revision breadth with GJR-GARCH sizing and an HMM/trend regime tilt. Development Calmar 1.77, later shown to be carried by net market exposure (selection book: Sharpe 0.63).
* *Six audit rounds.* More than fifty pre-registered hypotheses were tested and rejected (paper, Table 2).
* *Gen 2: First-Mover Flow.* The leader-minus-herd signal and a horizon-matched construction.
