# Installation and troubleshooting

Supported: Python 3.10 to 3.14 on macOS (Intel and Apple Silicon), Windows and Linux. CI tests all fifteen combinations of the three systems and five Python versions.

## 1. No installation: run in the browser

* **Google Colab.** Open any notebook from the "Run online" column of the README. The first cell fetches the repository and installs the package in the Colab session, which takes about a minute. That cell works just as well in any Jupyter that lacks the package. It uses git when available (falling back on GitHub's zip archive) and pip (or uv, inside environments uv created)
* **Binder.** Click the Binder badge and JupyterLab opens with the package already installed. Expect the first build to take a few minutes
* **GitHub Codespaces.** The Codespaces badge opens VS Code in your browser, and the dev container installs everything and registers the "Python (fmflow)" kernel

All three only work while the GitHub repository is **public**. If you are working from a fork, point the environment variable `FMFLOW_REPO_URL` at it before running a notebook's first cell.

## 2. One command (recommended on your computer)

| System | Command |
|---|---|
| macOS, Linux | `./setup.sh` |
| Windows | double-click `setup.cmd`, or in PowerShell `.\setup.ps1` (if scripts are blocked: `powershell -ExecutionPolicy Bypass -File setup.ps1`) |

Here is what the script does.
1. It looks for [uv](https://docs.astral.sh/uv/) and, if uv is missing, installs it from astral.sh (you don't need administrator rights).
2. If there's no suitable Python, uv downloads Python 3.12. The script then builds `.venv` from the lock file `uv.lock`, which guarantees every machine the same tested versions.
3. It registers a Jupyter kernel called "Python (fmflow)" that Jupyter, VS Code and Anaconda Navigator can all pick up.
4. It finishes by running `fmflow doctor`.

There are three options. `--full` / `-Full` adds the extras for the research notebook and for data gathering. `--minimal` / `-Minimal` installs only the package. `--no-kernel` / `-NoKernel` leaves out the kernel. On some corporate networks and behind some proxies uv can't be installed or can't fetch packages. In that case the script falls back to `python -m venv` and pip, using whatever Python 3.10 or newer it can find. Note that `setup.ps1` also runs under PowerShell 7 on macOS and Linux.

From then on, you can either put `uv run` in front of each command (`uv run python tasks.py demo`) or activate the environment once per session.

| System | Activate |
|---|---|
| macOS, Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |
| Git Bash on Windows | `source .venv/Scripts/activate` |

## 3. Manual installation

Install into a **fresh** virtual environment. A big shared environment such as Anaconda's `base` is asking for trouble (the NumPy issue further down is the typical example).

```bash
git clone https://github.com/bahakh19/first-mover-flow.git
cd first-mover-flow
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[notebooks,dev]"            # see the extras below
fmflow doctor
python tasks.py test
```

If you use uv directly, the equivalent is `uv sync --extra notebooks --extra dev` (or `--all-extras`). With conda it looks like this:

```bash
conda create -n fmflow python=3.12 -y
conda activate fmflow
pip install -e ".[notebooks,dev]"
```

| Extra | Adds | Needed for |
|---|---|---|
| (core) | numpy ≥ 1.24, pandas ≥ 2.1.2, scipy ≥ 1.10, statsmodels ≥ 0.14.5, arch ≥ 6.2 | the strategy, the CLI, all statistics |
| `plot` | matplotlib | figures |
| `notebooks` | matplotlib, JupyterLab, ipykernel, nbclient | running the notebooks |
| `gen` | JAX, scikit-learn | the diffusion model and TimeGAN (bootstrap and FHS need neither) |
| `io` | openpyxl, pyarrow | Excel and Parquet input files |
| `research` | hmmlearn, LightGBM, scikit-learn, yfinance | the full research notebook |
| `data` | kaggle, yfinance, pyarrow | rebuilding the open dataset (`notebooks/Data_Gathering.ipynb`) |
| `dev` | pytest, pytest-xdist, ruff, pre-commit, nbclient, ipykernel, matplotlib, scikit-learn | tests, lint, notebook execution |
| `all` | everything above | |

## 4. Docker

```bash
docker build -t fmflow .
docker run --rm -p 127.0.0.1:8888:8888 fmflow                 # JupyterLab on http://localhost:8888
docker run --rm fmflow python tasks.py test                   # tests inside the container
```

## 5. Everyday commands

`python tasks.py <task>` behaves identically on every system, and on macOS and Linux `make <task>` is a shortcut for it.

| Task | Does |
|---|---|
| `doctor` | environment check with fixes |
| `demo` | the pre-registered out-of-sample test on the shipped data |
| `lab` | JupyterLab on the notebooks |
| `test` / `test-all` | fast tests / all tests, in parallel when pytest-xdist is installed |
| `notebooks` | re-execute notebooks 00 and 02–09 |
| `lint`, `docs` | ruff, regenerate `docs/API.md` |
| `synth`, `development`, `audit`, `power` | synthetic data, paper Sections 6–8 and 10–11 (the last two need `data/private`), power study |
| `clean` | remove caches and generated outputs |

## 6. `fmflow doctor`

`fmflow doctor` imports each dependency and puts it through the same paces the package does, printing a line for every component. It covers the Python interpreter and the `fmflow` installation in use (both with their paths). Then NumPy, pandas, SciPy, statsmodels, arch (it runs a small GJR-GARCH fit), matplotlib, JAX, scikit-learn, the Excel and Parquet readers, JupyterLab and the data files. Every line begins with one of these statuses.

| Status | Meaning |
|---|---|
| `ok` | installed and working |
| `FAIL` | a required component is missing or broken: the line gives the fix, and the command exits with status 1 |
| `warn` | an optional component is installed but broken (for example JAX on a CPU without AVX) |
| `opt` | an optional component is not installed: the line says which extra adds it |
| `info` | a data file that is not shipped (the proprietary development data) |

When something breaks, run it before anything else, and include its output in your bug report.

## 7. Known issues

### macOS: JAX on Intel Macs, LightGBM and libomp

* **JAX** has no wheels for Intel Macs, so on those machines the `gen` extra leaves it out. You lose the diffusion model and TimeGAN, and `fmflow doctor` will tell you so. Everything else still works, the bootstrap and FHS stress tests included
* **LightGBM**, which only the research notebook uses (`research` extra), needs the OpenMP runtime on macOS: `brew install libomp`

### Windows: PowerShell refuses to run `setup.ps1`

Windows' default execution policy won't run unsigned scripts. Either double-click `setup.cmd` or run `powershell -ExecutionPolicy Bypass -File setup.ps1`. Neither changes any system-wide setting.

### Windows: `python` opens the Microsoft Store

What you're seeing is a placeholder from the Store, and not an actual Python. You don't need Python for `setup.cmd`. Should you want one anyway, use `winget install Python.Python.3.12`, or the python.org installer with "Add python.exe to PATH" ticked.

### `fmflow: command not found`

The `fmflow` command is installed inside the virtual environment. Activate that environment (section 2), put `uv run` in front of the command, or call `python -m fmflow ...` with the environment's own Python.

### CSV files from Excel

The loader recognises on its own semicolon-separated files with decimal commas (as European Excel writes them), tab-separated files, UTF-8 with a byte-order mark, old Windows-1252 files and compressed files. For Excel and Parquet you need the `io` extra.

### Google Colab: "could not read Username for 'https://github.com'"

This means the repository is private. You can make it public, work from a fork with `FMFLOW_REPO_URL` set, or upload the repository to Colab and open the notebook from the upload.

### `ImportError: cannot import name '_lazywhere' from 'scipy._lib._util'`

**Cause.** Versions of statsmodels up to 0.14.4 import a private SciPy function that was dropped in SciPy 1.16. Pair any SciPy ≥ 1.16 with an old statsmodels and the failure comes the moment `statsmodels.api` is imported (for us, in the CAPM and factor regressions).

**Fix.** `pip install -U "statsmodels>=0.14.5"`. The package has required this version since 2.1.0, so a fresh `pip install -e .` upgrades statsmodels for you.

### `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` / `numpy.core.multiarray failed to import` (in matplotlib)

**Cause.** The environment holds a matplotlib compiled against NumPy 1.x alongside NumPy 2.x. Typically pip has upgraded NumPy inside an Anaconda `base` environment whose matplotlib was installed by conda, and the notebooks then fail on `import matplotlib.pyplot`.

**Fix.** Start from a fresh environment (see above), or upgrade matplotlib where it is with `pip install -U --force-reinstall matplotlib`.

### `RuntimeError: This version of jaxlib was built using AVX instructions, which your CPU and/or operating system do not support`

**Cause.** Recent jaxlib wheels require a CPU with AVX instructions. Most CPUs made since 2011 have them; some older or low-power chips do not.

**Effect.** JAX is used only by the diffusion model and TimeGAN. Since version 2.1.0, `fmflow.generative.jax_status()` spots the problem. The JAX tests are skipped with the reason shown, and notebook 06 runs the bootstrap and FHS generators without the two JAX models. `fmflow stress --generator diffusion` explains what is wrong and suggests `--generator fhs`. Nothing else is affected. If you need the JAX models, run them on a different computer or build jaxlib from source.

### GARCH `ConvergenceWarning` messages

**Cause.** Inside `fit()`, `arch` turns its own convergence warnings back on, which overrides whatever filters the caller has set.

**Since 2.1.0** the package passes `show_warning=False` and writes each fit's optimiser status to the fit log (`garch_forecast_panel(...)[1]`, columns `optimizer_status` and `new_estimate`). As before, a fit that failed to converge is discarded and the previous estimate stays in use.

### Fresh GARCH re-estimation differs slightly from the reference on another computer

**Cause.** With a few hundred daily observations, the GJR-GARCH-t likelihood often has two local maxima that are almost the same height. For one stock in the open universe, for instance, they sit at beta = 0.83 and beta = 0.91, and their log-likelihoods differ by 0.0002. Which of the two the SLSQP optimiser ends up at depends on the machine's floating-point details (the BLAS kernel, AVX/FMA instructions). Switching BLAS kernels on the reference machine reproduces it. Weights move by up to 0.023 and the out-of-sample Sharpe ratio drops to 1.02 from 1.05.

**How the package handles it.** The published numbers are reproduced **exactly** from the shipped forecasts with `run_strategy(..., sig_vol=pd.read_csv("tests/fixtures/open_sig_gjr.csv.gz", index_col=0, parse_dates=True))`, which is what notebooks 00 and 02–09 do. For fresh estimation there is `DataConfig(garch_method="grid")`, a deterministic estimator (grid maximum likelihood with variance targeting, no iterative optimiser). Its forecasts are bit-identical across Python 3.12/3.13, NumPy 2.2/2.5, BLAS kernels and CPUs with or without AVX, and `test_grid_garch_is_bitwise_reproducible_on_any_machine` checks this on every machine. Results barely move with it: out-of-sample Sharpe 1.09 (1.05 with SLSQP), development 2017–21 Sharpe 1.12 (1.13). Finally, `test_open_universe_fresh_slsqp_garch_agrees_economically` checks that a fresh SLSQP estimation agrees with the reference within optimiser noise (weights correlation > 0.999, Sharpe within 0.05), though not bit for bit.

### Windows: `RuntimeWarning: Proactor event loop does not implement add_reader`

You can ignore this. pyzmq prints it when Jupyter runs on Windows. `scripts/run_notebooks.py` switches to the selector event loop, so you won't see it there.

### Tests take long

`python tasks.py test` runs the fast tests in parallel with pytest-xdist. On a recent laptop that takes around a minute and a half, on older hardware a few minutes. `python tasks.py test-all` adds the slow tests (GARCH re-estimation, placebo loops, training small neural networks). The whole run then takes roughly ten minutes.
