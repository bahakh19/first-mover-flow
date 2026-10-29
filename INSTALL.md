# Installation and troubleshooting

Python 3.10 to 3.14 on macOS (Intel and Apple Silicon), Windows and Linux. Continuous integration tests every combination of the three systems with the five Python versions.

## 1. No installation: run in the browser

* **Google Colab:** open any notebook from the README's "Run online" column. Its first cell downloads the repository and installs the package into the Colab session (about a minute). The same cell works in any Jupyter without the package: it uses git if installed (otherwise GitHub's zip archive) and pip (or uv, in environments created by uv)
* **Binder:** the Binder badge opens JupyterLab with the package installed (the first build takes a few minutes)
* **GitHub Codespaces:** the Codespaces badge opens VS Code in the browser; the dev container installs everything and registers the "Python (fmflow)" kernel

These three routes need the GitHub repository to be **public**. For a fork, set the environment variable `FMFLOW_REPO_URL` to your fork's URL before running a notebook's first cell.

## 2. One command (recommended on your computer)

| System | Command |
|---|---|
| macOS, Linux | `./setup.sh` |
| Windows | double-click `setup.cmd`, or in PowerShell `.\setup.ps1` (if scripts are blocked: `powershell -ExecutionPolicy Bypass -File setup.ps1`) |

The script:
1. finds [uv](https://docs.astral.sh/uv/), or installs it from astral.sh (no administrator rights needed)
2. lets uv download Python 3.12 if no suitable Python is present, then creates `.venv` from the lock file `uv.lock`, so every machine gets the same tested versions
3. registers the Jupyter kernel "Python (fmflow)", so Jupyter, VS Code or Anaconda Navigator can use the environment
4. runs `fmflow doctor`

Options: `--full` / `-Full` adds the research-notebook and data-gathering extras, `--minimal` / `-Minimal` installs the package only, `--no-kernel` / `-NoKernel` skips the kernel. When uv cannot be installed or cannot download the packages (some corporate networks and proxies), the script falls back to `python -m venv` and pip with any Python 3.10 or newer it finds. `setup.ps1` also runs under PowerShell 7 on macOS and Linux.

Afterwards, either prefix commands with `uv run` (`uv run python tasks.py demo`) or activate the environment once:

| System | Activate |
|---|---|
| macOS, Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |
| Git Bash on Windows | `source .venv/Scripts/activate` |

## 3. Manual installation

Use a **fresh** virtual environment, never a large shared one such as Anaconda's `base` (see the NumPy issue below).

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

With uv directly: `uv sync --extra notebooks --extra dev` (or `--all-extras`). With conda:

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

`python tasks.py <task>` works the same on every system (`make <task>` is a shortcut on macOS and Linux):

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

`fmflow doctor` imports and exercises every dependency the way the package uses it and prints one line per component: the Python interpreter and the `fmflow` installation in use (with their paths), NumPy, pandas, SciPy, statsmodels, arch (a real GJR-GARCH fit), matplotlib, JAX, scikit-learn, the Excel and Parquet readers, JupyterLab, and the data files. Each line starts with a status:

| Status | Meaning |
|---|---|
| `ok` | installed and working |
| `FAIL` | a required component is missing or broken: the line gives the fix, and the command exits with status 1 |
| `warn` | an optional component is installed but broken (for example JAX on a CPU without AVX) |
| `opt` | an optional component is not installed: the line says which extra adds it |
| `info` | a data file that is not shipped (the proprietary development data) |

Run it first whenever something breaks, and paste its output into any bug report.

## 7. Known issues

### macOS: JAX on Intel Macs, LightGBM and libomp

* **JAX** publishes no wheels for Intel Macs, so the `gen` extra skips it there. The diffusion model and TimeGAN are then unavailable (`fmflow doctor` says so); everything else works, including the bootstrap and FHS stress tests
* **LightGBM** (only for the research notebook, `research` extra) needs the OpenMP runtime on macOS: `brew install libomp`

### Windows: PowerShell refuses to run `setup.ps1`

The default execution policy blocks unsigned scripts. Double-click `setup.cmd` instead, or run `powershell -ExecutionPolicy Bypass -File setup.ps1`. Nothing is changed system-wide.

### Windows: `python` opens the Microsoft Store

That is the Store's placeholder, not a Python installation. `setup.cmd` does not need Python. To install one anyway: `winget install Python.Python.3.12`, or the installer from python.org with "Add python.exe to PATH" ticked.

### `fmflow: command not found`

The `fmflow` command lives in the virtual environment. Activate it (section 2), prefix the command with `uv run`, or use `python -m fmflow ...` with the environment's Python.

### CSV files from Excel

Semicolon-separated files with decimal commas (European Excel), tab-separated files, UTF-8 files with a byte-order mark, legacy Windows-1252 files and compressed files are detected automatically. Excel and Parquet files need the `io` extra.

### Google Colab: "could not read Username for 'https://github.com'"

The repository is private. Make it public, or use a fork and set `FMFLOW_REPO_URL`, or upload the repository to Colab and open the notebook from there.

### `ImportError: cannot import name '_lazywhere' from 'scipy._lib._util'`

**Cause:** statsmodels 0.14.4 and older import a private SciPy function that SciPy removed in version 1.16. Any SciPy ≥ 1.16 with an old statsmodels fails as soon as `statsmodels.api` is imported (here, in CAPM and factor regressions).

**Fix:** `pip install -U "statsmodels>=0.14.5"`. Since version 2.1.0 the package requires it, so a fresh `pip install -e .` upgrades statsmodels automatically.

### `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` / `numpy.core.multiarray failed to import` (in matplotlib)

**Cause:** A matplotlib build compiled against NumPy 1.x sits in the same environment as NumPy 2.x. This happens when pip upgrades NumPy inside an Anaconda `base` environment whose matplotlib came from conda. Notebooks fail at `import matplotlib.pyplot`.

**Fix:** Use a fresh environment (above), or upgrade matplotlib in place: `pip install -U --force-reinstall matplotlib`.

### `RuntimeError: This version of jaxlib was built using AVX instructions, which your CPU and/or operating system do not support`

**Cause:** Recent jaxlib wheels need a CPU with AVX instructions (most CPUs since 2011, but not some older or low-power processors).

**Effect:** Only the diffusion model and TimeGAN need JAX. Since version 2.1.0, `fmflow.generative.jax_status()` detects the problem; the JAX tests skip with the reason; notebook 06 runs the bootstrap and FHS generators and skips the two JAX models; `fmflow stress --generator diffusion` explains and suggests `--generator fhs`. Everything else is unaffected. To use the JAX models on such a machine, run them on another computer or build jaxlib from source.

### GARCH `ConvergenceWarning` messages

**Cause:** `arch` re-enables its own convergence warnings inside `fit()`, overriding filters set by the caller.

**Since 2.1.0** the package passes `show_warning=False` and records each fit's optimiser status in the fit log (`garch_forecast_panel(...)[1]`, columns `optimizer_status` and `new_estimate`). A fit that did not converge is not used; the previous estimate is kept, as before.

### Fresh GARCH re-estimation differs slightly from the reference on another computer

**Cause:** On a few hundred daily observations, the GJR-GARCH-t likelihood often has two local maxima of almost equal height (for one stock of the open universe, beta = 0.83 and beta = 0.91 with a log-likelihood gap of 0.0002). The SLSQP optimiser stops at one or the other depending on floating-point details of the machine (BLAS kernel, AVX/FMA instructions). Switching the BLAS kernel on the reference machine reproduces the effect: the maximum weight difference is 0.023 and the out-of-sample Sharpe ratio is 1.02 instead of 1.05.

**What the package does.**
* The published numbers are reproduced **exactly** from the shipped forecasts: `run_strategy(..., sig_vol=pd.read_csv("tests/fixtures/open_sig_gjr.csv.gz", index_col=0, parse_dates=True))`. Notebooks 00 and 02–09 do this.
* `DataConfig(garch_method="grid")` is a deterministic estimator (grid maximum likelihood with variance targeting, no iterative optimiser). Its forecasts are bit-identical across Python 3.12/3.13, NumPy 2.2/2.5, BLAS kernels and CPUs with or without AVX, and the test `test_grid_garch_is_bitwise_reproducible_on_any_machine` checks this on every machine. The strategy's results barely move with it: out-of-sample Sharpe 1.09 (1.05 with SLSQP), development 2017–21 Sharpe 1.12 (1.13).
* `test_open_universe_fresh_slsqp_garch_agrees_economically` checks that a fresh SLSQP estimation agrees with the reference within optimiser noise (weights correlation > 0.999, Sharpe within 0.05), not bit for bit.

### Windows: `RuntimeWarning: Proactor event loop does not implement add_reader`

Harmless message from pyzmq when Jupyter runs on Windows. `scripts/run_notebooks.py` sets the selector event loop and does not print it.

### Tests take long

`python tasks.py test` runs the fast tests in parallel (pytest-xdist): about a minute and a half on a recent laptop, a few minutes on older hardware. `python tasks.py test-all` adds the slow tests, which re-estimate GARCH models, run placebo loops and train small neural networks: about ten minutes.
