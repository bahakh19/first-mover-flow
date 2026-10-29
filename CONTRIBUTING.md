# Contributing

Issues and pull requests are welcome. The project's standards are the ones the research was held to:

1. **Causality first:** Any new feature or signal must pass the truncation test pattern in `tests/test_causality.py`: values on data truncated at T must equal full-sample values up to T. Add a negative control if the feature uses a new kind of window, and run `python scripts/fuzz_causality.py` (messy universes with late listings and delistings)
2. **No numbers outside `config.py`:** New parameters go into `DataConfig` or `StrategyConfig` with a comment and a default
3. **Same construction for comparisons:** A new comparable or placebo must be traded with `fmflow.pipeline.construct_book` (as `robustness.book_from_score` does), not a bespoke rule
4. **Pre-register out-of-sample claims:** Use `fmflow register` before evaluating a new universe, and report the result whatever it is
5. **Tests and lint:** `python tasks.py lint` and `python tasks.py test` must pass. Mark tests that take more than a few seconds with `@pytest.mark.slow`
6. **Portable code:** Read and write text files with `encoding="utf-8"`, build paths with `pathlib`, and never assume a shell: anything a user runs goes through `tasks.py`, the `fmflow` CLI or a Python script, so it works on macOS, Windows and Linux

## Development set-up

```bash
./setup.sh --full                    # Windows: setup.cmd -Full   (or: uv sync --all-extras)
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
uv run python tasks.py test
```

## Where things go

| You add | Put it in | And |
|---|---|---|
| a signal | `src/fmflow/signals/`, registered with `@register_signal` | a truncation test, an entry in METHODOLOGY.md |
| a statistical test | `src/fmflow/analytics/` | a unit test on simulated data with a known answer |
| a portfolio rule | `src/fmflow/portfolio/construction.py`, called from `pipeline.construct_book`, with a `StrategyConfig` field | a test that the default reproduces the reference weights (`tests/test_parity.py`) |
| a CLI command | `src/fmflow/cli.py` | a test in `tests/test_cli.py`, a line in the README's "Use it" table |
| a notebook | `notebooks/NN_name.ipynb`, starting with the setup cell (`python scripts/refresh_setup_cell.py notebooks/NN_name.ipynb` inserts it) | it must run with `python tasks.py notebooks NN` from the shipped data |
| a dependency | `pyproject.toml` (core only if every user needs it, otherwise an extra, with the lowest version that works), then `uv lock` | check it has wheels for Windows, macOS and Linux; CI tests the lowest allowed versions |

To refresh the pinned versions of `uv.lock` (Dependabot only updates the GitHub Actions): `uv lock --upgrade`, then `python tasks.py test`.

Please do not commit data you are not licensed to redistribute. `data/private/` is git-ignored, and a pre-commit hook refuses any file placed there.
