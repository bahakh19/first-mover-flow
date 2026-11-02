# Contributing

Issues and pull requests are very welcome. We ask contributions to meet the same standards the research itself had to meet.

1. **Causality comes first.** Whatever you add, feature or signal, has to pass the truncation test pattern in `tests/test_causality.py`. Computed on data truncated at T, its values must equal the full-sample values up to T. If your feature introduces a new kind of window, add a negative control as well, and run `python scripts/fuzz_causality.py` (it throws messy universes with late listings and delistings at the code)
2. **Numbers belong in `config.py` and nowhere else.** Put any new parameter in `DataConfig` or `StrategyConfig`, with a comment and a default
3. **Comparisons share one construction.** New comparables and placebos are traded through `fmflow.pipeline.construct_book` (the way `robustness.book_from_score` does it) rather than through some rule written for the occasion
4. **Out-of-sample claims are pre-registered.** Run `fmflow register` before you evaluate a new universe, and report what comes out, good or bad
5. **Tests and lint pass.** Both `python tasks.py lint` and `python tasks.py test` have to succeed. Any test that runs longer than a few seconds gets `@pytest.mark.slow`
6. **Code stays portable.** Open text files with `encoding="utf-8"`, build paths with `pathlib`, and don't assume there's a shell. Everything a user runs goes through `tasks.py`, the `fmflow` CLI or a Python script, so it works on macOS, Windows and Linux alike

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

Dependabot only keeps the GitHub Actions up to date. To refresh the versions pinned in `uv.lock`, run `uv lock --upgrade` and then `python tasks.py test`.

Please don't commit any data you aren't licensed to redistribute. Git ignores `data/private/`, and on top of that a pre-commit hook rejects any file put there.
