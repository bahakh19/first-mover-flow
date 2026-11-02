#!/usr/bin/env python3
"""One task runner for macOS, Windows and Linux (standard library only).

    python tasks.py                 # list the tasks
    python tasks.py doctor          # check the environment
    python tasks.py demo            # the pre-registered out-of-sample test on the shipped data (about a minute)
    python tasks.py test            # fast tests, in parallel when pytest-xdist is installed
    python tasks.py test-all        # + slow tests (GARCH re-estimation, placebo loops, neural networks)
    python tasks.py lab             # open JupyterLab in notebooks/
    python tasks.py notebooks       # re-execute the use-case notebooks (stops at the first failure)
    python tasks.py lint            # ruff
    python tasks.py docs            # regenerate docs/API.md
    python tasks.py synth           # synthetic datasets calibrated to both real universes
    python tasks.py development     # paper Sections 6-8 (needs data/private)
    python tasks.py audit           # paper Sections 10-11 (needs data/private)
    python tasks.py power           # synthetic power study: does the pipeline find alpha only when it exists?
    python tasks.py clean           # remove caches and generated outputs

Run it with the project's environment: `uv run python tasks.py test` (uv), or activate your virtual environment
first (`source .venv/bin/activate`, Windows: `.venv\\Scripts\\activate`) and run `python tasks.py test`.
Extra arguments are passed through, e.g. `python tasks.py test -k garch`.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def _run(*cmd: str) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(list(cmd), cwd=ROOT)


def _need(module: str, extra: str) -> bool:
    if importlib.util.find_spec(module) is None:
        print(f"'{module}' is not installed in this environment. Install it with:  pip install -e \".[{extra}]\"  "
              f"(or: uv sync --extra {extra})")
        return False
    return True


def _parallel() -> list[str]:
    return ["-n", "auto"] if importlib.util.find_spec("xdist") else []


def doctor(*a): return _run(PY, "-m", "fmflow.cli", "doctor", *a)
def demo(*a): return _run(PY, "scripts/reproduce_open_universe.py", "--fast", *a)
def test(*a): return _run(PY, "-m", "pytest", "-m", "not slow", *_parallel(), *a) if _need("pytest", "dev") else 1
def test_all(*a): return _run(PY, "-m", "pytest", *_parallel(), *a) if _need("pytest", "dev") else 1
def lint(*a): return _run(PY, "-m", "ruff", "check", "src", "tests", "scripts", "tasks.py", *a) if _need("ruff", "dev") else 1
def docs(*a): return _run(PY, "scripts/make_api_docs.py", *a)
def notebooks(*a): return _run(PY, "scripts/run_notebooks.py", *a) if _need("nbclient", "notebooks") else 1
def development(*a): return _run(PY, "scripts/reproduce_development.py", "--placebo", "--loo", *a)
def audit(*a): return _run(PY, "scripts/reproduce_audit.py", *a)
def power(*a): return _run(PY, "scripts/synthetic_power_study.py", "--seeds", "40", *a)


def lab(*a):
    if not _need("jupyterlab", "notebooks"):
        return 1
    return _run(PY, "-m", "jupyter", "lab", "--notebook-dir", str(ROOT / "notebooks"), *a)


def synth(*a):
    rc = _run(PY, "-m", "fmflow.cli", "synth", "--preset", "development", "--seed", "7", "--out", "data/synthetic/development")
    return rc or _run(PY, "-m", "fmflow.cli", "synth", "--preset", "open", "--seed", "7", "--out", "data/synthetic/open")


def clean(*a):
    for p in ["results", "build", "dist", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".fmflow_cache", "htmlcov", ".coverage",
              "data/raw", "data/rebuilt", *[str(x) for x in ROOT.glob("src/*.egg-info")]]:
        q = ROOT / p
        if q.is_dir():
            shutil.rmtree(q, ignore_errors=True)
        elif q.exists():
            q.unlink()
    for d in ROOT.rglob("__pycache__"):
        if not {".venv", "venv", ".git"} & set(d.relative_to(ROOT).parts):     # never touch the environment
            shutil.rmtree(d, ignore_errors=True)
    for d in (ROOT / "data" / "synthetic").glob("*/"):
        shutil.rmtree(d, ignore_errors=True)
    print("cleaned")
    return 0


TASKS = {"doctor": doctor, "demo": demo, "test": test, "test-all": test_all, "lint": lint, "docs": docs,
         "lab": lab, "notebooks": notebooks, "synth": synth, "development": development, "audit": audit,
         "power": power, "clean": clean}


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help") or argv[0] not in TASKS:
        print(__doc__)
        if argv and argv[0] not in ("-h", "--help", "help"):
            print(f"unknown task: {argv[0]}")
            return 2
        return 0
    return TASKS[argv[0]](*argv[1:])


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except BrokenPipeError:                       # output piped into `head`: not an error
        raise SystemExit(0) from None
