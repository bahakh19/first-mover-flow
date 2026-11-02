#!/usr/bin/env bash
# One-command setup for macOS and Linux. Python does not need to be installed.
#
#   ./setup.sh             standard: package + notebooks + tests + generative models + Excel/Parquet input
#   ./setup.sh --full      also the research-notebook and data-gathering extras
#   ./setup.sh --minimal   the package only (strategy, CLI, statistics)
#   ./setup.sh --no-kernel do not register the "Python (fmflow)" Jupyter kernel
#
# How: uses uv (https://docs.astral.sh/uv/), installing it if needed; uv downloads a suitable Python by itself and
# creates .venv/ from the lock file. Without internet access to astral.sh, it falls back to python3 -m venv + pip.
set -euo pipefail
cd "$(dirname "$0")"

EXTRAS=(--extra plot --extra notebooks --extra dev --extra gen --extra io)
PIP_EXTRAS="plot,notebooks,dev,gen,io"
KERNEL=1
for arg in "$@"; do
  case "$arg" in
    --full)      EXTRAS=(--all-extras); PIP_EXTRAS="all" ;;
    --minimal)   EXTRAS=(); PIP_EXTRAS="" ;;
    --no-kernel) KERNEL=0 ;;
    -h|--help)   sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see ./setup.sh --help)"; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

find_uv() {
  for c in uv "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if command -v "$c" >/dev/null 2>&1; then command -v "$c"; return 0; fi
  done
  return 1
}

UV="$(find_uv || true)"
if [ -z "$UV" ]; then
  say "Installing uv (a fast Python package and version manager)"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || true
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh || true
  fi
  UV="$(find_uv || true)"
fi

pip_fallback() {
  PY=""
  for c in python3.12 python3.13 python3.11 python3.14 python3.10 python3 python; do   # 3.12 first: the version uv uses (.python-version)
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
      PY="$c"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "No Python >= 3.10 found and uv could not be used."
    echo "Install Python from https://www.python.org/downloads/ (or: brew install python / sudo apt install python3-venv), then rerun."
    exit 1
  fi
  say "Creating .venv with $("$PY" --version 2>&1) and pip"
  "$PY" -m venv --clear .venv
  .venv/bin/python -m pip install --upgrade pip
  if [ -n "$PIP_EXTRAS" ]; then .venv/bin/python -m pip install -e ".[$PIP_EXTRAS]"; else .venv/bin/python -m pip install -e .; fi
}

if [ -n "$UV" ]; then
  say "Creating .venv with uv ($("$UV" --version))"
  if ! "$UV" sync "${EXTRAS[@]+"${EXTRAS[@]}"}"; then
    say "uv sync failed (network or proxy?): falling back to python -m venv + pip"
    UV=""
    pip_fallback
  fi
else
  say "uv is unavailable: falling back to python -m venv + pip"
  pip_fallback
fi

py() { if [ -n "${UV:-}" ]; then "$UV" run --no-sync python "$@"; else .venv/bin/python "$@"; fi; }

if [ "$KERNEL" = 1 ] && py -c "import ipykernel" 2>/dev/null; then
  say "Registering the Jupyter kernel 'Python (fmflow)'"
  py -m ipykernel install --user --name fmflow --display-name "Python (fmflow)" >/dev/null && echo "done"
fi

say "Checking the environment"
py -m fmflow.cli doctor || true

cat <<EOF

Setup complete. Activate the environment once per terminal, from this folder:

  source .venv/bin/activate

then:

  python tasks.py demo        # the pre-registered out-of-sample test (about a minute)
  python tasks.py lab         # JupyterLab on the notebooks
  python tasks.py test        # fast tests
  fmflow --help               # the command-line tools
EOF
if [ -n "${UV:-}" ]; then
  # uv may have been installed by this script into a folder the current shell does not have on its PATH yet
  UV_CMD="uv"; command -v uv >/dev/null 2>&1 || UV_CMD="$UV"
  echo ""
  echo "Without activating:  $UV_CMD run python tasks.py demo"
fi
