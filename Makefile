# Shortcuts for macOS / Linux. Every target calls tasks.py, which also works on Windows: python tasks.py <target>.
# Uses uv's environment when uv is installed, otherwise the active Python.
PY := $(shell command -v uv >/dev/null 2>&1 && [ -d .venv ] && echo "uv run --no-sync python" || echo python)

.PHONY: setup doctor demo test test-all lint docs lab notebooks synth development audit power clean help

help:               ## list the targets
	@$(PY) tasks.py

setup:              ## create .venv with everything (installs uv and Python if needed)
	./setup.sh

doctor demo test test-all lint docs lab notebooks synth development audit power clean:
	$(PY) tasks.py $@
