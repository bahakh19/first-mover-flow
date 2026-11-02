"""Insert or refresh the self-contained setup cell at the top of every notebook (maintainers only; idempotent).

    python scripts/refresh_setup_cell.py            # all notebooks except the research notebook

The cell lets a notebook run as-is in any Jupyter: inside the repository with the package installed it only finds the
repository; on Google Colab, Binder or a bare Jupyter it downloads the repository (git, or the GitHub zip archive when
git is missing) and installs the package (pip, or uv when the kernel has no pip). Edit TEMPLATE, then rerun.
Re-execute the notebooks afterwards (`python tasks.py notebooks`) so that the stored outputs match the new cell.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARK = "# fmflow-setup"
TEMPLATE = '''# fmflow-setup: runs as-is everywhere. In an installed environment it only locates the repository; on Google Colab,
# Binder or a Jupyter without the package, it downloads the repository once and installs it (about a minute).
import importlib, importlib.util, os, shutil, subprocess, sys
from pathlib import Path

REPO_URL = os.environ.get("FMFLOW_REPO_URL", "https://github.com/bahakh19/first-mover-flow")   # set it to use a fork


def _run(cmd: list, what: str, hint: str = "") -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{what} failed:\\n{(r.stderr or r.stdout).strip()[-1500:]}\\n{hint}")


_here = Path.cwd()
_root = next((p for p in (_here, *_here.parents, _here / "first-mover-flow") if (p / "src" / "fmflow").is_dir()), None)
if _root is None:                                                  # notebooks need the repository's data/ folder
    _root = (_here / "first-mover-flow").resolve()
    print(f"Downloading {REPO_URL} ...")
    if shutil.which("git"):
        _run(["git", "clone", "-q", "--depth", "1", REPO_URL, str(_root)], f"git clone {REPO_URL}",
             "Is the repository public? For a fork or a local copy, set the environment variable FMFLOW_REPO_URL.")
    else:                                                          # no git: the GitHub zip archive of the main branch
        import io, urllib.request, zipfile
        with urllib.request.urlopen(REPO_URL.rstrip("/") + "/archive/refs/heads/main.zip") as _r:
            zipfile.ZipFile(io.BytesIO(_r.read())).extractall(_here)
        (_here / (REPO_URL.rstrip("/").split("/")[-1] + "-main")).rename(_root)
if importlib.util.find_spec("fmflow") is None:
    print("Installing fmflow and its dependencies (about a minute) ...")
    _target = f"{_root}[EXTRAS]"
    if importlib.util.find_spec("pip"):
        _run([sys.executable, "-m", "pip", "install", "-q", "-e", _target], "pip install")
    elif shutil.which("uv"):                                       # uv-created environments have no pip
        _run(["uv", "pip", "install", "-q", "--python", sys.executable, "-e", _target], "uv pip install")
    else:
        raise RuntimeError("this kernel has neither pip nor uv: install the package from a terminal (INSTALL.md)")
    sys.path.insert(0, str(_root / "src"))
    importlib.invalidate_caches()
os.chdir(_root / "notebooks")
print("fmflow", importlib.import_module("fmflow").__version__, "| repository:", _root)'''

EXTRAS = {"06": "plot,gen", "Da": "plot,data"}           # by the first two characters of the notebook's file name


def refresh(path: Path) -> None:
    import nbformat
    nb = nbformat.read(path, as_version=4)
    code = TEMPLATE.replace("EXTRAS", EXTRAS.get(path.name[:2], "plot"))
    existing = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source.startswith(MARK)]
    if existing:
        nb.cells[existing[0]].source = code
    else:
        first_code = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code")
        nb.cells.insert(first_code, nbformat.v4.new_code_cell(code))
    nbformat.validator.normalize(nb)
    nbformat.write(nb, path)
    print("setup cell ->", path.relative_to(ROOT))


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or [p for p in sorted((ROOT / "notebooks").glob("*.ipynb"))
                                        if not p.name.startswith("Research")]
    for p in paths:
        refresh(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
