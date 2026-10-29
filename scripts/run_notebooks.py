"""Execute the use-case notebooks (00, 02-09) on any OS (no `make` or bash needed).

    python scripts/run_notebooks.py                 # all of them, in place; stop at the first failure
    python scripts/run_notebooks.py 03 06           # a selection, by number
    python scripts/run_notebooks.py --keep-going    # run everything and report failures at the end
    python scripts/run_notebooks.py --check         # execute without saving (outputs in the repository unchanged)
    python scripts/run_notebooks.py research        # the full research notebook (needs data/private)

The full research notebook (Research_notebook_FMF.ipynb) needs the proprietary development data in data/private/ and is skipped
unless that data is present. Data_Gathering.ipynb downloads data from Kaggle and Yahoo Finance and needs your own
Kaggle credentials, so it is never run automatically.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("numbers", nargs="*", help="two-digit notebook numbers, or 'research' (default: 00 and 02-09)")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--check", action="store_true", help="execute without writing the outputs back")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per cell")
    args = ap.parse_args(argv)
    if sys.platform.startswith("win"):            # silences pyzmq's Proactor warning on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import nbformat
    from nbclient import NotebookClient
    have_private = (ROOT / "data/private/MarketData.csv").exists() and (ROOT / "data/private/Ratings.csv").exists()
    books = sorted(p for p in (ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    if args.numbers:
        books = [p for p in books if p.name[:2] in args.numbers]
        if "research" in args.numbers:
            books.append(ROOT / "notebooks" / "Research_notebook_FMF.ipynb")
    failures = []
    for p in books:
        if p.name.startswith("Research_") and not have_private:
            print(f"skip  {p.name}  (needs data/private/MarketData.csv and Ratings.csv, see DATA_SOURCES.md)")
            continue
        t0 = time.time()
        nb = nbformat.read(p, as_version=4)
        try:
            NotebookClient(nb, timeout=args.timeout, kernel_name="python3", resources={"metadata": {"path": str(p.parent)}}).execute()
            if not args.check:
                nbformat.write(nb, p)
            print(f"ok    {p.name}  ({time.time() - t0:.0f} s)")
        except Exception as e:  # noqa: BLE001
            failures.append(p.name)
            print(f"FAIL  {p.name}: {type(e).__name__}: {str(e).strip().splitlines()[-1][:200]}")
            if not args.keep_going:
                break
    if failures:
        print(f"\n{len(failures)} notebook(s) failed: {', '.join(failures)}. Run `fmflow doctor` to check the environment.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
