"""Generate docs/API.md: every public module, class and function of fmflow with its signature and docstring summary."""
from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
from pathlib import Path

import fmflow

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"fmflow.cli", "fmflow.__main__"}          # __main__ runs the CLI when imported


def first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    return inspect.cleandoc(doc).split("\n\n")[0].replace("\n", " ")


class _Text:
    """Renders as the given text in a signature (annotations without quotes, short dataclass defaults)."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def signature(o) -> str:
    """inspect.signature, made readable: `cfg: DataConfig = DataConfig()` instead of quoted annotations and a full repr
    of every dataclass default."""
    sig = inspect.signature(o)

    def clean(ann):
        return ann if ann is inspect.Parameter.empty else _Text(ann if isinstance(ann, str) else inspect.formatannotation(ann))

    params = []
    for p in sig.parameters.values():
        d = p.default
        if dataclasses.is_dataclass(d) and not isinstance(d, type) and d == type(d)():
            d = _Text(f"{type(d).__name__}()")
        params.append(p.replace(annotation=clean(p.annotation), default=d))
    return str(sig.replace(parameters=params, return_annotation=clean(sig.return_annotation)))


def main() -> None:
    lines = ["# API reference", "", f"Generated from the docstrings of `fmflow` {fmflow.__version__} by "
             "`scripts/make_api_docs.py`. Run `python tasks.py docs` after changing the code.", ""]
    mods = sorted(m.name for m in pkgutil.walk_packages(fmflow.__path__, "fmflow."))
    for name in ["fmflow", *mods]:
        if name in SKIP:
            continue
        try:
            mod = importlib.import_module(name)
        except ImportError as e:                       # optional dependency missing (e.g. JAX)
            lines += [f"## `{name}`", "", f"*Not importable here: {e}*", ""]
            continue
        members = [(n, o) for n, o in inspect.getmembers(mod) if not n.startswith("_")
                   and (inspect.isfunction(o) or inspect.isclass(o)) and getattr(o, "__module__", "") == name]
        if not members and name != "fmflow":
            continue
        lines += [f"## `{name}`", "", first_paragraph(mod.__doc__), ""]
        for n, o in members:
            try:
                sig = signature(o)
            except (TypeError, ValueError):
                sig = "(...)"
            kind = "class" if inspect.isclass(o) else "def"
            lines += [f"- **`{kind} {n}{sig}`**", f"  {first_paragraph(o.__doc__)}" if o.__doc__ else ""]
        lines.append("")
    (ROOT / "docs" / "API.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote docs/API.md ({len(lines)} lines)")


if __name__ == "__main__":
    main()
