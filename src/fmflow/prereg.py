"""Pre-registration: freeze a test specification to disk and fingerprint it BEFORE any result is computed.

The fingerprint is the first 16 hex characters of the SHA-256 of the canonical JSON body (sorted keys, indent 2),
excluding the two fields written at registration time (`sha256_16`, `registered_utc`). Any later edit of the body,
including a changed parameter, breaks `verify`.

    doc = {"title": "...", "frozen": StrategyConfig.fmf_n().to_dict(), "decision_rule": "..."}
    register(doc, "configs/my_test.json")      # before looking at returns
    verify("configs/my_test.json")             # in the evaluation script / notebook
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

_STAMP_FIELDS = ("sha256_16", "registered_utc")


class PreregistrationError(RuntimeError):
    """The pre-registration file was edited after it was hashed."""


def fingerprint(doc: dict) -> str:
    """First 16 hex characters of the SHA-256 of the canonical JSON body (stamp fields excluded)."""
    body = {k: v for k, v in doc.items() if k not in _STAMP_FIELDS}
    return hashlib.sha256(json.dumps(body, sort_keys=True, indent=2).encode()).hexdigest()[:16]


def register(doc: dict, path: str | Path, overwrite: bool = False) -> dict:
    """Stamp `doc` with its fingerprint and the UTC time, and write it to `path` (refuses to overwrite by default)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists: a registration is never silently replaced")
    out = {**{k: v for k, v in doc.items() if k not in _STAMP_FIELDS}, "sha256_16": fingerprint(doc),
           "registered_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def verify(path: str | Path) -> dict:
    """Load a registration and check that its body still matches the stored fingerprint."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if "sha256_16" not in doc:
        raise PreregistrationError(f"{path}: no fingerprint (was it registered with fmflow.prereg.register?)")
    if fingerprint(doc) != doc["sha256_16"]:
        raise PreregistrationError(f"{path}: body does not match fingerprint {doc['sha256_16']} (edited after registration)")
    return doc
