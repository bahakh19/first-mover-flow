"""On-disk cache for the GJR-GARCH forecasts (the only slow step of the pipeline: ~30 s for 10 stocks x 6 years).

The key is a SHA-256 of the return panel's bytes, index and columns plus the GARCH settings, so a cache entry can never
be reused for different data or settings."""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import pandas as pd

from fmflow.config import DataConfig


def _key(ret: pd.DataFrame, cfg: DataConfig) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(ret, index=True).values.tobytes())
    h.update(repr(list(ret.columns)).encode())
    h.update(repr((cfg.garch_min_obs, cfg.garch_refit, cfg.garch_method)).encode())
    return h.hexdigest()[:24]


def cached_garch(ret: pd.DataFrame, cfg: DataConfig = DataConfig(), cache_dir: str | Path = ".fmflow_cache") -> pd.DataFrame:
    """GJR-GARCH forecasts for `ret`, read from or written to a content-addressed on-disk cache."""
    from fmflow.risk.garch import garch_forecast_panel
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"garch_{_key(ret, cfg)}.pkl"
    if f.exists():
        try:
            cached = pickle.loads(f.read_bytes())
            if isinstance(cached, pd.DataFrame):
                return cached
        except Exception:       # truncated entry, or written by an incompatible pandas version: recompute it
            pass
    sig, _ = garch_forecast_panel(ret, cfg)
    tmp = f.with_name(f"{f.name}.{os.getpid()}.tmp")   # atomic replace: safe with parallel workers and on Windows
    try:
        tmp.write_bytes(pickle.dumps(sig))
        os.replace(tmp, f)
    except OSError:             # read-only or full disk: the forecasts are still returned, just not cached
        tmp.unlink(missing_ok=True)
    return sig
