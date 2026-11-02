"""Analyst-data signals in common institutional use (comparables for FMF-N)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fmflow.data.snapshot import TPSnapshot


def analyst_signals(px: pd.DataFrame, snap: TPSnapshot) -> dict[str, pd.DataFrame]:
    """Institutional analyst-data comparables: consensus upside, target-price momentum, low dispersion."""
    out = {
        "tp_upside": np.log(snap.cons_median / px),                                      # consensus implied return
        "cons_change_63": np.log(snap.cons_median) - np.log(snap.cons_median.shift(63)),  # target-price momentum
        "low_dispersion": -snap.dispersion,                                               # Diether-Malloy-Scherbina
    }
    return {k: v.replace([np.inf, -np.inf], np.nan).reindex(index=px.index, columns=px.columns) for k, v in out.items()}
