"""Cross-sectional ranks and the equal-weight rank composite (no estimated weights)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


def cs_rank(panel: pd.DataFrame, min_names: int = 5) -> pd.DataFrame:
    """Date-by-date percentile rank minus 0.5 (NaN on dates with fewer than `min_names` valid names).

    Note: the cross-sectional mean of this rank is +0.5/N, not 0. Use `centred_rank` wherever weights are formed
    directly from ranks (a rank-proportional book built from `cs_rank` would carry a net-long bias)."""
    r = panel.rank(axis=1, pct=True) - 0.5
    return r.where(panel.notna().sum(axis=1) >= min_names)


def centred_rank(panel: pd.DataFrame, min_names: int = 5) -> pd.DataFrame:
    """Cross-sectional rank demeaned date by date (sums to exactly zero)."""
    c = cs_rank(panel, min_names)
    return c.sub(c.mean(axis=1), axis=0)


def composite_score(panels: Mapping[str, pd.DataFrame], names: Sequence[str], index: pd.Index | None = None,
                    min_names: int = 5) -> pd.DataFrame:
    """Equal-weight average of the available cross-sectional ranks of `names` (NaN-aware)."""
    out = pd.concat([cs_rank(panels[n], min_names) for n in names]).groupby(level=0).mean()
    return out.reindex(index) if index is not None else out
