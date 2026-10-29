"""Matplotlib helpers (optional dependency: pip install fmflow[plot])."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#b4b2ab"]


def equity_curves(books: Mapping[str, pd.Series], window: slice | None = None, title: str = "", path: str | None = None):
    """Growth-of-one chart of several return series over a window (matplotlib)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, (name, r) in enumerate(books.items()):
        rr = r.loc[window] if window is not None else r
        eq = (1 + rr.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, label=name, color=PALETTE[i % len(PALETTE)], lw=2 if i == 0 else 1.3)
    ax.axhline(1, color="#b4b2ab", lw=0.8)
    ax.set_ylabel("Growth of 1")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
