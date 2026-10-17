"""Shared helper: compute target weights on a fixed rebalance calendar.

Used by allocation and selection modules. The weight function at date t only
ever receives rows up to and including t, so nothing after t can leak in.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def rebalance_dates(index: pd.Index, first: int, every: int) -> list:
    """Positions ``first, first+every, ...`` in ``index`` (as labels)."""
    if every < 1:
        raise ValueError("rebalance_every must be >= 1")
    if first >= len(index):
        return []
    return list(index[np.arange(first, len(index), every)])


def rolling_weights(asset_returns: pd.DataFrame,
                    weight_fn: Callable[[pd.DataFrame, pd.Timestamp], pd.Series],
                    first: int, every: int) -> pd.DataFrame:
    """Call ``weight_fn(history_up_to_t, t)`` on each rebalance date and hold.

    Weights between rebalance dates are carried forward (the target does not
    change, so the backtester charges no turnover). Before the first
    rebalance date the target is zero (flat).
    """
    out = pd.DataFrame(np.nan, index=asset_returns.index,
                       columns=asset_returns.columns, dtype=float)
    for t in rebalance_dates(asset_returns.index, first, every):
        hist = asset_returns.loc[:t]
        w = weight_fn(hist, t)
        out.loc[t] = w.reindex(asset_returns.columns).fillna(0.0).to_numpy()
    return out.ffill().fillna(0.0)
