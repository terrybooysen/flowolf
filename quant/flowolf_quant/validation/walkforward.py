"""Walk-forward degradation.

Choose the best parameter on a training window, trade it on the next
window, roll forward, repeat. Compare how well the chosen parameters did
in-sample (where they were picked) with how they did out-of-sample (where
they were used). The ratio OOS/IS is the degradation: near 1 means the
research generalises; near 0 or negative means it was fitted to noise.

This is standard practice (Pardo, The Evaluation and Optimization of
Trading Strategies, 2nd ed., 2008); López de Prado (2018) ch. 11-12
discusses its limits: one path through history, so one lucky or unlucky
stretch can dominate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ..backtest import sharpe


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    is_sharpe: float          # average in-sample Sharpe of the chosen parameters
    oos_sharpe: float         # Sharpe of the stitched out-of-sample returns
    degradation: float        # oos_sharpe / is_sharpe (nan if is_sharpe <= 0)
    steps: list = field(default_factory=list)


def walk_forward(strategy_returns: Callable[[object], pd.Series], param_grid: Iterable,
                 train_days: int = 756, test_days: int = 126, anchored: bool = False
                 ) -> WalkForwardResult:
    grid = list(param_grid)
    series = {p: strategy_returns(p) for p in grid}
    n = len(next(iter(series.values())))
    pieces, steps, is_srs = [], [], []
    start = train_days
    while start + 1 <= n:
        end = min(start + test_days, n)
        if n - end < max(test_days // 2, 1):
            end = n                                   # fold a short tail into the last window
        lo = 0 if anchored else start - train_days
        best = max(grid, key=lambda p: sharpe(series[p].iloc[lo:start]))
        is_sr = sharpe(series[best].iloc[lo:start])
        oos = series[best].iloc[start:end]
        pieces.append(oos)
        is_srs.append(is_sr)
        steps.append({"train": (lo, start), "test": (start, end), "param": best,
                      "is_sharpe": is_sr, "oos_sharpe": sharpe(oos)})
        start = end
    if not pieces:
        raise ValueError("series too short for one walk-forward step")
    oos_all = pd.concat(pieces)
    is_mean = float(np.mean(is_srs))
    oos_sr = sharpe(oos_all)
    deg = oos_sr / is_mean if is_mean > 0 else float("nan")
    return WalkForwardResult(oos_returns=oos_all, is_sharpe=is_mean, oos_sharpe=oos_sr,
                             degradation=deg, steps=steps)
