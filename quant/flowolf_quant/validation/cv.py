"""Purged k-fold cross-validation with an embargo.

Reference
- López de Prado, M. (2018). Advances in Financial Machine Learning, Wiley,
  ch. 7 (purging and embargo).

Why
Ordinary k-fold leaks: a training day next to the test block shares data
with it. Two leaks matter for trading rules:
- a training observation whose *outcome window* (label horizon) runs into
  the test block, and
- a training observation just after the test block whose *features* (the
  look-back window of the rule) reach back into the test block.
Purging drops both. The embargo drops a further buffer after each test
block, for serial correlation the purge doesn't catch.

What it gives you
``purged_kfold`` returns train/test index arrays. ``cv_select_and_score``
uses them to pick a parameter on the training folds only and score it on
the test fold, so the out-of-sample returns it stitches together were never
used to choose anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ..backtest import sharpe


def purged_kfold(n_obs: int, k: int = 5, label_horizon: int = 1,
                 feature_lookback: int = 0, embargo: int | float = 0.01
                 ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous k-fold splits with purging and an embargo.

    label_horizon: days after an observation that its outcome depends on.
    feature_lookback: days before an observation that its inputs use.
    embargo: extra days dropped after each test block (int), or a fraction of
    n_obs (float < 1).
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    emb = int(np.ceil(embargo * n_obs)) if isinstance(embargo, float) and embargo < 1 else int(embargo)
    edges = np.linspace(0, n_obs, k + 1).astype(int)
    idx = np.arange(n_obs)
    folds = []
    for a, b in zip(edges[:-1], edges[1:]):
        test = idx[a:b]
        # Before the test block: drop obs whose outcome window reaches into it.
        lo = max(0, a - label_horizon)
        # After the test block: drop obs whose look-back reaches into it, plus embargo.
        hi = min(n_obs, b + max(feature_lookback, 0) + emb)
        train = np.concatenate([idx[:lo], idx[hi:]])
        folds.append((train, test))
    return folds


@dataclass
class CVResult:
    oos_returns: pd.Series
    oos_sharpe: float
    chosen: list = field(default_factory=list)     # parameter chosen in each fold
    fold_sharpes: list = field(default_factory=list)


def cv_select_and_score(strategy_returns: Callable[[object], pd.Series],
                        param_grid: Iterable, *, feature_lookback: int,
                        n_obs: int | None = None, k: int = 5,
                        label_horizon: int = 1, embargo: int | float = 0.01) -> CVResult:
    """Choose a parameter on purged training folds; score it on the test fold.

    ``strategy_returns(param)`` must return the strategy's daily returns over
    the full sample (the rule itself must only use past data).
    ``feature_lookback`` is required: set it to the longest look-back any
    parameter in the grid uses, so training days just after a test block,
    whose positions were computed from test-block returns, are purged.
    """
    grid = list(param_grid)
    series = {p: strategy_returns(p) for p in grid}
    first = next(iter(series.values()))
    n = n_obs or len(first)
    pieces, chosen, fold_sr = [], [], []
    for train, test in purged_kfold(n, k, label_horizon, feature_lookback, embargo):
        best = max(grid, key=lambda p: sharpe(series[p].iloc[train]))
        oos = series[best].iloc[test]
        pieces.append(oos)
        chosen.append(best)
        fold_sr.append(sharpe(oos))
    oos_all = pd.concat(pieces)
    return CVResult(oos_returns=oos_all, oos_sharpe=sharpe(oos_all), chosen=chosen,
                    fold_sharpes=fold_sr)
