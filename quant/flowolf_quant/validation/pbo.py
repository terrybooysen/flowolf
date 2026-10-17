"""Probability of backtest overfitting by combinatorially symmetric
cross-validation (CSCV).

Reference
- Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2017). The
  Probability of Backtest Overfitting. Journal of Computational Finance 20(4).

How it works
Take the daily returns of every variant you tried (columns) over the same
dates (rows). Cut the dates into S equal blocks. For every way of choosing
half the blocks as "in-sample" (the rest are "out-of-sample"):
  1. pick the variant with the best in-sample Sharpe, as you would in
     research;
  2. see where that variant ranks out-of-sample.
PBO is the share of splits in which the in-sample winner lands in the bottom
half out-of-sample. Pure noise gives about 0.5. A real, robust edge gives a
number near 0.

Limits
- Needs the full set of variants actually tried, on the same dates.
- With few variants (under ~10) the rank is coarse and PBO is noisy.
- Blocks are contiguous, which keeps time structure, but S/2 blocks still
  mix early and late periods in-sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass
class PBOResult:
    pbo: float                       # probability of backtest overfitting
    n_splits: int
    n_trials: int
    n_blocks: int
    logits: np.ndarray = field(repr=False)
    is_best_sharpe: np.ndarray = field(repr=False)     # in-sample Sharpe of the chosen variant per split
    oos_sharpe_of_best: np.ndarray = field(repr=False) # its out-of-sample Sharpe per split
    prob_oos_loss: float = 0.0       # share of splits where the chosen variant lost money out of sample
    degradation_slope: float = 0.0   # regression slope of OOS on IS Sharpe (negative = overfit)


def pbo_cscv(trial_returns: pd.DataFrame | np.ndarray, n_blocks: int = 16,
             max_splits: int | None = None, seed: int = 0) -> PBOResult:
    """PBO for a matrix of trial returns (rows = dates, columns = variants)."""
    M = np.asarray(trial_returns, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("need a 2-D matrix with at least two trials (columns)")
    M = M[~np.isnan(M).any(axis=1)]
    T, N = M.shape
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks must be an even number >= 2")
    if T < n_blocks * 5:
        raise ValueError(f"too few rows ({T}) for {n_blocks} blocks")

    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    cnt = np.diff(edges).astype(float)                                  # (S,)
    s1 = np.stack([M[a:b].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])        # (S, N)
    s2 = np.stack([(M[a:b] ** 2).sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])  # (S, N)

    combos = list(combinations(range(n_blocks), n_blocks // 2))
    if max_splits is not None and len(combos) > max_splits:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_splits, replace=False)]
    C = np.zeros((len(combos), n_blocks))
    for i, c in enumerate(combos):
        C[i, list(c)] = 1.0

    def sharpe_from(mask):
        n = mask @ cnt                                   # (K,)
        a = mask @ s1                                    # (K, N)
        b = mask @ s2
        mean = a / n[:, None]
        var = (b - n[:, None] * mean ** 2) / (n[:, None] - 1)
        return mean / np.sqrt(np.maximum(var, 1e-18))

    sr_is = sharpe_from(C)
    sr_oos = sharpe_from(1.0 - C)
    best = sr_is.argmax(axis=1)
    k = np.arange(len(combos))
    # Relative rank of the chosen variant out of sample, in (0, 1).
    oos_ranks = sr_oos.argsort(axis=1).argsort(axis=1)[k, best] + 1     # 1..N
    omega = oos_ranks / (N + 1.0)
    logits = np.log(omega / (1.0 - omega))
    is_best = sr_is[k, best]
    oos_best = sr_oos[k, best]
    slope = float(np.polyfit(is_best, oos_best, 1)[0]) if np.ptp(is_best) > 0 else 0.0
    return PBOResult(pbo=float((logits <= 0).mean()), n_splits=len(combos), n_trials=N,
                     n_blocks=n_blocks, logits=logits, is_best_sharpe=is_best,
                     oos_sharpe_of_best=oos_best, prob_oos_loss=float((oos_best < 0).mean()),
                     degradation_slope=slope)
