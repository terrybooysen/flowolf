"""Monte Carlo risk via the stationary block bootstrap.

Reference
---------
Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap."
Journal of the American Statistical Association 89(428), 1303-1313.

The stationary bootstrap resamples blocks of consecutive returns whose
lengths are geometric with mean ``block_mean``, wrapping around the end of
the sample. Keeping blocks together preserves short-range dependence such as
volatility clustering, which an i.i.d. bootstrap would destroy; random block
lengths make the resampled series stationary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap(returns: pd.Series | np.ndarray, n_paths: int = 2000,
                    block_mean: float = 20, horizon: int | None = None,
                    seed: int | None = None) -> np.ndarray:
    """Resampled return paths, shape (n_paths, horizon).

    Reference: Politis & Romano (1994).

    Known at t: whatever history you pass in. To use it in a live or
    walk-forward setting, pass only returns up to t.

    ``horizon`` defaults to the sample length. ``block_mean`` is the average
    block length in days (p = 1 / block_mean of starting a new block each
    day); ``block_mean=1`` reduces to the i.i.d. bootstrap.

    Limits: it can only replay the past. Crashes larger than any in the
    sample, regime changes, and dependence longer than a few blocks are not
    represented. Results depend on block_mean; 10-40 days is typical for
    daily data.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n == 0:
        raise ValueError("no returns to bootstrap")
    if block_mean < 1:
        raise ValueError("block_mean must be >= 1")
    H = n if horizon is None else int(horizon)
    rng = np.random.default_rng(seed)
    p = 1.0 / block_mean
    new_block = rng.random((n_paths, H)) < p
    new_block[:, 0] = True
    starts = rng.integers(0, n, size=(n_paths, H))
    idx = np.empty((n_paths, H), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, H):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return r[idx]


def _max_drawdowns(paths: np.ndarray) -> np.ndarray:
    wealth = np.cumprod(1.0 + paths, axis=1)
    wealth = np.concatenate([np.ones((paths.shape[0], 1)), wealth], axis=1)
    peak = np.maximum.accumulate(wealth, axis=1)
    return (wealth / peak - 1.0).min(axis=1)


def drawdown_distribution(returns: pd.Series | np.ndarray, n_paths: int = 2000,
                          block_mean: float = 20, horizon: int | None = None,
                          seed: int | None = None) -> dict:
    """Distribution of maximum drawdown across bootstrapped paths.

    Reference: Politis & Romano (1994) for the resampling.

    Returns a dict with ``median``, ``p5`` and ``p1`` (the 50th, 5th and 1st
    percentile of max drawdown; negative numbers, so p1 is the worst),
    plus ``mean``, ``historical`` (the realised max drawdown of the input),
    ``n_paths`` and ``horizon``.

    Known at t: whatever history you pass in. Limits: as for
    ``block_bootstrap`` - the tails are bounded by what the sample contains.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    paths = block_bootstrap(r, n_paths=n_paths, block_mean=block_mean,
                            horizon=horizon, seed=seed)
    mdd = _max_drawdowns(paths)
    return {
        "median": float(np.median(mdd)),
        "p5": float(np.quantile(mdd, 0.05)),
        "p1": float(np.quantile(mdd, 0.01)),
        "mean": float(mdd.mean()),
        "historical": float(_max_drawdowns(r[None, :])[0]),
        "n_paths": int(n_paths),
        "horizon": int(paths.shape[1]),
    }


def ruin_probability(returns: pd.Series | np.ndarray, ruin_level: float = -0.5,
                     horizon: int = 252 * 5, n_paths: int = 2000,
                     block_mean: float = 20, seed: int | None = None) -> float:
    """Share of bootstrapped paths whose drawdown reaches ``ruin_level``.

    Reference: Politis & Romano (1994) for the resampling.

    ``ruin_level`` is a drawdown from the running peak (e.g. -0.5 = losing
    half from the high-water mark) at any point within ``horizon`` days.

    Known at t: whatever history you pass in. Limits: a probability of 0
    only means no resampled path got there - it is not proof of safety,
    especially when the sample has no crash in it. Leverage changes the
    answer non-linearly; bootstrap the levered return series, not the
    unlevered one scaled afterwards.
    """
    if ruin_level >= 0:
        raise ValueError("ruin_level must be negative, e.g. -0.5")
    paths = block_bootstrap(returns, n_paths=n_paths, block_mean=block_mean,
                            horizon=horizon, seed=seed)
    return float((_max_drawdowns(paths) <= ruin_level).mean())
