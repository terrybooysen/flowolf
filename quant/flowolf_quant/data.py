"""Synthetic market data for tests and smoke runs.

Real prices come from a PriceSource in the app's workers. These generators
exist so every module can be proven on data where the right answer is known:
a pure random walk (no strategy should pass the gate), a series with a
planted, persistent edge (a correct strategy should pass), and a
regime-switching series (regime detectors should find the switches).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def business_days(n: int, start: str = "2010-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def random_walk(n: int = 2520, vol: float = 0.01, drift: float = 0.0,
                seed: int | None = 0) -> pd.Series:
    """Daily simple returns with no predictability."""
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    return pd.Series(r, index=business_days(n), name="ret")


def random_walk_panel(n: int = 2520, k: int = 20, vol: float = 0.015,
                      seed: int | None = 0) -> pd.DataFrame:
    """k independent random-walk assets (daily simple returns)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, vol, (n, k))
    cols = [f"A{i:02d}" for i in range(k)]
    return pd.DataFrame(r, index=business_days(n), columns=cols)


def planted_trend(n: int = 2520, vol: float = 0.01, strength: float = 0.15,
                  lookback: int = 20, seed: int | None = 0) -> pd.Series:
    """Returns whose sign is partly predictable from the trailing mean.

    Each day's expected return is ``strength * vol * sign(mean of last
    `lookback` returns)``. A time-series momentum rule with a similar lookback
    should find this edge; a random-walk test should not.
    """
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    eps = rng.normal(0.0, vol, n)
    for t in range(n):
        if t >= lookback:
            s = np.sign(r[t - lookback:t].mean())
        else:
            s = 0.0
        r[t] = strength * vol * s + eps[t]
    return pd.Series(r, index=business_days(n), name="ret")


def planted_momentum_panel(n: int = 2520, k: int = 20, vol: float = 0.015,
                           strength: float = 0.12, lookback: int = 60,
                           seed: int | None = 0) -> pd.DataFrame:
    """Panel where recent cross-sectional winners keep winning, slightly."""
    rng = np.random.default_rng(seed)
    r = np.zeros((n, k))
    eps = rng.normal(0.0, vol, (n, k))
    for t in range(n):
        if t >= lookback:
            past = r[t - lookback:t].sum(axis=0)
            z = (past - past.mean()) / (past.std() + 1e-12)
            r[t] = strength * vol * np.clip(z, -2, 2) + eps[t]
        else:
            r[t] = eps[t]
    cols = [f"A{i:02d}" for i in range(k)]
    return pd.DataFrame(r, index=business_days(n), columns=cols)


def regime_switching(n: int = 2520, vols: tuple[float, ...] = (0.006, 0.012, 0.03),
                     means: tuple[float, ...] = (0.0006, 0.0, -0.001),
                     stay: float = 0.985, seed: int | None = 0
                     ) -> tuple[pd.Series, pd.Series]:
    """Returns from a hidden Markov chain of calm / choppy / stress states.

    Returns (returns, true_state).
    """
    rng = np.random.default_rng(seed)
    k = len(vols)
    P = np.full((k, k), (1 - stay) / (k - 1))
    np.fill_diagonal(P, stay)
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = rng.choice(k, p=P[s[t - 1]])
    r = rng.normal(np.take(means, s), np.take(vols, s))
    idx = business_days(n)
    return pd.Series(r, index=idx, name="ret"), pd.Series(s, index=idx, name="state")


def prices_from_returns(r: pd.Series | pd.DataFrame, start: float = 100.0):
    return start * (1.0 + r).cumprod()
