"""Cross-sectional momentum (winners minus losers).

Reference
---------
Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling
Losers: Implications for Stock Market Efficiency." Journal of Finance 48(1),
65-91. The "12-1" formation window (skip the most recent month to sidestep
the one-month reversal) follows standard practice, e.g. Fama & French (1996)
and Asness, Moskowitz & Pedersen (2013).

Crash risk: Daniel, K. & Moskowitz, T. (2016). "Momentum Crashes."
Journal of Financial Economics 122(2), 221-247.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..allocation._rebalance import rolling_weights


def momentum_scores(asset_returns: pd.DataFrame, lookback: int = 252,
                    skip: int = 21) -> pd.DataFrame:
    """Compounded return from t-lookback+1 to t-skip, per asset.

    Reference: Jegadeesh & Titman (1993).

    Known at t: the score at t uses returns dated t-lookback+1 ... t-skip,
    all observed by the close of t. With ``skip=0`` it includes day t itself.
    Scores are NaN until ``lookback`` days are available, and NaN for an
    asset with a missing return anywhere in its window.

    Limits: a simple total-return score; it does not risk-adjust, so
    high-vol names dominate both tails.
    """
    if skip >= lookback:
        raise ValueError("skip must be smaller than lookback")
    lr = np.log1p(asset_returns.astype(float))
    cum = lr.cumsum()
    # cum(t-skip) - cum(t-lookback) = sum of log returns over (t-lookback, t-skip]
    score = cum.shift(skip) - cum.shift(lookback)
    # Invalidate windows containing NaNs (cumsum skips NaN silently).
    has_nan = asset_returns.isna().astype(float).rolling(lookback - skip, min_periods=1).sum().shift(skip)
    score = score.where(has_nan == 0)
    return np.expm1(score)


def _quantile_portfolio(scores: pd.Series, top_q: float, bottom_q: float,
                        long_only: bool) -> pd.Series:
    s = scores.dropna()
    w = pd.Series(0.0, index=scores.index)
    k = len(s)
    if k < 2:
        return w
    n_top = max(1, int(np.floor(k * top_q)))
    n_bot = max(1, int(np.floor(k * bottom_q)))
    ranked = s.sort_values(kind="mergesort")
    winners = ranked.index[-n_top:]
    if long_only:
        w[winners] = 1.0 / n_top
        return w
    if n_top + n_bot > k:
        n_top = n_bot = k // 2
        winners = ranked.index[-n_top:]
    losers = ranked.index[:n_bot]
    w[winners] = 1.0 / n_top
    w[losers] = -1.0 / n_bot
    return w


def xsmom_weights(asset_returns: pd.DataFrame, lookback: int = 252, skip: int = 21,
                  top_q: float = 0.2, bottom_q: float = 0.2, long_only: bool = False,
                  rebalance_every: int = 21) -> pd.DataFrame:
    """Target weights for a cross-sectional momentum portfolio.

    Reference: Jegadeesh & Titman (1993).

    Long-short (default): equal-weight long the top ``top_q`` of assets by
    momentum score, equal-weight short the bottom ``bottom_q``. Each leg has
    gross exposure 1 ($1 long, $1 short), so weights sum to 0 (dollar
    neutral) with gross leverage 2. Long-only: equal-weight the winners,
    summing to 1.

    Known at t: on each rebalance date t the ranking uses ``momentum_scores``
    at t (data to t-skip); the target is held until the next rebalance.
    Before the first full lookback the target is zero. Pass straight to
    ``backtest.run``.

    Limits: momentum crashes - the long-short version can lose very heavily
    in sharp market rebounds after bear markets, when the short leg (past
    losers, often high-beta) rallies (Daniel & Moskowitz 2016). It is
    turnover-heavy, so costs matter. With few assets the quantile buckets
    hold one or two names and results are noisy. The short leg assumes you
    can borrow every name at no fee.
    """
    scores = momentum_scores(asset_returns, lookback, skip)

    def _w(hist: pd.DataFrame, t) -> pd.Series:
        return _quantile_portfolio(scores.loc[t], top_q, bottom_q, long_only)

    return rolling_weights(asset_returns, _w, first=lookback, every=rebalance_every)
