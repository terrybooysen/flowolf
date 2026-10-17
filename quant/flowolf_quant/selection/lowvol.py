"""Low-volatility selection.

References
----------
- Baker, M., Bradley, B. & Wurgler, J. (2011). "Benchmarks as Limits to
  Arbitrage: Understanding the Low-Volatility Anomaly." Financial Analysts
  Journal 67(1), 40-54.
- Frazzini, A. & Pedersen, L. H. (2014). "Betting Against Beta." Journal of
  Financial Economics 111(1), 1-25 - the leverage-constraint explanation for
  why low-risk assets earn higher risk-adjusted returns.

The anomaly: historically, low-volatility stocks have delivered similar or
higher returns than high-volatility stocks with much less risk, contrary to
the CAPM. Explanations include benchmarked managers' reluctance to hold
low-beta stocks and investors' leverage constraints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import TRADING_DAYS
from ..allocation._rebalance import rolling_weights


def lowvol_scores(asset_returns: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Trailing annualised volatility per asset (lower = more attractive).

    Reference: Baker, Bradley & Wurgler (2011).

    Known at t: standard deviation of returns over (t-lookback, t]. NaN until
    a full window is available (or if the window has missing data).

    Limits: past volatility predicts future volatility well, but says
    nothing about future *returns* on its own.
    """
    return asset_returns.rolling(lookback, min_periods=lookback).std(ddof=1) * np.sqrt(TRADING_DAYS)


def lowvol_weights(asset_returns: pd.DataFrame, lookback: int = 252, top_q: float = 0.3,
                   rebalance_every: int = 21) -> pd.DataFrame:
    """Long-only, equal-weight the ``top_q`` fraction of lowest-vol assets.

    Reference: Baker, Bradley & Wurgler (2011); Frazzini & Pedersen (2014).

    Known at t: ranking on ``lowvol_scores`` at rebalance date t; held until
    the next rebalance. Zero before the first full window. Weights sum to 1.

    Limits: the portfolio concentrates in defensive sectors (utilities,
    staples), carries interest-rate sensitivity and can lag badly in strong
    risk-on rallies. Being long-only and unlevered it captures the
    risk-reduction, not the leveraged BAB factor return. The anomaly's size
    varies a lot by period and market.
    """
    vols = lowvol_scores(asset_returns, lookback)

    def _w(hist: pd.DataFrame, t) -> pd.Series:
        s = vols.loc[t].dropna()
        w = pd.Series(0.0, index=asset_returns.columns)
        if len(s) == 0:
            return w
        n = max(1, int(np.floor(len(s) * top_q)))
        pick = s.sort_values(kind="mergesort").index[:n]
        w[pick] = 1.0 / n
        return w

    return rolling_weights(asset_returns, _w, first=lookback - 1, every=rebalance_every)
