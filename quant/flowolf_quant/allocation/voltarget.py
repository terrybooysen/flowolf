"""Volatility targeting.

References
----------
- Moreira, A. & Muir, T. (2017). "Volatility-Managed Portfolios."
  Journal of Finance 72(4), 1611-1644.
- Harvey, C. R., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M. &
  Van Hemert, O. (2018). "The Impact of Volatility Targeting."
  Journal of Portfolio Management 45(1), 14-33.
- RiskMetrics (J.P. Morgan, 1996) for the EWMA variance estimator.

Idea: hold ``target / forecast_vol`` units of a strategy, so exposure falls
when recent volatility is high and rises when it is low. Harvey et al. find
this reliably reduces tail events and vol-of-vol for risk assets; any Sharpe
improvement is smaller and less reliable than Moreira & Muir's headline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import TRADING_DAYS


def _vol_forecast(returns: pd.Series, lookback: int, method: str) -> pd.Series:
    """Annualised volatility estimate using returns up to and including t."""
    r = returns.astype(float)
    if method == "ewma":
        # Zero-mean EWMA variance (RiskMetrics style); span = lookback.
        var = (r ** 2).ewm(span=lookback, min_periods=lookback, adjust=False).mean()
    elif method == "rolling":
        var = r.rolling(lookback, min_periods=lookback).var(ddof=1)
    else:
        raise ValueError("method must be 'ewma' or 'rolling'")
    return np.sqrt(var * TRADING_DAYS)


def vol_target_scale(returns: pd.Series, target: float = 0.10, lookback: int = 60,
                     method: str = "ewma", max_leverage: float = 2.0) -> pd.Series:
    """Scale factor to apply to a return stream so it runs at ``target`` vol.

    Reference: Moreira & Muir (2017); Harvey et al. (2018).

    Known at t: the value at date t uses only ``returns`` up to and including
    t (the close of day t). Pass it to ``backtest.run`` as a target weight;
    the backtester applies it to day t+1. Do not shift it yourself.

    Returns a Series in [0, max_leverage]. During the warm-up (fewer than
    ``lookback`` observations) the scale is 0, i.e. stay flat until there is
    enough data to estimate volatility.

    Limits: volatility forecasts lag sudden jumps (the first day of a crash is
    taken at the pre-crash exposure); the scheme raises leverage in quiet
    markets, which is exactly when a calm-before-the-storm regime can bite;
    and it adds turnover. ``max_leverage`` is the main guard - keep it modest.
    """
    if target <= 0:
        raise ValueError("target must be positive")
    vol = _vol_forecast(returns, lookback, method)
    scale = (target / vol.replace(0.0, np.nan)).clip(upper=max_leverage)
    return scale.fillna(0.0).rename("vol_scale")


def apply_vol_target(weights: pd.DataFrame | pd.Series, asset_returns: pd.DataFrame | pd.Series,
                     target: float = 0.10, lookback: int = 60, method: str = "ewma",
                     max_leverage: float = 2.0) -> pd.DataFrame:
    """Scale a portfolio's target weights so its *ex-ante* vol hits ``target``.

    Reference: Moreira & Muir (2017); Harvey et al. (2018).

    For each date t the current target weights w_t are applied to the trailing
    ``lookback`` days of asset returns (up to and including t); the volatility
    of that hypothetical portfolio is the forecast. ``method='ewma'`` weights
    recent days more (span = lookback); ``'rolling'`` is an equal-weighted
    sample. The weights are then multiplied by ``target / forecast``, with
    gross leverage capped at ``max_leverage`` (sum of |weights|).

    Known at t: only weights decided at t and asset returns up to t. Output is
    still a set of *target* weights at the close of t - pass straight to
    ``backtest.run``.

    Limits: as for ``vol_target_scale``; plus the covariance estimate is noisy
    when the number of assets approaches ``lookback``. Rows before enough
    history exists are set to zero.
    """
    W = weights.to_frame() if isinstance(weights, pd.Series) else weights
    R = asset_returns.to_frame() if isinstance(asset_returns, pd.Series) else asset_returns
    if isinstance(weights, pd.Series) and isinstance(asset_returns, pd.Series):
        W = W.copy()
        W.columns = R.columns
    W = W.reindex(index=R.index, columns=R.columns).fillna(0.0)
    Rv = R.fillna(0.0).to_numpy(dtype=float)
    Wv = W.to_numpy(dtype=float)
    n = len(R)
    if method == "ewma":
        lam = 1.0 - 2.0 / (lookback + 1.0)
        kern = lam ** np.arange(lookback - 1, -1, -1)   # oldest .. newest
        kern = kern / kern.sum()
    elif method == "rolling":
        kern = None
    else:
        raise ValueError("method must be 'ewma' or 'rolling'")

    out = np.zeros_like(Wv)
    for t in range(lookback - 1, n):
        w = Wv[t]
        if not np.any(w):
            continue
        p = Rv[t - lookback + 1:t + 1] @ w
        if kern is None:
            var = p.var(ddof=1)
        else:
            var = float(np.sum(kern * p ** 2))       # zero-mean EWMA
        vol = np.sqrt(var * TRADING_DAYS)
        if vol <= 0:
            continue
        scaled = w * (target / vol)
        gross = np.abs(scaled).sum()
        if gross > max_leverage:
            scaled *= max_leverage / gross
        out[t] = scaled
    res = pd.DataFrame(out, index=R.index, columns=R.columns)
    if isinstance(weights, pd.Series):
        return res.iloc[:, 0].rename(weights.name)
    return res
