"""Time-series momentum (TSMOM).

Reference
---------
Moskowitz, T. J., Ooi, Y. H. & Pedersen, L. H. (2012). "Time series
momentum." Journal of Financial Economics 104(2), 228-250.

The paper's rule: go long an instrument whose own excess return over the
past 12 months is positive, short if negative, and size each position to a
constant ex-ante volatility (40% per instrument in the paper, using an EWMA
variance with centre of mass 60 days). Here the lookback, vol target and vol
estimator are parameters.

Timing
------
Every output at index t uses returns up to and including day t only (the
close of t). Do not shift: ``flowolf_quant.backtest.run`` applies weights
decided at t to the return of t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import TRADING_DAYS


def _trailing_return(returns: pd.Series, lookback: int, skip: int) -> pd.Series:
    """Compounded return over days t-skip-lookback+1 .. t-skip (log-sum)."""
    lr = np.log1p(pd.Series(returns).astype(float))
    s = lr.rolling(lookback, min_periods=lookback).sum()
    return s.shift(skip) if skip else s


def tsmom_signal(returns: pd.Series, lookback: int = 252, skip: int = 0) -> pd.Series:
    """Sign of the trailing ``lookback``-day return: +1 / 0 / -1.

    Moskowitz, Ooi & Pedersen (2012). ``skip`` excludes the most recent
    ``skip`` days (e.g. skip=21 for a 12-1 month variant).

    Known at time t: uses returns[t-skip-lookback+1 .. t-skip]. Value is 0
    (flat) until enough history exists, and when the trailing return is
    exactly zero.

    Limits: the sign throws away magnitude; a single lookback whipsaws in
    range-bound markets and is late at turning points (it stays long into
    the first part of a crash and re-enters late).
    """
    tr = _trailing_return(returns, lookback, skip)
    return np.sign(tr).fillna(0.0).rename("tsmom_signal")


def ex_ante_vol(returns: pd.Series, vol_lookback: int = 60, method: str = "ewma",
                annualise: bool = True) -> pd.Series:
    """Ex-ante daily vol estimate known at the close of day t.

    ``method='ewma'`` follows MOP (2012): exponentially weighted variance with
    centre of mass ``vol_lookback`` days (squared deviations from the EWMA
    mean). ``'rolling'`` is a plain rolling standard deviation. Both use
    returns up to and including t. NaN until ``min(vol_lookback, 20)``
    observations exist.
    """
    r = pd.Series(returns).astype(float)
    minp = min(vol_lookback, 20)
    if method == "ewma":
        v = r.ewm(com=vol_lookback, min_periods=minp, adjust=True).std(bias=True)
    elif method == "rolling":
        v = r.rolling(vol_lookback, min_periods=minp).std(ddof=1)
    else:
        raise ValueError("method must be 'ewma' or 'rolling'")
    if annualise:
        v = v * np.sqrt(TRADING_DAYS)
    return v.rename("ex_ante_vol")


def tsmom_weights(returns: pd.Series, lookback: int = 252, vol_target: float = 0.10,
                  vol_lookback: int = 60, max_leverage: float = 2.0,
                  long_only: bool = False, skip: int = 0,
                  vol_method: str = "ewma") -> pd.Series:
    """Vol-scaled TSMOM target weight decided at the close of day t.

    weight_t = signal_t * min(vol_target / sigma_t, max_leverage), with
    sigma_t the annualised ex-ante vol from ``ex_ante_vol`` (past data only)
    and signal_t from ``tsmom_signal``. ``long_only`` maps short signals to
    flat. Pass straight into ``backtest.run`` (it does the one-day shift).

    Reference: Moskowitz, Ooi & Pedersen (2012), section on the
    diversified TSMOM factor (40% target per asset there; 10% default here).

    Limits: realised vol will differ from target (vol estimates lag, vol
    clusters); the leverage cap binds in very calm periods; costs not
    included here (the backtester charges them on weight changes).
    """
    sig = tsmom_signal(returns, lookback=lookback, skip=skip)
    if long_only:
        sig = sig.clip(lower=0.0)
    vol = ex_ante_vol(returns, vol_lookback=vol_lookback, method=vol_method)
    scale = (vol_target / vol).clip(upper=max_leverage)
    w = (sig * scale).fillna(0.0)
    return w.rename("tsmom_weight")


def trend_regime(returns: pd.Series, lookbacks: tuple[int, ...] = (21, 63, 252)) -> pd.Series:
    """Trend score in [-1, 1]: average of TSMOM signs over several lookbacks.

    +1 means every horizon is up, -1 every horizon down, near 0 mixed. A
    lookback contributes 0 until it has enough history. Known at the close
    of t (returns up to t).

    Reference: multi-horizon averaging as in the TSMOM literature (MOP 2012
    report 1-48 month lookbacks; combining horizons smooths the signal).

    Limits: describes direction of past returns, not future returns; slow
    lookbacks lag at turning points, fast ones whipsaw.
    """
    if not lookbacks:
        raise ValueError("need at least one lookback")
    sigs = [tsmom_signal(returns, lookback=lb) for lb in lookbacks]
    return (sum(sigs) / len(sigs)).rename("trend_regime")
