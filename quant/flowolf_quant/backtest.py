"""A small, strict backtester.

Rules that keep it honest:
- Positions decided with data up to day t are applied to day t+1's return.
  The caller passes *target weights as of the close of day t*; this module
  does the one-day shift. Never shift weights yourself before calling it.
- Costs are charged on every change in weight, in basis points of turnover.
- Holding a constant weight means rebalancing back to it every day at no
  cost; drift between rebalances isn't modelled. Fine for daily rules,
  slightly flattering for strategies that would let positions drift.
- ``trades`` counts position entries and exits (sign changes), not daily
  resizing.
- No leverage or shorting limits are imposed here; strategies set their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import TRADING_DAYS


@dataclass
class BacktestResult:
    returns: pd.Series          # net daily strategy returns
    gross_returns: pd.Series
    weights: pd.DataFrame       # weights actually held each day (after shift)
    turnover: pd.Series
    meta: dict = field(default_factory=dict)

    @property
    def trades(self) -> int:
        return count_trades(self.weights)

    @property
    def metrics(self) -> dict:
        return performance(self.returns, self.turnover, trades=self.trades)


def run(asset_returns: pd.Series | pd.DataFrame,
        target_weights: pd.Series | pd.DataFrame,
        cost_bps: float = 5.0) -> BacktestResult:
    """Apply target weights (decided at each close) to the next day's returns."""
    R = asset_returns.to_frame() if isinstance(asset_returns, pd.Series) else asset_returns
    W = target_weights.to_frame() if isinstance(target_weights, pd.Series) else target_weights
    if isinstance(asset_returns, pd.Series) and isinstance(target_weights, pd.Series):
        W.columns = R.columns
    W = W.reindex(index=R.index, columns=R.columns).fillna(0.0)
    held = W.shift(1).fillna(0.0)                     # no look-ahead
    gross = (held * R).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    net = gross - turnover * cost_bps / 1e4
    return BacktestResult(returns=net.rename("strategy"), gross_returns=gross,
                          weights=held, turnover=turnover,
                          meta={"cost_bps": cost_bps})


def count_trades(weights: pd.DataFrame | pd.Series) -> int:
    """Position entries and exits: days on which an asset's position changes
    sign (flat to long, long to short, short to flat, ...). Daily resizing of
    an unchanged position, as volatility scaling does, is not a trade."""
    W = weights.to_frame() if isinstance(weights, pd.Series) else weights
    s = np.sign(W.fillna(0.0).round(12))
    changes = s.diff().fillna(s.iloc[:1].where(s.iloc[:1] != 0, 0)).ne(0)
    return int(changes.values.sum())


def sharpe(r: pd.Series | np.ndarray, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


def max_drawdown(r: pd.Series | np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    return float((wealth / peak - 1.0).min()) if len(r) else 0.0


def cagr(r: pd.Series | np.ndarray, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return 0.0
    total = np.prod(1.0 + r)
    years = len(r) / periods
    return float(total ** (1 / years) - 1) if total > 0 else -1.0


def performance(r: pd.Series, turnover: pd.Series | None = None,
                trades: int | None = None) -> dict:
    out = {
        "days": int(len(r)),
        "cagr": cagr(r),
        "vol": float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0,
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
    }
    if turnover is not None:
        out["turnover_per_year"] = float(turnover.mean() * TRADING_DAYS)
    if trades is not None:
        out["trades"] = int(trades)
    elif turnover is not None:
        out["trades"] = int((turnover > 1e-9).sum())   # fallback: days with any trading
    return out
