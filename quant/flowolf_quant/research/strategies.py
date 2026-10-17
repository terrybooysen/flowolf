"""Strategy families the research loop can test.

A family is one idea (say, time-series momentum) with a declared parameter
space. Every point tried in that space becomes a trial, and the gate judges
the best one against all of them.

Weight functions return target weights known at each close; the backtester
applies them to the next day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Param:
    kind: str                       # "int", "float" or "choice"
    low: float | None = None
    high: float | None = None
    choices: tuple = ()

    def clip(self, v):
        if self.kind == "choice":
            return v if v in self.choices else self.choices[0]
        v = min(max(v, self.low), self.high)
        return int(round(v)) if self.kind == "int" else float(v)

    def grid(self, n: int = 5) -> list:
        if self.kind == "choice":
            return list(self.choices)
        vals = np.linspace(self.low, self.high, n)
        if self.kind == "int":
            return sorted({int(round(v)) for v in vals})
        return [float(v) for v in vals]


@dataclass
class Family:
    name: str
    description: str
    reference: str
    weights: Callable[..., pd.Series | pd.DataFrame]   # (asset_returns, **params) -> target weights
    space: dict[str, Param]
    universe: str = "single"                            # "single" (one series) or "panel"
    defaults: dict = field(default_factory=dict)

    def clip(self, params: dict) -> dict:
        out = dict(self.defaults)
        for k, p in self.space.items():
            if k in params:
                out[k] = p.clip(params[k])
        return out

    def grid(self, n: int = 5) -> list[dict]:
        keys = list(self.space)
        combos = [{}]
        for k in keys:
            combos = [dict(c, **{k: v}) for c in combos for v in self.space[k].grid(n)]
        return [self.clip(c) for c in combos]


# --- built-in families -----------------------------------------------------------------

def _trend_sign(r: pd.Series, lookback: int = 20, long_only: bool = False) -> pd.Series:
    s = np.sign(r.rolling(lookback).mean()).fillna(0.0)
    return s.clip(lower=0.0) if long_only else s


def _ma_cross(r: pd.Series, fast: int = 20, slow: int = 100, long_only: bool = True) -> pd.Series:
    p = (1.0 + r).cumprod()
    if fast >= slow:
        return pd.Series(0.0, index=r.index)
    s = np.sign(p.rolling(fast).mean() - p.rolling(slow).mean()).fillna(0.0)
    return s.clip(lower=0.0) if long_only else s


def _tsmom(r: pd.Series, lookback: int = 252, vol_target: float = 0.10, long_only: bool = False):
    from ..regimes.tsmom import tsmom_weights
    return tsmom_weights(r, lookback=lookback, vol_target=vol_target, long_only=long_only)


def _xsmom(R: pd.DataFrame, lookback: int = 252, skip: int = 21, top_q: float = 0.2,
           long_only: bool = False):
    from ..selection.xsmom import xsmom_weights
    return xsmom_weights(R, lookback=lookback, skip=skip, top_q=top_q, bottom_q=top_q,
                         long_only=long_only)


def _lowvol(R: pd.DataFrame, lookback: int = 252, top_q: float = 0.3):
    from ..selection.lowvol import lowvol_weights
    return lowvol_weights(R, lookback=lookback, top_q=top_q)


FAMILIES: dict[str, Family] = {
    "trend_sign": Family(
        "trend_sign", "Long when the trailing average return is positive, short (or flat) when negative.",
        "Moskowitz, Ooi & Pedersen (2012), J. Financial Economics 104(2) - simplified form",
        _trend_sign, {"lookback": Param("int", 5, 250)}, defaults={"long_only": False}),
    "ma_cross": Family(
        "ma_cross", "Long when the fast moving average of price is above the slow one.",
        "Brock, Lakonishok & LeBaron (1992), J. Finance 47(5)",
        _ma_cross, {"fast": Param("int", 5, 60), "slow": Param("int", 50, 250)},
        defaults={"long_only": True}),
    "tsmom": Family(
        "tsmom", "Time-series momentum with volatility scaling.",
        "Moskowitz, Ooi & Pedersen (2012), J. Financial Economics 104(2)",
        _tsmom, {"lookback": Param("int", 21, 252), "vol_target": Param("float", 0.05, 0.20)}),
    "xsmom": Family(
        "xsmom", "Cross-sectional momentum: own recent winners, short recent losers.",
        "Jegadeesh & Titman (1993), J. Finance 48(1)",
        _xsmom, {"lookback": Param("int", 63, 252), "top_q": Param("float", 0.1, 0.3)},
        universe="panel", defaults={"skip": 21, "long_only": False}),
    "lowvol": Family(
        "lowvol", "Own the lowest-volatility names.",
        "Baker, Bradley & Wurgler (2011), Financial Analysts Journal 67(1)",
        _lowvol, {"lookback": Param("int", 63, 252), "top_q": Param("float", 0.2, 0.5)},
        universe="panel"),
}
