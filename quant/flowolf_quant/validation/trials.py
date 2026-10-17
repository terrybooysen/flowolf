"""Trials registry: every variant ever tested, failures included.

Deflated Sharpe and PBO are only honest if they see every variant that was
tried. This registry is append-only: you can add trials, never remove them.
In the app the same records live in the ``trials`` table (append-only by
database trigger); this is the library-side equivalent.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..backtest import performance


@dataclass
class Trial:
    family: str
    params: dict
    metrics: dict
    returns: pd.Series = field(repr=False)
    id: str = ""


class TrialRegistry:
    def __init__(self, path: str | None = None):
        self.path = path
        self._trials: list[Trial] = []
        if path and os.path.exists(path):
            self._load()

    def add(self, family: str, params: dict, returns: pd.Series,
            turnover: pd.Series | None = None) -> Trial:
        key = json.dumps({"family": family, "params": params}, sort_keys=True, default=str)
        tid = hashlib.sha1(f"{key}|{len(self._trials)}".encode()).hexdigest()[:12]
        t = Trial(family=family, params=dict(params), metrics=performance(returns, turnover),
                  returns=returns.astype(float), id=tid)
        self._trials.append(t)
        if self.path:
            self._append(t)
        return t

    def family(self, family: str) -> list[Trial]:
        return [t for t in self._trials if t.family == family]

    def count(self, family: str | None = None) -> int:
        return len(self._trials if family is None else self.family(family))

    def matrix(self, family: str) -> pd.DataFrame:
        """Daily returns of every trial in a family, aligned on common dates."""
        ts = self.family(family)
        if not ts:
            raise KeyError(f"no trials for family {family!r}")
        return pd.concat({t.id: t.returns for t in ts}, axis=1).dropna()

    def best(self, family: str, key: str = "sharpe") -> Trial:
        return max(self.family(family), key=lambda t: t.metrics[key])

    # --- persistence (JSON lines; returns stored alongside) ---
    def _append(self, t: Trial):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        rec = {"id": t.id, "family": t.family, "params": t.params, "metrics": t.metrics,
               "dates": [str(d.date()) for d in t.returns.index],
               "returns": [float(x) for x in t.returns.values],
               "at": datetime.now(timezone.utc).isoformat()}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def _load(self):
        with open(self.path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                r = pd.Series(rec["returns"], index=pd.to_datetime(rec["dates"]))
                self._trials.append(Trial(rec["family"], rec["params"], rec["metrics"], r, rec["id"]))
