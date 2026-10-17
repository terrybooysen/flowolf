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


def effective_trials(trial_returns: pd.DataFrame, duplicate_corr: float = 0.95) -> int:
    """Number of distinct trials, counting near-duplicates once.

    Variants whose daily returns correlate at ``duplicate_corr`` or more (for
    example the same signal at a different position size) are one idea, not
    several. Complete-linkage clustering keeps this conservative: every pair
    inside a cluster must clear the threshold, so genuinely different
    variants are never merged. A higher threshold merges less (stricter).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    M = trial_returns.dropna()
    if M.shape[1] < 2:
        return int(M.shape[1])
    std = M.std(ddof=1).values
    if (std == 0).any():
        # Flat trials (no position ever) carry no information; count them once.
        flat = std == 0
        return effective_trials(M.loc[:, ~flat], duplicate_corr) + (1 if flat.any() else 0)
    C = np.clip(np.nan_to_num(np.corrcoef(M.values.T)), -1.0, 1.0)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="complete")
    return int(len(set(fcluster(Z, 1.0 - duplicate_corr, criterion="distance"))))


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
