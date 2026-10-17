"""Locked holdout.

The most recent stretch of history (default two years) is hidden from all
design work. A strategy is scored on it once. The ledger records every
holdout score; scoring the same strategy family on the same holdout again is
allowed but flagged, because a holdout you keep peeking at stops being
out-of-sample.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .. import TRADING_DAYS


def split_holdout(r: pd.Series | pd.DataFrame, years: float = 2.0):
    """Return (design, holdout) by position: the last ``years`` of rows are held out."""
    n_hold = int(round(years * TRADING_DAYS))
    if n_hold >= len(r):
        raise ValueError("series shorter than the holdout")
    return r.iloc[:-n_hold], r.iloc[-n_hold:]


@dataclass
class HoldoutUse:
    family: str
    holdout_start: str
    holdout_end: str
    times_used_before: int

    @property
    def spent(self) -> bool:
        return self.times_used_before > 0


class HoldoutLedger:
    """Append-only record of holdout scoring (JSON lines)."""

    def __init__(self, path: str):
        self.path = path

    def _entries(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def record(self, family: str, holdout: pd.Series | pd.DataFrame, score: dict) -> HoldoutUse:
        start, end = str(holdout.index[0].date()), str(holdout.index[-1].date())
        before = sum(1 for e in self._entries()
                     if e["family"] == family and e["holdout_start"] == start and e["holdout_end"] == end)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"family": family, "holdout_start": start, "holdout_end": end,
                                "score": score, "at": datetime.now(timezone.utc).isoformat()}) + "\n")
        return HoldoutUse(family, start, end, before)
