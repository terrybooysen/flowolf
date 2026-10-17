"""Locked holdout.

The most recent stretch of history (default two years) is hidden from all
design work. A strategy is scored on it once. The ledger records every
holdout score. Scoring the same strategy family on the same universe again,
on any holdout that *overlaps* an earlier one, is allowed but flagged,
because a holdout you keep peeking at stops being out-of-sample. Changing
the holdout length or trimming a day does not reset the flag.
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
    key: str
    holdout_start: str
    holdout_end: str
    times_used_before: int         # earlier scores on an overlapping holdout

    @property
    def spent(self) -> bool:
        return self.times_used_before > 0


class HoldoutLedger:
    """Append-only record of holdout scoring.

    With a ``path`` it persists as JSON lines; without one it lives in memory
    (fine for tests and one-off runs, useless for keeping research honest
    across sessions, so the app always passes a path or a database-backed
    equivalent).
    """

    def __init__(self, path: str | None = None):
        self.path = path
        self._mem: list[dict] = []

    def _entries(self) -> list[dict]:
        if self.path is None:
            return list(self._mem)
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def record(self, key: str, holdout: pd.Series | pd.DataFrame, score: dict) -> HoldoutUse:
        start, end = holdout.index[0], holdout.index[-1]
        before = 0
        for e in self._entries():
            if e["key"] != key:
                continue
            s, t = pd.Timestamp(e["holdout_start"]), pd.Timestamp(e["holdout_end"])
            if s <= end and start <= t:          # date ranges overlap
                before += 1
        entry = {"key": key, "holdout_start": str(start.date()), "holdout_end": str(end.date()),
                 "score": score, "at": datetime.now(timezone.utc).isoformat()}
        if self.path is None:
            self._mem.append(entry)
        else:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        return HoldoutUse(key, entry["holdout_start"], entry["holdout_end"], before)
