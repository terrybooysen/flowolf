"""Who proposes the next variants to test.

The loop doesn't care whether a grid, a random search or a language model
proposes parameters. What it enforces is the same for all of them:
- proposals are clipped to the family's declared bounds;
- every proposal that gets run is recorded as a trial, so the gate's trial
  count stays honest;
- a proposer only ever sees design-data results, never the holdout.

``LLMProposer`` builds the prompt and parses the reply; the actual model call
is injected (``complete``), so this library has no API keys and no network
access. In the app, the worker passes a function that calls Claude.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .strategies import Family


@dataclass
class TrialSummary:
    params: dict
    sharpe: float
    max_drawdown: float
    trades: int


class Proposer(Protocol):
    def propose(self, family: Family, history: list[TrialSummary], n: int) -> list[dict]: ...


class GridProposer:
    def __init__(self, points_per_param: int = 5):
        self.points = points_per_param
        self._queue: list[dict] | None = None

    def propose(self, family, history, n):
        if self._queue is None:
            self._queue = family.grid(self.points)
        tried = {json.dumps(h.params, sort_keys=True) for h in history}
        out = []
        while self._queue and len(out) < n:
            p = self._queue.pop(0)
            if json.dumps(p, sort_keys=True) not in tried:
                out.append(p)
        return out


class RandomProposer:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def propose(self, family, history, n):
        out = []
        for _ in range(n):
            p = {}
            for k, sp in family.space.items():
                if sp.kind == "choice":
                    p[k] = sp.choices[self.rng.integers(len(sp.choices))]
                else:
                    p[k] = self.rng.uniform(sp.low, sp.high)
            out.append(family.clip(p))
        return out


PROMPT = """You are proposing parameter settings for a trading-strategy family in a research loop.
Family: {name}: {description}
Reference: {reference}
Parameter space (stay inside these bounds): {space}
Results so far on design data only (Sharpe is annualised; every variant you propose will count
against the strategy in a multiple-testing correction, so propose few, well-reasoned variants
rather than many):
{history}
Reply with a JSON list of up to {n} parameter objects and nothing else."""


class LLMProposer:
    """Asks a language model for the next variants. ``complete(prompt) -> str`` is injected."""

    def __init__(self, complete: Callable[[str], str]):
        self.complete = complete

    def build_prompt(self, family: Family, history: list[TrialSummary], n: int) -> str:
        space = {k: ({"choices": list(v.choices)} if v.kind == "choice"
                     else {"type": v.kind, "low": v.low, "high": v.high})
                 for k, v in family.space.items()}
        rows = "\n".join(f"- {json.dumps(h.params, default=str)}: Sharpe {h.sharpe:.2f}, "
                         f"max drawdown {h.max_drawdown:.0%}, {h.trades} trades"
                         for h in history[-40:]) or "(none yet)"
        return PROMPT.format(name=family.name, description=family.description,
                             reference=family.reference, space=json.dumps(space),
                             history=rows, n=n)

    def propose(self, family, history, n):
        reply = self.complete(self.build_prompt(family, history, n))
        m = re.search(r"\[.*\]", reply, re.S)
        if not m:
            return []
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        return [family.clip(p) for p in items[:n] if isinstance(p, dict)]
