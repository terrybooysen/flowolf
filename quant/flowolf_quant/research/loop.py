"""The research loop: propose, test, record every trial, let the gate decide.

This is the "self-improving" loop done the honest way. The Hermes template
that lost money let one noisy result change the strategy. Here:
1. the last two years are locked away as a holdout before anything runs;
2. a proposer (grid, random or a language model) suggests variants;
3. every variant is backtested on the design data and recorded as a trial;
4. the best variant by design Sharpe is scored once on the holdout, and the
   use is logged (overlapping reuse is flagged);
5. the gate judges it against *all* trials tried on exactly the same design
   data (deflated Sharpe, PBO, holdout, data length, trade count).
Nothing is ever promoted on a single number.

Trials are keyed by family, universe and a fingerprint of the exact design
data (its assets, dates and values). Trials from a run with a different
holdout boundary, different data or a different asset can never be pooled
into this run's selection or statistics.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .. import backtest as bt
from ..validation import GateConfig, HoldoutLedger, TrialRegistry, Verdict, evaluate, split_holdout
from .proposers import GridProposer, Proposer, TrialSummary
from .strategies import FAMILIES, Family


@dataclass
class ResearchReport:
    family: str
    universe: str
    trial_key: str
    best_params: dict
    design_metrics: dict
    holdout_metrics: dict
    n_trials: int
    verdict: Verdict
    trials: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.to_dict()
        return d


def fingerprint(x: pd.Series | pd.DataFrame) -> str:
    """Short hash of the assets, dates and values of a return series or panel."""
    df = x.to_frame() if isinstance(x, pd.Series) else x
    h = hashlib.sha1()
    h.update("|".join(map(str, df.columns)).encode())
    h.update(np.asarray(df.index.asi8).tobytes())
    h.update(np.round(df.to_numpy(dtype=float), 12).tobytes())
    return h.hexdigest()[:12]


def _assets(x: pd.Series | pd.DataFrame) -> str:
    cols = [x.name] if isinstance(x, pd.Series) else list(x.columns)
    return hashlib.sha1("|".join(map(str, cols)).encode()).hexdigest()[:8]


def research(asset_returns: pd.Series | pd.DataFrame, family: Family | str, *,
             universe: str, ledger: HoldoutLedger,
             proposer: Proposer | None = None, rounds: int = 1, per_round: int = 50,
             registry: TrialRegistry | None = None, holdout_years: float = 2.0,
             cost_bps: float = 5.0, config: GateConfig = GateConfig()) -> ResearchReport:
    """Research one family on one universe.

    universe: a stable name for what's being traded (e.g. "NPN.JO",
        "house-universe"). Required, so different assets are never pooled.
    ledger: where holdout use is recorded. Required: an unlogged holdout can
        be peeked at without anyone knowing. Pass a persistent ledger in real use.
    """
    if not universe or not isinstance(universe, str):
        raise ValueError("universe must be a non-empty name")
    if ledger is None:
        raise ValueError("a HoldoutLedger is required")
    fam = FAMILIES[family] if isinstance(family, str) else family
    proposer = proposer or GridProposer()
    registry = registry or TrialRegistry()
    design, holdout = split_holdout(asset_returns, years=holdout_years)

    holdout_key = f"{fam.name}:{universe}:{_assets(asset_returns)}"
    key = f"{holdout_key}:{fingerprint(design)}"

    history: list[TrialSummary] = [
        TrialSummary(t.params, t.metrics["sharpe"], t.metrics["max_drawdown"], t.metrics.get("trades", 0))
        for t in registry.family(key)]
    for _ in range(rounds):
        for params in proposer.propose(fam, history, per_round):
            w = fam.weights(design, **params)
            res = bt.run(design, w, cost_bps=cost_bps)
            registry.add(key, params, res.returns, res.turnover, res.trades)
            m = res.metrics
            history.append(TrialSummary(params, m["sharpe"], m["max_drawdown"], m["trades"]))

    if registry.count(key) == 0:
        raise RuntimeError("the proposer produced no variants")
    best = registry.best(key)
    M = registry.matrix(key)
    # Belt and braces: every trial must sit exactly on this run's design dates.
    if not (M.index.equals(design.index) and best.returns.index.equals(design.index)):
        raise RuntimeError("trial dates don't match the design data; refusing to evaluate")

    # Score the chosen variant on the holdout: run it on the full history (its
    # rule only uses past data) and keep the holdout days.
    full = bt.run(asset_returns, fam.weights(asset_returns, **best.params), cost_bps=cost_bps)
    hold_ret = full.returns.loc[holdout.index]
    hold_metrics = bt.performance(hold_ret, full.turnover.loc[holdout.index],
                                  trades=bt.count_trades(full.weights.loc[holdout.index]))
    spent = ledger.record(holdout_key, holdout, hold_metrics).spent

    # Trials run earlier on the same family and universe but a different
    # design window (another holdout length, older data) still count towards
    # the deflated-Sharpe trial total, so rerunning can't reset the count.
    n_total = sum(registry.count(k) for k in registry.keys() if k.startswith(holdout_key + ":"))
    verdict = evaluate(best.returns, M, hold_ret, n_trades=int(best.metrics.get("trades", 0)),
                       config=config, holdout_spent=spent, n_trials_total=n_total)
    return ResearchReport(family=fam.name, universe=universe, trial_key=key,
                          best_params=best.params, design_metrics=best.metrics,
                          holdout_metrics=hold_metrics, n_trials=registry.count(key),
                          verdict=verdict,
                          trials=[{"params": t.params, **t.metrics} for t in registry.family(key)])
