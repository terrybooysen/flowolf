"""The research loop: propose, test, record every trial, let the gate decide.

This is the "self-improving" loop done the honest way. The Hermes template
that lost money let one noisy result change the strategy. Here:
1. the last two years are locked away as a holdout before anything runs;
2. a proposer (grid, random or a language model) suggests variants;
3. every variant is backtested on the design data and recorded as a trial;
4. the best variant by design Sharpe is scored once on the holdout;
5. the gate judges it against *all* trials tried (deflated Sharpe, PBO,
   holdout, data length, trade count).
Nothing is ever promoted on a single number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from .. import backtest as bt
from ..validation import GateConfig, HoldoutLedger, TrialRegistry, Verdict, evaluate, split_holdout
from .proposers import GridProposer, Proposer, TrialSummary
from .strategies import FAMILIES, Family


@dataclass
class ResearchReport:
    family: str
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


def _key(family: str, universe_name: str) -> str:
    return f"{family}:{universe_name}"


def research(asset_returns: pd.Series | pd.DataFrame, family: Family | str,
             proposer: Proposer | None = None, rounds: int = 1, per_round: int = 50,
             registry: TrialRegistry | None = None, ledger: HoldoutLedger | None = None,
             universe_name: str = "default", holdout_years: float = 2.0,
             cost_bps: float = 5.0, config: GateConfig = GateConfig()) -> ResearchReport:
    fam = FAMILIES[family] if isinstance(family, str) else family
    proposer = proposer or GridProposer()
    registry = registry or TrialRegistry()
    key = _key(fam.name, universe_name)
    design, holdout = split_holdout(asset_returns, years=holdout_years)

    history: list[TrialSummary] = [
        TrialSummary(t.params, t.metrics["sharpe"], t.metrics["max_drawdown"], t.metrics.get("trades", 0))
        for t in registry.family(key)]
    results: dict[str, bt.BacktestResult] = {}
    for _ in range(rounds):
        for params in proposer.propose(fam, history, per_round):
            w = fam.weights(design, **params)
            res = bt.run(design, w, cost_bps=cost_bps)
            trial = registry.add(key, params, res.returns, res.turnover)
            results[trial.id] = res
            m = res.metrics
            history.append(TrialSummary(params, m["sharpe"], m["max_drawdown"], m["trades"]))

    if registry.count(key) == 0:
        raise RuntimeError("the proposer produced no variants")
    best = registry.best(key)
    M = registry.matrix(key)

    # Score the chosen variant on the holdout: run it on the full history (its
    # rule only uses past data) and keep the holdout days.
    full = bt.run(asset_returns, fam.weights(asset_returns, **best.params), cost_bps=cost_bps)
    hold_ret = full.returns.loc[holdout.index]
    hold_metrics = bt.performance(hold_ret, full.turnover.loc[holdout.index])
    spent = False
    if ledger is not None:
        spent = ledger.record(key, holdout, hold_metrics).spent

    n_trades = best.metrics.get("trades")
    if n_trades is None and best.id in results:
        n_trades = results[best.id].metrics["trades"]
    verdict = evaluate(best.returns, M, hold_ret, n_trades=int(n_trades or 0),
                       config=config, holdout_spent=spent)
    return ResearchReport(family=key, best_params=best.params, design_metrics=best.metrics,
                          holdout_metrics=hold_metrics, n_trials=registry.count(key),
                          verdict=verdict,
                          trials=[{"params": t.params, **t.metrics} for t in registry.family(key)])
