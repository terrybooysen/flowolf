"""The validation gate: Pass, Fail or Not enough data.

Every strategy result Flowolf shows goes through here first. The numbers
are computed by code, never by a model. A Pass is evidence to review, not a
promise about future returns.

Default thresholds are a proposal awaiting Numoro's sign-off (see
docs/validation.md). They can be tightened, never loosened: use
``GateConfig.tightened(...)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from .. import TRADING_DAYS
from ..backtest import sharpe
from .pbo import pbo_cscv
from .sharpe import deflated_sharpe, sharpe_per_period
from .trials import effective_trials


@dataclass(frozen=True)
class GateConfig:
    min_dsr: float = 0.95          # deflated Sharpe probability
    max_pbo: float = 0.25          # probability of backtest overfitting
    min_holdout_sharpe: float = 0.0
    min_years: float = 3.0         # design data, before the holdout
    min_trades: int = 30
    pbo_blocks: int = 16
    duplicate_corr: float = 0.95   # trials this correlated count as one in the deflated Sharpe

    _HIGHER_IS_STRICTER = ("min_dsr", "min_holdout_sharpe", "min_years", "min_trades", "duplicate_corr")
    _LOWER_IS_STRICTER = ("max_pbo",)

    def tightened(self, **changes) -> "GateConfig":
        """Return a stricter config. Raises ValueError on any loosening."""
        for k, v in changes.items():
            cur = getattr(self, k)
            if k in self._HIGHER_IS_STRICTER and v < cur:
                raise ValueError(f"{k} can only be raised (current {cur}, asked {v})")
            if k in self._LOWER_IS_STRICTER and v > cur:
                raise ValueError(f"{k} can only be lowered (current {cur}, asked {v})")
            if k not in self._HIGHER_IS_STRICTER + self._LOWER_IS_STRICTER:
                raise ValueError(f"{k} is not a gate threshold")
        return replace(self, **changes)


@dataclass
class Check:
    name: str
    value: float | None
    threshold: float | None
    passed: bool | None            # None = could not be computed
    reason: str


@dataclass
class Verdict:
    status: str                    # "Pass", "Fail" or "Not enough data"
    checks: list[Check] = field(default_factory=list)
    summary: str = ""
    config: dict = field(default_factory=dict)
    note: str = "A pass is evidence to review, not proof of future returns."

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(selected_returns: pd.Series, trial_returns: pd.DataFrame | None,
             holdout_returns: pd.Series | None, n_trades: int,
             config: GateConfig = GateConfig(), holdout_spent: bool = False) -> Verdict:
    """Judge one selected strategy.

    selected_returns: daily returns of the chosen variant on the design data.
    trial_returns: daily returns of *every* variant tried in its family on the
        same design data (columns), including the selected one.
    holdout_returns: the chosen variant's returns on the locked holdout.
    n_trades: number of position changes in the design data.
    """
    checks: list[Check] = []
    years = len(selected_returns.dropna()) / TRADING_DAYS

    enough = True
    if years < config.min_years:
        enough = False
        checks.append(Check("data length", round(years, 2), config.min_years, None,
                            f"only {years:.1f} years of design data; need {config.min_years:g}"))
    else:
        checks.append(Check("data length", round(years, 2), config.min_years, True,
                            f"{years:.1f} years of design data"))
    if n_trades < config.min_trades:
        enough = False
        checks.append(Check("trades", n_trades, config.min_trades, None,
                            f"only {n_trades} trades; need {config.min_trades}"))
    else:
        checks.append(Check("trades", n_trades, config.min_trades, True, f"{n_trades} trades"))

    n_raw = 1 if trial_returns is None else trial_returns.shape[1]
    n_trials, trial_srs = n_raw, None
    if trial_returns is not None and trial_returns.shape[1] >= 2:
        n_trials = effective_trials(trial_returns, config.duplicate_corr)
        trial_srs = [sharpe_per_period(trial_returns[c]) for c in trial_returns.columns]
    ds = deflated_sharpe(selected_returns, n_trials=n_trials, trial_sharpes=trial_srs)
    counted = (f"{n_trials} variant(s)" if n_trials == n_raw else
               f"{n_trials} distinct variant(s) ({n_raw} tried; near-duplicates counted once)")
    checks.append(Check(
        "deflated Sharpe", round(ds.dsr, 4), config.min_dsr, bool(ds.dsr >= config.min_dsr),
        f"annualised Sharpe {ds.sr_annual:.2f} against {ds.sr0_annual:.2f} expected from the luckiest "
        f"of {counted}; probability of real skill {ds.dsr:.0%}"))

    if trial_returns is not None and trial_returns.shape[1] >= 2:
        try:
            p = pbo_cscv(trial_returns, n_blocks=config.pbo_blocks)
            checks.append(Check(
                "overfitting probability", round(p.pbo, 4), config.max_pbo, bool(p.pbo <= config.max_pbo),
                f"in {p.pbo:.0%} of {p.n_splits} splits the in-sample winner landed in the bottom half "
                f"out of sample"))
        except ValueError as e:
            checks.append(Check("overfitting probability", None, config.max_pbo, None, str(e)))
            enough = False
    else:
        checks.append(Check("overfitting probability", None, config.max_pbo, None,
                            "only one variant recorded; PBO needs every variant tried"))
        enough = False

    if holdout_returns is None or len(holdout_returns.dropna()) < 20:
        checks.append(Check("holdout", None, config.min_holdout_sharpe, None, "no holdout data"))
        enough = False
    else:
        hs = sharpe(holdout_returns)
        reason = f"annualised Sharpe {hs:.2f} on {len(holdout_returns)} unseen days"
        if holdout_spent:
            reason += " (this holdout has been used before for this family, so treat it with caution)"
        checks.append(Check("holdout", round(hs, 4), config.min_holdout_sharpe,
                            bool(hs > config.min_holdout_sharpe), reason))

    failed = [c for c in checks if c.passed is False]
    if failed:
        status = "Fail"
        summary = "Failed: " + "; ".join(c.reason for c in failed)
    elif not enough:
        status = "Not enough data"
        summary = "Not enough data: " + "; ".join(c.reason for c in checks if c.passed is None)
    else:
        status = "Pass"
        summary = "Passed every check. " + "; ".join(c.reason for c in checks)
    cfg = {k: v for k, v in asdict(config).items() if not k.startswith("_")}
    return Verdict(status=status, checks=checks, summary=summary, config=cfg)
