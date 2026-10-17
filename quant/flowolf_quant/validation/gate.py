"""The validation gate: Pass, Fail or Not enough data.

Every strategy result Flowolf shows goes through here first. The numbers
are computed by code, never by a model. A Pass is evidence to review, not a
promise about future returns.

Default thresholds are a proposal awaiting Numoro's sign-off (see
docs/validation.md). They can be tightened, never loosened: constructing a
looser ``GateConfig`` raises an error, and ``GateConfig.tightened(...)``
refuses loosening changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np
import pandas as pd

from .. import TRADING_DAYS
from ..backtest import sharpe
from .pbo import pbo_cscv
from .sharpe import deflated_sharpe, sharpe_per_period
from .trials import effective_trials

# The floor. Any config may be stricter than this, never looser.
SIGNED_OFF = {"min_dsr": 0.95, "max_pbo": 0.25, "min_holdout_sharpe": 0.0,
              "min_years": 3.0, "min_trades": 30, "duplicate_corr": 0.95}
HIGHER_IS_STRICTER = ("min_dsr", "min_holdout_sharpe", "min_years", "min_trades", "duplicate_corr")
LOWER_IS_STRICTER = ("max_pbo",)


@dataclass(frozen=True)
class GateConfig:
    min_dsr: float = 0.95          # deflated Sharpe probability
    max_pbo: float = 0.25          # probability of backtest overfitting
    min_holdout_sharpe: float = 0.0
    min_years: float = 3.0         # design data, before the holdout, after warm-up
    min_trades: int = 30           # position entries and exits, not daily resizing
    pbo_blocks: int = 16
    duplicate_corr: float = 0.95   # trials this correlated count as one (never makes the bar lower than raw N)

    def __post_init__(self):
        for k in HIGHER_IS_STRICTER:
            if getattr(self, k) < SIGNED_OFF[k]:
                raise ValueError(f"{k}={getattr(self, k)} is looser than the signed-off {SIGNED_OFF[k]}")
        for k in LOWER_IS_STRICTER:
            if getattr(self, k) > SIGNED_OFF[k]:
                raise ValueError(f"{k}={getattr(self, k)} is looser than the signed-off {SIGNED_OFF[k]}")

    def tightened(self, **changes) -> "GateConfig":
        """Return a stricter config. Raises ValueError on any loosening."""
        for k, v in changes.items():
            cur = getattr(self, k)
            if k in HIGHER_IS_STRICTER and v < cur:
                raise ValueError(f"{k} can only be raised (current {cur}, asked {v})")
            if k in LOWER_IS_STRICTER and v > cur:
                raise ValueError(f"{k} can only be lowered (current {cur}, asked {v})")
            if k not in HIGHER_IS_STRICTER + LOWER_IS_STRICTER:
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


def _trim_warmup(r: pd.Series) -> pd.Series:
    """Drop the leading run of exactly-zero returns (the rule had no position yet)."""
    nz = np.flatnonzero(r.fillna(0.0).values != 0.0)
    return r.iloc[nz[0]:] if len(nz) else r.iloc[0:0]


def evaluate(selected_returns: pd.Series, trial_returns: pd.DataFrame | None,
             holdout_returns: pd.Series | None, n_trades: int,
             config: GateConfig = GateConfig(), holdout_spent: bool = False,
             n_trials_total: int | None = None) -> Verdict:
    """Judge one selected strategy.

    selected_returns: daily returns of the chosen variant on the design data.
    trial_returns: daily returns of *every* variant tried in its family on the
        same design data (columns), including the selected one.
    holdout_returns: the chosen variant's returns on the locked holdout; it
        must start after the design data ends.
    n_trades: position entries and exits in the design data.
    n_trials_total: every variant ever tried for this family and universe,
        including earlier runs on other design windows. Never fewer than the
        matrix's columns.
    """
    if not isinstance(config, GateConfig):
        raise TypeError("config must be a GateConfig")
    if trial_returns is not None:
        if not selected_returns.index.isin(trial_returns.index).all():
            raise ValueError("selected returns must lie on the trial matrix's dates")
    if holdout_returns is not None and len(holdout_returns) and len(selected_returns):
        if holdout_returns.index[0] <= selected_returns.index[-1]:
            raise ValueError("holdout overlaps the design data")

    checks: list[Check] = []
    live = _trim_warmup(selected_returns.dropna())
    years = len(live) / TRADING_DAYS

    enough = True
    if years < config.min_years:
        enough = False
        checks.append(Check("data length", round(years, 2), config.min_years, None,
                            f"only {years:.1f} years of design data after warm-up; need {config.min_years:g}"))
    else:
        checks.append(Check("data length", round(years, 2), config.min_years, True,
                            f"{years:.1f} years of design data after warm-up"))
    if n_trades < config.min_trades:
        enough = False
        checks.append(Check("trades", n_trades, config.min_trades, None,
                            f"only {n_trades} entries and exits; need {config.min_trades}"))
    else:
        checks.append(Check("trades", n_trades, config.min_trades, True,
                            f"{n_trades} entries and exits"))

    # Deflated Sharpe: take the stricter of two benchmarks, so counting
    # near-duplicates once can never lower the bar below the paper's own
    # formula with the raw trial count.
    n_cols = 1 if trial_returns is None else trial_returns.shape[1]
    n_raw = max(n_cols, int(n_trials_total or 0))
    if trial_returns is not None and n_cols >= 2:
        trial_srs = [sharpe_per_period(trial_returns[c]) for c in trial_returns.columns]
        n_eff = effective_trials(trial_returns, config.duplicate_corr) + (n_raw - n_cols)
        ds_raw = deflated_sharpe(live, n_trials=n_raw, trial_sharpes=trial_srs, floor_variance=False)
        ds_eff = deflated_sharpe(live, n_trials=n_eff, trial_sharpes=trial_srs, floor_variance=True)
        ds = ds_raw if ds_raw.sr0 >= ds_eff.sr0 else ds_eff
        counted = (f"{n_raw} variant(s)" if n_eff == n_raw else
                   f"{n_raw} variant(s) tried ({n_eff} distinct)")
    else:
        ds = deflated_sharpe(live, n_trials=n_raw)
        counted = f"{n_raw} variant(s)"
    dsr_ok = bool(np.isfinite(ds.dsr) and ds.dsr >= config.min_dsr)
    checks.append(Check(
        "deflated Sharpe", round(float(ds.dsr), 4), config.min_dsr, dsr_ok,
        f"annualised Sharpe {ds.sr_annual:.2f} against {ds.sr0_annual:.2f} expected from the luckiest "
        f"of {counted}; probability of real skill {ds.dsr:.0%}"))

    if trial_returns is not None and n_cols >= 2:
        try:
            p = pbo_cscv(trial_returns, n_blocks=config.pbo_blocks)
            checks.append(Check(
                "overfitting probability", round(p.pbo, 4), config.max_pbo, bool(p.pbo <= config.max_pbo),
                f"in {p.pbo:.0%} of {p.n_splits} splits the in-sample winner landed in the bottom half "
                f"of its in-sample competitors out of sample; it lost money out of sample in "
                f"{p.prob_oos_loss:.0%}"))
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
            reason += (" (an overlapping holdout has been scored before for this family and universe, "
                       "so treat it with caution)")
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
    cfg = {f.name: getattr(config, f.name) for f in fields(config)}
    return Verdict(status=status, checks=checks, summary=summary, config=cfg)
