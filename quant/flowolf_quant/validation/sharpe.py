"""Probabilistic and deflated Sharpe ratios.

References
- Bailey, D. H. & López de Prado, M. (2012). The Sharpe Ratio Efficient
  Frontier. Journal of Risk 15(2). (Probabilistic Sharpe ratio, minimum
  track record length.)
- Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
  Journal of Portfolio Management 40(5).

Conventions
- Sharpe ratios inside these formulas are *per period* (daily), not
  annualised. Helpers convert where needed.
- Kurtosis is the raw fourth standardised moment (3 for a normal
  distribution), not excess kurtosis.

What these answer
- PSR(SR*): the probability that the true Sharpe ratio exceeds SR*, given
  the observed Sharpe, the sample length, skew and fat tails.
- DSR: PSR measured against the Sharpe ratio you would expect from the
  *best of N* strategies that all have zero true skill. If you tried 50
  variants and kept the best, its Sharpe has to beat the expected best of 50
  lucky ones, not zero.

Limits
- Assumes returns are roughly independent over time. Strong
  autocorrelation makes the Sharpe estimate's variance larger than these
  formulas assume.
- N must be the honest number of variants tried, including failures. An
  undercounted N makes every strategy look better. That is what the trials
  registry is for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .. import TRADING_DAYS

EULER_GAMMA = 0.5772156649015329


def _clean(r) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return r[~np.isnan(r)]


def sharpe_per_period(r) -> float:
    r = _clean(r)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if len(r) > 1 and sd > 0 else 0.0


def moments(r) -> tuple[float, float, int]:
    """Skew, raw kurtosis and sample size."""
    r = _clean(r)
    if len(r) < 4:
        return 0.0, 3.0, len(r)
    return float(stats.skew(r, bias=False)), float(stats.kurtosis(r, fisher=False, bias=False)), len(r)


def sr_std_error(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of a per-period Sharpe estimate (Mertens 2002 form)."""
    v = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    return float(np.sqrt(max(v, 1e-12) / max(n - 1, 1)))


def probabilistic_sharpe(r, sr_benchmark: float = 0.0) -> float:
    """PSR: probability the true per-period Sharpe exceeds ``sr_benchmark``."""
    sr = sharpe_per_period(r)
    skew, kurt, n = moments(r)
    if n < 4:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) / sr_std_error(sr, n, skew, kurt)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` zero-skill
    strategies whose Sharpe estimates have variance ``sr_variance``
    (Bailey & López de Prado 2014, eq. for SR_0)."""
    if n_trials <= 1:
        return 0.0
    n = float(n_trials)
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


@dataclass
class DeflatedSharpe:
    dsr: float                 # probability the strategy has real skill after the multiple-testing correction
    sr: float                  # observed Sharpe, per period
    sr_annual: float
    sr0: float                 # expected best-of-N luck, per period
    sr0_annual: float
    n_trials: int
    sr_variance: float
    skew: float
    kurtosis: float
    n_obs: int
    variance_source: str


def deflated_sharpe(r, n_trials: int, trial_sharpes=None,
                    sr_variance: float | None = None) -> DeflatedSharpe:
    """Deflated Sharpe ratio of the selected strategy's returns ``r``.

    The variance of Sharpe ratios across trials comes from, in order:
    ``sr_variance`` if given; the spread of ``trial_sharpes`` (per period) if
    at least two are given; otherwise the sampling variance of a zero Sharpe
    estimate, 1/(T-1), which is the right scale when every trial is noise.
    """
    sr = sharpe_per_period(r)
    skew, kurt, n = moments(r)
    if sr_variance is not None:
        var, src = float(sr_variance), "given"
    elif trial_sharpes is not None and len(np.atleast_1d(trial_sharpes)) >= 2:
        ts = np.asarray(trial_sharpes, dtype=float)
        var, src = float(np.var(ts, ddof=1)), "trials"
        # Never let a suspiciously tight cluster of trials shrink the bar
        # below pure sampling noise.
        var = max(var, 1.0 / max(n - 1, 1))
    else:
        var, src = 1.0 / max(n - 1, 1), "sampling"
    sr0 = expected_max_sharpe(n_trials, var)
    dsr = float(stats.norm.cdf((sr - sr0) / sr_std_error(sr, n, skew, kurt))) if n >= 4 else float("nan")
    k = np.sqrt(TRADING_DAYS)
    return DeflatedSharpe(dsr=dsr, sr=sr, sr_annual=sr * k, sr0=sr0, sr0_annual=sr0 * k,
                          n_trials=int(n_trials), sr_variance=var, skew=skew,
                          kurtosis=kurt, n_obs=n, variance_source=src)


def min_track_record_length(r, sr_benchmark: float = 0.0, confidence: float = 0.95) -> float:
    """Minimum number of observations for the observed Sharpe to beat
    ``sr_benchmark`` with the given confidence. Infinite if it doesn't beat it."""
    sr = sharpe_per_period(r)
    skew, kurt, _ = moments(r)
    if sr <= sr_benchmark:
        return float("inf")
    z = stats.norm.ppf(confidence)
    return float(1.0 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) * (z / (sr - sr_benchmark)) ** 2)
