"""Fractional Kelly sizing that refuses to size without statistical evidence.

References
----------
- Kelly, J. L. (1956). "A New Interpretation of Information Rate."
  Bell System Technical Journal 35(4), 917-926.
- Thorp, E. O. (2006). "The Kelly Criterion in Blackjack, Sports Betting and
  the Stock Market." In Handbook of Asset and Liability Management, Vol. 1.
- MacLean, Thorp & Ziemba (2010), "Good and bad properties of the Kelly
  criterion" - on fractional Kelly's growth/drawdown trade-off.

For a continuous return stream with mean mu and variance sigma^2 per period,
the growth-optimal (full Kelly) leverage is approximately f* = mu / sigma^2.

Why full Kelly is dangerous with *estimated* parameters: growth is a concave
function of leverage that falls off steeply past f*; betting 2x Kelly gives
zero expected growth, beyond that it is negative. Estimates of mu are very
noisy (the standard error of an annual mean is roughly sigma/sqrt(years)), so
an estimated f* is frequently far above the true one. Overbetting is much
more costly than underbetting, hence fractional Kelly (typically 0.25-0.5),
and hence this module refuses to size at all when the evidence that mu > 0
is weak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class KellyResult:
    fraction: float | None      # leverage to apply, or None = do not size
    reason: str
    mean: float                 # estimated mean per-period return
    var: float                  # estimated per-period variance
    t_stat: float
    n_obs: int = 0
    p_value: float = float("nan")   # one-sided, H1: mean > 0
    full_kelly: float = float("nan")


def kelly_fraction(returns: pd.Series | np.ndarray, fraction: float = 0.5,
                   min_obs: int = 252, confidence: float = 0.95,
                   max_leverage: float = 1.0) -> KellyResult:
    """Fractional Kelly leverage for a strategy, or None if there is no edge.

    Reference: Kelly (1956); Thorp (2006) on fractional Kelly.

    Rule:
      1. Fewer than ``min_obs`` non-missing observations -> None.
      2. One-sided t-test of H0: mean <= 0 against H1: mean > 0 on the
         per-period returns (t = mean / (sd / sqrt(n)), Student-t with n-1
         d.o.f.). If p >= 1 - confidence -> None, "no evidence of an edge".
      3. Otherwise f = fraction * mean / var, capped at ``max_leverage``.

    Known at t: pass only returns up to t (e.g. an expanding or trailing
    window ending at t). The function does no time slicing itself.

    Limits: the t-test assumes roughly independent returns; autocorrelated or
    fat-tailed returns overstate the evidence. If the strategy was chosen
    from many candidates, this single test is too lenient - deflate for
    multiple testing first (see the validation gate). Passing the test only
    says mean > 0 is likely, not that the point estimate of mu is accurate,
    which is why ``fraction`` should stay well below 1.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 2:
        return KellyResult(None, f"not enough data ({n} observations, need {min_obs})",
                           float("nan"), float("nan"), float("nan"), n)
    mu = float(r.mean())
    var = float(r.var(ddof=1))
    sd = np.sqrt(var)
    t = mu / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    p = float(stats.t.sf(t, df=n - 1)) if np.isfinite(t) else float("nan")
    full = mu / var if var > 0 else float("nan")
    base = dict(mean=mu, var=var, t_stat=float(t), n_obs=n, p_value=p, full_kelly=full)

    if n < min_obs:
        return KellyResult(None, f"not enough data ({n} observations, need {min_obs})", **base)
    if not np.isfinite(t) or p >= 1.0 - confidence:
        return KellyResult(None, f"no evidence of an edge (t={t:.1f}, p={p:.2f})", **base)
    f = min(fraction * full, max_leverage)
    capped = " (capped)" if fraction * full > max_leverage else ""
    return KellyResult(float(f), f"edge significant (t={t:.1f}, p={p:.3f}); "
                       f"{fraction:g}x Kelly = {f:.2f}{capped}", **base)
