"""Tail-risk measures: historical VaR / CVaR and a correlation stress test.

References
----------
- Rockafellar, R. T. & Uryasev, S. (2000). "Optimization of Conditional
  Value-at-Risk." Journal of Risk 2(3), 21-41.
- Acerbi, C. & Tasche, D. (2002). "On the coherence of expected shortfall."
  Journal of Banking & Finance 26(7) - CVaR / expected shortfall is coherent;
  VaR is not.
- Longin, F. & Solnik, B. (2001). "Extreme Correlation of International
  Equity Markets." Journal of Finance 56(2) - correlations rise in sell-offs,
  the motivation for the stress test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import TRADING_DAYS


def var_cvar(returns: pd.Series | np.ndarray,
             levels: tuple[float, ...] = (0.95, 0.99)) -> pd.DataFrame:
    """Historical Value-at-Risk and Conditional VaR (expected shortfall).

    Reference: Rockafellar & Uryasev (2000).

    Returned in *return units* (losses are negative), one row per level:
      - ``var``: the (1 - level) empirical quantile of returns, i.e. the
        return that is worse than this only (1 - level) of the time.
      - ``cvar``: the average return on the days at or below ``var``.
    So ``cvar <= var`` always.

    Known at t: whatever history you pass in (pass returns up to t).

    Limits: historical simulation only knows the sample; at 99% with a few
    hundred days the estimate rests on a handful of observations. It assumes
    the next period looks like the sample period, and one-day figures do not
    scale cleanly to longer horizons when returns cluster.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        raise ValueError("no returns")
    rows = {}
    for lvl in levels:
        if not 0 < lvl < 1:
            raise ValueError("levels must be in (0, 1)")
        q = float(np.quantile(r, 1.0 - lvl))
        tail = r[r <= q]
        rows[lvl] = {"var": q, "cvar": float(tail.mean()), "n_tail": int(len(tail))}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "level"
    return out


def correlation_stress(asset_returns: pd.DataFrame, weights: pd.Series | dict,
                       stress_corr: float = 0.9) -> dict:
    """Annualised portfolio vol now vs. if every pairwise correlation jumps.

    Uses each asset's sample volatility from ``asset_returns`` and replaces
    all off-diagonal correlations with ``stress_corr`` (vols unchanged).

    Reference: motivated by Longin & Solnik (2001) - correlations rise in
    market stress, so diversification fails when it is most needed.

    Returns ``{'normal_vol', 'stressed_vol', 'ratio', 'stress_corr'}``.

    Known at t: pass asset returns up to t and the weights held at t.

    Limits: a crude what-if. Real crises also raise volatilities (often by
    2-3x), which this leaves out - combine with a vol shock if needed. For a
    long-short book, raising correlations *reduces* risk, so the stressed
    figure can be below the normal one; that is correct arithmetic, not
    safety, since correlations can also flip sign.
    """
    if not -1.0 <= stress_corr <= 1.0:
        raise ValueError("stress_corr must be in [-1, 1]")
    w = pd.Series(weights, dtype=float).reindex(asset_returns.columns).fillna(0.0).to_numpy()
    X = asset_returns.dropna(how="any")
    cov = X.cov().to_numpy()
    sd = np.sqrt(np.diag(cov))
    k = len(sd)
    C = np.full((k, k), stress_corr)
    np.fill_diagonal(C, 1.0)
    cov_s = np.outer(sd, sd) * C
    normal = float(np.sqrt(max(w @ cov @ w, 0.0) * TRADING_DAYS))
    stressed = float(np.sqrt(max(w @ cov_s @ w, 0.0) * TRADING_DAYS))
    return {"normal_vol": normal, "stressed_vol": stressed,
            "ratio": stressed / normal if normal > 0 else float("nan"),
            "stress_corr": stress_corr}
