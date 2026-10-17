"""GARCH(1,1) volatility model and volatility regimes.

Reference
---------
Bollerslev, T. (1986). "Generalized autoregressive conditional
heteroskedasticity." Journal of Econometrics 31(3), 307-327.

Model (daily returns r_t, constant mean mu)::

    e_t       = r_t - mu
    sigma2_t  = omega + alpha * e_{t-1}**2 + beta * sigma2_{t-1}
    e_t | past ~ N(0, sigma2_t)

with omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1 (covariance
stationary). Parameters are estimated by Gaussian maximum likelihood.

Timing convention used throughout this module
---------------------------------------------
- ``sigma2_t`` (the variance *for* day t) uses returns up to day t-1.
- ``next_vol_daily`` at index t is the one-step-ahead forecast for day t+1,
  made at the close of day t from returns up to and including day t. This is
  the quantity regimes are built on, so a regime label at day t uses nothing
  after the close of day t.
- Parameters are only free of look-ahead if they were fitted on data that
  ends on or before the day they are applied to. ``fit_garch`` on a full
  series is an in-sample fit; use ``walk_forward_vol_regimes`` for an honest
  out-of-sample label series.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from .. import TRADING_DAYS

_LOG2PI = np.log(2.0 * np.pi)
_MAX_PERSISTENCE = 0.9999


@dataclass
class GarchFit:
    """Result of a GARCH(1,1) Gaussian MLE fit.

    Parameters (omega, mu) are in the units of the input returns.
    ``conditional_vol`` is annualised (sqrt(252)); ``conditional_vol_daily``
    is the same series in daily units. Both are sigma_t, the vol *for* day t
    based on returns to t-1. ``next_vol_daily`` is the forecast for t+1
    made at the close of t.
    """

    omega: float
    alpha: float
    beta: float
    mu: float
    loglik: float
    conditional_vol: pd.Series
    conditional_vol_daily: pd.Series
    next_vol_daily: pd.Series
    sigma2_init: float
    n_obs: int
    converged: bool
    meta: dict = field(default_factory=dict)

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_vol_daily(self) -> float:
        p = self.persistence
        return float(np.sqrt(self.omega / (1.0 - p))) if p < 1 else float("nan")

    @property
    def half_life(self) -> float:
        """Days for a variance shock to decay by half."""
        p = self.persistence
        return float(np.log(0.5) / np.log(p)) if 0 < p < 1 else float("inf")


# ---------------------------------------------------------------------------
# core recursion
# ---------------------------------------------------------------------------

def garch_variance(eps: np.ndarray, omega: float, alpha: float, beta: float,
                   sigma2_init: float) -> np.ndarray:
    """Run the GARCH(1,1) variance recursion.

    Returns an array of length n+1: element t (t < n) is sigma2_t, the
    variance for day t using eps up to t-1; the last element is the
    one-step-ahead forecast for day n. Strictly causal.
    """
    eps = np.asarray(eps, dtype=float)
    n = len(eps)
    out = np.empty(n + 1)
    out[0] = sigma2_init
    if n == 0:
        return out
    u = omega + alpha * eps ** 2                      # drives sigma2_{t+1}
    out[1:] = lfilter([1.0], [1.0, -beta], u, zi=[beta * sigma2_init])[0]
    return out


def _neg_loglik(params: np.ndarray, eps: np.ndarray, sigma2_init: float) -> float:
    omega, alpha, beta = params
    s2 = garch_variance(eps, omega, alpha, beta, sigma2_init)[:-1]
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return 1e12
    return 0.5 * float(np.sum(_LOG2PI + np.log(s2) + eps ** 2 / s2))


def simulate_garch(n: int = 5000, omega: float = 2e-6, alpha: float = 0.08,
                   beta: float = 0.90, mu: float = 0.0, seed: int | None = 0,
                   burn: int = 500) -> pd.Series:
    """Simulate daily returns from a Gaussian GARCH(1,1) (for testing)."""
    if alpha + beta >= 1:
        raise ValueError("alpha + beta must be < 1")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + burn)
    s2 = omega / (1 - alpha - beta)
    r = np.empty(n + burn)
    for t in range(n + burn):
        e = np.sqrt(s2) * z[t]
        r[t] = mu + e
        s2 = omega + alpha * e * e + beta * s2
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(r[burn:], index=idx, name="ret")


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def fit_garch(returns: pd.Series) -> GarchFit:
    """Fit GARCH(1,1) (Bollerslev 1986) by Gaussian maximum likelihood.

    Steps: demean with the sample mean (mu), rescale to unit variance for
    numerical stability, maximise the Gaussian log-likelihood with SLSQP
    under omega > 0, alpha >= 0, beta >= 0, alpha + beta <= 0.9999, from a
    few standard starting points (alpha ~0.05-0.10, beta ~0.85-0.90), keep
    the best. The recursion starts at the sample variance.

    Known at time t: the fitted *parameters* use the whole input sample, so
    applying them inside that sample is in-sample (look-ahead in the
    parameters, not in the recursion). Fit on a training window and apply to
    later data, or use ``walk_forward_vol_regimes``.

    Limits: Gaussian likelihood on fat-tailed returns gives consistent
    (quasi-ML) but inefficient estimates; symmetric response (no leverage
    effect, unlike GJR/EGARCH); constant mean; needs several hundred
    observations, ideally 1000+, for stable alpha/beta.
    """
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 100:
        raise ValueError("fit_garch needs at least 100 observations")
    x = r.to_numpy()
    mu = float(x.mean())
    eps = x - mu
    scale = float(eps.std(ddof=0))
    if scale <= 0:
        raise ValueError("returns have zero variance")
    z = eps / scale
    s2_init_z = 1.0                                       # sample var of z

    bounds = [(1e-8, 10.0), (0.0, 1.0), (0.0, 1.0)]
    cons = [{"type": "ineq", "fun": lambda p: _MAX_PERSISTENCE - p[1] - p[2]}]
    best = None
    for a0, b0 in ((0.05, 0.90), (0.10, 0.85), (0.03, 0.95), (0.15, 0.60)):
        p0 = np.array([max(1e-6, 1.0 - a0 - b0), a0, b0])
        res = minimize(_neg_loglik, p0, args=(z, s2_init_z), method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    omega_z, alpha, beta = (float(v) for v in best.x)
    alpha, beta = max(alpha, 0.0), max(beta, 0.0)
    omega = omega_z * scale ** 2
    sigma2_init = scale ** 2
    loglik = -float(best.fun) - len(z) * np.log(scale)   # back to return units

    s2 = garch_variance(eps, omega, alpha, beta, sigma2_init)
    vol_d = pd.Series(np.sqrt(s2[:-1]), index=r.index, name="garch_vol_daily")
    nxt = pd.Series(np.sqrt(s2[1:]), index=r.index, name="garch_next_vol_daily")
    return GarchFit(omega=omega, alpha=alpha, beta=beta, mu=mu, loglik=loglik,
                    conditional_vol=(vol_d * np.sqrt(TRADING_DAYS)).rename("garch_vol"),
                    conditional_vol_daily=vol_d, next_vol_daily=nxt,
                    sigma2_init=sigma2_init, n_obs=len(r),
                    converged=bool(best.success),
                    meta={"message": str(best.message), "train_end": r.index[-1]})


def apply_garch(fit: GarchFit, returns: pd.Series) -> pd.Series:
    """Next-day daily vol forecasts for ``returns`` under fixed ``fit`` params.

    Value at t is sigma_{t+1} forecast at the close of t (uses returns up to
    and including t). Out-of-sample if ``returns`` after fit.meta['train_end']
    are what you care about.
    """
    r = pd.Series(returns).astype(float)
    eps = (r - fit.mu).fillna(0.0).to_numpy()
    s2 = garch_variance(eps, fit.omega, fit.alpha, fit.beta, fit.sigma2_init)
    return pd.Series(np.sqrt(s2[1:]), index=r.index, name="garch_next_vol_daily")


def forecast_vol(fit: GarchFit, horizon: int = 10, annualise: bool = False) -> pd.Series:
    """h-step-ahead vol forecasts from the end of the fitted sample.

    sigma2_{T+h} = V + (alpha+beta)**(h-1) * (sigma2_{T+1} - V),
    V = omega / (1 - alpha - beta). Returns per-day vol for h = 1..horizon
    (not cumulative). Known at the close of the last fitted day.

    Limits: forecasts mean-revert geometrically to the long-run level; they
    cannot anticipate scheduled events or jumps.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    p = fit.persistence
    s2_next = float(fit.next_vol_daily.iloc[-1]) ** 2
    h = np.arange(1, horizon + 1)
    if p < 1:
        V = fit.omega / (1 - p)
        s2 = V + p ** (h - 1) * (s2_next - V)
    else:
        s2 = s2_next + fit.omega * (h - 1)
    vol = np.sqrt(s2) * (np.sqrt(TRADING_DAYS) if annualise else 1.0)
    return pd.Series(vol, index=pd.Index(h, name="horizon"), name="vol_forecast")


# ---------------------------------------------------------------------------
# regimes
# ---------------------------------------------------------------------------

REGIME_NAMES = {0: "calm", 1: "choppy", 2: "stress"}


def _label_by_history(vol: pd.Series, quantiles: tuple[float, float],
                      lookback: int | None, min_history: int) -> pd.Series:
    q_lo, q_hi = quantiles
    if not 0 < q_lo < q_hi < 1:
        raise ValueError("quantiles must satisfy 0 < q_lo < q_hi < 1")
    if lookback is None:
        roll = vol.expanding(min_periods=min_history)
    else:
        roll = vol.rolling(lookback, min_periods=min(min_history, lookback))
    lo = roll.quantile(q_lo)
    hi = roll.quantile(q_hi)
    lab = pd.Series(np.where(vol <= lo, 0, np.where(vol <= hi, 1, 2)),
                    index=vol.index, dtype=float)
    lab[lo.isna() | vol.isna()] = np.nan
    return lab.rename("vol_regime")


def vol_regimes(returns: pd.Series, fit: GarchFit | None = None,
                quantiles: tuple[float, float] = (0.33, 0.66),
                lookback: int | None = None, min_history: int = 63) -> pd.Series:
    """Label each day calm (0) / choppy (1) / stress (2) from GARCH vol.

    The GARCH next-day vol forecast made at the close of day t is compared
    with the q_lo / q_hi quantiles of its own history up to and including
    day t (expanding window, or rolling ``lookback`` days). No future vol
    enters the thresholds.

    If ``fit`` is None the model is fitted on ``returns`` itself, which puts
    look-ahead into the *parameters* (fine for description, not for a
    backtest). Pass a fit from an earlier training window, or use
    ``walk_forward_vol_regimes``, for an honest series.

    Limits: these regimes describe volatility, not direction; labels are
    relative to the series' own history (a "calm" day in a crisis-heavy
    history can still be volatile in absolute terms); first ``min_history``
    days are NaN.
    """
    r = pd.Series(returns).astype(float)
    if fit is None:
        fit = fit_garch(r)
    vol = apply_garch(fit, r)
    return _label_by_history(vol, quantiles, lookback, min_history)


def walk_forward_vol_regimes(returns: pd.Series, train_days: int = 756,
                             refit_every: int = 63,
                             quantiles: tuple[float, float] = (0.33, 0.66),
                             lookback: int | None = None, min_history: int = 63,
                             return_vol: bool = False):
    """Out-of-sample GARCH vol regimes with periodic refits.

    At each refit position k = train_days, train_days + refit_every, ...
    GARCH is fitted on returns[:k] (days 0..k-1 only) and its parameters
    are used to produce next-day vol forecasts for days k..k+refit_every-1.
    The vol history before the first refit (in-sample filtered vol of the
    first training window) is used only as threshold history, never
    labelled. Labels for days before ``train_days`` are NaN.

    Known at time t: label[t] depends only on returns[:t+1]; truncating the
    input at t leaves label[t] unchanged (tested).

    Limits: an expanding training window (all data up to the refit date);
    parameters jump at each refit; same caveats as ``vol_regimes``.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n <= train_days:
        raise ValueError("need more than train_days observations")
    vol = pd.Series(np.nan, index=r.index, name="garch_next_vol_daily")
    fits = []
    for k in range(train_days, n, refit_every):
        f = fit_garch(r.iloc[:k])
        end = min(k + refit_every, n)
        v = apply_garch(f, r.iloc[:end])
        if not fits:
            vol.iloc[:k] = v.iloc[:k].to_numpy()          # warm-up history
        vol.iloc[k:end] = v.iloc[k:end].to_numpy()
        fits.append((r.index[k], f))
    lab = _label_by_history(vol, quantiles, lookback, min_history)
    lab.iloc[:train_days] = np.nan
    if return_vol:
        return lab, vol, fits
    return lab
