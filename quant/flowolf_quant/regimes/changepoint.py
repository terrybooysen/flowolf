"""Bayesian online change-point detection (BOCPD).

Reference
---------
Adams, R. P. & MacKay, D. J. C. (2007). "Bayesian Online Changepoint
Detection." arXiv:0710.3742.

Model: the series is split into segments; within a segment observations are
i.i.d. Gaussian with unknown mean and variance, Normal-Gamma prior
(mu0, kappa0, alpha0, beta0), so the posterior predictive is Student-t:

    x | segment data ~ t_{2 alpha}(mu, beta (kappa + 1) / (alpha kappa))

A new segment starts each day with constant hazard H = 1 / hazard_lambda.
The run length L_t is the number of observations in the segment that
contains x_t (L_t = 1 means x_t is the first point of a new segment). The
algorithm propagates P(L_t | x_1..x_t) exactly (up to truncation), one day
at a time, in log space.

Known at time t: every output at index t is computed from x_1..x_t only.
Change points are only *recognised* with a delay: the probability that a
change happened exactly at t is usually small at t and grows as more data
arrive, so ``detect_changepoints`` alarms on "a change happened within the
last ``window`` days" and reports both the estimated change date and the
(later) alarm date.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp


@dataclass
class BocpdResult:
    """Run-length posterior summaries, one row per observation.

    cp_prob          P(L_t = 1 | x_1..t): x_t starts a new segment (filtering).
    recent_cp_prob   P(L_t <= window | x_1..t): a change within the last
                     ``window`` observations (the usable alarm statistic).
    map_run_length   argmax_L P(L_t = L | x_1..t).
    expected_run_length  E[L_t | x_1..t].
    run_length_posterior  optional (n x max_run_length) array, row t =
                     P(L_t = l+1 | x_1..t); only if ``store_posterior``.
    """

    cp_prob: pd.Series
    recent_cp_prob: pd.Series
    map_run_length: pd.Series
    expected_run_length: pd.Series
    window: int
    run_length_posterior: np.ndarray | None = None
    params: dict = field(default_factory=dict)


def _student_t_logpdf(x: float, mu: np.ndarray, kappa: np.ndarray,
                      alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    nu = 2.0 * alpha
    scale2 = beta * (kappa + 1.0) / (alpha * kappa)
    z2 = (x - mu) ** 2 / scale2
    return (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi * scale2)
            - (nu + 1.0) / 2.0 * np.log1p(z2 / nu))


def bocpd(x: pd.Series | np.ndarray, hazard_lambda: float = 250.0,
          mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0,
          beta0: float | None = None, window: int = 20,
          max_run_length: int = 1000, store_posterior: bool = False,
          prior_calibration: int = 0) -> BocpdResult:
    """Bayesian online change-point detection, Adams & MacKay (2007).

    Gaussian observations with unknown mean and variance (Normal-Gamma
    conjugate prior, Student-t predictive), constant hazard 1/hazard_lambda,
    computed in log space with the run length truncated at
    ``max_run_length`` (mass beyond it is dropped and the rest
    renormalised). 2,500 points take well under a second.

    Prior scale ``beta0``: the prior mean of the precision is alpha0/beta0,
    so beta0 = alpha0 * (typical variance). If ``beta0`` is None it is set
    to ``alpha0 * 1e-4`` (daily vol ~1%, right for daily equity returns)
    unless ``prior_calibration`` = k > 0, in which case the variance of the
    first k observations is used. That calibration is the one place
    information flows backwards (into the first k days' prior); leave it at
    0 for strictly causal output, or standardise the data yourself.

    Known at time t: all outputs at t use x_1..x_t. The first observation is
    treated as the start of a segment (cp_prob = 1 there).

    Limits: assumes i.i.d. Gaussian data within a segment, so volatility
    clustering or fat tails inside a "regime" get read as change points
    (consider running it on standardised or GARCH-filtered returns); the
    hazard and prior strongly affect sensitivity; detection is delayed by
    a few to tens of observations depending on the size of the shift.
    """
    if isinstance(x, pd.Series):
        idx = x.index
        arr = x.astype(float).to_numpy()
    else:
        arr = np.asarray(x, dtype=float)
        idx = pd.RangeIndex(len(arr))
    if np.isnan(arr).any():
        raise ValueError("bocpd does not accept NaN; drop or fill them first")
    n = len(arr)
    if n == 0:
        raise ValueError("empty input")
    if beta0 is None:
        if prior_calibration > 1:
            v = float(np.var(arr[:prior_calibration], ddof=1))
            beta0 = alpha0 * (v if v > 0 else 1e-4)
        else:
            beta0 = alpha0 * 1e-4
    logH = -np.log(hazard_lambda)
    log1mH = np.log1p(-1.0 / hazard_lambda)
    K = int(max_run_length)

    # Posterior params for runs of length 1..m (index j <-> L = j + 1).
    mu = np.empty(0); kappa = np.empty(0); alpha = np.empty(0); beta = np.empty(0)
    logP = np.empty(0)

    cp = np.empty(n); recent = np.empty(n); mapL = np.empty(n, dtype=int)
    expL = np.empty(n)
    post = np.zeros((n, K), dtype=np.float32) if store_posterior else None

    for t in range(n):
        xt = arr[t]
        # candidate predictive params: L_t = 1 (prior) and L_t = l + 1 (grow)
        cmu = np.concatenate(([mu0], mu))
        ckap = np.concatenate(([kappa0], kappa))
        calp = np.concatenate(([alpha0], alpha))
        cbet = np.concatenate(([beta0], beta))
        if t == 0:
            logprior = np.array([0.0])
        else:
            logprior = np.concatenate(([logH], log1mH + logP))
        logjoint = logprior + _student_t_logpdf(xt, cmu, ckap, calp, cbet)
        if len(logjoint) > K:
            logjoint = logjoint[:K]
            cmu, ckap, calp, cbet = cmu[:K], ckap[:K], calp[:K], cbet[:K]
        logP = logjoint - logsumexp(logjoint)

        # update every run's params with x_t (Normal-Gamma conjugate update)
        d = xt - cmu
        beta = cbet + ckap * d * d / (2.0 * (ckap + 1.0))
        mu = (ckap * cmu + xt) / (ckap + 1.0)
        kappa = ckap + 1.0
        alpha = calp + 0.5

        p = np.exp(logP)
        cp[t] = p[0]
        recent[t] = p[:window].sum()
        mapL[t] = int(np.argmax(p)) + 1
        expL[t] = float(np.dot(np.arange(1, len(p) + 1), p))
        if post is not None:
            post[t, :len(p)] = p

    return BocpdResult(
        cp_prob=pd.Series(cp, index=idx, name="cp_prob"),
        recent_cp_prob=pd.Series(recent, index=idx, name="recent_cp_prob"),
        map_run_length=pd.Series(mapL, index=idx, name="map_run_length"),
        expected_run_length=pd.Series(expL, index=idx, name="expected_run_length"),
        window=window, run_length_posterior=post,
        params=dict(hazard_lambda=hazard_lambda, mu0=mu0, kappa0=kappa0,
                    alpha0=alpha0, beta0=beta0, max_run_length=K))


def detect_changepoints(x: pd.Series | np.ndarray, threshold: float = 0.5,
                        min_gap: int = 20, window: int | None = None,
                        with_alarms: bool = False, **bocpd_kwargs):
    """Change dates from BOCPD, recognised online.

    An alarm fires at day t when P(L_t <= window | x_1..t) > threshold
    (default window = min_gap). The change date is estimated at the alarm as
    t - L* + 1, where L* is the most probable run length among L <= window.
    Alarms whose estimated change date lies within ``min_gap`` observations
    of the previous reported change (or of the start of the series), or
    that fire within ``min_gap`` days of the previous alarm, are merged
    into it.

    Returns a list of change dates (index labels), or with
    ``with_alarms=True`` a list of (change_date, alarm_date) tuples. The
    change date is only known at the alarm date, which is later; a trading
    rule must act on the alarm date.

    Reference: Adams & MacKay (2007). Limits: see ``bocpd``; threshold and
    hazard trade detection delay against false alarms.
    """
    if window is None:
        window = min_gap
    res = bocpd(x, window=window, store_posterior=True, **bocpd_kwargs)
    idx = res.cp_prob.index
    post = res.run_length_posterior
    out: list = []
    last_pos = 0                                  # series start acts as a change
    last_alarm = -10 ** 9
    for t in np.flatnonzero(res.recent_cp_prob.to_numpy() > threshold):
        if t < window:
            continue                              # trivially short runs at the start
        Lstar = int(np.argmax(post[t, :window])) + 1
        c = t - Lstar + 1
        if c - last_pos < min_gap or t - last_alarm < min_gap:
            continue
        out.append((idx[c], idx[t]) if with_alarms else idx[c])
        last_pos, last_alarm = c, t
    return out
