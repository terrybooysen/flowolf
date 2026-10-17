# Regimes

`flowolf_quant.regimes` describes the state of a market with three separate tools. One measures how volatile it is (GARCH), one measures which way it has been trending (time-series momentum) and one detects when its behaviour has changed (Bayesian online change-point detection).

**Timing rule for all three:** every value at day *t* is computed from data up to and including the close of day *t*. None of these functions shifts its output. `backtest.run` applies weights decided at *t* to the return of *t+1*, so pass outputs straight in.

---

## 1. GARCH volatility regimes (`regimes/garch.py`)

**What it does.** It fits a GARCH(1,1) model to daily returns. Today's variance is a long-run level plus a reaction to yesterday's squared surprise (`alpha`) plus a carry-over of yesterday's variance (`beta`). The model is fitted by Gaussian maximum likelihood with constraints omega > 0, alpha, beta >= 0 and alpha + beta < 1. Each day's one-day-ahead vol forecast is then compared with the history of that same forecast, and the day is labelled **calm (0)**, **choppy (1)** or **stress (2)** by the 33rd and 66th percentiles.

**Reference.** Bollerslev, T. (1986), "Generalized autoregressive conditional heteroskedasticity", *Journal of Econometrics* 31(3), 307–327.

**Functions.**
- `fit_garch(returns)` returns a `GarchFit` holding omega, alpha, beta, mu, the log-likelihood, `conditional_vol` (annualised), `conditional_vol_daily` and `next_vol_daily`.
- `forecast_vol(fit, horizon)` gives vol forecasts 1..h days ahead. They decay geometrically toward the long-run vol.
- `apply_garch(fit, returns)` runs fixed parameters over new data.
- `vol_regimes(returns, fit=None, quantiles=(0.33, 0.66), lookback=None)` assigns the labels. Thresholds come from an expanding window by default, or a rolling `lookback`.
- `walk_forward_vol_regimes(returns, train_days=756, refit_every=63)` gives honest out-of-sample labels.
- `simulate_garch(...)` generates test data.

**Known at time t.** The label at *t* uses the vol forecast made at the close of *t* and percentiles of that forecast's history up to *t*. `vol_regimes` without a `fit` estimates parameters on the whole series. That puts look-ahead into the parameters, though not into the recursion, so use it for description only. `walk_forward_vol_regimes` refits only on data before each refit date, and a test checks that cutting the data at *t* leaves the label at *t* unchanged.

**Limits.**
- The labels describe **volatility, not direction**. A calm market can still be falling.
- Labels are relative to the series' own history. "Calm" in a history full of crises can still be quite volatile in absolute terms.
- The model is symmetric, so it has no leverage effect: falls and rises of equal size raise vol equally. It uses Gaussian likelihood on fat-tailed data, which is consistent but not efficient.
- Stable alpha and beta need 1,000 or more observations. The first `train_days` days get no walk-forward label.

---

## 2. Time-series momentum (`regimes/tsmom.py`)

**What it does.** It goes long when an asset's own trailing return is positive and short when it is negative. The position is scaled so its expected volatility hits a target, with a leverage cap. `trend_regime` averages the sign over several lookbacks (21, 63 and 252 days by default) into a trend score from −1 (down on every horizon) to +1 (up on every horizon).

**Reference.** Moskowitz, T. J., Ooi, Y. H. & Pedersen, L. H. (2012), "Time series momentum", *Journal of Financial Economics* 104(2), 228–250. The paper uses a 12-month lookback, EWMA ex-ante variance with a 60-day centre of mass and a 40% per-asset vol target. Here the defaults are 252 days, a 60-day EWMA and a 10% target.

**Functions.** `tsmom_signal(returns, lookback=252, skip=0)` returns −1, 0 or +1. `tsmom_weights(returns, lookback, vol_target=0.10, vol_lookback=60, max_leverage=2.0, long_only=False)` returns target weights. `trend_regime(returns, lookbacks)` returns the trend score. `ex_ante_vol(...)` returns the EWMA or rolling vol used for sizing.

**Known at time t.** The signal uses returns from *t − skip − lookback + 1* to *t − skip*. The vol estimate uses returns up to *t*. Until there is enough history the signal is 0 (flat). A test confirms that cutting the data at *t* gives the same value at *t*.

**Limits.**
- **Momentum fails at turning points.** It stays long into the start of a crash and buys back in late.
- In sideways markets it whipsaws and pays costs for nothing.
- Vol targeting uses lagged vol, so realised vol drifts from target around vol spikes. The leverage cap binds in very quiet periods.
- The score summarises past direction. It does not forecast. On a pure random walk the strategy has no edge: mean Sharpe across 20 seeds is about 0.

---

## 3. Bayesian online change-point detection (`regimes/changepoint.py`)

**What it does.** It treats the series as a chain of segments. Within each segment the data are Gaussian with an unknown mean and variance, under a Normal-Gamma prior with a Student-t predictive. A new segment can start on any day with probability 1/`hazard_lambda`. Each day it updates the probability distribution of the **run length**, meaning how many days the current segment has lasted. A sudden collapse of the run length means the behaviour of the series has changed, whether in mean, volatility or both.

**Reference.** Adams, R. P. & MacKay, D. J. C. (2007), "Bayesian Online Changepoint Detection", arXiv:0710.3742.

**Functions.**
- `bocpd(x, hazard_lambda=250, mu0=0, kappa0=1, alpha0=1, beta0=None, window=20, max_run_length=1000)` returns a `BocpdResult` with:
  - `cp_prob`: probability that today starts a new segment.
  - `recent_cp_prob`: probability of a change within the last `window` days.
  - `map_run_length` and `expected_run_length`.
  - Optionally, the full run-length posterior.
- `detect_changepoints(x, threshold=0.5, min_gap=20)` returns a list of estimated change dates. With `with_alarms=True` it returns (change date, alarm date) pairs.

`beta0` defaults to `alpha0 × 1e-4`, which implies a prior daily vol of about 1%. That suits daily returns. For other units, set it yourself or standardise the data. The algorithm runs in log space with run lengths capped at 1,000 and takes about half a second for 2,500 points.

**Known at time t.** Every output at *t* uses data up to *t*. A change is **recognised late**: the alarm fires a few to roughly 20 days after the change, and only then is the change date estimated backwards. Any trading rule must act on the **alarm date**, never the estimated change date. The optional `prior_calibration=k` sets the prior from the first *k* points, which is the only non-causal option. It is off by default.

**Limits.**
- Inside a segment the model assumes independent Gaussian data. Real returns have volatility clustering and fat tails, which it can mistake for change points. On pure noise it raises about 2–3 false alarms per 10 years at the default threshold, and fewer at 0.8. Running it on GARCH-standardised returns reduces this.
- Sensitivity depends heavily on `hazard_lambda`, the prior and the threshold. Smaller shifts are detected later or not at all.
- It says *that* something changed, not *what* the new regime is. Pair it with the GARCH labels and the trend score for that.
