# Allocation, selection and risk

These modules turn signals into positions and measure how bad things can get.
Every weight function returns **target weights at the close of day t, using
only data up to t**. `backtest.run` applies them to day t+1. Never shift the
weights yourself.

Nothing here is a validated strategy. Every result still has to pass the
validation gate before it is shown as anything but "not validated".

---

## Allocation

### Volatility targeting: `allocation/voltarget.py`

**What:** this scales exposure so the strategy runs at a chosen annual
volatility (10% by default). Exposure goes down when recent markets are
turbulent and up when they are quiet. `vol_target_scale` handles a single
return stream. `apply_vol_target` rescales a whole portfolio's weights using
that portfolio's own recent volatility. Leverage is capped (2x by default).

**References:** Moreira & Muir (2017), *Volatility-Managed Portfolios*,
J. Finance 72(4). Harvey et al. (2018), *The Impact of Volatility Targeting*,
JPM 45(1). RiskMetrics (1996) for the EWMA estimator.

**Known at t:** volatility estimated from returns up to and including t. The
scale is zero until `lookback` days of history exist.

**Limits:** the forecast lags sudden shocks, so the first day of a crash is
taken at full size. The method adds leverage in calm periods, which is when
risk can quietly build up. Harvey et al. find it reliably cuts tail risk, but
the Sharpe benefit is smaller and less consistent than the original paper
suggests.

### Hierarchical Risk Parity: `allocation/hrp.py`

**What:** a long-only allocation that groups similar assets into a tree, then
splits capital between branches in inverse proportion to their risk. No
matrix inversion is needed, so it is more stable than mean-variance
optimisation. `rolling_hrp` recomputes weights every 21 days on a trailing
252-day window. `inverse_vol_weights` and `equal_weights` are the baselines it
should be compared against.

**Reference:** López de Prado (2016), *Building Diversified Portfolios that
Outperform Out of Sample*, JPM 42(4).

**Split choice:** `split="bisect"` follows the paper: it cuts the ordered list
of assets in half at each step. `split="tree"` (the default) cuts at the
cluster tree's actual branches. With unequal cluster sizes, list halving can
cut through a cluster. On an 8-asset plus 2-asset two-cluster test, it put
about 70-80% of the capital in the big cluster, while tree splitting gave
roughly 50/50.

**Known at t:** the covariance comes from the trailing window ending at t, and
weights are held between rebalances. Assets missing data in the window get
zero weight.

**Limits:** correlations measured in calm periods break down in crises.
Single-linkage clustering can chain unrelated assets together. HRP has no
view on returns, so it is a risk allocator, not an alpha source.

### Fractional Kelly: `allocation/kelly.py`

**What:** sizing that maximises long-run growth, f* = mu / sigma^2, then
scaled down to a fraction (half Kelly by default) and capped at 1x. **It
refuses to size** when there is no evidence of an edge. If there are fewer
than 252 observations, or a one-sided t-test of mean > 0 is not significant at
95%, it returns `fraction=None` with a plain reason such as
"no evidence of an edge (t=0.8, p=0.21)".

**References:** Kelly (1956), *A New Interpretation of Information Rate*, Bell
System Technical Journal 35(4). Thorp (2006), *The Kelly Criterion in
Blackjack, Sports Betting and the Stock Market*.

**Known at t:** whatever return history is passed in. Pass only data up to t.

**Limits:** full Kelly is dangerous when the inputs are estimated. Growth falls
off steeply above f*: betting twice the true Kelly gives zero growth, and more
than that loses money. Estimates of the mean are very noisy, so the estimated
f* is often far too high. That is why the module uses a fraction and requires
evidence first. The t-test assumes independent returns. If the strategy was
picked from many tried, the test is too lenient, so run the validation gate's
multiple-testing correction first.

---

## Selection

### Cross-sectional momentum: `selection/xsmom.py`

**What:** each asset is ranked by its return over the past 12 months,
excluding the most recent month ("12-1"). The portfolio goes long the top 20%
and shorts the bottom 20%, equally weighted. The long-short version is dollar
neutral (weights sum to 0, gross exposure 2x). The long-only version holds
just the winners and sums to 1. It rebalances monthly.

**Reference:** Jegadeesh & Titman (1993), *Returns to Buying Winners and
Selling Losers*, J. Finance 48(1). The skip month follows standard practice.
Crash risk is covered in Daniel & Moskowitz (2016), *Momentum Crashes*, JFE
122(2).

**Known at t:** the score uses returns from t-251 to t-21, all known by the
close of t.

**Limits:** momentum crashes. The long-short version can lose badly when
markets rebound sharply after a bear market, because the shorted losers rally
hardest. Turnover is high, so costs matter. With few assets each bucket holds
only one or two names. Shorting assumes free borrow.

**Test evidence:** on the planted-momentum panel, Sharpe is about 3.4 after 5bp
costs. On 10 random-walk panels, the mean Sharpe is about 0.0.

### Low volatility: `selection/lowvol.py`

**What:** holds the 30% of assets with the lowest trailing 1-year volatility,
long-only and equally weighted, rebalanced monthly.

**References:** Baker, Bradley & Wurgler (2011), *Benchmarks as Limits to
Arbitrage: Understanding the Low-Volatility Anomaly*, FAJ 67(1). Frazzini &
Pedersen (2014), *Betting Against Beta*, JFE 111(1), explains why
leverage-constrained investors leave this anomaly in place.

**Known at t:** volatility over the 252 days ending at t.

**Limits:** the portfolio concentrates in defensive sectors and is sensitive to
interest rates. It can lag badly in strong rallies. Unlevered and long-only,
it delivers the risk reduction, not the levered betting-against-beta return.
The strength of the anomaly varies a lot across periods and markets.

---

## Risk

### Monte Carlo drawdown and ruin: `risk/montecarlo.py`

**What:** `block_bootstrap` builds thousands of alternative histories by
stitching together random blocks of real past returns. Block lengths are
random, averaging 20 days, which preserves volatility clustering.
`drawdown_distribution` reports the median, 5th and 1st percentile maximum
drawdown. `ruin_probability` is the share of paths that fall more than (say)
50% from a peak within 5 years.

**Reference:** Politis & Romano (1994), *The Stationary Bootstrap*, JASA
89(428).

**Known at t:** whatever history is passed in.

**Limits:** it assumes the future looks like the past distribution. It cannot
produce a crash bigger than anything in the sample, and it ignores regime
changes and dependence longer than a few blocks. A 0% ruin probability means
"not seen in resampling", not "safe". To test a levered strategy, bootstrap
the levered returns.

### Tail risk: `risk/tail.py`

**What:** `var_cvar` gives historical VaR (the return exceeded on the downside
only 5% or 1% of days) and CVaR / expected shortfall (the average return on
those worst days), in return units, so losses are negative and CVaR <= VaR.
`correlation_stress` reports portfolio volatility now versus if every pairwise
correlation jumped to 0.9.

**References:** Rockafellar & Uryasev (2000), *Optimization of Conditional
Value-at-Risk*, J. Risk 2(3). Acerbi & Tasche (2002) on why CVaR is coherent
and VaR is not. Longin & Solnik (2001) on correlations rising in sell-offs.

**Known at t:** whatever history and weights are passed in.

**Limits:** a 99% estimate on a few hundred days rests on a handful of
observations. One-day figures don't scale cleanly to longer horizons. The
stress test changes correlations only, not volatilities, which real crises
also raise. For long-short books, higher correlation can *lower* measured
risk, which is arithmetic, not safety.
