# flowolf-quant

Flowolf's own quant library: regimes, allocation, selection, risk, a research loop, and the validation gate every result has to pass. Built from published research, owned by Flowolf. It follows Numoro's process: Regimes → Allocation → Selection, with execution left to people.

| Folder | What | Docs |
|---|---|---|
| `flowolf_quant/validation/` | Deflated and probabilistic Sharpe, probability of backtest overfitting, purged cross-validation, walk-forward, locked holdout, trials registry, the gate verdict | `docs/validation.md` |
| `flowolf_quant/research/` | Strategy families, proposers (grid, random, language model), the loop that records every trial and lets the gate decide | `docs/research-loop.md` |
| `flowolf_quant/regimes/` | GARCH volatility regimes, time-series momentum, Bayesian change-point detection | `docs/regimes.md` |
| `flowolf_quant/allocation/` | Volatility targeting, hierarchical risk parity, Kelly sizing that refuses without evidence | `docs/allocation-selection-risk.md` |
| `flowolf_quant/selection/` | Cross-sectional momentum, low volatility | `docs/allocation-selection-risk.md` |
| `flowolf_quant/risk/` | Block-bootstrap drawdowns and ruin probability, VaR/CVaR, correlation stress | `docs/allocation-selection-risk.md` |
| `flowolf_quant/backtest.py` | Strict backtester: weights decided at a close apply to the next day, costs on every trade | |
| `flowolf_quant/data.py` | Synthetic data (random walk, planted edge, regime switching) for tests | |

The Markov/HMM engine stays in `skills/numoro-markov-regime/`; the app's workers import it alongside this library.

## Use

```
cd quant
pip install -e ".[test]"        # numpy, scipy, pandas; pytest for the tests
python -m pytest -q
```

```python
from flowolf_quant import data
from flowolf_quant.research import research, GridProposer

r = data.planted_trend(n=3024, strength=0.10, lookback=20, seed=4)   # or real daily returns
report = research(r, "trend_sign", GridProposer(points_per_param=50))
print(report.verdict.status)     # Pass / Fail / Not enough data
print(report.verdict.summary)
```

## House rules

- Every module is implemented from a named paper, in our own code. No proprietary libraries, and nothing from Numoro's own models: those plug in separately as house signals.
- Signals use only data up to each close; the backtester applies them to the next day.
- Every variant tested is recorded. The gate's numbers are only honest if the trial count is.
- A pass is evidence to review, not proof of future returns. Synthetic tests prove the machinery, not that any real strategy works.
