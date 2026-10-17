# Validation gate

Every strategy Flowolf shows has to survive tests for luck and overfitting first. The numbers are computed by code (`flowolf_quant/validation/`), never by a language model. A pass is evidence to review, not a promise about future returns.

## The verdict

Each run returns **Pass**, **Fail** or **Not enough data**, with every check's number and a plain reason. Any failed check means Fail. A check that couldn't be computed (too little data, only one variant recorded, no holdout) means Not enough data.

## Default thresholds

These are Claude's proposal and wait for Numoro's sign-off. Only a Numoro admin can change them, and only to make them stricter (`GateConfig.tightened`, which raises an error on any loosening).

| Check | Default | Plain meaning |
|---|---|---|
| Deflated Sharpe probability | at least 0.95 | At least 95% likely the strategy has real skill after counting every variant tried |
| Probability of backtest overfitting | at most 0.25 | The research winner lands in the bottom half out of sample in at most a quarter of splits |
| Holdout Sharpe | above 0 | It made money on the last two years, which nobody looked at while designing it |
| Design data | at least 3 years | Before the holdout |
| Trades | at least 30 | In the design data |

## The checks

**Deflated Sharpe ratio** (Bailey & López de Prado, *Journal of Portfolio Management* 40(5), 2014). Try 50 variants of a rule on pure noise and the best one will still show a decent Sharpe. The deflated Sharpe asks whether the chosen variant beats the Sharpe you'd expect from the luckiest of all the variants you tried, allowing for skew and fat tails. It needs the honest trial count, failures included, which is why every variant goes into the trials registry. Variants whose daily returns correlate at 0.95 or more (the same signal at a different position size, say) count as one idea; complete-linkage clustering keeps that conservative, so genuinely different variants are never merged. The threshold can only be raised.

**Probabilistic Sharpe ratio** (Bailey & López de Prado, *Journal of Risk* 15(2), 2012). The probability the true Sharpe is above a benchmark, given the sample length and the shape of the returns. The deflated Sharpe is this measured against the best-of-N benchmark. The same paper gives the minimum track record length: how many days you need before a Sharpe that size means anything.

**Probability of backtest overfitting** (Bailey, Borwein, López de Prado & Zhu, *Journal of Computational Finance* 20(4), 2017). The returns of every variant are cut into 16 time blocks. For each of the 12,870 ways to pick half the blocks as "research" data, the variant that did best there is checked on the other half. If it usually lands in the bottom half, the research was picking noise. Pure noise scores about 0.5; a real edge scores near 0.

**Purged cross-validation with an embargo** (López de Prado, *Advances in Financial Machine Learning*, 2018, ch. 7). When a parameter is chosen on training folds and scored on a test fold, training days that share data with the test block are dropped: days whose outcome runs into the test block, days just after it whose look-back window reaches into it, and an extra buffer for serial correlation.

**Walk-forward degradation** (standard practice; Pardo 2008). Pick the best parameter on a training window, trade it on the next window, roll forward. The ratio of out-of-sample to in-sample Sharpe shows how much of the research survives contact with new data.

**Locked holdout.** The last two years are hidden from all design work and scored once. The ledger records every use; scoring the same family on the same holdout again is allowed but flagged in the verdict, because a holdout you keep checking stops being unseen.

## How the tests prove it works

`tests/test_validation.py` and `tests/test_research.py`:
- the expected-best-of-N Sharpe formula matches a 300-run simulation within 10%;
- the probabilistic Sharpe matches the textbook normal-returns formula;
- the deflated Sharpe falls as the number of trials rises;
- PBO is about 0.5 on pure noise and below 0.1 when one variant is genuinely better;
- purged folds never overlap the test block, its look-back reach or the embargo;
- the gate fails noise in at least 9 of 10 random worlds and passes a planted edge;
- the gate fails a real edge when the search grid misses it, because the variant found is too weak;
- holdout reuse is flagged.

## Limits

- All of this assumes returns are roughly independent from day to day. Strong autocorrelation makes the Sharpe estimate noisier than the formulas assume.
- The trial count is only as honest as the registry. A variant tested outside the loop and never recorded makes every verdict too kind.
- Synthetic tests prove the machinery, not that any real strategy works. Real-data verdicts come from the staging workers.
- Backtests over dates before a language model's training cutoff are contaminated if a model chose the strategy. Only forward results are clean.
