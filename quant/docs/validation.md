# Validation gate

Every strategy Flowolf shows has to survive tests for luck and overfitting first. The numbers are computed by code (`flowolf_quant/validation/`), never by a language model. A pass is evidence to review, not a promise about future returns.

## The verdict

Each run returns **Pass**, **Fail** or **Not enough data**, with every check's number and a plain reason. Any failed check means Fail. A check that couldn't be computed (too little data, only one variant recorded, no holdout) means Not enough data.

## Default thresholds

These are Claude's proposal and wait for Numoro's sign-off. Only a Numoro admin can change them, and only to make them stricter: constructing a looser `GateConfig` raises an error, and so does `GateConfig.tightened` on any loosening change.

| Check | Default | Plain meaning |
|---|---|---|
| Deflated Sharpe probability | at least 0.95 | At least 95% likely the strategy has real skill after counting every variant tried |
| Probability of backtest overfitting | at most 0.25 | The research winner lands in the bottom half out of sample in at most a quarter of splits |
| Holdout Sharpe | above 0 | It made money on the last two years, which nobody looked at while designing it |
| Design data | at least 3 years | Before the holdout |
| Trades | at least 30 | Entries and exits in the design data |

## The checks

**Deflated Sharpe ratio** (Bailey & López de Prado, *Journal of Portfolio Management* 40(5), 2014). Try 50 variants of a rule on pure noise and the best one will still show a decent Sharpe. The deflated Sharpe asks whether the chosen variant beats the Sharpe you'd expect from the luckiest of all the variants you tried, allowing for skew and fat tails. It needs the honest trial count, failures included, which is why every variant goes into the trials registry. Variants whose daily returns correlate at 0.95 or more (the same signal at a different position size, say) are grouped as one idea. The gate then uses whichever benchmark is *stricter*: the paper's formula with every variant tried and their measured spread of Sharpes, or the grouped count with a sampling-noise floor on the spread. So grouping can rescue a real edge from being over-penalised for duplicates, but it can never make the bar lower than the paper's own. Trials from earlier runs on the same family and universe (another holdout length, older data) still count towards the total, so rerunning can't reset it.

**Probabilistic Sharpe ratio** (Bailey & López de Prado, *Journal of Risk* 15(2), 2012). The probability the true Sharpe is above a benchmark, given the sample length and the shape of the returns. The deflated Sharpe is this measured against the best-of-N benchmark. The same paper gives the minimum track record length: how many days you need before a Sharpe that size means anything.

**Probability of backtest overfitting** (Bailey, Borwein, López de Prado & Zhu, *Journal of Computational Finance* 20(4), 2017). The returns of every variant are cut into 16 time blocks. For each of the 12,870 ways to pick half the blocks as "research" data, the variant that did best there is checked on the other half. If it usually lands in the bottom half, the research was picking noise. Pure noise scores about 0.5; a real edge scores near 0. Variants that lost money significantly in the research half are left out of that ranking, so padding the trial set with sure losers can't make a noise winner look robust; moderately bad padding can still lower PBO, but it widens the spread of trial Sharpes and the deflated Sharpe then fails it (0 of 40 test worlds passed the full gate).

**Purged cross-validation with an embargo** (López de Prado, *Advances in Financial Machine Learning*, 2018, ch. 7). When a parameter is chosen on training folds and scored on a test fold, training days that share data with the test block are dropped: days whose outcome runs into the test block, days just after it whose look-back window reaches into it, and an extra buffer for serial correlation.

**Walk-forward degradation** (standard practice; Pardo 2008). Pick the best parameter on a training window, trade it on the next window, roll forward. The ratio of out-of-sample to in-sample Sharpe shows how much of the research survives contact with new data.

**Locked holdout.** The last two years are hidden from all design work and scored once. The ledger is required and records every use; scoring the same family on the same universe again on any *overlapping* holdout (a different length, a day trimmed) is allowed but flagged in the verdict, because a holdout you keep checking stops being unseen.

**Keeping trials apart.** Trials are keyed by family, universe and a fingerprint of the exact design data. A run with a different holdout boundary, different data or a different asset can never feed this run's selection or statistics, and the research loop refuses to evaluate if any trial's dates differ from the design window. A universe name is required, so two assets can't be pooled by accident.

**Trades.** The trade count is position entries and exits (changes of direction), not days on which a volatility-scaled position was resized.

**Data length.** Counted after the warm-up, the leading days before a rule has enough history to take a position.

## How the tests prove it works

`tests/test_validation.py` and `tests/test_research.py`:
- the expected-best-of-N Sharpe formula matches a 300-run simulation within 10%;
- the probabilistic Sharpe matches the textbook normal-returns formula;
- the deflated Sharpe falls as the number of trials rises;
- PBO is about 0.5 on pure noise and below 0.1 when one variant is genuinely better;
- purged folds never overlap the test block, its look-back reach or the embargo;
- the gate fails noise in at least 9 of 10 random worlds and passes a planted edge;
- the gate fails a real edge when the search grid misses it, because the variant found is too weak;
- holdout reuse is flagged, including on an overlapping holdout of a different length;
- an independent review (18 Oct 2026) re-derived the formulas and attacked the gate; its findings are fixed and each has a regression test: trials can't leak across holdout boundaries or assets, near-duplicates and padding can't lower the bar, vol-scaled resizing doesn't count as trades, looser thresholds can't be constructed.

## Limits

- All of this assumes returns are roughly independent from day to day. Strong autocorrelation makes the Sharpe estimate noisier than the formulas assume.
- The trial count is only as honest as the registry. A variant tested outside the loop and never recorded makes every verdict too kind.
- Synthetic tests prove the machinery, not that any real strategy works. Real-data verdicts come from the staging workers.
- Backtests over dates before a language model's training cutoff are contaminated if a model chose the strategy. Only forward results are clean.
