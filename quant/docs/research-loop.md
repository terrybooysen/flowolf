# Research loop

`flowolf_quant/research/` runs the "self-improving" loop the honest way.

1. The last two years are locked away as a holdout before anything runs.
2. A proposer suggests parameter settings for a strategy family: a grid, random search, or a language model (`LLMProposer`).
3. Every proposal is clipped to the family's declared bounds, backtested on the design data with costs, and recorded in the trials registry.
4. The best variant by design Sharpe is scored once on the holdout, and the use is logged.
5. The gate judges it against every variant ever tried in that family on that universe.

Nothing is promoted on a single number. Compare the viral Hermes template, where one cycle's result changed the live strategy and a labelling bug drove every change.

## Strategy families (first set)

| Family | Idea | Reference |
|---|---|---|
| `trend_sign` | Long when the trailing average return is positive, short or flat otherwise | Moskowitz, Ooi & Pedersen (2012), simplified |
| `ma_cross` | Long when the fast moving average is above the slow one | Brock, Lakonishok & LeBaron (1992) |
| `tsmom` | Time-series momentum with volatility scaling | Moskowitz, Ooi & Pedersen (2012) |
| `xsmom` | Own recent winners, short recent losers, across a universe | Jegadeesh & Titman (1993) |
| `lowvol` | Own the lowest-volatility names | Baker, Bradley & Wurgler (2011) |

New families are one entry in `FAMILIES`: a weight function, a parameter space and a reference.

## The language-model proposer

`LLMProposer` builds the prompt and parses the reply; the model call is passed in, so the library holds no API keys and makes no network calls. The prompt shows design-data results only, never the holdout, and tells the model that every variant it proposes counts against the strategy in the multiple-testing correction, so it should propose few, reasoned variants.

## Limits

- More variants make the deflated-Sharpe bar higher. That's deliberate: a model that proposes hundreds of variants will find it harder to pass, not easier.
- The holdout is one stretch of history. A strategy can pass it by luck; the forward track record is the final test.
