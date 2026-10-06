# flowolf

Flowolf, powered by Numoro: a trading agent built from A to Z. MVP/POC for Lone Bull Group: start simple, grow it over time. **Paper trading only until Terry signs off.**

## A to Z
| Stage | What | Status |
|---|---|---|
| Data | Ten years of daily closes from Yahoo Finance | Built: `fetch_prices.py` (JSE tickers still need manual fixes) |
| Analysis | Markov chain on daily returns; hidden-Markov calm/choppy/stress regimes | Built: `analyze.py` |
| Backtest | Walk-forward after 5 bp costs, with bootstrap, timing and threshold checks | Built: `analyze.py` |
| Report | One-page read; the verdict is written by fixed rules so it matches the numbers | Built: `build_report.py`, `assets/template.html` |
| Strategy rules | Daily rule set on the watchlist (e.g. trend filter, fixed size, stop, regime filter) | Phase 1, not started |
| Paper trading | Alpaca paper account, shown in TradingView | Phase 1, not started |
| Runtime | Hermes Agent on its own Hostinger server, one daily job. Server setup: `docs/SETUP-GUIDE.md` in hermes-bots | Phase 1, not started |
| Reporting | Telegram: daily trades, positions and P&L against buy-and-hold; weekly scorecard | Phase 1, not started |
| Guardrails | Paper keys only, kill switch, bounds on every parameter, rollback | Phase 1, not started |
| AI second opinion | A second paper account where the AI must agree to each rule-triggered entry | Phase 2 |
| Self-improvement | A change goes live only if it beats the current version on unseen data | Phase 3 |
| Real money | Only after Terry signs off and a FAIS check | Later |

## Layout
| Path | What |
|---|---|
| `skills/numoro-markov-regime/` | The analysis engine as a Claude skill ("run flowolf on TICKER"): SKILL.md, scripts, report template |

## Run the analysis
From the repo root, in a shell that can reach Yahoo (e.g. your Mac):
```
pip install pandas numpy scipy hmmlearn
S=skills/numoro-markov-regime/scripts
python $S/fetch_prices.py AAPL --years 10 --out prices.csv
python $S/analyze.py --csv prices.csv --ticker AAPL
python $S/build_report.py --results results.json --out report.html
```
Outputs land in the current folder and are ignored by git.

## House rules
- Nothing secret in git. Keys and `.env` files live only on the server.
- Paper only: no live broker keys and no real money without Terry's sign-off.
- Reports are a statistical description of past prices, not investment advice.

## Build record
Server, accounts, decisions and open items: `claude/flowolf-build-record.md` in the Claude project "Numoro AI Trading Agent".
