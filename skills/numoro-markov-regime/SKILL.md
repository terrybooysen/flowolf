---
name: numoro-markov-regime
description: Flowolf, powered by Numoro. Markov regime read for a stock, ETF, index or crypto ticker - Markov chain on daily returns, hidden-Markov calm/choppy/stress regimes, walk-forward backtest - published as a report page. Use whenever the user says "run flowolf on [ticker]" or "run numoro markov regime on [ticker]", asks for a Markov, regime or hedge-fund Markov method read on a ticker, or types it with typos like "flow wolf" or "numorro markov regine".
---

<!-- Flowolf — powered by Numoro -->

# Numoro Markov Regime · Flowolf, powered by Numoro

Turns about ten years of a ticker's daily closes into a one-page read. The page answers three questions:

- What does a Markov chain say about tomorrow?
- Which hidden regime is the stock in now?
- Would either signal have made money after costs?

`scripts/build_report.py` writes the page's verdict from fixed rules, so the prose always matches the numbers. Your job is to get clean prices in, run the scripts, publish the page and summarise it honestly.

**Trigger:** "run flowolf on TICKER" or "run numoro markov regime on TICKER", in any casing or spelling. Defaults:

- Ten years of daily data.
- Yahoo symbols: `AAPL`, `SPY`, `^GSPC`, `BTC-USD`, or JSE names like `NPN.JO`. Map company names to the symbol yourself and say which one you used.
- Several tickers means one run per ticker.

Paths below are relative to this skill's folder. Run the scripts in place and write outputs to the working directory. Set up a task list: prices, analysis, page, publish, reply.

## 1. Get prices into `prices.csv`

Try the routes in order and stop at the first that works.

**A. Direct download.** This works anywhere the shell has open internet, such as Claude Code on a laptop.

    python scripts/fetch_prices.py TICKER --years 10 --out prices.csv

Exit code 2 means the network refused the request. Go to route B.

**B. Through a browser.** Use this in cloud sessions such as claude.ai, where the shell can only reach package registries and GitHub. Don't retry Yahoo with curl, wget or a Python package: every shell route goes through the same allowlist.

1. Open `https://query1.finance.yahoo.com/v8/finance/chart/TICKER?range=10y&interval=1d&events=div%2Csplit&includeAdjustedClose=true`. Use the built-in browser on the user's linked computer if present, otherwise Claude in Chrome. Read that browser's skill first and request site access when asked.
2. Run the contents of `scripts/browser_extract.js` with the browser's javascript tool on that page.
3. Write the returned object, exactly as returned, to `chunks.json` with the Write tool. It is about 20 KB of digits. Copy it with care; the checksums exist because a single slipped digit would corrupt returns.
4. Run `python scripts/decode_chunks.py chunks.json --out prices.csv`. It must print `ALL CHECKS PASSED`. If a chunk fails:
   - set `ONLY` to that chunk's index in the JS and run it again;
   - replace that chunk in `chunks.json`;
   - decode again.

**C. A file from the user.** If the user attached a Yahoo, Nasdaq or Stooq export, or anything with a date and close column:

    python scripts/fetch_prices.py --from-csv the_file.csv --out prices.csv

If the file only has unadjusted closes, say so in the reply, since dividends are then ignored.

If no route works, ask the user to attach a CSV, for example from Yahoo Finance's Historical Data download.

`prices_meta.json` records a live intraday price when the market is open. That bar is left out of the analysis on purpose: an unfinished day would be misclassified.

## 2. Analyse

    pip install --break-system-packages hmmlearn scipy    # only if missing
    python scripts/analyze.py --csv prices.csv --ticker TICKER

This writes `results.json` and `arrays.npz` and takes about a minute for ten years. It needs at least about 600 sessions. With less history, the backtest warm-up shrinks and the page says so.

## 3. Build the page

    python scripts/build_report.py --results results.json --out report.html [options]

| Option | Use |
|---|---|
| `--verified` | Data came through route B; the page then says it was checksum-verified. |
| `--live-price P --live-label "Monday 5 October at 11:08 New York"` | Copy these from `prices_meta.json` when it has a `live` entry. |
| `--standalone site/index.html` | Also write a complete HTML document for Vercel or any static host. |
| `--credit ""` | Remove the "Flowolf · powered by Numoro" footer line. Keep it by default. |

The build also writes `facts.json`, which holds the verdict text and headline numbers for your reply.

## 4. Publish

- **With an Artifact tool (claude.ai, Cowork):** publish `report.html` with icon `chart` and a one-sentence description. The page is already designed, theme-aware and phone-friendly, so publish it as built rather than rewriting it.
- **Without one:** send `report.html`, or `site/index.html`, as a file.
- **On Vercel, when the user asks:**
  - Put `site/index.html` at the root of a repo connected to Vercel, with no build step and framework "Other", then push.
  - Or run `vercel deploy --prod` from that folder.
  - This publishes publicly, so confirm the project name first.

## 5. Reply in chat

Keep it short. Base it on `facts.json`:

1. Open with the verdict lead sentence.
2. Add three bullets:
   - **Chain:** today's state, the chance tomorrow closes up against the base rate, and the direction-test p-value.
   - **Regime:** calm, choppy and stress probabilities now, and 20-day volatility against its median.
   - **Backtest:** the chain rule's annual return against buy-and-hold; the filter's annual return and worst drawdown against buy-and-hold; and whether the Sharpe gain's 95% range includes zero.
3. Add one line on the data source, period and any data caveat.
4. End with: "Statistical description of past prices, not investment advice."

Report the walk-forward rules' current positions (long or cash) as model outputs, never as advice to buy, sell or hold.

## Reading the results honestly

These rules keep the reply consistent with the page.

- **Direction.** When the 5-band and 3-band direction tests both give p ≥ 0.05, there is no directional edge, whatever the matrix looks like. Big-move rows that send weight to both tails are volatility clustering, not direction.
- **Multiple comparisons.** With five bands, one row's t-statistic near 2 is expected by chance. A pattern only counts if it survives the split-sample check and the walk-forward chain rule.
- **Regimes.** Regimes are volatility states. The stress filter's usual trade-off is shallower crashes in exchange for missed V-shaped rebounds. Check the crash-and-rebound table before praising it.
- **Significance.** A bootstrap Sharpe range that includes zero means "not statistically secure". The timing test shows whether exits beat chance, not whether the strategy beats buy-and-hold.
- **No forecasts.** Don't turn the backtest into a forecast, a price target or a recommendation.

## Files

| File | What it does |
|---|---|
| `scripts/fetch_prices.py` | Routes A and C: direct download or CSV clean-up into `prices.csv`. |
| `scripts/browser_extract.js` and `scripts/decode_chunks.py` | Route B: checksummed browser transfer. |
| `scripts/analyze.py` | Chain, HMM regimes, walk-forward backtest and robustness checks into `results.json`. |
| `scripts/build_report.py` | Rule-based prose, tables and chart data; writes the page and `facts.json`. |
| `assets/template.html` | The page: charts with hover, light and dark themes, phone layout. |
