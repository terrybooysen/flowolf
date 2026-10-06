#!/usr/bin/env python3
# Flowolf — powered by Numoro
"""Get daily adjusted closes into prices.csv (date,adj_close).

Route A — direct download (works where the shell can reach Yahoo, e.g. Claude Code on a laptop):
  python fetch_prices.py AAPL --years 10 --out prices.csv

Route C — normalise a CSV the user attached (Yahoo/Stooq/Nasdaq exports, any file with a date and a close column):
  python fetch_prices.py --from-csv upload.csv --out prices.csv

Also writes prices_meta.json (symbol, currency, last session, live price if the market is open).
Exit code 2 means the network refused the request; switch to the browser route in SKILL.md.
"""
import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval=1d&events=div%2Csplit&includeAdjustedClose=true"


def parse_chart(j):
    """Yahoo chart JSON -> (DataFrame[date, adj_close] of completed sessions, meta dict)."""
    r = j["chart"]["result"][0]
    m = r["meta"]
    gmt = m.get("gmtoffset", 0)
    ts = r["timestamp"]
    adj = r["indicators"].get("adjclose", [{}])[0].get("adjclose") or r["indicators"]["quote"][0]["close"]
    rows = [(dt.datetime.utcfromtimestamp(t + gmt).strftime("%Y-%m-%d"), v) for t, v in zip(ts, adj) if v is not None]
    df = pd.DataFrame(rows, columns=["date", "adj_close"]).drop_duplicates("date", keep="last")
    reg = (m.get("currentTradingPeriod") or {}).get("regular") or {}
    live = None
    now = time.time()
    if reg and reg.get("start", 0) <= now < reg.get("end", 0) and len(df):
        today = dt.datetime.utcfromtimestamp(reg["start"] + gmt).strftime("%Y-%m-%d")
        if df["date"].iloc[-1] == today:          # in-progress bar: keep it out of the analysis
            live = dict(date=today, price=m.get("regularMarketPrice"),
                        time_utc=dt.datetime.utcfromtimestamp(m.get("regularMarketTime", now)).strftime("%Y-%m-%dT%H:%M:%SZ"))
            df = df.iloc[:-1]
    meta = dict(symbol=m.get("symbol"), currency=m.get("currency"), exchange=m.get("exchangeName"),
                timezone=m.get("exchangeTimezoneName"), last_completed=df["date"].iloc[-1] if len(df) else None, live=live)
    return df, meta


def from_csv(path):
    df = pd.read_csv(path)
    cols = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    dcol = cols.get("date") or cols.get("timestamp") or df.columns[0]
    pcol = next((cols[k] for k in ("adj_close", "adjclose", "adj._close", "close", "close/last", "price") if k in cols), None)
    if pcol is None:
        sys.exit(f"can't find a close column in {list(df.columns)}")
    px = pd.to_numeric(df[pcol].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    out = pd.DataFrame({"date": pd.to_datetime(df[dcol]).dt.strftime("%Y-%m-%d"), "adj_close": px}).dropna()
    out = out[out.adj_close > 0].drop_duplicates("date").sort_values("date")
    note = "" if "adj" in pcol.lower() else " (unadjusted closes: dividends ignored, splits may distort returns)"
    return out, dict(symbol=None, source=f"user CSV {path}{note}", last_completed=out["date"].iloc[-1], live=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?")
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--from-csv")
    ap.add_argument("--out", default="prices.csv")
    a = ap.parse_args()
    if a.from_csv:
        df, meta = from_csv(a.from_csv)
    else:
        if not a.ticker:
            sys.exit("give a ticker or --from-csv")
        req = urllib.request.Request(URL.format(sym=a.ticker.upper(), rng=f"{a.years}y"),
                                     headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                j = json.load(resp)
        except (urllib.error.URLError, OSError) as err:
            print(f"Yahoo unreachable from this shell ({err}). Use the browser route or ask the user for a CSV.", file=sys.stderr)
            sys.exit(2)
        if not j.get("chart", {}).get("result"):
            sys.exit(f"Yahoo returned no data for {a.ticker}: {j.get('chart', {}).get('error')}")
        df, meta = parse_chart(j)
    df.to_csv(a.out, index=False, float_format="%.4f")
    json.dump(meta, open(a.out.rsplit(".", 1)[0] + "_meta.json", "w"), indent=1)
    print(f"{len(df)} sessions {df['date'].iloc[0]} -> {df['date'].iloc[-1]} written to {a.out}"
          + (f"; live {meta['live']['price']} on {meta['live']['date']}" if meta.get("live") else ""))


if __name__ == "__main__":
    main()
