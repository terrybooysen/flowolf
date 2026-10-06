// Flowolf — powered by Numoro
// Route B: run with the browser's javascript tool on the Yahoo chart page, e.g.
//   https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=10y&interval=1d&events=div%2Csplit&includeAdjustedClose=true
// It returns the adjusted closes in compact checksummed chunks. Copy the returned object verbatim into
// chunks.json, then run: python decode_chunks.py chunks.json --out prices.csv
// Token format: first character = calendar-day gap from the previous row (0 for the first), rest = adj close x 1000.
// To re-pull one chunk after a checksum failure, set ONLY to its index (0-based).
const ONLY = null;
const CHUNKS = 4;
const j = JSON.parse(document.body.innerText);
const r = j.chart.result[0];
const m = r.meta, gmt = m.gmtoffset;
const ts = r.timestamp;
const adjRaw = (r.indicators.adjclose && r.indicators.adjclose[0].adjclose) || r.indicators.quote[0].close;
const rows = ts.map((t, i) => [new Date((t + gmt) * 1000).toISOString().slice(0, 10), adjRaw[i]]).filter(x => x[1] != null);
const reg = (m.currentTradingPeriod || {}).regular || {};
const nowS = Date.now() / 1000;
let live = null;
if (reg.start <= nowS && nowS < reg.end) {
  const today = new Date((reg.start + gmt) * 1000).toISOString().slice(0, 10);
  if (rows.length && rows[rows.length - 1][0] === today) {
    live = { date: today, price: m.regularMarketPrice, time_utc: new Date(m.regularMarketTime * 1000).toISOString() };
    rows.pop();
  }
}
const dates = rows.map(x => x[0]);
const p = rows.map(x => Math.round(x[1] * 1000));
const g = dates.map((d, i) => i === 0 ? 0 : (Date.parse(d) - Date.parse(dates[i - 1])) / 86400000);
const N = rows.length, size = Math.ceil(N / CHUNKS), chunks = [];
for (let c = 0; c < CHUNKS; c++) {
  if (ONLY !== null && c !== ONLY) continue;
  const a = c * size, b = Math.min(N, a + size);
  if (a >= b) continue;
  let s = 0, w = 0, sg = 0;
  const toks = [];
  for (let i = a; i < b; i++) {
    if (g[i] > 9) throw new Error("gap > 9 days at " + dates[i] + "; split the request");
    toks.push(String(g[i]) + String(p[i]));
    s += p[i]; sg += g[i]; w = (w + (i - a + 1) * p[i]) % 1000000007;
  }
  chunks.push({ c, n: b - a, sumG: sg, sumP: s, wsum: w, frm: dates[a], to: dates[b - 1], data: toks.join(",") });
}
({ symbol: m.symbol, currency: m.currency, N, size, base: dates[0], live, chunks })
