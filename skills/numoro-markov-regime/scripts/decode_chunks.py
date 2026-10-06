#!/usr/bin/env python3
# Flowolf — powered by Numoro
"""Route B, second half: verify the checksummed chunks from browser_extract.js and write prices.csv.

Usage: python decode_chunks.py chunks.json --out prices.csv
chunks.json is the object browser_extract.js returned (all chunks; or merge re-pulled chunks into it by index).
Any checksum mismatch is reported per chunk; re-pull just that chunk (set ONLY in the JS) and try again.
"""
import argparse
import datetime as dt
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("--out", default="prices.csv")
    a = ap.parse_args()
    J = json.load(open(a.chunks))
    chunks = sorted(J["chunks"], key=lambda ch: ch["c"])
    size, ok, gaps, prices = J["size"], True, [], []
    if [ch["c"] for ch in chunks] != list(range(len(chunks))):
        sys.exit(f"chunks missing: have {[ch['c'] for ch in chunks]}")
    for ch in chunks:
        toks = ch["data"].split(",")
        g = [int(t[0]) for t in toks]
        p = [int(t[1:]) for t in toks]
        w = 0
        for i, v in enumerate(p):
            w = (w + (i + 1) * v) % 1000000007
        got = dict(n=len(toks), sumG=sum(g), sumP=sum(p), wsum=w)
        bad = {k: (got[k], ch[k]) for k in got if got[k] != ch[k]}
        print(f"chunk {ch['c']}: {'OK' if not bad else 'MISMATCH ' + str(bad)}")
        ok &= not bad
        gaps += g
        prices += p
    if not ok:
        sys.exit("checksum failure: re-pull the failing chunk(s)")
    d = dt.date.fromisoformat(J["base"])
    rows = []
    for i, (g, v) in enumerate(zip(gaps, prices)):
        if i:
            d += dt.timedelta(days=g)
        rows.append((d.isoformat(), v / 1000))
    for k, ch in enumerate(chunks):
        a_, b_ = k * size, k * size + ch["n"] - 1
        if rows[a_][0] != ch["frm"] or rows[b_][0] != ch["to"]:
            sys.exit(f"chunk {k} dates {rows[a_][0]}..{rows[b_][0]} don't match {ch['frm']}..{ch['to']}")
    if any(dt.date.fromisoformat(r[0]).weekday() >= 5 for r in rows):
        sys.exit("weekend date decoded: transcription error")
    with open(a.out, "w") as fh:
        fh.write("date,adj_close\n")
        fh.writelines(f"{r[0]},{r[1]:.3f}\n" for r in rows)
    json.dump(dict(symbol=J.get("symbol"), currency=J.get("currency"), last_completed=rows[-1][0], live=J.get("live"),
                   source="Yahoo Finance chart API via browser, checksum-verified"),
              open(a.out.rsplit(".", 1)[0] + "_meta.json", "w"), indent=1)
    print(f"ALL CHECKS PASSED: {len(rows)} sessions {rows[0][0]} -> {rows[-1][0]} written to {a.out}")


if __name__ == "__main__":
    main()
