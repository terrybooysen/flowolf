#!/usr/bin/env python3
# Flowolf — powered by Numoro
"""Build the report page from analyze.py output. The prose is generated from the numbers by fixed rules,
so every claim on the page matches results.json.

Usage:
  python build_report.py --results results.json --out report.html
         [--standalone site/index.html]                # full HTML document for Vercel / any static host
         [--live-price 333.67 --live-label "Monday 5 October at 11:08 New York"]
         [--source "Yahoo Finance daily closes adjusted for splits and dividends"] [--verified]
         [--credit "Flowolf · powered by Numoro"]    # pass --credit "" to drop the credit line
         [--template path/to/template.html]            # defaults to ../assets/template.html

report.html is the page body for the Artifact tool, which adds the document wrapper.
Also writes facts.json (verdict text and headline numbers) next to --out, for the chat reply.
"""
import argparse
import datetime as dt
import html
import json
import math
import os
import re

import numpy as np
import pandas as pd

M = "−"
HERE = os.path.dirname(os.path.abspath(__file__))
WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
         11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen"}


def pct(x, d=1, sign=False):
    s = f"{abs(x) * 100:.{d}f}%"
    if x < 0 and float(s[:-1]) != 0:
        return M + s
    return ("+" + s) if (sign and x > 0 and float(s[:-1]) != 0) else s


def num(x, d=2, sign=False):
    s = f"{abs(x):.{d}f}"
    if x < 0 and float(s) != 0:
        return M + s
    return ("+" + s) if (sign and x > 0 and float(s) != 0) else s


def usd(x):
    return f"${x:,.2f}"


def ts(s):
    return pd.Timestamp(s)


def dlong(s):                      # 2 Oct 2026
    d = ts(s)
    return f"{d.day} {d.strftime('%b %Y')}"


def dfull(s):                      # Friday 2 October 2026
    d = ts(s)
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B %Y')}"


def esc(s):
    return html.escape(str(s), quote=False)


def article(label):
    return "an" if label[0].lower() in "aeiou" else "a"


def log_ticks(lo, hi):
    """Tick values for a log axis: prefer 1-2-3-5 steps, thin to 1-2-5 or 1s when crowded, densify when sparse."""
    def pick(mults):
        out = set()
        for k in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 2):
            for m_ in mults:
                v = round(m_ * 10 ** k, 10)
                if lo <= v <= hi:
                    out.add(v)
        return sorted(out)
    out = pick((1, 2, 3, 5))
    if len(out) > 7:
        out = pick((1, 2, 5))
    if len(out) > 7:
        out = pick((1,))
    if len(out) < 3:
        out = pick((1, 1.5, 2, 3, 4, 5, 7))
    lab = lambda v: f"${v:,.0f}" if v >= 10 else f"${v:g}"
    return [[float(v), lab(v)] for v in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--standalone")
    ap.add_argument("--template", default=os.path.join(HERE, "..", "assets", "template.html"))
    ap.add_argument("--live-price", type=float)
    ap.add_argument("--live-label")
    ap.add_argument("--source", default="Yahoo Finance daily closes adjusted for splits and dividends")
    ap.add_argument("--verified", action="store_true", help="data came through the checksummed browser transfer")
    ap.add_argument("--credit", default="Flowolf · powered by Numoro")
    a = ap.parse_args()

    R = json.load(open(a.results))
    meta, c, h, bt, ex, ctx = R["meta"], R["chain"], R["hmm"], R["backtest"], R["extras"], R["context"]
    T = meta["ticker"]
    labels, e, cur = c["labels"], c["edges_pct"], c["current_state"]
    cond, um = c["conditional"], c["uncond_mean"]
    rob = {r_["k"]: r_ for r_ in R["robustness"]}
    p5, p3, p2 = rob[5]["p_direction"], rob[3]["p_direction"], rob[2]["p_direction"]
    last = meta["last_date"]
    L = labels[cur]
    years = R["meta"]["n_prices"] / 252
    span_years = round(years)
    now, n5, n20 = h["now"], h["next5"], h["next20"]
    rg = h["regimes"]
    boot = ex["sharpe_diff_bootstrap"]
    bh, ch, hm = bt["buy_hold"], bt["chain"], bt["hmm"]

    ranges_long = [f"below {num(e[0], 2, True)}%", f"{num(e[0], 2, True)}% to {num(e[1], 2, True)}%",
                   f"{num(e[1], 2, True)}% to {num(e[2], 2, True)}%", f"{num(e[2], 2, True)}% to {num(e[3], 2, True)}%",
                   f"above {num(e[3], 2, True)}%"]
    ranges_short = [f"< {num(e[0], 2, True)}%", f"{num(e[0], 2, True)} to {num(e[1], 2, True)}",
                    f"{num(e[1], 2, True)} to {num(e[2], 2, True)}", f"{num(e[2], 2, True)} to {num(e[3], 2, True)}",
                    f"> {num(e[3], 2, True)}%"]

    # ---------------- judgements (fixed rules) ----------------
    vol_cluster = (c["p_extreme_after_extreme"] - c["p_extreme_after_calm"] >= 0.05) and c["ac1_abs"] >= 0.10
    dir_sig = p5 < 0.05 or p3 < 0.05
    ch_oos = boot["chain"]["lo"] > 0
    hm_secure = boot["hmm"]["lo"] > 0
    names = ["calm", "choppy", "stress"]
    order = list(np.argsort(now)[::-1])
    top, second = int(order[0]), int(order[1])
    if now[top] - now[second] < 0.20:
        lo_i, hi_i = sorted((top, second))
        regime_phrase = f"between {names[lo_i]} and {names[hi_i]}"
        regime_tile = f"{names[lo_i].capitalize()} / {names[hi_i]}"
    else:
        regime_phrase = f"in the {names[top]} regime ({pct(now[top], 0)})"
        regime_tile = names[top].capitalize()
    stress_clause = ("" if top == 2 else
                     ", with almost no stress probability" if now[2] < 0.02 else f", with a {pct(now[2])} stress probability")

    hl = c["memory_halflife_days"]
    mem_txt = "a day" if hl < 1 else f"about {round(hl)} days"
    bt_start_my = ts(bt["start"]).strftime("%B %Y")

    # crash / rebound pairs
    legs = ex["legs"]
    pairs = {}
    for lg in legs:
        if lg["pair"]:
            pairs.setdefault(lg["pair"], []).append(lg)
    pair_info = []
    for key, items in pairs.items():
        if len(items) != 2:
            continue
        crash, reb = items
        comb = {k: (1 + crash[k]) * (1 + reb[k]) - 1 for k in ("buy_hold", "hmm")}
        saved = crash["hmm"] > crash["buy_hold"] + 0.02
        missed = reb["buy_hold"] > 0 and reb["hmm"] < 0.6 * reb["buy_hold"]
        label = "COVID" if key == "covid" else "April 2025"
        year = ts(crash["start"]).year
        pair_info.append(dict(label=label, year=year, saved=saved, missed=missed, **comb))

    # verdict
    if not dir_sig:
        lead = "No usable directional edge."
    elif not ch_oos:
        lead = "A statistical pattern in direction, but it didn't pay out of sample."
    else:
        lead = "A directional pattern that held up out of sample."
    lead += f" {T}'s Markov memory is in volatility, not direction." if (vol_cluster and not ch_oos) else ""
    s1 = (f"After {ts(last).strftime('%A')}'s {pct(c['last_return'], 2, True)} day the chain gives the next session a "
          f"{pct(cond[cur]['p_up'])} chance of closing up, against {pct(c['uncond_pup'])} for an average day, and that memory fades within {mem_txt}.")
    if p5 >= 0.05:
        s2 = f"A test of whether today's state predicts tomorrow's direction finds nothing (p = {p5:.2f})."
    else:
        s2 = (f"A test of whether today's state predicts tomorrow's direction flags a dependence (p = {p5:.2f}), "
              + ("and the chain rule built on it beat buy-and-hold out of sample." if ch_oos else
                 "but the chain rule built on it didn't beat buy-and-hold out of sample."))
    s3 = f"The regime model has {T} {regime_phrase}" + (f", with a {pct(now[2])} stress probability." if top != 2 else ".")
    dd_gain = hm["maxdd"] - bh["maxdd"]
    dd_txt = (f"cut the worst drawdown from {pct(bh['maxdd'])} to {pct(hm['maxdd'])}" if dd_gain > 0.02 else
              f"left the worst drawdown about the same ({pct(hm['maxdd'])} against {pct(bh['maxdd'])})" if dd_gain > -0.02 else
              f"deepened the worst drawdown ({pct(hm['maxdd'])} against {pct(bh['maxdd'])})")
    cg = hm["cagr"] - bh["cagr"]
    ret_txt = (f"for a similar return ({pct(hm['cagr'])} a year against {pct(bh['cagr'])})" if abs(cg) <= 0.015 else
               f"and beat buy-and-hold ({pct(hm['cagr'])} a year against {pct(bh['cagr'])})" if cg > 0 else
               f"but trailed buy-and-hold ({pct(hm['cagr'])} a year against {pct(bh['cagr'])})")
    tail = ""
    if boot["hmm"]["point"] > 0:
        tail = ", and the Sharpe improvement is statistically secure" if hm_secure else ", but that gain isn't statistically secure"
    if pair_info and all(p_["saved"] and p_["missed"] for p_ in pair_info):
        tail += " and the filter missed most of the rebound after each crash" if tail else ", and the filter missed most of the rebound after each crash"
    s4 = f"Used as a stress filter since {bt_start_my}, it {dd_txt} {ret_txt}{tail}."
    verdict_body = " ".join([s1, s2, s3, s4])

    # chain section
    P_all = np.array(c["P"])
    tk = ("Tomorrow's direction barely depends on today's state." if p5 >= 0.05 else
          f"Tomorrow's direction depends on today's state more than chance would explain (p = {p5:.2f}).")
    chain_takeaway = (tk + f" Every daily return falls into one of five equal bands, so a chain with no memory would put 20% on each; "
                      f"the real transition probabilities range from {pct(P_all.min())} to {pct(P_all.max())}"
                      + (", and most of that spread is big moves following big moves." if vol_cluster else "."))
    tvals = [abs(cd["t_vs_avg"]) for cd in cond]
    jmax = int(np.argmax(tvals))
    if max(tvals) < 1.96:
        cn1 = f"No row moves the next-day average more than {max(tvals):.1f} standard errors from an average day"
    else:
        cn1 = f"The {labels[jmax]} row moves the next-day average {max(tvals):.1f} standard errors from an average day"
    chain_note = (cn1 + f", and the chi-square test on next-day direction gives p = {p5:.2f} (p = {p3:.2f} with three bands, {p2:.2f} with two). "
                  f"Further out the chain adds little: from today's state the expected 5-day return is {pct(c['exp_cum_path'][4], 2, True)} "
                  f"against {pct(c['uncond_cum_path'][4], 2, True)} for any day, and {pct(c['exp_cum_path'][19], 2, True)} against "
                  f"{pct(c['uncond_cum_path'][19], 2, True)} over 20 days. The chain's memory half-life is {hl:.1f} trading days.")

    # volatility section
    acp, act, ntr = ex["ac1_pearson"], ex["ac1_trimmed"], ex["n_trimmed_pairs"]
    if ntr == 0 or act is None:
        ac_note = f"The ordinary return autocorrelation is {num(acp, 2, True)}."
    elif acp < -0.03 and abs(act) < 0.03:
        ac_note = (f"The ordinary return autocorrelation is {num(acp, 2, True)}, which looks like mean reversion, but it comes from "
                   f"{ntr} days with moves beyond ±5%. Without them it is {num(act, 2, True)}.")
    else:
        ac_note = f"The ordinary return autocorrelation is {num(acp, 2, True)} ({num(act, 2, True)} without the {ntr} days with moves beyond ±5%)."

    # regime section
    rv, rvm = ctx["rv20"], ctx["rv20_median"]
    rel = rv / rvm - 1
    rv_phrase = (f"right at its {span_years}-year median" if abs(rel) < 0.10 else
                 f"above its {span_years}-year median of {pct(rvm, 0)}" if rel > 0 else
                 f"below its {span_years}-year median of {pct(rvm, 0)}")
    reg_takeaway = (f"A three-state hidden Markov model splits {T}'s history into calm, choppy and stress regimes. "
                    f"Today it sits {regime_phrase}{stress_clause}, and 20-day realized volatility of {pct(rv, 0)} is {rv_phrase}.")
    bic = {b["k"]: b["bic"] for b in R["hmm_bic"]}
    bic_best = min(bic, key=bic.get)
    if bic_best == 3:
        bic_txt = f"BIC picks three regimes over two or four ({bic[3]:,.0f} against {bic[2]:,.0f} and {bic[4]:,.0f})."
        bic_sentence = "BIC prefers three states over two or four."
    else:
        bic_txt = f"BIC prefers {WORDS[bic_best].lower()} regimes ({bic[bic_best]:,.0f} against {bic[3]:,.0f} for three); three are shown to keep the labels readable."
        bic_sentence = f"BIC prefers {WORDS[bic_best].lower()} states; three are shown for readability."
    st = rg[2]
    spell = "short" if st["duration_days"] < 15 else "persistent"
    rare = "rare" if st["share"] < 0.08 else "not rare"
    reg_note = (f"{bic_txt} Stress spells are {spell} (about {st['duration_days']:.0f} trading days) and {rare} ({pct(st['share'])} of days)"
                + (f", and the losses sit there: the model's average stress day is {pct(st['mean_ann'] / 252, 2, True)}." if st["mean_ann"] < 0 else
                   f"; the model's average stress day is {pct(st['mean_ann'] / 252, 2, True)}, so stress here means wide swings rather than steady losses."))

    # backtest section
    gap = bh["cagr"] - ch["cagr"]
    if gap > 0.01:
        bt1 = ("Not as a forecaster. " if not ch_oos else "") + f"The chain rule lost to buy-and-hold by {gap * 100:.1f} percentage points a year."
    elif gap < -0.01:
        bt1 = f"The chain rule beat buy-and-hold by {-gap * 100:.1f} percentage points a year."
    else:
        bt1 = "The chain rule matched buy-and-hold."
    verb = "kept pace with" if abs(cg) <= 0.015 else ("beat" if cg > 0 else "trailed")
    bt2 = (f" The stress filter {verb} buy-and-hold" + (" with a smaller drawdown" if dd_gain > 0.02 else "")
           + (", and its Sharpe gain is statistically secure." if hm_secure else
              (", but its edge is thin and inside the noise." if verb != "trailed" else ".")))
    bt_takeaway = bt1 + bt2

    def ci(b):
        return f"{num(b['point'], 2, True)} ({num(b['lo'], 2, True)} to {num(b['hi'], 2, True)})"

    zero = [b["lo"] <= 0 <= b["hi"] for b in (boot["hmm"], boot["chain"])]
    zero_txt = ("Both ranges include zero." if all(zero) else
                "Neither range includes zero." if not any(zero) else
                ("The stress filter's range sits above zero." if boot["hmm"]["lo"] > 0 else
                 "The stress filter's range sits below zero." if not zero[0] else
                 "The chain rule's range sits above zero." if boot["chain"]["lo"] > 0 else "The chain rule's range sits below zero."))
    checks = [f"<li><b>Statistical check.</b> Block-bootstrap range for the Sharpe difference against buy-and-hold: stress filter "
              f"{ci(boot['hmm'])}, chain rule {ci(boot['chain'])}. {zero_txt}</li>"]
    tt = []
    if hm.get("shift_percentile") is not None:
        sp = hm["shift_percentile"]
        tt.append(f"The stress filter's exits beat {pct(sp)} of time-shifted copies of its own schedule, "
                  + ("so the timing wasn't luck." if sp >= 0.95 else "which is within what luck could produce."))
    if ch.get("shift_percentile") is not None:
        sp = ch["shift_percentile"]
        tt.append(f"The chain rule's beat {pct(sp, 0)}" + (", no better than chance." if sp < 0.95 else ", better than chance."))
    if tt:
        checks.append("<li><b>Timing test.</b> " + " ".join(tt) + "</li>")
    sens = ex["hmm_threshold_sensitivity"]
    checks.append("<li><b>Exit threshold.</b> Going to cash at a stress probability of 30%, 50%, 70% or 90% gives Sharpe ratios of "
                  + ", ".join(num(s_["sharpe"]) for s_ in sens[:3]) + f" and {num(sens[3]['sharpe'])} against {num(bh['sharpe'])} for buy-and-hold. "
                  "Worst drawdowns: " + ", ".join(pct(s_["maxdd"]) for s_ in sens[:3]) + f" and {pct(sens[3]['maxdd'])} against {pct(bh['maxdd'])}.</li>")
    checks.append(f"<li><b>Direction calls.</b> The chain called the next day's direction right {pct(ch['hit_rate'])} of the time, "
                  + ("below" if ch["hit_rate"] < bt["always_up_hit_rate"] else "against") + f" the {pct(bt['always_up_hit_rate'])} you'd get by always calling up.</li>")
    sd = ex["split_dates"]
    stable = [s_ for s_ in ex["split"] if np.sign(s_["first_vs_avg"]) == np.sign(s_["second_vs_avg"])
              and min(abs(s_["first_vs_avg"]), abs(s_["second_vs_avg"])) >= 0.0005]
    halves = f"{ts(sd[0]).year}–{ts(sd[1]).year} against {ts(sd[2]).year}–{ts(sd[3]).year}"
    if stable:
        parts = [f"the day after {article(s_['state'])} {s_['state']} day ({num(s_['first_vs_avg'] * 100, 2, True)}, then "
                 f"{num(s_['second_vs_avg'] * 100, 2, True)} points against an average day)" for s_ in stable]
        checks.append(f"<li><b>Split sample.</b> Comparing {halves}, only " + "; ".join(parts) + " kept the same edge in both halves. "
                      "Every other band's edge changed sign or faded.</li>")
    else:
        checks.append(f"<li><b>Split sample.</b> Comparing {halves}, no band kept a consistent edge over an average day in both halves.</li>")

    # legs block
    legs_block = ""
    if legs:
        rows = "".join(f'<tr><th scope="row">{esc(lg["name"])}</th><td style="font-family:var(--font-body)">{dlong(lg["start"])} – {dlong(lg["end"])}</td>'
                       f'<td>{pct(lg["buy_hold"], 1, True)}</td><td>{pct(lg["chain"], 1, True)}</td><td>{pct(lg["hmm"], 1, True)}</td></tr>' for lg in legs)
        legs_block = ('    <div class="table-card"><p class="tbl-title">Crashes and the rebounds after them</p><div class="table-scroll"><table>'
                      '<thead><tr><th scope="col">Episode</th><th scope="col">Dates</th><th scope="col">Buy &amp; hold</th><th scope="col">Chain rule</th>'
                      f'<th scope="col">Stress filter</th></tr></thead><tbody>{rows}</tbody></table></div></div>\n')
        if pair_info:
            s_m = [p_ for p_ in pair_info if p_["saved"] and p_["missed"]]
            comp = "; ".join(f"{pct(p_['hmm'], 1, True)} against {pct(p_['buy_hold'], 1, True)} for {p_['year']}" for p_ in pair_info)
            rel_word = ("behind" if all(p_["hmm"] < p_["buy_hold"] for p_ in pair_info) else
                        "ahead of" if all(p_["hmm"] > p_["buy_hold"] for p_ in pair_info) else "mixed against")
            txt = ""
            if s_m:
                txt = (f"The filter's protection has a price. It stepped aside in the {' and '.join(p_['label'] for p_ in s_m)} sell-off"
                       + ("s" if len(s_m) > 1 else "") + ", then sat out most of the rebound. ")
            txt += f"Across each crash-and-rebound pair it finished {rel_word} buy-and-hold ({comp})."
            legs_block += f'    <p class="note">{txt}</p>\n'

    # ---------------- tables ----------------
    wd = ts(last).strftime("%a")
    rows = []
    for j in range(5):
        cd = cond[j]
        today = j == cur
        cells = "".join(f'<td class="heat" data-p="{p:.4f}">{pct(p)}</td>' for p in c["P"][j])
        rows.append(f'<tr{" class=\"today\"" if today else ""}><th scope="row">{labels[j]}{f"<span class=\"chip\">{wd}</span>" if today else ""}'
                    f'<span class="sub">{ranges_long[j]}</span></th>{cells}<td>{pct(cd["p_up"])}</td><td>{pct(cd["mean"], 2, True)}</td>'
                    f'<td>{num((cd["mean"] - um) * 100, 2, True)} pts<span class="sub">t = {num(cd["t_vs_avg"], 1, True)}</span></td></tr>')
    tbl_heat = ('<thead><tr><th scope="col">Today</th>' + "".join(f'<th scope="col">{l}</th>' for l in labels)
                + '<th scope="col">Up next day</th><th scope="col">Avg next day</th><th scope="col">vs average day</th></tr></thead><tbody>'
                + "".join(rows) + '</tbody><tfoot><tr><th scope="row">Any day</th>' + "".join(f"<td>{pct(p)}</td>" for p in c["stationary"])
                + f'<td>{pct(c["uncond_pup"])}</td><td>{pct(um, 2, True)}</td><td>{M}</td></tr></tfoot>')
    tbl_regimes = ('<thead><tr><th scope="col">Regime</th><th scope="col">Average day</th><th scope="col">Annual volatility</th>'
                   '<th scope="col">Typical spell</th><th scope="col">Share of days</th><th scope="col">Now</th><th scope="col">In 5 days</th>'
                   '<th scope="col">In 20 days</th></tr></thead><tbody>'
                   + "".join(f'<tr><th scope="row"><span class="swatch bg-r{j}"></span>{g["name"]}</th><td>{pct(g["mean_ann"] / 252, 2, True)}</td>'
                             f'<td>{pct(g["vol_ann"])}</td><td>{g["duration_days"]:.0f} days</td><td>{pct(g["share"])}</td><td>{pct(now[j])}</td>'
                             f'<td>{pct(n5[j])}</td><td>{pct(n20[j])}</td></tr>' for j, g in enumerate(rg)) + "</tbody>")
    pr = R["prices"]
    px = pd.Series(pr["close"], index=pd.to_datetime(pr["dates"]))
    regs = pd.Series([int(x) for x in R["regime_string"]], index=pd.to_datetime(pr["dates"][1:]))
    yrows = []
    y0, y1 = px.index[0].year, px.index[-1].year
    for yr in range(y0, y1 + 1):
        p_y = px[px.index.year == yr]
        prev = px[px.index.year < yr]
        base = prev.iloc[-1] if len(prev) else p_y.iloc[0]
        rr = regs[regs.index.year == yr]
        note = f" (from {px.index[0].day} {px.index[0].strftime('%b')})" if (yr == y0 and px.index[0].dayofyear > 7) else \
               (f" (to {px.index[-1].day} {px.index[-1].strftime('%b')})" if yr == y1 else "")
        yrows.append(f'<tr><th scope="row">{yr}{note}</th><td>{pct(p_y.iloc[-1] / base - 1, 1, True)}</td>'
                     + "".join(f"<td>{pct((rr == k).mean(), 0) if len(rr) else M}</td>" for k in range(3)) + "</tr>")
    tbl_years = ('<thead><tr><th scope="col">Year</th><th scope="col">' + esc(T) + ' return</th><th scope="col">Calm days</th>'
                 '<th scope="col">Choppy days</th><th scope="col">Stress days</th></tr></thead><tbody>' + "".join(yrows) + "</tbody>")
    names_bt = {"buy_hold": ("Buy &amp; hold", "bg-s1"), "chain": ("Chain rule", "bg-s2"), "hmm": ("Stress filter (HMM)", "bg-s3")}
    tbl_metrics = ('<thead><tr><th scope="col">Strategy</th><th scope="col">Annual return</th><th scope="col">Volatility</th><th scope="col">Sharpe</th>'
                   '<th scope="col">Worst drawdown</th><th scope="col">Time invested</th><th scope="col">Switches</th><th scope="col">Total return</th></tr></thead><tbody>'
                   + "".join(f'<tr><th scope="row"><span class="key-line {cls}" style="margin-right:8px;vertical-align:3px"></span>{nm}</th>'
                             f'<td>{pct(bt[k]["cagr"])}</td><td>{pct(bt[k]["vol"])}</td><td>{num(bt[k]["sharpe"])}</td><td>{pct(bt[k]["maxdd"])}</td>'
                             f'<td>{pct(bt[k]["exposure"], 0)}</td><td>{bt[k]["switches"]}</td><td>{pct(bt[k]["total"], 0, True)}</td></tr>'
                             for k, (nm, cls) in names_bt.items()) + "</tbody>")
    eqd = R["equity"]
    start = eqd["start_index"]
    od = pd.to_datetime(pr["dates"][start + 1:])
    daily = {k: np.array(eqd[k][1:]) / np.array(eqd[k][:-1]) - 1 for k in ("buy_hold", "chain", "hmm")}
    byrows = []
    for yr in sorted(set(od.year)):
        msk = np.asarray(od.year == yr)
        vals = [np.prod(1 + daily[k][msk]) - 1 for k in ("buy_hold", "chain", "hmm")]
        note = f" (from {od[0].day} {od[0].strftime('%b')})" if yr == od[0].year else (f" (to {od[-1].day} {od[-1].strftime('%b')})" if yr == od[-1].year else "")
        byrows.append(f'<tr><th scope="row">{yr}{note}</th>' + "".join(f"<td>{pct(v, 1, True)}</td>" for v in vals) + "</tr>")
    tbl_bt_years = ('<thead><tr><th scope="col">Year</th><th scope="col">Buy &amp; hold</th><th scope="col">Chain rule</th>'
                    '<th scope="col">Stress filter</th></tr></thead><tbody>' + "".join(byrows) + "</tbody>")

    # ---------------- simple fields ----------------
    close = ctx["close"]
    yt = WORDS.get(span_years, str(span_years))
    dek = (f"{yt} years of daily closes run through a Markov chain and a hidden Markov regime model, with a walk-forward backtest. "
           f"Data to the close on {dfull(last)} ({usd(close)}).")
    if a.live_price and a.live_label:
        dek += f" {a.live_label}: {usd(a.live_price)}, {pct(a.live_price / close - 1, 2, True)} on the day."
    tile1 = f"{pct(ctx['r1d'], 2, True)} on the day · " + (f"{pct(-ctx['from_hi'])} below the 52-week high" if ctx["from_hi"] < -0.001 else "at the 52-week high")
    method_data = (f"{esc(a.source)}: {meta['n_prices']:,} sessions from {dlong(meta['first_date'])} to {dlong(last)}"
                   + (", checksum-verified on transfer" if a.verified else "") + "."
                   + (" The current session was still trading and is left out." if a.live_price else ""))
    today = dt.date.today()
    F = dict(
        ticker=esc(T), dek=dek, verdict_lead=lead, verdict_body=verdict_body,
        last_short=f"{wd} {ts(last).day} {ts(last).strftime('%b')}", close=usd(close), tile1_sub=tile1,
        state_label=L, state_article=article(L), r1d=pct(c["last_return"], 2, True),
        band_phrase=(f"the band {ranges_long[cur]}" if cur in (0, 4) else f"the {ranges_long[cur]} band"),
        band_desc=(ranges_long[cur] if cur in (0, 4) else f"of {ranges_long[cur]}"),
        pup_cur_r=pct(cond[cur]["p_up"], 0), pup_all_r=pct(c["uncond_pup"], 0),
        gap_sig="gap not significant" if abs(cond[cur]["t_vs_avg"]) < 1.96 else "gap significant",
        regime_tile=regime_tile, calm_now=pct(now[0], 0), chop_now=pct(now[1], 0), stress_now=pct(now[2]),
        chain_takeaway=chain_takeaway, n_cur=f"{cond[cur]['n']:,}",
        span_txt=f"{ts(meta['first_date']).strftime('%b %Y')} to {ts(last).strftime('%b %Y')}",
        last_dow_long=ts(last).strftime("%A"), tbl_heat=tbl_heat, chain_note=chain_note,
        vol_h2="Where the memory is: volatility" if vol_cluster else "Volatility memory",
        vol_takeaway=(f"Big moves cluster. That is the strongest Markov effect in {esc(T)}, and it is what the regime model below picks up."
                      if vol_cluster else f"Big moves cluster only weakly in {esc(T)}, so the regime model below has less to work with."),
        ext_ext=pct(c["p_extreme_after_extreme"]), ext_calm=pct(c["p_extreme_after_calm"]),
        ac_abs=num(c["ac1_abs"], 2, True), ac_rank=num(ex["ac1_spearman"], 2, True), ac_note=ac_note,
        reg_takeaway=reg_takeaway, tbl_regimes=tbl_regimes, reg_note=reg_note, tbl_years=tbl_years,
        bt_takeaway=bt_takeaway, eq_start_long=dlong(bt["start_close_date"]), n_spells=str(len(ex["hmm_flat_spells"])),
        bt_start=dlong(bt["start"]), bt_end=dlong(bt["end"]), bt_days=f"{bt['days']:,}", tbl_metrics=tbl_metrics,
        legs_block=legs_block, checks="\n".join("      " + x for x in checks), tbl_bt_years=tbl_bt_years,
        method_data=method_data, edges=", ".join(f"{num(x, 2, True)}%" for x in e[:3]) + f" and {num(e[3], 2, True)}%",
        n_pairs=f"{meta['n_returns'] - 1:,}", bic_sentence=bic_sentence,
        warmup_txt="Three-year" if bt["burn"] == 756 else f"{bt['burn']}-session",
        credit_line=(f"{esc(a.credit)} · prepared {today.day} {today.strftime('%B %Y')}." if a.credit else f"Prepared {today.day} {today.strftime('%B %Y')}."),
    )

    # ---------------- chart payload ----------------
    closes = pr["close"]
    eq_all = eqd["buy_hold"] + eqd["chain"] + eqd["hmm"]
    pmax_cur = max(c["P"][cur])
    nd_max = max(0.3, math.ceil((pmax_cur + 0.03) / 0.05) * 0.05)
    payload = dict(
        ticker=T, px=dict(d=pr["dates"], c=closes), reg=R["regime_string"],
        eq=dict(start=start, bh=eqd["buy_hold"], ch=eqd["chain"], hm=eqd["hmm"]),
        cash=[[s_["start"], s_["end"]] for s_ in ex["hmm_flat_spells"]],
        chain=dict(P=c["P"], cur=cur, labels=labels, ranges=ranges_short),
        ui=dict(price=dict(domain=[min(closes) * 0.88, max(closes) * 1.12], ticks=log_ticks(min(closes) * 0.9, max(closes) * 1.1)),
                eq=dict(domain=[min(eq_all) * 0.88, max(eq_all) * 1.12], ticks=log_ticks(min(eq_all) * 0.9, max(eq_all) * 1.1)),
                nd=dict(max=nd_max, ticks=[round(x, 2) for x in np.arange(0, nd_max + 1e-9, 0.1) if abs(x - 0.2) > 1e-9]),
                heat=[math.floor((P_all.min() - 0.01) * 100) / 100, math.ceil((P_all.max() + 0.01) * 100) / 100]),
    )
    assert len(payload["eq"]["bh"]) == len(pr["dates"]) - start, "equity length mismatch"
    assert len(payload["reg"]) == len(pr["dates"]) - 1, "regime string length mismatch"

    page = open(a.template).read()
    for k, v in F.items():
        page = page.replace(f"[[{k}]]", v)
    left = re.findall(r"\[\[[a-z0-9_]+\]\]", page)
    assert not left, f"unfilled placeholders: {left}"
    page = page.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write(page)

    if a.standalone:
        head, body = page.split("<main", 1)
        title = re.search(r"<title>.*?</title>", head).group(0)
        rest = head.replace(title, "").strip()
        icon = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' "
                "fill='%230f1a2b'/%3E%3Cpath d='M6 22 L12 16 L17 19 L26 9' fill='none' stroke='%23a99ff0' stroke-width='3' "
                "stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E")
        doc = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
               f'<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n{title}\n'
               f'<meta name="description" content="Markov chain and hidden Markov regime analysis of {esc(T)} daily returns with a walk-forward backtest.">\n'
               f'<link rel="icon" href="{icon}">\n'
               '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
               'img{max-width:100%}[hidden]{display:none!important}</style>\n'
               f'{rest}\n</head>\n<body>\n<main{body.rstrip()}\n</body>\n</html>\n')
        os.makedirs(os.path.dirname(os.path.abspath(a.standalone)), exist_ok=True)
        open(a.standalone, "w").write(doc)

    facts = dict(ticker=T, last_date=last, close=close, verdict_lead=lead, verdict_body=verdict_body,
                 chain_state=L, p_up_next=cond[cur]["p_up"], p_up_any=c["uncond_pup"], p_direction=p5,
                 regime_now={"calm": now[0], "choppy": now[1], "stress": now[2]}, rv20=rv, rv20_median=rvm,
                 backtest={k: {m_: bt[k][m_] for m_ in ("cagr", "sharpe", "maxdd", "exposure")} for k in ("buy_hold", "chain", "hmm")},
                 sharpe_diff_ci=boot, live_signal=R.get("live_signal"))
    with open(os.path.join(os.path.dirname(os.path.abspath(a.out)), "facts.json"), "w") as fh:
        json.dump(facts, fh, indent=1, default=str)
    print(f"wrote {a.out} ({len(page) // 1024} KB)" + (f" and {a.standalone}" if a.standalone else ""))
    print("VERDICT:", lead)
    print(verdict_body)


if __name__ == "__main__":
    main()
