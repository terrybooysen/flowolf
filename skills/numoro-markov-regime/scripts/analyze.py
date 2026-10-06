#!/usr/bin/env python3
# Flowolf — powered by Numoro
"""Markov regime analysis of one ticker: discrete Markov chain on daily returns, 3-state Gaussian HMM,
walk-forward backtest, robustness checks. Writes results.json and arrays.npz.

Usage:
  python analyze.py --csv prices.csv --ticker AAPL [--outdir .]

prices.csv needs a date column (YYYY-MM-DD) and an adjusted close column (adj_close / Adj Close / close).
Needs: numpy pandas scipy hmmlearn   (pip install --break-system-packages hmmlearn scipy)
Runtime: about a minute for ten years of data (most of it is the quarterly HMM refits).
"""
import argparse
import json
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    sys.exit("hmmlearn missing: pip install --break-system-packages hmmlearn")

COST = 0.0005            # 5 bp per unit change in position
LABELS = {2: ["Down", "Up"], 3: ["Down", "Flat", "Up"], 5: ["Big down", "Down", "Flat", "Up", "Big up"]}
REG_NAMES = ["Calm", "Choppy", "Stress"]
# Market-wide episodes; each is reported only when it sits inside the backtest window.
EPISODES = [
    ("COVID crash", "2020-02-20", "2020-03-23", "covid"),
    ("COVID rebound", "2020-03-24", "2020-06-08", "covid"),
    ("2022 bear market", "2022-01-03", "2022-12-30", None),
    ("Tariff shock", "2025-04-03", "2025-04-08", "tariff"),
    ("Tariff rebound", "2025-04-09", "2025-04-30", "tariff"),
]


def load_prices(path):
    df = pd.read_csv(path)
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    dcol = cols.get("date") or df.columns[0]
    pcol = next((cols[k] for k in ("adj_close", "adjclose", "adj._close", "close") if k in cols), None)
    if pcol is None:
        sys.exit(f"no adjusted close column in {path}: {list(df.columns)}")
    out = pd.DataFrame({"date": pd.to_datetime(df[dcol]).dt.strftime("%Y-%m-%d"), "px": pd.to_numeric(df[pcol], errors="coerce")})
    out = out.dropna().query("px > 0").drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return out


def chain(Rv, k):
    edges = np.quantile(Rv, np.linspace(0, 1, k + 1)[1:-1])
    s = np.digitize(Rv, edges)
    C = np.zeros((k, k), int)
    np.add.at(C, (s[:-1], s[1:]), 1)
    return edges, s, C, C / np.maximum(C.sum(1, keepdims=True), 1)


def stationary(P):
    w, v = np.linalg.eig(P.T)
    i = np.argmin(np.abs(w - 1))
    pi = np.real(v[:, i])
    return pi / pi.sum(), float(sorted(np.abs(w))[-2])


def fit_hmm(Xf, k, seeds=range(8), n_iter=300):
    best, bestll = None, -np.inf
    for sd in seeds:
        m = GaussianHMM(n_components=k, covariance_type="diag", n_iter=n_iter, tol=1e-5, random_state=sd)
        try:
            m.fit(Xf)
            ll = m.score(Xf)
        except Exception:
            continue
        if np.isfinite(ll) and ll > bestll:
            best, bestll = m, ll
    return best, bestll


def params(m):
    k = m.n_components
    return m.means_.ravel(), m.covars_.reshape(k, -1)[:, 0], m.transmat_, m.startprob_


def forward_filter(m, Xf):
    """Filtered (causal) state probabilities: each row uses data up to that day only."""
    mu, var, A, p0 = params(m)
    x = Xf.ravel()
    ll = -0.5 * np.log(2 * np.pi * var)[None, :] - 0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :]
    lik = np.exp(ll - ll.max(1, keepdims=True))
    f = np.empty((len(x), len(mu)))
    a = p0 * lik[0]
    a /= a.sum()
    f[0] = a
    for t in range(1, len(x)):
        a = (a @ A) * lik[t]
        a /= a.sum()
        f[t] = a
    return f


def sharpe(x):
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = load_prices(a.csv)
    pdates, px = df["date"].values, df["px"].values
    if len(px) < 600:
        sys.exit(f"only {len(px)} prices; need at least ~600 trading days (2.5 years) for a meaningful read")
    R = px[1:] / px[:-1] - 1
    LR = np.log(px[1:] / px[:-1])
    rdates = pdates[1:]
    n = len(R)
    out = {"meta": {"ticker": a.ticker.upper(), "first_date": pdates[0], "last_date": pdates[-1],
                    "n_prices": int(len(px)), "n_returns": int(n)}}

    # ---------------- A. discrete Markov chain on return bands ----------------
    robust = []
    for k in (2, 3, 5):
        edges, s, C, P = chain(R, k)
        up_next = R[1:] > 0
        T = np.array([[np.sum(up_next[s[:-1] == j]), np.sum(~up_next[s[:-1] == j])] for j in range(k)])
        _, p_dir, _, _ = stats.chi2_contingency(T)
        _, p_all, _, _ = stats.chi2_contingency(C)
        pup = T[:, 0] / np.maximum(T.sum(1), 1)
        robust.append(dict(k=k, p_direction=float(p_dir), p_any=float(p_all), pup_by_state=[float(x) for x in pup],
                           spread_pp=float((pup.max() - pup.min()) * 100)))
    out["robustness"] = robust

    K = 5
    edges, s, C, P = chain(R, K)
    pi, slem = stationary(P)
    cur = int(s[-1])
    um = float(R.mean())
    cond = []
    for j in range(K):
        nxt = R[1:][s[:-1] == j]
        se = float(nxt.std(ddof=1) / np.sqrt(len(nxt)))
        cond.append(dict(state=LABELS[K][j], n=int(len(nxt)), p_up=float((nxt > 0).mean()), mean=float(nxt.mean()), se=se,
                         t_vs_avg=float((nxt.mean() - um) / se)))
    m_vec = np.array([c["mean"] for c in cond])
    v = np.eye(K)[cur]
    exp_path, cum = [], 0.0
    for h in range(1, 21):
        cum += float(v @ m_vec)
        exp_path.append(cum)
        v = v @ P
    extreme = np.isin(s, [0, K - 1])
    out["chain"] = dict(
        labels=LABELS[K], edges_pct=[float(e * 100) for e in edges], counts=C.tolist(), P=P.tolist(),
        stationary=[float(x) for x in pi], slem=slem, memory_halflife_days=float(np.log(0.5) / np.log(max(slem, 1e-9))),
        current_state=cur, current_label=LABELS[K][cur], last_return=float(R[-1]), last_date=rdates[-1],
        conditional=cond, uncond_mean=um, uncond_pup=float((R > 0).mean()),
        exp_cum_path=exp_path, uncond_cum_path=[um * h for h in range(1, 21)],
        p_extreme_after_extreme=float(extreme[1:][extreme[:-1]].mean()),
        p_extreme_after_calm=float(extreme[1:][~extreme[:-1]].mean()),
        ac1=float(np.corrcoef(R[:-1], R[1:])[0, 1]), ac1_abs=float(np.corrcoef(np.abs(R[:-1]), np.abs(R[1:]))[0, 1]),
    )

    # ---------------- B. hidden Markov regimes ----------------
    X = (LR * 100).reshape(-1, 1)
    bic, models = [], {}
    for k in (2, 3, 4):
        m, ll = fit_hmm(X, k)
        npar = (k - 1) + k * (k - 1) + 2 * k
        bic.append(dict(k=k, loglik=float(ll), bic=float(-2 * ll + npar * np.log(len(X)))))
        models[k] = m
    out["hmm_bic"] = bic
    m = models[3]
    mu, var, A, p0 = params(m)
    order = np.argsort(var)
    mu, var, A = mu[order], var[order], A[np.ix_(order, order)]
    smooth = m.predict_proba(X)[:, order]
    filt = forward_filter(m, X)[:, order]
    reg = []
    lab = smooth.argmax(1)
    for j in range(3):
        days = lab == j
        reg.append(dict(name=REG_NAMES[j], mean_ann=float(mu[j] * 252 / 100), vol_ann=float(np.sqrt(var[j]) * np.sqrt(252) / 100),
                        stay=float(A[j, j]), duration_days=float(1 / max(1 - A[j, j], 1e-9)), share=float(days.mean()),
                        realized_ann=float(LR[days].mean() * 252) if days.any() else None))
    f_now = filt[-1]
    out["hmm"] = dict(names=REG_NAMES, regimes=reg, A=A.tolist(), now=[float(x) for x in f_now],
                      next5=[float(x) for x in f_now @ np.linalg.matrix_power(A, 5)],
                      next20=[float(x) for x in f_now @ np.linalg.matrix_power(A, 20)])
    out["regime_string"] = "".join(str(int(x)) for x in lab)          # hindsight label per return day

    # ---------------- C. walk-forward backtest ----------------
    BURN = 756 if n >= 1500 else max(252, n // 3)
    pos1 = np.zeros(n)
    live1 = None
    for t in range(BURN - 1, n):
        hist = R[: t + 1]
        e = np.quantile(hist, [0.2, 0.4, 0.6, 0.8])
        sh = np.digitize(hist, e)
        st = sh[-1]
        nxt = hist[1:][sh[:-1] == st]
        cm = nxt.mean() if len(nxt) else 0.0
        if t + 1 < n:
            pos1[t + 1] = 1.0 if cm > 0 else 0.0
        else:
            live1 = dict(state=LABELS[5][int(st)], cond_mean=float(cm), position="long" if cm > 0 else "cash")
    pos2 = np.zeros(n)
    pstress = np.full(n, np.nan)
    live2 = None
    REFIT = 63
    t = BURN - 1
    while t <= n - 1:
        mm, _ = fit_hmm(X[: t + 1], 3, seeds=range(4), n_iter=200)
        stress = int(np.argmax(params(mm)[1]))
        end = min(t + REFIT, n)
        f = forward_filter(mm, X[:end])
        for u in range(t, end):
            ps = float(f[u, stress])
            pstress[u] = ps
            if u + 1 < n:
                pos2[u + 1] = 0.0 if ps > 0.5 else 1.0
            else:
                live2 = dict(p_stress=ps, position="cash" if ps > 0.5 else "long")
        t += REFIT

    Ro = R[BURN:]
    od = rdates[BURN:]

    def run(pos):
        p = pos[BURN:]
        return p * Ro - COST * np.abs(np.diff(np.r_[0.0, p])), p

    def metrics(ret, p):
        eq = np.cumprod(1 + ret)
        peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
        return dict(cagr=float(eq[-1] ** (252 / len(ret)) - 1), vol=float(ret.std(ddof=1) * np.sqrt(252)), sharpe=sharpe(ret),
                    maxdd=float((eq / peak - 1).min()), total=float(eq[-1] - 1), exposure=float(p.mean()),
                    switches=int(np.sum(np.abs(np.diff(p)) > 0)))

    rng = np.random.default_rng(7)

    def shift_pct(p, actual):
        L = len(p)
        sims = []
        for k in rng.integers(21, L - 21, 2000):
            q = np.roll(p, k)
            sims.append(sharpe(q * Ro - COST * np.abs(np.diff(np.r_[0.0, q]))))
        return float((np.array(sims) < actual).mean())

    r1, p1 = run(pos1)
    r2, p2 = run(pos2)
    bt = {"buy_hold": metrics(Ro.copy(), np.ones_like(Ro)), "chain": metrics(r1, p1), "hmm": metrics(r2, p2)}
    for key, (rr, pp) in {"chain": (r1, p1), "hmm": (r2, p2)}.items():
        bt[key]["shift_percentile"] = shift_pct(pp, sharpe(rr)) if 0 < pp.mean() < 1 else None
    bt["chain"]["hit_rate"] = float(((p1 == 1) == (Ro > 0)).mean())
    bt["always_up_hit_rate"] = float((Ro > 0).mean())
    bt["start"], bt["end"], bt["days"], bt["burn"] = od[0], od[-1], int(len(Ro)), int(BURN)
    bt["start_close_date"] = pdates[BURN]   # equity curve starts at this close (= $1)

    # ---------------- D. robustness ----------------
    ex = {}
    a_, b_ = R[:-1], R[1:]
    keep = (np.abs(a_) < 0.05) & (np.abs(b_) < 0.05)
    ex["ac1_pearson"] = float(np.corrcoef(a_, b_)[0, 1])
    ex["ac1_spearman"] = float(stats.spearmanr(a_, b_).correlation)
    ex["ac1_trimmed"] = float(np.corrcoef(a_[keep], b_[keep])[0, 1]) if keep.sum() > 30 else None
    ex["n_trimmed_pairs"] = int((~keep).sum())
    half = n // 2
    um1, um2 = float(R[1:half].mean()), float(R[half + 1:].mean())
    split = []
    for j in range(5):
        m1 = R[1:half][s[: half - 1] == j]
        m2 = R[half + 1:][s[half: -1] == j]
        split.append(dict(state=LABELS[5][j], first_mean=float(m1.mean()), second_mean=float(m2.mean()),
                          first_vs_avg=float(m1.mean() - um1), second_vs_avg=float(m2.mean() - um2)))
    ex["split"] = split
    ex["split_dates"] = [str(rdates[0]), str(rdates[half - 1]), str(rdates[half]), str(rdates[-1])]
    brng = np.random.default_rng(11)
    L = len(Ro)

    def block_idx():
        idx, i = np.empty(L, int), 0
        while i < L:
            start, blen = brng.integers(0, L), brng.geometric(1 / 20)
            for k in range(blen):
                if i >= L:
                    break
                idx[i] = (start + k) % L
                i += 1
        return idx

    boot = {}
    for key, rr in (("chain", r1), ("hmm", r2)):
        diffs = []
        for _ in range(2000):
            ix = block_idx()
            diffs.append(sharpe(rr[ix]) - sharpe(Ro[ix]))
        diffs = np.array(diffs)
        boot[key] = dict(point=sharpe(rr) - sharpe(Ro), lo=float(np.quantile(diffs, 0.025)), hi=float(np.quantile(diffs, 0.975)))
    ex["sharpe_diff_bootstrap"] = boot
    sens = []
    for thr in (0.3, 0.5, 0.7, 0.9):
        p = np.zeros(n)
        p[BURN:] = np.where(pstress[BURN - 1:n - 1] > thr, 0.0, 1.0)
        rr, pp = run(p)
        sens.append(dict(threshold=thr, **{k: v for k, v in metrics(rr, pp).items() if k in ("cagr", "sharpe", "maxdd", "exposure")}))
    ex["hmm_threshold_sensitivity"] = sens
    legs = []
    for name, d0, d1, pair in EPISODES:
        if d0 < od[0] or d1 > od[-1]:
            continue
        i0, i1 = np.searchsorted(od, d0), np.searchsorted(od, d1, side="right")
        if i1 - i0 < 3:
            continue
        legs.append(dict(name=name, start=d0, end=d1, pair=pair,
                         **{k: float(np.prod(1 + x[i0:i1]) - 1) for k, x in (("buy_hold", Ro), ("chain", r1), ("hmm", r2))}))
    ex["legs"] = legs
    spells, i, p = [], 0, pos2[BURN:]
    while i < len(p):
        if p[i] == 0:
            j = i
            while j + 1 < len(p) and p[j + 1] == 0:
                j += 1
            spells.append(dict(start=str(od[i]), end=str(od[j]), days=int(j - i + 1), bh_return=float(np.prod(1 + Ro[i:j + 1]) - 1)))
            i = j + 1
        else:
            i += 1
    ex["hmm_flat_spells"] = spells
    out["extras"] = ex
    out["backtest"] = bt
    out["live_signal"] = dict(chain=live1, hmm=live2, for_session_after=rdates[-1])

    # ---------------- E. context ----------------
    prev_year = str(int(pdates[-1][:4]) - 1)
    ytd_idx = np.where(pdates <= f"{prev_year}-12-31")[0]
    rv20 = pd.Series(R).rolling(20).std() * np.sqrt(252)
    look = min(252, len(px) - 1)
    out["context"] = dict(close=float(px[-1]), r1d=float(R[-1]),
                          r21d=float(px[-1] / px[-22] - 1), r252d=float(px[-1] / px[-1 - look] - 1),
                          ytd=float(px[-1] / px[ytd_idx[-1]] - 1) if len(ytd_idx) else None,
                          rv20=float(rv20.iloc[-1]), rv20_median=float(rv20.median()),
                          hi52=float(px[-look:].max()), from_hi=float(px[-1] / px[-look:].max() - 1))
    out["prices"] = dict(dates=list(pdates), close=[round(float(x), 4) for x in px])
    eq = lambda x: [1.0] + [round(float(v), 5) for v in np.cumprod(1 + x)]
    out["equity"] = dict(start_index=int(BURN), buy_hold=eq(Ro), chain=eq(r1), hmm=eq(r2))

    with open(os.path.join(a.outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    np.savez(os.path.join(a.outdir, "arrays.npz"), R=R, rdates=rdates, pos1=pos1, pos2=pos2, pstress=pstress, burn=BURN)

    c = out["chain"]
    print(f"{out['meta']['ticker']}: {n + 1} prices {pdates[0]} -> {pdates[-1]}")
    print(f"chain: state {c['current_label']} ({c['last_return']*100:+.2f}%), P(up next) {cond[cur]['p_up']:.3f} vs {c['uncond_pup']:.3f}, "
          f"direction p(5/3/2) = {robust[2]['p_direction']:.3f}/{robust[1]['p_direction']:.3f}/{robust[0]['p_direction']:.3f}")
    print("hmm now (calm/choppy/stress):", [round(x, 3) for x in f_now], "| BIC best k =", min(bic, key=lambda b: b["bic"])["k"])
    for key in ("buy_hold", "chain", "hmm"):
        b = bt[key]
        print(f"bt {key:>8}: CAGR {b['cagr']*100:.1f}% Sharpe {b['sharpe']:.2f} MaxDD {b['maxdd']*100:.1f}% exposure {b['exposure']*100:.0f}%")
    print("live:", out["live_signal"])


if __name__ == "__main__":
    main()
