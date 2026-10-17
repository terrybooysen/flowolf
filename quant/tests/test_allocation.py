import numpy as np
import pandas as pd
import pytest

from flowolf_quant import TRADING_DAYS, backtest, data
from flowolf_quant.allocation import (apply_vol_target, equal_weights, hrp_weights,
                                      inverse_vol_weights, kelly_fraction, rolling_hrp,
                                      vol_target_scale)


# ---------------------------------------------------------------- vol targeting

@pytest.mark.parametrize("method", ["ewma", "rolling"])
def test_vol_target_hits_target_on_random_walk(method):
    r = data.random_walk(n=2520, vol=0.02, seed=1)
    scale = vol_target_scale(r, target=0.10, lookback=60, method=method, max_leverage=5.0)
    res = backtest.run(r, scale, cost_bps=0.0)
    live = res.returns.iloc[61:]
    realised = live.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert 0.075 <= realised <= 0.125


def test_vol_target_with_changing_vol():
    r, _ = data.regime_switching(seed=3)
    scale = vol_target_scale(r, target=0.10, lookback=40, method="ewma", max_leverage=5.0)
    out = backtest.run(r, scale, cost_bps=0.0).returns.iloc[41:]
    realised = out.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert 0.075 <= realised <= 0.125


@pytest.mark.parametrize("method", ["ewma", "rolling"])
def test_vol_target_no_lookahead(method):
    r = data.random_walk(n=800, vol=0.015, seed=2)
    full = vol_target_scale(r, method=method)
    for t in [100, 333, 650]:
        trunc = vol_target_scale(r.iloc[:t + 1], method=method)
        assert trunc.iloc[-1] == pytest.approx(full.iloc[t], rel=1e-12, abs=1e-12)


def test_vol_target_caps_and_warmup():
    r = data.random_walk(n=300, vol=0.001, seed=0)          # tiny vol -> wants huge leverage
    s = vol_target_scale(r, target=0.10, lookback=60, max_leverage=2.0)
    assert (s.iloc[:59] == 0).all()
    assert s.max() <= 2.0 + 1e-12


def test_apply_vol_target_portfolio():
    P = data.random_walk_panel(n=1500, k=5, vol=0.02, seed=4)
    W = pd.DataFrame(0.2, index=P.index, columns=P.columns)
    Ws = apply_vol_target(W, P, target=0.10, lookback=60, max_leverage=5.0)
    out = backtest.run(P, Ws, cost_bps=0.0).returns.iloc[61:]
    realised = out.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert 0.075 <= realised <= 0.125
    # no look-ahead
    t = 700
    trunc = apply_vol_target(W.iloc[:t + 1], P.iloc[:t + 1], target=0.10, max_leverage=5.0)
    np.testing.assert_allclose(trunc.iloc[-1].to_numpy(), Ws.iloc[t].to_numpy())


# ---------------------------------------------------------------- HRP

def _two_cluster_panel(n=1000, sizes=(6, 3), rho=0.8, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    cols, blocks = [], []
    for c, k in enumerate(sizes):
        f = rng.normal(0, 1, (n, 1))
        e = rng.normal(0, 1, (n, k))
        blocks.append(vol * (np.sqrt(rho) * f + np.sqrt(1 - rho) * e))
        cols += [f"C{c}_{i}" for i in range(k)]
    return pd.DataFrame(np.hstack(blocks), index=data.business_days(n), columns=cols)


def test_hrp_sums_to_one_long_only():
    P = data.random_walk_panel(n=500, k=12, seed=5)
    w = hrp_weights(P)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    for fn in (inverse_vol_weights, equal_weights):
        b = fn(P)
        assert b.sum() == pytest.approx(1.0) and (b >= 0).all()


@pytest.mark.parametrize("sizes", [(6, 3), (4, 4), (2, 6), (8, 2)])
def test_hrp_balances_clusters(sizes):
    P = _two_cluster_panel(sizes=sizes, seed=sum(sizes))
    w = hrp_weights(P)                      # tree split (default)
    c0 = w[[c for c in w.index if c.startswith("C0")]].sum()
    c1 = w[[c for c in w.index if c.startswith("C1")]].sum()
    assert c0 + c1 == pytest.approx(1.0)
    assert 0.3 <= c0 <= 0.7 and 0.3 <= c1 <= 0.7


@pytest.mark.parametrize("sizes", [(6, 3), (4, 4), (2, 6)])
def test_hrp_paper_bisection_balances_moderate_clusters(sizes):
    P = _two_cluster_panel(sizes=sizes, seed=sum(sizes))
    w = hrp_weights(P, split="bisect")
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
    c0 = w[[c for c in w.index if c.startswith("C0")]].sum()
    assert 0.3 <= c0 <= 0.7


def test_hrp_split_modes_agree_on_balanced_tree():
    # Equal-size clusters: the top split is the same in both modes, so the
    # cluster totals agree (within-cluster splits can differ).
    P = _two_cluster_panel(sizes=(4, 4), seed=11)
    c0 = [c for c in P.columns if c.startswith("C0")]
    assert hrp_weights(P, split="tree")[c0].sum() == pytest.approx(
        hrp_weights(P, split="bisect")[c0].sum(), abs=1e-9)


@pytest.mark.parametrize("split", ["tree", "bisect"])
def test_hrp_favours_low_vol_both_splits(split):
    rng = np.random.default_rng(7)
    vols = np.array([0.005, 0.01, 0.02, 0.04])
    X = rng.normal(0, 1, (1000, 4)) * vols
    P = pd.DataFrame(X, index=data.business_days(1000), columns=list("abcd"))
    w = hrp_weights(P, split=split)
    assert w["a"] > w["b"] > w["c"] > w["d"]


def test_hrp_favours_low_vol():
    rng = np.random.default_rng(7)
    vols = np.array([0.005, 0.01, 0.02, 0.04])
    X = rng.normal(0, 1, (1000, 4)) * vols
    P = pd.DataFrame(X, index=data.business_days(1000), columns=list("abcd"))
    w = hrp_weights(P)
    assert w["a"] > w["b"] > w["c"] > w["d"]
    assert w["a"] > 0.5


def test_hrp_handles_missing_and_constant_assets():
    P = data.random_walk_panel(n=300, k=5, seed=1)
    P["A00"] = np.nan
    P["A01"] = 0.0
    w = hrp_weights(P)
    assert w["A00"] == 0 and w["A01"] == 0 and w.sum() == pytest.approx(1.0)


def test_rolling_hrp_shape_and_no_lookahead():
    P = data.random_walk_panel(n=700, k=8, seed=9)
    W = rolling_hrp(P, window=252, rebalance_every=21)
    assert (W.iloc[:251] == 0).all().all()
    live = W.iloc[251:]
    np.testing.assert_allclose(live.sum(axis=1), 1.0)
    t = 251 + 21 * 10
    W2 = rolling_hrp(P.iloc[:t + 1], window=252, rebalance_every=21)
    np.testing.assert_allclose(W2.iloc[-1].to_numpy(), W.iloc[t].to_numpy())
    # held constant between rebalances
    np.testing.assert_allclose(W.iloc[t + 1].to_numpy(), W.iloc[t].to_numpy())


# ---------------------------------------------------------------- Kelly

def test_kelly_refuses_on_random_walk():
    for seed in range(5):
        r = data.random_walk(n=2520, vol=0.01, seed=seed)
        res = kelly_fraction(r)
        if res.fraction is None:
            assert "evidence" in res.reason
    # seed 0 specifically must refuse
    res = kelly_fraction(data.random_walk(n=2520, seed=0))
    assert res.fraction is None and "evidence" in res.reason


def test_kelly_refuses_on_short_sample():
    r = data.random_walk(n=100, drift=0.005, seed=0)
    res = kelly_fraction(r, min_obs=252)
    assert res.fraction is None and "not enough data" in res.reason


def test_kelly_sizes_planted_drift():
    r = data.random_walk(n=2520, vol=0.01, drift=0.001, seed=0)
    res = kelly_fraction(r, fraction=0.5, max_leverage=1.0)
    assert res.fraction is not None
    assert 0 < res.fraction <= 1.0
    assert res.t_stat > 2
    # uncapped value is fraction * mu / var
    res2 = kelly_fraction(r, fraction=0.5, max_leverage=100.0)
    assert res2.fraction == pytest.approx(0.5 * res2.mean / res2.var)


def test_kelly_false_positive_rate_bounded():
    hits = sum(kelly_fraction(data.random_walk(n=500, seed=s)).fraction is not None
               for s in range(200))
    assert hits / 200 < 0.10      # nominal 5%
