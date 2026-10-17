import numpy as np
import pandas as pd
import pytest

from flowolf_quant import backtest, data
from flowolf_quant.selection import (lowvol_scores, lowvol_weights, momentum_scores,
                                     xsmom_weights)


def test_momentum_scores_window():
    P = data.random_walk_panel(n=400, k=3, seed=0)
    s = momentum_scores(P, lookback=252, skip=21)
    t = 300
    expected = (1 + P.iloc[t - 251:t - 20]).prod() - 1   # rows t-251 .. t-21
    np.testing.assert_allclose(s.iloc[t].to_numpy(), expected.to_numpy(), rtol=1e-10)
    assert s.iloc[:252].isna().all().all()


def test_momentum_no_lookahead():
    P = data.random_walk_panel(n=600, k=10, seed=1)
    W = xsmom_weights(P)
    t = 252 + 21 * 8
    W2 = xsmom_weights(P.iloc[:t + 1])
    np.testing.assert_allclose(W2.iloc[-1].to_numpy(), W.iloc[t].to_numpy())


def test_xsmom_planted_panel_sharpe():
    P = data.planted_momentum_panel()
    W = xsmom_weights(P)
    assert backtest.run(P, W).metrics["sharpe"] > 0.5


def test_xsmom_random_walk_near_zero():
    sharpes = []
    for seed in range(10):
        P = data.random_walk_panel(seed=seed)
        sharpes.append(backtest.run(P, xsmom_weights(P)).metrics["sharpe"])
    assert abs(np.mean(sharpes)) < 0.3


def test_xsmom_weight_sums():
    P = data.planted_momentum_panel(n=800)
    ls = xsmom_weights(P).iloc[252:]
    np.testing.assert_allclose(ls.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(ls.abs().sum(axis=1), 2.0)
    lo = xsmom_weights(P, long_only=True).iloc[252:]
    np.testing.assert_allclose(lo.sum(axis=1), 1.0)
    assert (lo >= 0).all().all()
    assert ((lo > 0).sum(axis=1) == 4).all()          # 20% of 20


def _vol_ladder(n=800, seed=0):
    rng = np.random.default_rng(seed)
    vols = np.linspace(0.005, 0.03, 10)
    X = rng.normal(0, 1, (n, 10)) * vols
    cols = [f"V{i}" for i in range(10)]
    return pd.DataFrame(X, index=data.business_days(n), columns=cols)


def test_lowvol_picks_low_vol_assets():
    P = _vol_ladder()
    W = lowvol_weights(P, lookback=252, top_q=0.3)
    live = W.iloc[251:]
    np.testing.assert_allclose(live.sum(axis=1), 1.0)
    assert (live >= 0).all().all()
    held = set(live.columns[live.iloc[-1] > 0])
    assert held == {"V0", "V1", "V2"}
    assert (W.iloc[:251] == 0).all().all()


def test_lowvol_scores_no_lookahead():
    P = _vol_ladder(n=500)
    s = lowvol_scores(P)
    t = 400
    np.testing.assert_allclose(lowvol_scores(P.iloc[:t + 1]).iloc[-1].to_numpy(),
                               s.iloc[t].to_numpy())
