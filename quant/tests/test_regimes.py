import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from flowolf_quant import backtest, data
from flowolf_quant.regimes import (bocpd, detect_changepoints, fit_garch, forecast_vol,
                                   simulate_garch, trend_regime, tsmom_signal,
                                   tsmom_weights, vol_regimes, walk_forward_vol_regimes)


# --------------------------------------------------------------------- GARCH

@pytest.mark.parametrize("seed", [0, 1])
def test_garch_recovers_parameters(seed):
    r = simulate_garch(5000, omega=2e-6, alpha=0.08, beta=0.90, seed=seed)
    f = fit_garch(r)
    assert abs(f.alpha - 0.08) < 0.05
    assert abs(f.beta - 0.90) < 0.05
    assert f.omega > 0 and f.alpha + f.beta < 1
    assert (f.conditional_vol > 0).all() and (f.conditional_vol_daily > 0).all()
    assert np.allclose(f.conditional_vol, f.conditional_vol_daily * np.sqrt(252))
    assert np.isfinite(f.loglik)


def test_forecast_vol_reverts_to_long_run():
    f = fit_garch(simulate_garch(3000, seed=3))
    fc = forecast_vol(f, horizon=2000)
    assert (fc > 0).all()
    assert fc.iloc[0] == pytest.approx(f.next_vol_daily.iloc[-1])
    assert fc.iloc[-1] == pytest.approx(f.long_run_vol_daily, rel=0.02)


def test_vol_regimes_track_true_state():
    r, state = data.regime_switching()
    lab = vol_regimes(r)
    m = lab.notna()
    assert set(lab[m].unique()) <= {0.0, 1.0, 2.0}
    assert (lab[m] == state[m]).mean() > 0.5
    assert spearmanr(lab[m], state[m])[0] > 0.5


def test_walk_forward_vol_regimes_track_true_state():
    r, state = data.regime_switching()
    lab = walk_forward_vol_regimes(r, train_days=756, refit_every=63)
    assert lab.iloc[:756].isna().all()
    m = lab.notna()
    assert m.sum() > 1500
    assert (lab[m] == state[m]).mean() > 0.5
    assert spearmanr(lab[m], state[m])[0] > 0.5


def test_walk_forward_vol_regimes_no_lookahead():
    r, _ = data.regime_switching(n=1200, seed=4)
    kw = dict(train_days=500, refit_every=100)
    full = walk_forward_vol_regimes(r, **kw)
    for t in (520, 650, 899, 1000):
        trunc = walk_forward_vol_regimes(r.iloc[:t + 1], **kw)
        assert trunc.iloc[-1] == full.iloc[t]


# ---------------------------------------------------------------------- TSMOM

def test_tsmom_no_lookahead():
    r = data.planted_trend(n=800, seed=2)
    sig = tsmom_signal(r, lookback=60, skip=5)
    w = tsmom_weights(r, lookback=60, vol_lookback=30)
    tr = trend_regime(r, lookbacks=(10, 40, 120))
    for t in (5, 70, 150, 400, 799):
        cut = r.iloc[:t + 1]
        assert tsmom_signal(cut, lookback=60, skip=5).iloc[-1] == sig.iloc[t]
        assert tsmom_weights(cut, lookback=60, vol_lookback=30).iloc[-1] == pytest.approx(w.iloc[t])
        assert trend_regime(cut, lookbacks=(10, 40, 120)).iloc[-1] == tr.iloc[t]


def test_tsmom_signal_values_and_leverage_cap():
    r = data.random_walk(n=600, seed=1)
    sig = tsmom_signal(r, lookback=50)
    assert set(sig.unique()) <= {-1.0, 0.0, 1.0}
    assert (sig.iloc[:49] == 0).all()
    w = tsmom_weights(r, lookback=50, max_leverage=1.5, vol_target=0.5)
    assert w.abs().max() <= 1.5 + 1e-12
    assert (tsmom_weights(r, lookback=50, long_only=True) >= 0).all()
    tr = trend_regime(r)
    assert tr.between(-1, 1).all()


def test_tsmom_finds_planted_trend():
    r = data.planted_trend(strength=0.15, lookback=20)
    res = backtest.run(r, tsmom_weights(r, lookback=20))
    assert res.metrics["sharpe"] > 1.0


def test_tsmom_random_walk_has_no_edge():
    sharpes = []
    for seed in range(20):
        r = data.random_walk(seed=seed)
        sharpes.append(backtest.run(r, tsmom_weights(r, lookback=20)).metrics["sharpe"])
    assert abs(np.mean(sharpes)) < 0.3


# ----------------------------------------------------------------------- BOCPD

def _shift_series(seed):
    rng = np.random.default_rng(seed)
    y = np.concatenate([rng.normal(0.0, 0.01, 600),     # baseline
                        rng.normal(0.008, 0.01, 400),   # mean shift at 600
                        rng.normal(0.0, 0.03, 500),     # vol shift at 1000
                        rng.normal(0.0, 0.01, 500)])    # back to calm at 1500
    return pd.Series(y, index=data.business_days(len(y)))


def test_bocpd_finds_known_changes():
    y = _shift_series(2)
    found = detect_changepoints(y, threshold=0.5, min_gap=20)
    pos = [y.index.get_loc(d) for d in found]
    for true in (600, 1000, 1500):
        assert any(abs(p - true) <= 20 for p in pos), (true, pos)
    assert len(pos) <= 6


def test_bocpd_recall_across_seeds():
    hits = 0
    for seed in range(5):
        y = _shift_series(seed)
        pos = [y.index.get_loc(d) for d in detect_changepoints(y)]
        hits += sum(any(abs(p - t) <= 20 for p in pos) for t in (600, 1000, 1500))
    assert hits >= 13            # of 15


def test_bocpd_few_false_alarms_on_noise():
    counts = [len(detect_changepoints(data.random_walk(seed=s))) for s in range(10)]
    assert np.mean(counts) <= 4          # ~10 years of daily data each
    strict = [len(detect_changepoints(data.random_walk(seed=s), threshold=0.8))
              for s in range(10)]
    assert np.mean(strict) <= 1


def test_bocpd_outputs_and_no_lookahead():
    y = _shift_series(0)
    res = bocpd(y)
    assert res.cp_prob.between(0, 1).all()
    assert res.recent_cp_prob.between(0, 1 + 1e-9).all()
    assert (res.map_run_length >= 1).all()
    assert res.map_run_length.iloc[590] > 500         # long run before the shift
    assert res.map_run_length.iloc[1100] < 150        # reset after the shift
    t = 1010
    cut = bocpd(y.iloc[:t + 1])
    assert cut.cp_prob.iloc[-1] == pytest.approx(res.cp_prob.iloc[t])
    assert cut.map_run_length.iloc[-1] == res.map_run_length.iloc[t]


def test_bocpd_speed():
    import time
    t0 = time.perf_counter()
    bocpd(data.random_walk(n=2500))
    assert time.perf_counter() - t0 < 5.0
