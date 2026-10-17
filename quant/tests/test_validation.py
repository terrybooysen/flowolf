import numpy as np
import pandas as pd
import pytest
from scipy import stats

from flowolf_quant import backtest as bt
from flowolf_quant import data
from flowolf_quant.validation import (GateConfig, HoldoutLedger, TrialRegistry, cv_select_and_score,
                                      deflated_sharpe, evaluate, expected_max_sharpe, pbo_cscv,
                                      probabilistic_sharpe, purged_kfold, split_holdout,
                                      walk_forward)

LOOKBACKS = [5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 150, 200]


def trend_returns(r: pd.Series, lookback: int) -> bt.BacktestResult:
    w = np.sign(r.rolling(lookback).mean()).fillna(0.0)
    return bt.run(r, w, cost_bps=2.0)


def family_matrix(r: pd.Series, lookbacks=LOOKBACKS):
    res = {lb: trend_returns(r, lb) for lb in lookbacks}
    M = pd.DataFrame({lb: x.returns for lb, x in res.items()})
    return M, res


def run_gate(r: pd.Series, config=GateConfig()):
    design, hold = split_holdout(r, years=2)
    M, res = family_matrix(design)
    best = M.apply(bt.sharpe).idxmax()
    hold_ret = trend_returns(r, best).returns.loc[hold.index]
    return evaluate(M[best], M, hold_ret, n_trades=res[best].metrics["trades"], config=config)


# --- Sharpe statistics -------------------------------------------------------------

def test_psr_matches_normal_formula_for_gaussian_returns():
    r = data.random_walk(n=5000, drift=0.0004, vol=0.01, seed=3).values
    sr = r.mean() / r.std(ddof=1)
    approx = stats.norm.cdf(sr / np.sqrt((1 + 0.5 * sr ** 2) / (len(r) - 1)))
    assert abs(probabilistic_sharpe(r) - approx) < 0.02


def test_expected_max_sharpe_matches_simulation():
    n_trials, T = 50, 1000
    rng = np.random.default_rng(7)
    maxes = []
    for _ in range(300):
        x = rng.normal(0, 0.01, (T, n_trials))
        srs = x.mean(axis=0) / x.std(axis=0, ddof=1)
        maxes.append(srs.max())
    formula = expected_max_sharpe(n_trials, 1.0 / (T - 1))
    assert abs(np.mean(maxes) - formula) / formula < 0.10


def test_dsr_falls_as_trials_rise():
    r = data.random_walk(n=2520, drift=0.0005, vol=0.01, seed=5)
    values = [deflated_sharpe(r, n_trials=n).dsr for n in (1, 5, 20, 100, 1000)]
    assert all(a > b for a, b in zip(values, values[1:]))


# --- PBO ---------------------------------------------------------------------------

def test_pbo_near_half_for_noise():
    pbos = []
    for seed in range(5):
        M = data.random_walk_panel(n=2000, k=30, seed=seed)
        pbos.append(pbo_cscv(M, n_blocks=10).pbo)
    assert 0.3 < np.mean(pbos) < 0.7


def test_pbo_low_when_one_variant_is_really_better():
    M = data.random_walk_panel(n=2000, k=30, vol=0.01, seed=11)
    M["real"] = data.random_walk(n=2000, drift=0.0012, vol=0.01, seed=12).values
    assert pbo_cscv(M, n_blocks=10).pbo < 0.1


# --- purged CV, walk-forward, holdout ------------------------------------------------

def test_purged_kfold_has_no_leakage():
    n, horizon, lookback, emb = 1000, 5, 20, 10
    for train, test in purged_kfold(n, k=5, label_horizon=horizon, feature_lookback=lookback, embargo=emb):
        a, b = test.min(), test.max() + 1
        assert not np.any((train >= a - horizon) & (train < b + lookback + emb))
        assert len(set(train) & set(test)) == 0


def test_cv_and_walk_forward_find_planted_edge_but_not_noise():
    edge = data.planted_trend(strength=0.10, lookback=20, seed=2)
    noise = data.random_walk(seed=2)
    f_edge = lambda lb: trend_returns(edge, lb).returns
    f_noise = lambda lb: trend_returns(noise, lb).returns
    grid = [5, 10, 20, 40, 80]
    assert cv_select_and_score(f_edge, grid, feature_lookback=80).oos_sharpe > 0.8
    assert walk_forward(f_edge, grid).oos_sharpe > 0.8
    assert cv_select_and_score(f_noise, grid, feature_lookback=80).oos_sharpe < 0.8


def test_holdout_ledger_flags_reuse(tmp_path):
    r = data.random_walk()
    _, hold = split_holdout(r)
    ledger = HoldoutLedger(str(tmp_path / "holdout.jsonl"))
    assert not ledger.record("tsmom", hold, {"sharpe": 0.1}).spent
    assert ledger.record("tsmom", hold, {"sharpe": 0.1}).spent


# --- trials registry ----------------------------------------------------------------

def test_trial_registry_persists_and_counts_everything(tmp_path):
    path = str(tmp_path / "trials.jsonl")
    reg = TrialRegistry(path)
    r = data.random_walk(n=800)
    for lb in (5, 10, 20):
        reg.add("tsmom", {"lookback": lb}, trend_returns(r, lb).returns)
    again = TrialRegistry(path)
    assert again.count("tsmom") == 3
    assert again.matrix("tsmom").shape == (800, 3)


# --- the gate end to end -------------------------------------------------------------

def test_gate_fails_noise_almost_always():
    statuses = [run_gate(data.random_walk(n=2520 + 504, seed=s)).status for s in range(10)]
    assert statuses.count("Pass") <= 1


def test_gate_passes_a_real_planted_edge():
    r = data.planted_trend(n=2520 + 504, strength=0.10, lookback=20, seed=4)
    v = run_gate(r)
    assert v.status == "Pass", v.summary


def test_gate_reports_not_enough_data_for_short_history():
    r = data.planted_trend(n=600 + 504, strength=0.10, lookback=20, seed=4)
    assert run_gate(r).status in ("Not enough data", "Fail")
    assert any(c.name == "data length" and c.passed is None for c in run_gate(r).checks)


def test_gate_config_can_only_tighten():
    cfg = GateConfig()
    assert cfg.tightened(min_dsr=0.99, max_pbo=0.1).min_dsr == 0.99
    with pytest.raises(ValueError):
        cfg.tightened(min_dsr=0.5)
    with pytest.raises(ValueError):
        cfg.tightened(max_pbo=0.5)


def test_effective_trials_counts_scaled_copies_once():
    from flowolf_quant.validation.trials import effective_trials
    M = data.random_walk_panel(n=1500, k=6, seed=9)
    copies = pd.concat([M * s for s in (0.5, 1.0, 2.0)], axis=1, keys=["a", "b", "c"])
    copies.columns = [f"{a}{b}" for a, b in copies.columns]
    assert effective_trials(copies) == 6
    assert effective_trials(M) == 6
