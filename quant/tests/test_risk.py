import numpy as np
import pandas as pd
import pytest

from flowolf_quant import data
from flowolf_quant.risk import (block_bootstrap, correlation_stress, drawdown_distribution,
                                ruin_probability, var_cvar)


def test_block_bootstrap_shape_and_mean():
    r = data.random_walk(n=2520, vol=0.01, drift=0.0005, seed=0)
    paths = block_bootstrap(r, n_paths=500, block_mean=20, horizon=252, seed=1)
    assert paths.shape == (500, 252)
    assert paths.mean() == pytest.approx(r.mean(), abs=0.0002)
    assert paths.std() == pytest.approx(r.std(), rel=0.05)
    # every resampled value comes from the sample
    assert np.isin(paths[:5], r.to_numpy()).all()


def test_block_bootstrap_keeps_blocks():
    r = np.arange(1000, dtype=float)
    paths = block_bootstrap(r, n_paths=200, block_mean=20, horizon=300, seed=0)
    consecutive = (np.diff(paths, axis=1) == 1).mean()
    assert 0.9 < consecutive < 0.99            # ~ 1 - 1/20
    iid = block_bootstrap(r, n_paths=200, block_mean=1, horizon=300, seed=0)
    assert (np.diff(iid, axis=1) == 1).mean() < 0.01


def test_block_bootstrap_seeded():
    r = data.random_walk(n=300, seed=0)
    a = block_bootstrap(r, n_paths=10, seed=42)
    b = block_bootstrap(r, n_paths=10, seed=42)
    np.testing.assert_array_equal(a, b)


def test_drawdown_quantiles_ordered():
    r = data.random_walk(n=1500, vol=0.012, drift=0.0003, seed=3)
    d = drawdown_distribution(r, n_paths=1000, horizon=504, seed=0)
    assert d["p1"] <= d["p5"] <= d["median"] <= 0
    assert d["historical"] <= 0


def test_ruin_higher_for_riskier_series():
    safe = data.random_walk(n=2520, vol=0.008, drift=0.0003, seed=5)
    risky = data.random_walk(n=2520, vol=0.03, drift=0.0003, seed=5)
    p_safe = ruin_probability(safe, ruin_level=-0.5, n_paths=500, seed=0)
    p_risky = ruin_probability(risky, ruin_level=-0.5, n_paths=500, seed=0)
    assert p_risky > p_safe
    assert 0 <= p_safe <= p_risky <= 1


def test_var_cvar_ordering():
    r = data.random_walk(n=2520, vol=0.01, seed=0)
    t = var_cvar(r, levels=(0.95, 0.99))
    for lvl in (0.95, 0.99):
        assert t.loc[lvl, "cvar"] <= t.loc[lvl, "var"] < 0
    assert t.loc[0.99, "var"] <= t.loc[0.95, "var"]
    # normal check: 95% VaR ~ -1.645 sigma
    assert t.loc[0.95, "var"] == pytest.approx(-1.645 * 0.01, rel=0.1)


def test_correlation_stress_raises_long_only_vol():
    P = data.random_walk_panel(n=1000, k=10, vol=0.01, seed=0)
    w = pd.Series(0.1, index=P.columns)
    s = correlation_stress(P, w, stress_corr=0.9)
    assert s["stressed_vol"] > 2 * s["normal_vol"]
    # with corr = 1 the stressed vol equals the weighted sum of vols
    s1 = correlation_stress(P, w, stress_corr=1.0)
    expected = (w * P.std(ddof=1)).sum() * np.sqrt(252)
    assert s1["stressed_vol"] == pytest.approx(expected, rel=1e-9)
