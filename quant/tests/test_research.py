import json

from flowolf_quant import data
from flowolf_quant.research import FAMILIES, GridProposer, LLMProposer, RandomProposer, research
from flowolf_quant.validation import HoldoutLedger, TrialRegistry


def R(r, fam, prop, **kw):
    kw.setdefault("universe", "test")
    kw.setdefault("ledger", HoldoutLedger())
    return research(r, fam, proposer=prop, **kw)


def test_loop_rejects_noise():
    r = data.random_walk(n=3024, seed=21)
    rep = R(r, "trend_sign", GridProposer(points_per_param=12))
    assert rep.verdict.status != "Pass"
    assert rep.n_trials >= 10


def test_loop_accepts_planted_edge():
    # A 50-point grid (step 5) includes the planted 20-day lookback; 50 trials
    # also raises the deflated-Sharpe bar, and the edge still clears it.
    r = data.planted_trend(n=3024, strength=0.10, lookback=20, seed=4)
    rep = R(r, "trend_sign", GridProposer(points_per_param=50))
    assert rep.verdict.status == "Pass", rep.verdict.summary
    assert rep.best_params["lookback"] == 20


def test_loop_fails_when_the_search_misses_the_edge():
    # A coarse grid that skips the real lookback finds a weaker variant, and
    # the gate should not wave it through.
    r = data.planted_trend(n=3024, strength=0.10, lookback=20, seed=4)
    rep = R(r, "trend_sign", GridProposer(points_per_param=12))
    assert rep.verdict.status != "Pass"


def test_every_proposal_counts_as_a_trial_across_rounds(tmp_path):
    r = data.random_walk(n=3024, seed=1)
    reg = TrialRegistry(str(tmp_path / "t.jsonl"))
    R(r, "ma_cross", RandomProposer(seed=1), rounds=2, per_round=5, registry=reg)
    rep = R(r, "ma_cross", RandomProposer(seed=2), rounds=1, per_round=5, registry=reg)
    assert rep.n_trials == 15
    ds = [c for c in rep.verdict.checks if c.name == "deflated Sharpe"][0]
    assert "15 variant" in ds.reason or "15 tried" in ds.reason


def test_holdout_reuse_is_flagged(tmp_path):
    r = data.planted_trend(n=3024, strength=0.10, lookback=20, seed=4)
    ledger = HoldoutLedger(str(tmp_path / "h.jsonl"))
    reg = TrialRegistry()
    R(r, "trend_sign", GridProposer(6), registry=reg, ledger=ledger)
    rep = R(r, "trend_sign", RandomProposer(3), per_round=3, registry=reg, ledger=ledger)
    hold = [c for c in rep.verdict.checks if c.name == "holdout"][0]
    assert "scored before" in hold.reason


def test_llm_proposer_is_clipped_to_bounds_and_sees_no_holdout():
    seen = {}

    def fake_model(prompt: str) -> str:
        seen["prompt"] = prompt
        return 'Here you go: [{"lookback": 9999}, {"lookback": 30}, "junk"]'

    fam = FAMILIES["trend_sign"]
    props = LLMProposer(fake_model).propose(fam, [], 3)
    assert props == [{"lookback": 250, "long_only": False}, {"lookback": 30, "long_only": False}]
    assert "holdout" not in seen["prompt"].lower()


def test_report_serialises():
    r = data.random_walk(n=3024, seed=5)
    rep = R(r, "trend_sign", GridProposer(4))
    json.dumps(rep.to_dict(), default=str)


def test_tsmom_family_passes_planted_edge_and_fails_noise():
    r = data.planted_trend(n=3024, strength=0.10, lookback=20, seed=4)
    assert R(r, "tsmom", GridProposer(12)).verdict.status == "Pass"
    statuses = [R(data.random_walk(n=3024, seed=s), "tsmom", GridProposer(12)).verdict.status
                for s in range(10)]
    assert statuses.count("Pass") <= 1


# --- regression tests for the independent review's findings -------------------------

def test_different_holdout_lengths_never_pool_trials_into_selection():
    # C1: a run with a shorter holdout must not leak its later-dated trials
    # into a run whose holdout starts earlier; earlier trials still count in N.
    r = data.random_walk(n=3024, seed=31)
    reg, ledger = TrialRegistry(), HoldoutLedger()
    R(r, "trend_sign", GridProposer(10), registry=reg, ledger=ledger, holdout_years=0.5)
    rep = R(r, "trend_sign", GridProposer(10), registry=reg, ledger=ledger, holdout_years=2.0)
    M = reg.matrix(rep.trial_key)
    assert M.index[-1] < r.index[-504]
    ds = [c for c in rep.verdict.checks if c.name == "deflated Sharpe"][0]
    assert "20 variant" in ds.reason            # both runs' trials count towards N
    hold = [c for c in rep.verdict.checks if c.name == "holdout"][0]
    assert "scored before" in hold.reason       # overlapping holdout flagged (M3)


def test_two_assets_in_one_registry_are_never_pooled():
    # C2: an edge on asset B must not be attached to asset A's report.
    b = data.planted_trend(n=3024, strength=0.15, lookback=20, seed=4).rename("B")
    a = data.random_walk(n=3024, seed=7).rename("A")
    reg = TrialRegistry()
    R(b, "trend_sign", GridProposer(50), registry=reg, universe="B")
    rep = R(a, "trend_sign", GridProposer(50), registry=reg, universe="A")
    assert rep.verdict.status != "Pass"
    assert rep.design_metrics["sharpe"] < 1.5
    rep2 = R(a, "trend_sign", GridProposer(50), registry=reg, universe="B")   # same name, other data
    assert rep2.trial_key != reg.keys()[0] or rep2.design_metrics["sharpe"] < 1.5


def test_research_requires_universe_and_ledger():
    import pytest
    r = data.random_walk(n=3024)
    with pytest.raises(TypeError):
        research(r, "trend_sign")
    with pytest.raises(ValueError):
        research(r, "trend_sign", universe="", ledger=HoldoutLedger())


def test_trade_count_ignores_daily_resizing():
    # M4: vol-scaled positions resize every day but rarely change direction.
    from flowolf_quant import backtest as bt
    from flowolf_quant.regimes.tsmom import tsmom_weights
    r = data.random_walk(n=1500, seed=1)
    res = bt.run(r, tsmom_weights(r, lookback=63))
    assert res.trades < 0.2 * (res.turnover > 1e-9).sum()
