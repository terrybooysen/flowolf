import json

from flowolf_quant import data
from flowolf_quant.research import FAMILIES, GridProposer, LLMProposer, RandomProposer, research
from flowolf_quant.validation import HoldoutLedger, TrialRegistry


def test_loop_rejects_noise():
    r = data.random_walk(n=3024, seed=21)
    rep = research(r, "trend_sign", GridProposer(points_per_param=12))
    assert rep.verdict.status != "Pass"
    assert rep.n_trials >= 10


def test_loop_accepts_planted_edge():
    # A 50-point grid (step 5) includes the planted 20-day lookback; 50 trials
    # also raises the deflated-Sharpe bar, and the edge still clears it.
    r = data.planted_trend(n=3024, strength=0.10, lookback=20, seed=4)
    rep = research(r, "trend_sign", GridProposer(points_per_param=50))
    assert rep.verdict.status == "Pass", rep.verdict.summary
    assert rep.best_params["lookback"] == 20


def test_loop_fails_when_the_search_misses_the_edge():
    # A coarse grid that skips the real lookback finds a weaker variant, and
    # the gate should not wave it through.
    r = data.planted_trend(n=3024, strength=0.10, lookback=20, seed=4)
    rep = research(r, "trend_sign", GridProposer(points_per_param=12))
    assert rep.verdict.status != "Pass"


def test_every_proposal_counts_as_a_trial_across_rounds(tmp_path):
    r = data.random_walk(n=3024, seed=1)
    reg = TrialRegistry(str(tmp_path / "t.jsonl"))
    research(r, "ma_cross", RandomProposer(seed=1), rounds=2, per_round=5, registry=reg)
    rep = research(r, "ma_cross", RandomProposer(seed=2), rounds=1, per_round=5, registry=reg)
    assert rep.n_trials == 15
    ds = [c for c in rep.verdict.checks if c.name == "deflated Sharpe"][0]
    assert "15 variant" in ds.reason or "15 tried" in ds.reason


def test_holdout_reuse_is_flagged(tmp_path):
    r = data.planted_trend(n=3024, strength=0.10, lookback=20, seed=4)
    ledger = HoldoutLedger(str(tmp_path / "h.jsonl"))
    reg = TrialRegistry()
    research(r, "trend_sign", GridProposer(6), registry=reg, ledger=ledger)
    rep = research(r, "trend_sign", RandomProposer(3), per_round=3, registry=reg, ledger=ledger)
    hold = [c for c in rep.verdict.checks if c.name == "holdout"][0]
    assert "used before" in hold.reason


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
    rep = research(r, "trend_sign", GridProposer(4))
    json.dumps(rep.to_dict(), default=str)


def test_tsmom_family_passes_planted_edge_and_fails_noise():
    r = data.planted_trend(n=3024, strength=0.10, lookback=20, seed=4)
    assert research(r, "tsmom", GridProposer(12)).verdict.status == "Pass"
    statuses = [research(data.random_walk(n=3024, seed=s), "tsmom", GridProposer(12)).verdict.status
                for s in range(10)]
    assert statuses.count("Pass") <= 1
