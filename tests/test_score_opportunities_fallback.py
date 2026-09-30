"""
Tests for the Bedrock-free fallback scorer. It stands in for
gap_scoring_agent while bedrock:InvokeModel is denied by the org SCP, so it
needs to behave sanely on the real Parallel-state output shape -- including
the degraded cases where a branch returned an error or is missing entirely.
"""
import os, json

from conftest import load_tool

handler = load_tool("score_opportunities_fallback")


def _findings(n_burden=1, n_drugs=5, n_papers=30, n_trials=10, sponsors=6):
    return [
        {"agent": "disease_agent", "therapeutic_area": "Colorectal Cancer",
         "records": [{"incidence": 1930000}] * n_burden},
        {"agent": "treatment_agent", "records": [{"brand_name": f"d{i}"} for i in range(n_drugs)]},
        {"agent": "research_agent", "records": [{"pmid": str(i)} for i in range(n_papers)]},
        {"agent": "clinical_trial_agent",
         "records": [{"nct_id": f"NCT{i}", "sponsor": f"S{i % sponsors}"} for i in range(n_trials)]},
    ]


def test_scores_within_range_and_is_labelled_a_fallback():
    r = handler.lambda_handler(_findings(), None)
    opp = r["opportunities"][0]
    assert 0 <= opp["score"] <= 100
    assert r["method"] == "deterministic_fallback"
    assert opp["method"] == "deterministic_fallback"
    assert "not a guarantee" in r["disclaimer"]


def test_fewer_existing_therapies_raises_the_score():
    crowded = handler.lambda_handler(_findings(n_drugs=25), None)["opportunities"][0]["score"]
    open_field = handler.lambda_handler(_findings(n_drugs=0), None)["opportunities"][0]["score"]
    assert open_field > crowded


def test_more_sponsors_lowers_the_score():
    few = handler.lambda_handler(_findings(sponsors=1), None)["opportunities"][0]["score"]
    many = handler.lambda_handler(_findings(n_trials=30, sponsors=15), None)["opportunities"][0]["score"]
    assert few > many


def test_accepts_dict_wrapped_payload():
    r = handler.lambda_handler({"agent_findings": _findings()}, None)
    assert r["opportunities"][0]["score"] >= 0


def test_surfaces_missing_and_errored_agents():
    partial = [_findings()[0], {"agent": "research_agent", "error": "timeout", "records": []}]
    r = handler.lambda_handler(partial, None)
    unc = " ".join(r["opportunities"][0]["key_uncertainties"])
    assert "missing" in unc and "treatment_agent" in unc
    assert "error" in unc and "research_agent" in unc


def test_survives_completely_empty_input():
    r = handler.lambda_handler([], None)
    assert r["opportunities"][0]["score"] >= 0
    assert r["agents_seen"] == []


def test_weights_come_from_env_when_present():
    os.environ["SCORING_WEIGHTS"] = json.dumps(
        {"weights": {"existing_treatment_gap": 1.0}})
    try:
        r = handler.lambda_handler(_findings(n_drugs=0), None)
        # gap dimension is 100 and carries all the weight
        assert r["opportunities"][0]["score"] == 100.0
    finally:
        del os.environ["SCORING_WEIGHTS"]
