"""
score_opportunities_fallback — a Bedrock-free stand-in for gap_scoring_agent.

WHY THIS EXISTS: the gap_scoring_agent needs bedrock:InvokeModel, which is
currently denied org-wide by a Service Control Policy. Every other stage of
the pipeline (the four data tools, the scope check, the report composer)
needs no model access at all, so this deterministic scorer lets the whole
ingest -> analyze -> score -> report pipeline run and demo end-to-end while
that policy exception is pending.

THIS IS NOT LLM REASONING. It applies the weights in data/scoring_weights.json
to countable signals in the six agents' outputs. It cannot identify a novel
opportunity or explain one in natural language -- it ranks the evidence it can
count. Every output is tagged "method": "deterministic_fallback" so a demo
never misrepresents it as agent reasoning.

To switch back once Bedrock is available, repoint the state machine's
Agent_GapAnalysisAndScoring state at ${InvokeAgentCoreAgentArn} (the original
Parameters block is preserved in a comment in statemachine.asl.json).
"""
import json
import os

# Injected at synth time by AgentStack from data/scoring_weights.json, so the
# weighting formula has exactly one source of truth.
_DEFAULT_WEIGHTS = {
    "unmet_medical_need": 0.30,
    "disease_burden": 0.20,
    "existing_treatment_gap": 0.20,
    "scientific_evidence": 0.15,
    "research_momentum": 0.10,
    "competitive_landscape": 0.05,
}

DISCLAIMER = (
    "This is a decision-support prioritization metric, not a guarantee of "
    "drug success, a clinical prediction, or a regulatory prediction. Scores "
    "on this run were produced by a deterministic fallback scorer, not by an "
    "LLM agent."
)


def _weights() -> dict:
    raw = os.environ.get("SCORING_WEIGHTS")
    if not raw:
        return _DEFAULT_WEIGHTS
    try:
        return json.loads(raw).get("weights", _DEFAULT_WEIGHTS)
    except (TypeError, ValueError):
        return _DEFAULT_WEIGHTS


def lambda_handler(event, context):
    """
    Input: the Parallel state's output -- a list of the six agents' results,
    in branch order (disease, treatment, research, clinical_trial,
    competition, trend). Tolerates a dict-wrapped payload too.
    """
    findings = event.get("agent_findings", event) if isinstance(event, dict) else event
    if not isinstance(findings, list):
        findings = [findings]

    by_agent = {}
    for item in findings:
        if isinstance(item, dict) and item.get("agent"):
            by_agent.setdefault(item["agent"], item)

    disease = by_agent.get("disease_agent", {})
    treatment = by_agent.get("treatment_agent", {})
    research = by_agent.get("research_agent", {})
    trials = by_agent.get("clinical_trial_agent", {})

    n_burden = len(disease.get("records") or [])
    n_drugs = len(treatment.get("records") or [])
    n_papers = len(research.get("records") or [])
    trial_records = trials.get("records") or []
    n_trials = len(trial_records)
    sponsors = {r.get("sponsor") for r in trial_records if r.get("sponsor")}

    # Each dimension is scored 0-100 from a countable signal. Saturating
    # divisors are deliberately conservative -- they are demo-calibrated, not
    # validated, which is exactly why the real scorer must be an agent.
    dims = {
        "disease_burden": _saturate(n_burden, 10),
        "existing_treatment_gap": 100 - _saturate(n_drugs, 25),
        "scientific_evidence": _saturate(n_papers, 30),
        "research_momentum": _saturate(n_papers, 30),
        "competitive_landscape": 100 - _saturate(len(sponsors), 15),
        # No countable proxy for unmet need without reasoning over the burden
        # text; approximate it as burden weighted against available therapies.
        "unmet_medical_need": _mean(
            _saturate(n_burden, 10), 100 - _saturate(n_drugs, 25)
        ),
    }

    weights = _weights()
    score = round(sum(dims[k] * weights.get(k, 0) for k in dims), 1)

    area = (
        disease.get("therapeutic_area")
        or research.get("therapeutic_area")
        or "Unknown"
    )

    missing = [a for a in ("disease_agent", "treatment_agent", "research_agent",
                           "clinical_trial_agent") if a not in by_agent]
    errored = [a for a, v in by_agent.items() if v.get("error")]

    opportunity = {
        "name": f"{area} — evidence-signal composite",
        "score": score,
        "method": "deterministic_fallback",
        "rationale": (
            f"Scored from countable signals: {n_burden} burden record(s), "
            f"{n_drugs} approved-label record(s), {n_papers} publication(s), "
            f"{n_trials} trial(s) across {len(sponsors)} distinct sponsor(s). "
            f"Treatment-gap and competition dimensions are inverse-scored, so "
            f"fewer existing therapies and fewer active sponsors raise the score."
        ),
        "dimension_scores": {k: round(v, 1) for k, v in dims.items()},
        "supporting_evidence": {
            "disease_burden": f"disease_agent: {n_burden} record(s)",
            "existing_treatment_gap": f"treatment_agent: {n_drugs} label(s)",
            "scientific_evidence": f"research_agent: {n_papers} publication(s)",
            "research_momentum": f"research_agent: publication count (no date analysis in fallback)",
            "competitive_landscape": f"clinical_trial_agent: {len(sponsors)} sponsor(s)",
            "unmet_medical_need": "derived: burden vs. available therapies",
        },
        "key_uncertainties": [
            "Scored by record counts, not by reasoning over record content.",
            "Saturating divisors are demo-calibrated and not validated.",
            "research_momentum duplicates scientific_evidence: assessing real "
            "momentum needs publication-date analysis the fallback does not do.",
        ]
        + ([f"Agent output missing: {', '.join(missing)}"] if missing else [])
        + ([f"Agent returned an error: {', '.join(errored)}"] if errored else []),
    }

    return {
        "opportunities": [opportunity],
        "method": "deterministic_fallback",
        "agents_seen": sorted(by_agent),
        "disclaimer": DISCLAIMER,
    }


def _saturate(value: int, full: int) -> float:
    """Map a count onto 0-100, saturating at `full`."""
    if full <= 0:
        return 0.0
    return min(100.0, 100.0 * value / full)


def _mean(*values: float) -> float:
    return sum(values) / len(values) if values else 0.0
