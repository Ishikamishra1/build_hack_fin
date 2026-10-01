"""
gap_scoring_agent — Gap Analysis & Opportunity Scoring Engine.

Uses Amazon Bedrock (Claude Sonnet 5 / configurable via BEDROCK_MODEL_ID) to
reason over the six specialist agents' findings, identify unmet-need
intersections, and rank the Top 5-6 therapeutic opportunities with explainable
scores and evidence-backed rationales.

Falls back to the deterministic weighted scorer if Bedrock is unavailable,
so the pipeline always produces a result.
"""
import json
import os
import boto3
from botocore.exceptions import ClientError

_REGION = os.environ.get("AWS_REGION", "us-east-1")
_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")

_DEFAULT_WEIGHTS = {
    "unmet_medical_need": 0.30,
    "disease_burden": 0.20,
    "existing_treatment_gap": 0.20,
    "scientific_evidence": 0.15,
    "research_momentum": 0.10,
    "competitive_landscape": 0.05,
}

DISCLAIMER = (
    "This is a decision-support prioritization draft, not a guarantee of "
    "drug success, a clinical prediction, or a regulatory prediction. A "
    "human researcher must review and validate before any resource is committed."
)


def lambda_handler(event, context):
    findings = event.get("agent_findings", event) if isinstance(event, dict) else event
    if not isinstance(findings, list):
        findings = [findings]

    by_agent = {}
    for item in findings:
        if isinstance(item, dict) and item.get("agent"):
            by_agent.setdefault(item["agent"], item)

    therapeutic_area = next(
        (v.get("therapeutic_area") for v in by_agent.values() if v.get("therapeutic_area")),
        "Unknown",
    )

    try:
        result = _score_with_bedrock(therapeutic_area, by_agent)
        result["method"] = "bedrock_llm"
        result["model"] = _MODEL_ID
    except Exception as exc:
        result = _score_deterministic(therapeutic_area, by_agent)
        result["bedrock_error"] = str(exc)

    result["agents_seen"] = sorted(by_agent)
    result["disclaimer"] = DISCLAIMER
    return result


def _score_with_bedrock(therapeutic_area: str, by_agent: dict) -> dict:
    prompt = _build_prompt(therapeutic_area, by_agent)
    raw_text = _invoke_bedrock(prompt)
    return _parse_llm_response(raw_text, therapeutic_area)


def _build_prompt(therapeutic_area: str, by_agent: dict) -> str:
    def fmt_records(agent_key: str, label: str) -> str:
        agent = by_agent.get(agent_key, {})
        records = agent.get("records", [])
        insights = agent.get("llm_insights", "")
        lines = [f"=== {label} ==="]
        if insights:
            lines.append(f"Agent analysis: {insights}")
        if records:
            lines.append(f"Records retrieved: {len(records)}")
            for r in records[:5]:
                lines.append(f"  - {json.dumps(r, default=str)[:200]}")
        else:
            lines.append("No records retrieved.")
        return "\n".join(lines)

    sections = "\n\n".join([
        fmt_records("disease_agent", "DISEASE AGENT — Global disease burden (GLOBOCAN)"),
        fmt_records("treatment_agent", "TREATMENT AGENT — Approved therapies (openFDA)"),
        fmt_records("research_agent", "RESEARCH AGENT — Scientific literature (PubMed)"),
        fmt_records("clinical_trial_agent", "CLINICAL TRIAL AGENT — Active trials (ClinicalTrials.gov)"),
        fmt_records("competition_agent", "COMPETITION AGENT — Competitive landscape"),
        fmt_records("trend_agent", "TREND AGENT — Research momentum (PubMed trends)"),
    ])

    return f"""You are a pharmaceutical research intelligence analyst specializing in therapeutic opportunity identification.

Analyze the following data from 6 specialist AI agents for the therapeutic area: {therapeutic_area}

{sections}

Based on this cross-source intelligence, identify and rank the TOP 5 therapeutic opportunities.
For each opportunity, consider: disease burden, existing treatment gaps, scientific evidence strength,
research momentum, competitive landscape, and unmet medical need.

Respond ONLY with a valid JSON object in this exact format (no other text before or after):
{{
  "opportunities": [
    {{
      "name": "Specific opportunity name (e.g. 'KRAS G12C targeted therapy for CRC')",
      "score": 85,
      "rationale": "2-3 sentence evidence-based rationale explaining WHY this ranks highly",
      "key_evidence": ["evidence point 1", "evidence point 2", "evidence point 3"],
      "dimension_scores": {{
        "unmet_medical_need": 90,
        "disease_burden": 85,
        "existing_treatment_gap": 80,
        "scientific_evidence": 75,
        "research_momentum": 70,
        "competitive_landscape": 65
      }}
    }}
  ]
}}

Return ONLY the JSON. No markdown, no explanation, no preamble."""


def _invoke_bedrock(prompt: str) -> str:
    client = boto3.client("bedrock-runtime", region_name=_REGION)
    response = client.converse(
        modelId=_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 2048},
    )
    return response["output"]["message"]["content"][0]["text"]


def _parse_llm_response(text: str, therapeutic_area: str) -> dict:
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    parsed = json.loads(text)
    opportunities = parsed.get("opportunities", [])

    # Normalise scores to ensure they are numeric
    for opp in opportunities:
        opp["score"] = float(opp.get("score", 0))
        for k, v in opp.get("dimension_scores", {}).items():
            opp["dimension_scores"][k] = float(v)

    return {"opportunities": opportunities}


# ---------------------------------------------------------------------------
# Deterministic fallback (used only when Bedrock is unavailable)
# ---------------------------------------------------------------------------

def _score_deterministic(therapeutic_area: str, by_agent: dict) -> dict:
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

    dims = {
        "disease_burden": _sat(n_burden, 10),
        "existing_treatment_gap": 100 - _sat(n_drugs, 25),
        "scientific_evidence": _sat(n_papers, 30),
        "research_momentum": _sat(n_papers, 30),
        "competitive_landscape": 100 - _sat(len(sponsors), 15),
        "unmet_medical_need": (_sat(n_burden, 10) + (100 - _sat(n_drugs, 25))) / 2,
    }

    weights = _DEFAULT_WEIGHTS
    score = round(sum(dims[k] * weights.get(k, 0) for k in dims), 1)

    return {
        "opportunities": [{
            "name": f"{therapeutic_area} — evidence-signal composite",
            "score": score,
            "method": "deterministic_fallback",
            "rationale": (
                f"Deterministic fallback: {n_burden} disease records, "
                f"{n_drugs} drug labels, {n_papers} publications, "
                f"{n_trials} trials across {len(sponsors)} sponsors."
            ),
            "key_evidence": [
                f"{n_papers} PubMed publications retrieved",
                f"{n_trials} active clinical trials found",
                f"{n_drugs} existing drug labels in FDA database",
            ],
            "dimension_scores": {k: round(v, 1) for k, v in dims.items()},
        }],
    }


def _sat(value: int, full: int) -> float:
    return min(100.0, 100.0 * value / full) if full > 0 else 0.0
