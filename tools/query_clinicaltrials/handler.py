"""
query_clinicaltrials — Clinical Trial Agent + Competition Agent tool.

Fetches trial records from ClinicalTrials.gov API v2, then uses Amazon Bedrock
to analyze trial phases, competitive landscape, and emerging sponsors — turning
raw trial data into agent intelligence.
"""
import json
import os
import urllib.request
import urllib.parse
import urllib.error
import boto3

CTGOV_BASE = "https://clinicaltrials.gov/api/v2/studies"
_REGION = os.environ.get("AWS_REGION", "us-east-1")
_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")


def lambda_handler(event, context):
    therapeutic_area = event.get("therapeutic_area", "Colorectal Cancer")
    max_results = event.get("max_results", 30)

    try:
        records = _search(therapeutic_area, max_results)
    except urllib.error.URLError as exc:
        return {
            "agent": "clinical_trial_agent",
            "tool": "query_clinicaltrials",
            "therapeutic_area": therapeutic_area,
            "error": str(exc),
            "records": [],
        }

    llm_insights = _analyze_with_bedrock(therapeutic_area, records)

    return {
        "agent": "clinical_trial_agent",
        "tool": "query_clinicaltrials",
        "therapeutic_area": therapeutic_area,
        "record_count": len(records),
        "records": records,
        "llm_insights": llm_insights,
    }


def _analyze_with_bedrock(therapeutic_area: str, records: list) -> str:
    if not records:
        return ""
    try:
        sponsors = list({r.get("sponsor") for r in records if r.get("sponsor")})[:10]
        phases = [r.get("phase") for r in records if r.get("phase")]
        statuses = [r.get("status") for r in records if r.get("status")]
        titles = [r.get("title", "") for r in records[:10] if r.get("title")]

        prompt = (
            f"You are a pharmaceutical competitive intelligence analyst. "
            f"Analyze this ClinicalTrials.gov data for '{therapeutic_area}':\n\n"
            f"Total trials: {len(records)}\n"
            f"Key sponsors: {', '.join(sponsors[:8])}\n"
            f"Trial phases: {', '.join(str(p) for p in phases[:10])}\n"
            f"Statuses: {', '.join(set(statuses[:10]))}\n"
            f"Sample trials: {'; '.join(titles[:5])}\n\n"
            "In 2-3 sentences: What does this trial landscape reveal about the competitive "
            "dynamics? Are there underexplored phases or indications? What opportunities "
            "exist given current trial activity? Be specific."
        )
        return _invoke_bedrock(prompt)
    except Exception:
        return ""


def _invoke_bedrock(prompt: str) -> str:
    client = boto3.client("bedrock-runtime", region_name=_REGION)
    response = client.converse(
        modelId=_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 512},
    )
    return response["output"]["message"]["content"][0]["text"]


def _search(term: str, max_results: int) -> list:
    params = urllib.parse.urlencode({
        "query.cond": term, "pageSize": max_results,
        "fields": "NCTId,BriefTitle,OverallStatus,Phase,LeadSponsorName,Condition",
    })
    req = urllib.request.Request(f"{CTGOV_BASE}?{params}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    records = []
    for study in body.get("studies", []):
        protocol = study.get("protocolSection", {})
        ident = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})
        design = protocol.get("designModule", {})
        sponsor = protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
        records.append({
            "nct_id": ident.get("nctId"),
            "title": ident.get("briefTitle"),
            "status": status.get("overallStatus"),
            "phase": design.get("phases"),
            "sponsor": sponsor.get("name"),
        })
    return records
