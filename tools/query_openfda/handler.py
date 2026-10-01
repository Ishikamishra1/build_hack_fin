"""
query_openfda — Treatment Agent tool.

Fetches approved drug label data from the openFDA API, then uses Amazon Bedrock
to analyze the existing treatment landscape and surface gaps — turning raw FDA
label data into treatment-gap intelligence.
"""
import json
import os
import urllib.request
import urllib.parse
import urllib.error
import boto3

OPENFDA_BASE = "https://api.fda.gov/drug/label.json"
_REGION = os.environ.get("AWS_REGION", "us-east-1")
_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")


def lambda_handler(event, context):
    therapeutic_area = event.get("therapeutic_area", "Colorectal Cancer")
    max_results = event.get("max_results", 30)

    try:
        records = _search(therapeutic_area, max_results)
    except urllib.error.URLError as exc:
        return {
            "agent": "treatment_agent",
            "tool": "query_openfda",
            "therapeutic_area": therapeutic_area,
            "error": str(exc),
            "records": [],
        }

    llm_insights = _analyze_with_bedrock(therapeutic_area, records)

    return {
        "agent": "treatment_agent",
        "tool": "query_openfda",
        "therapeutic_area": therapeutic_area,
        "record_count": len(records),
        "records": records,
        "llm_insights": llm_insights,
    }


def _analyze_with_bedrock(therapeutic_area: str, records: list) -> str:
    if not records:
        return ""
    try:
        drugs = [
            f"{r.get('brand_name', 'Unknown')} ({r.get('generic_name', '')})"
            for r in records[:12] if r.get("brand_name") or r.get("generic_name")
        ]
        indications_sample = (records[0].get("indications", "") if records else "")[:400]

        prompt = (
            f"You are a pharmaceutical treatment landscape analyst. "
            f"Analyze FDA-approved treatments for '{therapeutic_area}':\n\n"
            f"Approved drugs found: {len(records)}\n"
            f"Drug names: {', '.join(drugs[:10])}\n"
            f"Sample indication text: {indications_sample}\n\n"
            "In 2-3 sentences: What does the current treatment landscape look like? "
            "What patient populations or disease stages are underserved? "
            "Where are the most significant treatment gaps that represent R&D opportunities? "
            "Be specific and clinically grounded."
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
        "search": f'indications_and_usage:"{term}"',
        "limit": max_results,
    })
    req = urllib.request.Request(f"{OPENFDA_BASE}?{params}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    records = []
    for result in body.get("results", []):
        openfda = result.get("openfda", {})
        records.append({
            "brand_name": (openfda.get("brand_name") or [None])[0],
            "generic_name": (openfda.get("generic_name") or [None])[0],
            "manufacturer": (openfda.get("manufacturer_name") or [None])[0],
            "indications": (result.get("indications_and_usage") or [""])[0][:500],
            "warnings": (result.get("warnings") or [""])[0][:500],
        })
    return records
