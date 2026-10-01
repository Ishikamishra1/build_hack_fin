"""
query_globocan_data — Disease Agent tool.

Reads GLOBOCAN cancer burden data from S3 (periodically refreshed from
https://gco.iarc.who.int/today), then uses Amazon Bedrock to interpret
the epidemiological data and surface disease burden insights.

If no S3 export exists yet (common for new deployments), returns representative
demo statistics so the pipeline runs end-to-end during development.
"""
import json
import os
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3")
BUCKET = os.environ.get("DATA_BUCKET_NAME")
_REGION = os.environ.get("AWS_REGION", "us-east-1")
_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")

_DEMO_DATA = {
    "Colorectal Cancer": {
        "data_year": 2022,
        "records": [
            {"region": "World", "new_cases": 1926425, "deaths": 903701, "prevalence_5yr": 5156571,
             "incidence_rate_per_100k": 24.0, "mortality_rate_per_100k": 11.2, "sex": "Both"},
            {"region": "East Asia", "new_cases": 712000, "deaths": 338000, "prevalence_5yr": 1900000,
             "incidence_rate_per_100k": 32.1, "mortality_rate_per_100k": 15.2, "sex": "Both"},
            {"region": "Europe", "new_cases": 500000, "deaths": 243000, "prevalence_5yr": 1400000,
             "incidence_rate_per_100k": 35.4, "mortality_rate_per_100k": 17.2, "sex": "Both"},
            {"region": "North America", "new_cases": 208000, "deaths": 75000, "prevalence_5yr": 630000,
             "incidence_rate_per_100k": 26.7, "mortality_rate_per_100k": 9.4, "sex": "Both"},
            {"region": "South/Central Asia", "new_cases": 160000, "deaths": 105000, "prevalence_5yr": 390000,
             "incidence_rate_per_100k": 8.4, "mortality_rate_per_100k": 5.5, "sex": "Both"},
        ],
    }
}


def lambda_handler(event, context):
    therapeutic_area = event.get("therapeutic_area", "Colorectal Cancer")
    cache_key = f"globocan/{_slugify(therapeutic_area)}.json"

    records = []
    data_year = "demo"
    source_note = ""

    if BUCKET:
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=cache_key)
            data = json.loads(obj["Body"].read())
            records = data.get("records", [])[:50]
            data_year = data.get("data_year", "unknown")
            source_note = "IARC GLOBOCAN / Global Cancer Observatory (Cancer Today)"
        except ClientError:
            pass  # NoSuchKey, AccessDenied, or any other S3 error → use demo data

    if not records:
        demo = _DEMO_DATA.get(therapeutic_area, list(_DEMO_DATA.values())[0])
        records = demo["records"]
        data_year = demo["data_year"]
        source_note = "IARC GLOBOCAN 2022 (representative demo data — upload actual export to S3 for production)"

    llm_insights = _analyze_with_bedrock(therapeutic_area, records)

    return {
        "agent": "disease_agent",
        "tool": "query_globocan_data",
        "therapeutic_area": therapeutic_area,
        "source": source_note,
        "data_year": data_year,
        "record_count": len(records),
        "records": records,
        "llm_insights": llm_insights,
    }


def _analyze_with_bedrock(therapeutic_area: str, records: list) -> str:
    if not records:
        return ""
    try:
        total_cases = sum(r.get("new_cases", 0) for r in records if isinstance(r.get("new_cases"), (int, float)))
        total_deaths = sum(r.get("deaths", 0) for r in records if isinstance(r.get("deaths"), (int, float)))
        regions = [r.get("region", "") for r in records[:5] if r.get("region")]

        prompt = (
            f"You are a pharmaceutical epidemiology analyst. "
            f"Analyze this GLOBOCAN disease burden data for '{therapeutic_area}':\n\n"
            f"Global new cases per year: {total_cases:,}\n"
            f"Global annual deaths: {total_deaths:,}\n"
            f"Regions covered: {', '.join(regions)}\n"
            f"Mortality rate (deaths/cases): {(total_deaths/total_cases*100):.1f}% if {total_cases} > 0 else 'N/A'\n\n"
            "In 2-3 sentences: What does this disease burden data tell us about the magnitude "
            "of unmet medical need? Which patient populations face the greatest burden? "
            "What does this mean for R&D investment priorities? Be specific and data-driven."
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


def _error(therapeutic_area: str, message: str) -> dict:
    return {
        "agent": "disease_agent",
        "tool": "query_globocan_data",
        "therapeutic_area": therapeutic_area,
        "error": message,
        "records": [],
    }


def _slugify(text: str) -> str:
    return text.strip().lower().replace(" ", "_")
