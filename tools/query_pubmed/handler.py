"""
query_pubmed — Research Agent + Trend Agent tool.

Fetches recent scientific literature from PubMed via the NCBI E-utilities API,
then uses Amazon Bedrock to extract key research themes, emerging directions,
and evidence quality insights — turning raw citation data into agent intelligence.
"""
import json
import os
import urllib.request
import urllib.parse
import urllib.error
import boto3

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_REGION = os.environ.get("AWS_REGION", "us-east-1")
_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")


def lambda_handler(event, context):
    therapeutic_area = event.get("therapeutic_area", "Colorectal Cancer")
    max_results = event.get("max_results", 30)

    try:
        pmids = _search(therapeutic_area, max_results)
        records = _summarize(pmids) if pmids else []
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        return {
            "agent": "research_agent",
            "tool": "query_pubmed",
            "therapeutic_area": therapeutic_area,
            "error": str(exc),
            "records": [],
        }

    llm_insights = _analyze_with_bedrock(therapeutic_area, records)

    return {
        "agent": "research_agent",
        "tool": "query_pubmed",
        "therapeutic_area": therapeutic_area,
        "record_count": len(records),
        "records": records,
        "llm_insights": llm_insights,
    }


def _analyze_with_bedrock(therapeutic_area: str, records: list) -> str:
    if not records:
        return ""
    try:
        titles = [r.get("title", "") for r in records[:15] if r.get("title")]
        prompt = (
            f"You are a pharmaceutical research analyst. Analyze these {len(titles)} recent "
            f"PubMed publication titles for '{therapeutic_area}':\n\n"
            + "\n".join(f"- {t}" for t in titles)
            + "\n\nIn 2-3 sentences: What are the key research themes? What emerging "
            "treatment approaches show promise? What gaps in evidence exist? "
            "Be specific and evidence-focused."
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
        "db": "pubmed", "term": term, "retmax": max_results,
        "retmode": "json", "sort": "date",
    })
    with urllib.request.urlopen(f"{EUTILS_BASE}/esearch.fcgi?{params}", timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("esearchresult", {}).get("idlist", [])


def _summarize(pmids: list) -> list:
    params = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(pmids), "retmode": "json"})
    with urllib.request.urlopen(f"{EUTILS_BASE}/esummary.fcgi?{params}", timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    result = body.get("result", {})
    return [
        {
            "pmid": pmid,
            "title": result.get(pmid, {}).get("title"),
            "pub_date": result.get(pmid, {}).get("pubdate"),
            "journal": result.get(pmid, {}).get("fulljournalname"),
        }
        for pmid in result.get("uids", [])
    ]
