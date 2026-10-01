# TheraScout

**Agentic AI platform for pharmaceutical R&D opportunity prioritization.**

TheraScout takes a therapeutic area (e.g., Colorectal Cancer) and deploys four specialized AI agents in parallel to gather data from GLOBOCAN, PubMed, ClinicalTrials.gov, and openFDA. Amazon Nova Pro (via Bedrock) synthesizes the findings and ranks the **Top 5 therapeutic research opportunities** using a six-dimension weighted scoring model.

---

## Architecture

```
User selects therapeutic area
        │
        ▼
FastAPI Backend (scan.py)
        │  starts execution
        ▼
AWS Step Functions ── parallel ──┬── Disease Agent    (GLOBOCAN / WHO data)
                                 ├── Treatment Agent  (openFDA drug labels)
                                 ├── Research Agent   (PubMed publications)
                                 └── Clinical Trial Agent (ClinicalTrials.gov)
                                              │
                                              ▼
                              Amazon Nova Pro (Bedrock Converse API)
                              Scores & ranks Top 5 opportunities
                                              │
                                              ▼
                              React Frontend — card-based results UI
```

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10+ | Backend + CDK |
| Node.js | 18+ | Frontend (Vite + React) |
| AWS CDK | 2.x | `npm install -g aws-cdk` |
| saml2aws | 2.36.x | For Cognizant SSO login |
| AWS CLI | 2.x | Used by CDK and boto3 |

---

## AWS Setup (one-time)

### 1. Refresh your SAML token

Your AWS credentials expire every ~6 hours. Before running the backend or deploying, log in:

```powershell
cd "C:\Users\2469312\OneDrive - Cognizant\Desktop\build a thon\saml2aws_2.36.19_windows_amd64"
.\saml2aws.exe login
```

This writes temporary credentials to the `saml` profile in `~/.aws/credentials`.

### 2. Deploy infrastructure to AWS

```powershell
cd infra
pip install -r requirements.txt
cdk bootstrap
cdk deploy --all
```

After deploy, `deploy.ps1` writes the `STATE_MACHINE_ARN` to your `.env` automatically.

---

## Running Locally

### Step 1 — Environment files

```bash
# Root .env (backend + AWS)
cp .env.example .env
# Edit .env: set STATE_MACHINE_ARN from cdk_outputs.json after deploying

# Frontend .env
cp frontend/.env.example frontend/.env
# Default VITE_API_BASE=http://localhost:8000 works for local dev
```

### Step 2 — Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`.  
Swagger docs: `http://localhost:8000/docs`

> **Token expired?** Run `saml2aws login` (Step 1 above) and restart the backend — no `--reload` needed, restart fully.

### Step 3 — Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

---

## Scoring Model

Each opportunity is scored across six dimensions:

| Dimension | Weight |
|---|---|
| Unmet Medical Need | 30% |
| Disease Burden | 20% |
| Existing Treatment Gap | 20% |
| Scientific Evidence | 15% |
| Research Momentum | 10% |
| Competitive Landscape | 5% |

Scoring is performed by **Amazon Nova Pro** (`amazon.nova-pro-v1:0`) via the Bedrock Converse API from the FastAPI backend (SAML credentials). A deterministic fallback runs if Bedrock is unavailable.

---

## Data Sources

| Agent | Source |
|---|---|
| Disease Agent | IARC GLOBOCAN 2022 (WHO cancer burden data) |
| Treatment Agent | openFDA drug label API |
| Research Agent | PubMed / NCBI Entrez API |
| Clinical Trial Agent | ClinicalTrials.gov API v2 |

---

## Project Structure

```
TheraScout_Project/
├── backend/              # FastAPI app
│   ├── main.py
│   └── routers/
│       └── scan.py       # Step Functions + Bedrock scoring
├── frontend/             # React + Vite
│   └── src/App.jsx       # Card-based results UI
├── tools/                # Lambda handler functions
│   ├── query_globocan_data/
│   ├── query_pubmed/
│   ├── query_clinicaltrials/
│   ├── query_openfda/
│   ├── score_opportunities_fallback/
│   └── compose_pdf_report/
├── infra/                # AWS CDK stacks
│   └── stacks/
│       ├── data_stack.py
│       └── agent_stack.py
├── .env.example          # Copy to .env — never commit .env
└── README.md
```

---

## Disclaimer

TheraScout output is a **decision-support draft**, not a clinical prediction, regulatory guidance, or guarantee of drug success. A human researcher must review and validate all results before any resource is committed.
