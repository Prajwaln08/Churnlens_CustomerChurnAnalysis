# ChurnLens - Customer Churn Analytics Suite

[![Live app](https://img.shields.io/badge/live_demo-churnlenstool.streamlit.app-brightgreen)](https://churnlenstool.streamlit.app)
[![Tests](https://img.shields.io/badge/tests-125_passing-brightgreen)](#)
[![LLM](https://img.shields.io/badge/LLM-local_Ollama_or_hosted_free_tier-blue)](#)

One tool, five questions about every customer base: **who** will leave, **when**,
**who can be saved**, **which segments** exist, and **what it costs to do nothing**.

Built as a complete data-science project in two parts: a five-lens analytics suite
proven on three public datasets (`phase_1`), and a data-agnostic production pipeline
that takes any customer dataset - file, database or API - through a staged, audited
flow to a live dashboard (`phase_2`). Prototyped in notebooks, productionized into
importable packages, served through a Streamlit app, a REST API and an MCP server.

> **Live demo:** https://churnlenstool.streamlit.app - one-click sample data, or bring any CSV; models train on what you upload.

---

## The five lenses

| Lens | Question | Method | Headline result |
|---|---|---|---|
| A - Classification | Who is likely to churn? | HistGradientBoosting (tuned), LogReg + RF baselines | ROC-AUC **0.845**; top-decile lift **2.81x** |
| B - Survival | When will they leave? | Kaplan-Meier, Cox PH, Weibull AFT | C-index **0.86**; a quarter of the base has <=2 expected months left |
| C - Uplift | Who can a campaign actually save? | T-learner (best of 3 meta-learners) | uplift@10% = **44pp** vs +5pp average effect (~9x targeting efficiency) |
| D - Segmentation | What customer groups exist? | K-Means (silhouette-selected k) + RFM | churn 18.6% vs 31.7% across Telco clusters; 4 lifecycle segments in retail |
| E - CLV | What is each customer worth? | BG/NBD + Gamma-Gamma, holdout-validated | MAE ~1 purchase on 6-month holdout; mean 12-mo CLV 1,394 GBP |

**The synthesis:** Revenue-at-Risk = P(churn) x CLV per customer -
**$3.6M** across the 7,043-customer demo base, with 31% of it concentrated
in just 500 customers.

## Results at a glance

Model finds 2.8x more churners than random in the first decile - top two deciles
capture 51% of all churners:

![Cumulative gains](assets/gains_chart.png)

Commitment is the strongest retention lever the business owns - month-to-month
customers churn at 43%, two-year contracts at 3%:

![Survival by contract](assets/survival_by_contract.png)

Non-contractual churn is visible without any label - typical retail cohort keeps
~20% of customers one month after first purchase:

![Cohort retention](assets/cohort_retention.png)

## The platform, proven on demo integrations

Every scenario below was run end to end through the real UI against the demo
datasets; the numbers are actual outputs, illustrative of the workflow.

| Scenario | What the pipeline did | Result |
|---|---|---|
| Live deployment (Streamlit Cloud, hosted free-tier LLM) | visitor clicks a sample -> LLM maps the schema -> calibrated models train -> full dashboard | test ROC-AUC 0.897 (95% CI 0.86 - 0.93), $232K revenue at risk with CI, narrative by llama-3.1-8b-instant - number-audited |
| Telco batch (300 customers) | validated columns, scored lenses A/B/D/E | $152,017 total revenue at risk; risk bands + top-10 call list |
| Raw transaction log (3,928 rows) | recognised the shape, routed to lens E only | RFM segments + 12-month CLV of 217,594 GBP |
| Foreign bank export (2,000 rows, 9 renamed columns) | local LLM proposed the schema mapping, human gate confirmed, TRAIN mode trained fresh models | 8/9 columns mapped correctly, 1 correctly left unmapped; test ROC-AUC 0.8416; id-like column quarantined as a leakage risk |
| Second upload from the same source | replayed the confirmed mapping, no LLM call | ~36s first proposal vs ~0.1s replay |
| Health-app event log (1,001 users) | engagement churn at weekly grain, sliced by condition | 91.4% churned, most common exit in week 2; retention@8 of 0.221 vs 0.098 across the condition slice |
| Campaign file (1,500 customers) | uplift scoring for targeting | mean uplift +0.039, top-decile +0.336; 510 sleeping dogs flagged (contacting them increases churn) |
| Broken file (tenure removed) | validation refused to score | every lens reports "not available in this data" with the reason - nothing fails silently |
| Drifted scoring batch | PSI against the training baseline | PSI 3.88 alert with "retraining recommended" |

## What makes it a tool, not a notebook

- **Schema contracts + validation:** upload any customer file; the tool reports which
  lenses the columns support and exactly what is missing - nothing fails silently.
- **Leakage-safe by construction:** stratified split before any fitting, all
  transformers inside sklearn Pipelines, fitted preprocessors reused at scoring time
  (never refit), survival covariates screened for duration leakage, id-like and
  post-outcome columns quarantined before training.
- **Cost-aware decisions:** the classification threshold is chosen by a business cost
  matrix (missed churner vs retention offer), not the 0.5 default - and shown to be
  stable across cost assumptions.
- **Falsifiable CLV:** BG/NBD fitted on an 18-month calibration window and validated
  against what customers actually did in the final 180 days.
- **Self-bootstrapping deployment:** a fresh install trains its demo models from the
  public Telco dataset on first boot - no artifacts needed in git.
- **Statistically honest:** served probabilities are calibrated (Brier +
  bin table reported), headline metrics ship with bootstrap 95% CIs, datasets
  with a date column are validated train-on-past / test-on-future, and risk
  bands pivot on a cost-optimal threshold instead of 0.5.
- **Closes the loop:** upload realized outcomes later and every run gets
  graded (realized AUC vs training AUC -> "retrain recommended"); each
  trained artifact ships a model card; a fairness audit reports score gaps
  across user-flagged sensitive groups.
- **Tested:** 125 tests cover profiling, mapping, orchestration, training, drift,
  analytics, connectors and both serving doors; they run offline in CI on every push.

## The app - four steps

1. **What-if** - a single-customer form scored against demo models trained on the
   public IBM Telco dataset; a test account to try the tool before integrating data.
2. **Add your data** - bring a file, database table or REST API (or use the
   one-click synthetic samples if no data is handy). The pipeline
   profiles it, the local LLM proposes a mapping to the canonical schema, and an
   autonomy gate decides whether a human reviews it first (Level 0) or the run
   proceeds unattended (Level 1).
3. **Process** - the run registry: every stage with status, timing, row counts and
   a plain-English decision ledger, plus the audited LLM narrative of the run.
4. **Dashboard** - unlocked once a run completes. Population analytics per lens
   (A to E), each with its own KPIs, charts, drill tables, narration and lineage,
   plus a what-if section that scores hypothetical customers on the integrated
   dataset's own model with per-profile driver explanations. Lens C trains a
   campaign-specific uplift model on the fly when a connected campaign file
   does not fit the demo artifact.

## Quickstart

```bash
# environment
conda create -n customerchurn python=3.11 -y
conda activate customerchurn
pip install -r requirements.txt

# the app (first boot auto-trains demo models, ~1 min)
streamlit run app/streamlit_app.py
# the live demo runs this exact repo on Streamlit Community Cloud (free),
# with Groq's free tier as the hosted LLM engine (see LLM engine below)

# batch scoring from the CLI
cd Code
python -m phase_1.serving.pipeline score --input your_customers.csv

# retrain on new labeled data
python -m phase_1.serving.pipeline train --data labeled.csv

# REST serving door
uvicorn phase_2.api.rest:app

# MCP server (AI agents operate the pipeline conversationally)
python -m phase_2.api.mcp_server

# run the test suite
python -m pytest tests/ -q   # from the project root
```

Full analysis: run the notebooks in order (`Code/01_...` to `Code/05_...`).
Notebooks 01-02 need the datasets in `Dataset/` (sources below); the MegaFon uplift
dataset downloads itself on first use.

**LLM engine (pluggable):** schema mapping and run narration use a local
Ollama model (`qwen2.5:7b-instruct`) when one is running - data never leaves
the machine. Without Ollama, any OpenAI-compatible endpoint configured via
`CHURNLENS_LLM_BASE_URL` / `CHURNLENS_LLM_API_KEY` / `CHURNLENS_LLM_MODEL`
takes over (the live demo uses Groq's free tier; only column names, a few
sample values and computed run facts are ever sent - never raw datasets).
With neither, the pipeline falls back to deterministic fuzzy matching and
template narration. LLM footprint is tiny either way: two calls per run,
roughly 1-3K tokens. Every narrative is number-audited against computed
facts regardless of the engine.

## Repository structure

```
Code/
  01_data_loading.ipynb          # the lab: explore, hit data landmines, fix them
  02_eda_visualization.ipynb     # EDA for all three datasets
  03_feature_engineering_modeling.ipynb   # all five lenses trained
  04_result_analysis.ipynb       # thresholds, lift, error analysis, revenue-at-risk
  05_pipeline_tool.ipynb         # the tool proven on good, broken and foreign data
  phase_1/                       # the local analytics suite (promoted from notebooks)
    config.py
    loader/          data_loader.py
    preprocessing/   features.py
    visualization/   eda.py
    modeling/        models.py, bootstrap.py
    evaluation/      evaluation.py
    serving/         pipeline.py (contracts, validation, scoring, CLI)
  phase_2/                       # the production pipeline (any dataset, staged runs)
    ingestion/       connectors.py (file / database / REST API, watermarks)
    profiling/       profiler.py (roles, label candidates, collinearity)
    mapping/         mapping.py (local-LLM proposals, fuzzy fallback, value maps)
    orchestration/   orchestrator.py, run_manifest.py, registry.py
    training/        auto_train.py, leakage.py, mode.py
    monitoring/      drift.py (PSI vs training baseline)
    analytics/       insights.py, journeys.py (population + individual level)
    narration/       narrator.py (grounded summaries, number audit)
    api/             rest.py (FastAPI), mcp_server.py (MCP door)
tests/                           # 125 tests, offline by design, run in CI
app/streamlit_app.py             # the ChurnLens UI (four-step flow above)
assets/                          # README charts
docs/                            # local design and planning notes (not committed)
requirements.txt
```

Workflow convention: every function was prototyped in a notebook first, then promoted
to a package and imported from there - notebooks tell the story, packages ship it.

## Data

| Dataset | Role | Source |
|---|---|---|
| IBM Telco Customer Churn (7,043 x 21) | lenses A, B, D | public IBM sample |
| Online Retail II (~1.07M transactions) | lenses D, E | UCI ML Repository (id 502) |
| MegaFon Uplift (~600K, randomized) | lens C | `sklift.datasets.fetch_megafon` |

Datasets and trained artifacts are intentionally not committed; loaders and the
bootstrap rebuild everything. All results are illustrative of the workflow and
calibrated to these public datasets - production use requires retraining on
business-specific data (`python -m phase_1.serving.pipeline train`).

## The production pipeline (phase_2) - any dataset, end to end

`Code/phase_2/` is a data-agnostic churn platform that takes ANY customer dataset
and runs a staged, fully-audited pipeline:

```
ingest -> profile -> map -> validate -> clean -> EDA -> mode -> train -> score -> drift -> persist -> narrate
```

- **Connectors** (files / SQL databases / REST APIs) all end at one internal shape;
  credentials are redacted, predicate pushdown and incremental watermarks supported.
- **Local-LLM schema mapping** (Ollama, qwen2.5:7b) - foreign column names and value
  levels are mapped to the canonical schema; proposals are confidence-gated,
  human-confirmable, saved per source and reused forever. Data never leaves the machine.
- **Mode decision** per dataset - score with existing models, TRAIN fresh ones on all
  surviving columns (leakage-screened, quality-gated: no model ships below its metric
  bar), or explore label-free.
- **Full observability** - every run writes a JSON manifest (stage checklist, timings,
  a plain-English decision ledger) into a SQLite registry, rendered live in the app's
  Process tab.
- **Drift monitoring** - PSI against the training baseline; crossing the alert
  threshold surfaces "retraining recommended".
- **Grounded LLM narration** - run summaries written by the local model with a number
  audit: any figure not present in the computed facts rejects the text entirely.
- **Analytics at two levels** - population (drivers, segments, lift, retention curves,
  revenue at risk) and individual (local driver deltas, journey outlook, expected
  remaining tenure per customer).
- **Three doors, one engine**: the Streamlit app, a FastAPI service
  (`uvicorn phase_2.api.rest:app`), and an MCP server so AI agents can operate the
  pipeline conversationally (`python -m phase_2.api.mcp_server`).
