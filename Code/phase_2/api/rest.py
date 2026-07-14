"""The REST door - system-to-system scoring over the same pipeline engine.

Origin: architecture.md section 5 (Phase 7 of the build order).

Same engine, second door: every endpoint calls the exact functions the
Streamlit UI and CLI call - the API can never disagree with them.

Run (from Code/):
    ../.venv/bin/uvicorn phase_2.api.rest:app --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Code/ importable

from phase_1.serving import pipeline  # noqa: E402
from phase_2.orchestration import Registry, RunManifest, run_pipeline  # noqa: E402
from phase_2.training import auto_train, mode as mode_mod  # noqa: E402


class Records(BaseModel):
    records: list[dict] = Field(min_length=1)
    source: str | None = None


class RunRequest(BaseModel):
    records: list[dict] = Field(min_length=1)
    source: str
    autonomy_level: int = 1


def create_app() -> FastAPI:
    app = FastAPI(title="ChurnLens API",
                  description="Who leaves - when - who can be saved - "
                              "which segment - what it costs.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok",
                "global_artifacts": pipeline.artifact_exists("churn_pipeline"),
                "runs_recorded": int(len(Registry().all_runs(limit=1000)))}

    @app.post("/validate")
    def validate(payload: Records) -> dict:
        df = pd.DataFrame(payload.records)
        report = pipeline.validate(df)
        return {"lenses": {k: v["message"] for k, v in report["lenses"].items()},
                "ignored_columns": report["ignored_columns"]}

    @app.post("/score")
    def score(payload: Records) -> dict:
        df = pd.DataFrame(payload.records)
        if payload.source:
            dataset_id = mode_mod.dataset_id_for(payload.source)
            if pipeline.artifact_exists("churn_pipeline", dataset_id=dataset_id):
                try:
                    scored = auto_train.score_for_source(df, payload.source)
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
                return {"engine": f"dataset:{dataset_id}",
                        "records": scored.to_dict("records")}
            raise HTTPException(404, f"no trained artifacts for source "
                                     f"'{payload.source}' - POST /run to train")
        scored, report = pipeline.score_any(df)
        return {"engine": "global", "applied": report["applied"],
                "errors": report.get("errors", {}),
                "records": scored.to_dict("records")}

    @app.post("/run")
    def run(payload: RunRequest) -> dict:
        df = pd.DataFrame(payload.records)
        manifest = run_pipeline(df, payload.source,
                                autonomy_level=payload.autonomy_level)
        return {"run_id": manifest.run_id, "status": manifest.status,
                "mode": (manifest.data["reports"].get("mode_decision") or {})
                        .get("mode"),
                "ledger": manifest.ledger_lines()}

    @app.get("/runs")
    def runs(limit: int = 20) -> list[dict]:
        return Registry().all_runs(limit=limit).to_dict("records")

    @app.get("/runs/{run_id}")
    def run_detail(run_id: str) -> dict:
        try:
            manifest = RunManifest.load(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, f"unknown run '{run_id}'") from exc
        return manifest.data

    return app


app = create_app()
