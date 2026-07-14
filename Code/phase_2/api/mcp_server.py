"""The agent door - ChurnLens as MCP tools.

Origin: architecture.md sections 6 and 19 (Phase 7, the final layer).

"MCP for the conversation, the pipeline for the truth": every tool below is a
thin wrapper over the SAME gated engine the UI/CLI/REST use. The agent can
never bypass a quality gate, an eligibility check, or the leakage screen -
it has no special powers, only a nicer voice.

Run standalone (from Code/):
    ../.venv/bin/python -m phase_2.api.mcp_server

Claude Desktop config (~/Library/Application Support/Claude/claude_desktop_config.json):
    {"mcpServers": {"churnlens": {
        "command": "<project>/.venv/bin/python",
        "args": ["-m", "phase_2.api.mcp_server"],
        "cwd": "<project>/Code"}}}
"""
from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Code/ importable

from phase_1.serving import pipeline  # noqa: E402
from phase_2.ingestion import connectors  # noqa: E402
from phase_2.orchestration import Registry, RunManifest  # noqa: E402
from phase_2.orchestration import run_pipeline as _run_pipeline  # noqa: E402
from phase_2.profiling import profiler  # noqa: E402
from phase_2.training import auto_train, mode as mode_mod  # noqa: E402

mcp = FastMCP("ChurnLens")


@mcp.tool()
def validate_data(file_path: str) -> dict:
    """Profile a customer data file (csv/xlsx/parquet/json) and report which
    churn-analytics lenses its columns support, plus label candidates."""
    src = connectors.from_file(file_path)
    profile = profiler.profile_table(src.frame)
    report = pipeline.validate(src.frame)
    return {
        "rows": src.rows,
        "columns": {c: p["role"] for c, p in profile["columns"].items()},
        "label_candidates": profiler.label_candidates(profile),
        "lenses": {k: v["message"] for k, v in report["lenses"].items()},
    }


@mcp.tool()
def run_pipeline(file_path: str, source_name: str, autonomy_level: int = 1) -> dict:
    """Run a data file through the full staged pipeline (profile -> map ->
    validate -> clean -> EDA -> mode decision -> train/score -> drift ->
    narrate). Returns the run id, status and the plain-English ledger."""
    src = connectors.from_file(file_path)
    manifest = _run_pipeline(src, source_name, autonomy_level=autonomy_level)
    reports = manifest.data["reports"]
    return {
        "run_id": manifest.run_id,
        "status": manifest.status,
        "mode": (reports.get("mode_decision") or {}).get("mode"),
        "summary": (reports.get("narrative") or {}).get("summary"),
        "ledger": manifest.ledger_lines(),
    }


@mcp.tool()
def score_customers(file_path: str, source_name: str) -> dict:
    """Score a file with a source's own trained model (train it first with
    run_pipeline if needed). Returns risk-band counts and the riskiest rows."""
    dataset_id = mode_mod.dataset_id_for(source_name)
    if not pipeline.artifact_exists("churn_pipeline", dataset_id=dataset_id):
        return {"error": f"no trained artifacts for '{source_name}' - "
                         "call run_pipeline first"}
    src = connectors.from_file(file_path)
    scored = auto_train.score_for_source(src.frame, source_name)
    top = scored.nlargest(5, "churn_probability")
    return {
        "rows_scored": len(scored),
        "risk_bands": scored["risk_band"].value_counts().to_dict(),
        "mean_churn_probability": round(float(scored["churn_probability"].mean()), 4),
        "riskiest_5": top.to_dict("records"),
    }


@mcp.tool()
def list_runs(limit: int = 10) -> list[dict]:
    """The most recent pipeline runs: id, source, status, rows, timestamps."""
    return Registry().all_runs(limit=limit).to_dict("records")


@mcp.tool()
def explain_run(run_id: str) -> dict:
    """Everything about one run: the stage checklist, the decision ledger,
    the mode decision, drift verdict and the narrated summary."""
    manifest = RunManifest.load(run_id)
    reports = manifest.data["reports"]
    return {
        "status": manifest.status,
        "source": manifest.data["source"],
        "stages": {name: s["status"]
                   for name, s in manifest.data["stages"].items()},
        "ledger": manifest.ledger_lines(),
        "mode_decision": reports.get("mode_decision"),
        "drift": reports.get("drift"),
        "summary": (reports.get("narrative") or {}).get("summary"),
    }


if __name__ == "__main__":
    mcp.run()          # stdio transport - what Claude Desktop expects
