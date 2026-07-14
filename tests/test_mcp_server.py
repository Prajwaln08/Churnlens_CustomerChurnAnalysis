"""Tests for Code/phase_2/api/mcp_server.py - the agent door.

The tools are called directly (they are plain functions after registration);
what matters is that they wrap the same gated engine and register under the
right names.
"""
import pytest

pytest.importorskip("mcp")

from phase_1 import config
from phase_2 import mapping
from phase_2.api import mcp_server


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REGISTRY_DB", tmp_path / "registry.sqlite")
    monkeypatch.setattr(mapping, "ollama_available", lambda *a, **k: False)


@pytest.fixture()
def csv_path(labeled_frame, tmp_path):
    path = tmp_path / "api-source.csv"
    labeled_frame.to_csv(path, index=False)
    return str(path)


def test_all_tools_registered():
    import anyio

    async def _names():
        return {t.name for t in await mcp_server.mcp.list_tools()}
    names = anyio.run(_names)
    assert {"validate_data", "run_pipeline", "score_customers",
            "list_runs", "explain_run"} <= names


def test_validate_data_tool(csv_path):
    out = mcp_server.validate_data(csv_path)
    assert out["rows"] == 400
    assert out["columns"]["tenure"] == "numeric"
    assert "Churn" in out["label_candidates"] or out["label_candidates"]


def test_full_agent_journey(csv_path):
    run_out = mcp_server.run_pipeline(csv_path, "agent-source", autonomy_level=1)
    assert run_out["status"] == "completed" and run_out["mode"] == "train"
    assert run_out["summary"]                      # narrated, template engine

    score_out = mcp_server.score_customers(csv_path, "agent-source")
    assert score_out["rows_scored"] == 400
    assert sum(score_out["risk_bands"].values()) == 400
    assert len(score_out["riskiest_5"]) == 5

    runs = mcp_server.list_runs()
    assert any(r["run_id"] == run_out["run_id"] for r in runs)

    detail = mcp_server.explain_run(run_out["run_id"])
    assert detail["stages"]["train"] == "done"
    assert detail["mode_decision"]["mode"] == "train"


def test_score_before_train_returns_named_error(csv_path):
    out = mcp_server.score_customers(csv_path, "never-trained")
    assert "no trained artifacts" in out["error"]
