"""Tests for Code/phase_2/api/rest.py - the system-to-system door."""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from phase_1 import config
from phase_2 import mapping
from phase_2.api.rest import app


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REGISTRY_DB", tmp_path / "registry.sqlite")
    monkeypatch.setattr(mapping, "ollama_available", lambda *a, **k: False)


@pytest.fixture()
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_validate_reports_lenses(client, bank_frame):
    resp = client.post("/validate",
                       json={"records": bank_frame.to_dict("records")})
    assert resp.status_code == 200
    body = resp.json()
    assert "missing" in body["lenses"]["C"]
    assert isinstance(body["ignored_columns"], list)


def test_run_then_score_via_dataset_model(client, labeled_frame):
    records = labeled_frame.to_dict("records")
    run_resp = client.post("/run", json={"records": records,
                                         "source": "api-source",
                                         "autonomy_level": 1})
    assert run_resp.status_code == 200
    body = run_resp.json()
    assert body["status"] == "completed" and body["mode"] == "train"

    score_resp = client.post("/score", json={"records": records[:25],
                                             "source": "api-source"})
    assert score_resp.status_code == 200
    scored = score_resp.json()
    assert scored["engine"] == "dataset:api-source"
    assert len(scored["records"]) == 25
    assert 0 <= scored["records"][0]["churn_probability"] <= 1

    runs_resp = client.get("/runs")
    assert any(r["run_id"] == body["run_id"] for r in runs_resp.json())
    detail = client.get(f"/runs/{body['run_id']}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "completed"


def test_score_unknown_source_404(client, bank_frame):
    resp = client.post("/score", json={"records": bank_frame.to_dict("records"),
                                       "source": "never-trained"})
    assert resp.status_code == 404


def test_unknown_run_404(client):
    assert client.get("/runs/run_nope").status_code == 404


def test_empty_records_rejected(client):
    assert client.post("/validate", json={"records": []}).status_code == 422


def test_outcomes_endpoint_grades_a_run(client, labeled_frame):
    run = client.post("/run", json={
        "records": labeled_frame.to_dict("records"),
        "source": "outcome-src", "autonomy_level": 1})
    assert run.status_code == 200
    run_id = run.json()["run_id"]
    actuals = [{"customer_id": cid, "actual": int(ch)}
               for cid, ch in zip(labeled_frame["customerID"],
                                  labeled_frame["Churn"])]
    resp = client.post("/outcomes", json={"run_id": run_id,
                                          "records": actuals})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["matched"] >= 20
    assert 0 <= body["realized_auc"] <= 1

    missing = client.post("/outcomes", json={"run_id": "run_nope",
                                             "records": actuals[:5]})
    assert missing.status_code == 404
