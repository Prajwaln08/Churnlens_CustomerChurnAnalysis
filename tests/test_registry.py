"""Tests for Code/phase_2/orchestration/registry.py - facts about the files."""
import pytest

from phase_2 import run_manifest as rm
from phase_2.orchestration.registry import Registry


@pytest.fixture()
def registry(tmp_path):
    return Registry(db_path=tmp_path / "registry.sqlite")


def _manifest(source="bank-crm", status="running"):
    m = rm.RunManifest(source, autonomy_level=1)
    m.data["input"] = {"rows": 500}
    m.data["status"] = status
    return m


def test_schema_creation_is_idempotent(tmp_path):
    db = tmp_path / "registry.sqlite"
    Registry(db_path=db)
    reg = Registry(db_path=db)          # second init must not destroy anything
    assert reg.all_runs().empty


def test_record_run_upserts_and_tracks_source(registry):
    m = _manifest(status="running")
    registry.record_run(m.data)
    m.data["status"] = "completed"
    m.data["finished_at"] = "2026-07-14T10:00:00Z"
    registry.record_run(m.data)          # upsert, not duplicate

    runs = registry.runs_for_source("bank-crm")
    assert len(runs) == 1
    assert runs.iloc[0]["status"] == "completed"
    assert runs.iloc[0]["rows"] == 500

    sources = registry.sources()
    assert list(sources["source_name"]) == ["bank-crm"]
    assert sources.iloc[0]["last_run_id"] == m.run_id


def test_runs_ordering_and_limit(registry):
    for i in range(3):
        m = _manifest(status="completed")
        m.data["created_at"] = f"2026-07-1{i}T00:00:00Z"
        registry.record_run(m.data)
    runs = registry.all_runs(limit=2)
    assert len(runs) == 2
    assert runs.iloc[0]["created_at"] > runs.iloc[1]["created_at"]  # newest first


def test_get_run_roundtrip(registry):
    m = _manifest()
    registry.record_run(m.data)
    row = registry.get_run(m.run_id)
    assert row["source_name"] == "bank-crm"
    assert registry.get_run("run_does_not_exist") is None


def test_mapping_versions_and_latest(registry):
    proposal = {"engine": "ollama:qwen2.5:7b-instruct",
                "mappings": [{"source": "cust_ref", "target": "customerID",
                              "confidence": 1.0, "value_map": None}],
                "profile": {"huge": "blob that must not be stored"}}
    v1 = registry.save_mapping("bank-crm", proposal, confirmed=False,
                               created_at="2026-07-14T09:00:00Z")
    v2 = registry.save_mapping("bank-crm", proposal, confirmed=True,
                               created_at="2026-07-14T10:00:00Z")
    assert (v1, v2) == (1, 2)

    latest = registry.latest_mapping("bank-crm")            # confirmed only
    assert latest["version"] == 2
    assert "profile" not in latest                          # bulk stripped
    assert latest["mappings"][0]["target"] == "customerID"

    any_latest = registry.latest_mapping("bank-crm", confirmed_only=False)
    assert any_latest["version"] == 2
    assert registry.latest_mapping("never-seen-source") is None


def test_unconfirmed_mapping_is_not_reused(registry):
    proposal = {"engine": "fuzzy", "mappings": []}
    registry.save_mapping("shop-data", proposal, confirmed=False,
                          created_at="2026-07-14T09:00:00Z")
    assert registry.latest_mapping("shop-data") is None     # nothing confirmed yet


def test_outcomes_roundtrip(registry):
    import pandas as pd
    frame = pd.DataFrame({"customer_id": ["a", "b"], "predicted": [0.8, 0.2],
                          "actual": [1, 0]})
    n = registry.record_outcomes("run_x", frame)
    assert n == 2
    back = registry.outcomes_for("run_x")
    assert len(back) == 2
    assert set(back["customer_id"]) == {"a", "b"}
    # upsert: re-recording overwrites, never duplicates
    registry.record_outcomes("run_x", frame.assign(actual=[0, 1]))
    back2 = registry.outcomes_for("run_x")
    assert len(back2) == 2
    assert set(back2["actual"]) == {0, 1}
