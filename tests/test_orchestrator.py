"""Tests for Code/phase_2/orchestration/orchestrator.py - the keystone.

All offline: Ollama mocked away, artifacts absent (score stage must SKIP,
not fail). The live end-to-end with real artifacts + real Ollama is a demo,
not a unit test.
"""
import pytest

from phase_1 import config
from phase_2 import mapping, run_manifest as rm
from phase_2.orchestration.orchestrator import run_pipeline
from phase_2.orchestration.registry import Registry


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")  # no artifacts
    monkeypatch.setattr(mapping, "ollama_available", lambda *a, **k: False)


@pytest.fixture()
def registry(tmp_path):
    return Registry(db_path=tmp_path / "registry.sqlite")


CONFIRMED = {
    "engine": "human",
    "mappings": [
        {"source": "cust_ref", "target": "customerID", "confidence": 0.99, "value_map": None},
        {"source": "months_active", "target": "tenure", "confidence": 0.95, "value_map": None},
        {"source": "monthly_fee", "target": "MonthlyCharges", "confidence": 0.95, "value_map": None},
        {"source": "churned_flag", "target": "Churn", "confidence": 0.95, "value_map": None},
    ],
    "quarantine": [{"source": "close_reason", "reason": "post-outcome"}],
}


def test_level1_full_run_completes_offline(bank_frame, registry):
    m = run_pipeline(bank_frame, "bank-crm", autonomy_level=1,
                     registry=registry, raise_on_error=True)

    assert m.status == rm.RUN_COMPLETED
    stages = m.data["stages"]
    for stage in ("ingest", "profile", "map", "validate", "clean",
                  "eda_report", "persist"):
        assert stages[stage]["status"] == rm.DONE, stage
    assert stages["score"]["status"] == rm.SKIPPED          # no artifacts here

    # ledger + artifacts on disk
    assert m.ledger_lines()
    assert (m.run_dir / "manifest.json").exists()
    assert (m.run_dir / "profile.csv").exists()
    assert (m.run_dir / "mapped.parquet").exists()
    assert any(str(p).endswith(".png") for p in stages["eda_report"]["artifacts"])

    # registry has the completed run and an auto-confirmed mapping
    row = registry.get_run(m.run_id)
    assert row["status"] == rm.RUN_COMPLETED
    assert registry.latest_mapping("bank-crm") is not None   # auto-pilot confirmed


def test_level0_pauses_and_saves_unconfirmed_proposal(bank_frame, registry):
    m = run_pipeline(bank_frame, "new-source", autonomy_level=0,
                     registry=registry, raise_on_error=True)
    assert m.status == rm.RUN_PAUSED
    assert m.data["stages"]["map"]["status"] == rm.PAUSED
    assert m.data["stages"]["validate"]["status"] == rm.PENDING   # never reached
    assert "mapping_proposal" in m.data["reports"]
    assert registry.latest_mapping("new-source") is None          # not confirmed
    assert registry.latest_mapping("new-source", confirmed_only=False) is not None
    assert registry.get_run(m.run_id)["status"] == rm.RUN_PAUSED


def test_level0_runs_through_with_confirmed_mapping(bank_frame, registry):
    registry.save_mapping("bank-crm", CONFIRMED, confirmed=True,
                          created_at="2026-07-14T09:00:00Z")
    m = run_pipeline(bank_frame, "bank-crm", autonomy_level=0,
                     registry=registry, raise_on_error=True)
    assert m.status == rm.RUN_COMPLETED                     # no pause: mapping known
    assert any("reusing confirmed mapping v1" in line for line in m.ledger_lines())
    assert "tenure" in m.data["reports"]["mapping"]["renamed"].values()


def test_proposal_override_resumes_a_level0_source(bank_frame, registry):
    m = run_pipeline(bank_frame, "resumed-source", autonomy_level=0,
                     registry=registry, proposal_override=CONFIRMED,
                     raise_on_error=True)
    assert m.status == rm.RUN_COMPLETED
    assert any("human-confirmed mapping" in line for line in m.ledger_lines())
    assert registry.latest_mapping("resumed-source")["engine"] == "human"


def test_quarantined_column_never_mapped(bank_frame, registry):
    m = run_pipeline(bank_frame, "bank-crm", autonomy_level=1,
                     registry=registry, raise_on_error=True)
    renamed = m.data["reports"]["mapping"]["renamed"]
    assert "close_reason" not in renamed                      # leak stays out
    assert "close_reason" in m.data["reports"]["mapping"]["quarantined"]


def test_failure_is_recorded_not_raised(bank_frame, registry, monkeypatch):
    from phase_2.orchestration import orchestrator

    def boom(*a, **k):
        raise RuntimeError("profiler exploded")
    monkeypatch.setattr(orchestrator.profiler, "profile_table", boom)

    m = run_pipeline(bank_frame, "bank-crm", registry=registry)   # default: no raise
    assert m.status == rm.RUN_FAILED
    assert "profiler exploded" in m.data["stages"]["profile"]["error"]
    assert registry.get_run(m.run_id)["status"] == rm.RUN_FAILED

    with pytest.raises(RuntimeError):
        run_pipeline(bank_frame, "bank-crm", registry=registry, raise_on_error=True)
