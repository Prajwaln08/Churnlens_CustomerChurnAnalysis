"""Tests for Code/phase_2/orchestration/orchestrator.py - the keystone.

All offline: Ollama mocked away, artifacts absent (score stage must SKIP,
not fail). The live end-to-end with real artifacts + real Ollama is a demo,
not a unit test.
"""
import pandas as pd
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
                  "eda_report", "mode", "persist", "narrate"):
        assert stages[stage]["status"] == rm.DONE, stage
    assert m.data["reports"]["narrative"]["engine"] == "template"  # LLM mocked off
    # fuzzy mapping leaves churned_flag unmapped -> explore -> no train, no score
    assert m.data["reports"]["mode_decision"]["mode"] == "explore"
    assert stages["train"]["status"] == rm.SKIPPED
    assert stages["score"]["status"] == rm.SKIPPED

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


def test_train_mode_full_path(labeled_frame, registry):
    """Labeled data with a new schema: mode=TRAIN, fresh artifacts pass the
    gate, and the data is scored with its OWN model."""
    m = run_pipeline(labeled_frame, "credit-union", autonomy_level=1,
                     registry=registry, raise_on_error=True)

    assert m.status == rm.RUN_COMPLETED
    assert m.data["reports"]["mode_decision"]["mode"] == "train"
    stages = m.data["stages"]
    assert stages["train"]["status"] == rm.DONE
    assert stages["score"]["status"] == rm.DONE

    training = m.data["reports"]["training"]
    assert training["lenses"]["A"]["status"] == "promoted"
    assert "close_reason" in training["leakage"]["dropped"]

    assert (config.MODEL_DIR / "credit-union" / "churn_pipeline.joblib").exists()
    assert (m.run_dir / "scored.parquet").exists()
    assert any("freshly trained model" in line for line in m.ledger_lines())

    # drift: baseline established at training, checked on the spot (PSI ~ 0)
    assert stages["drift"]["status"] == rm.DONE
    assert m.data["reports"]["drift"]["alerts"] == []
    assert (config.MODEL_DIR / "credit-union" / "drift_baseline.joblib").exists()

    # run 2 for the same source: per-dataset artifacts exist -> SCORE mode
    m2 = run_pipeline(labeled_frame.sample(150, random_state=2), "credit-union",
                      autonomy_level=1, registry=registry, raise_on_error=True)
    assert m2.data["reports"]["mode_decision"]["mode"] == "score"
    assert m2.data["stages"]["train"]["status"] == rm.SKIPPED
    assert any("dataset 'credit-union' model" in line for line in m2.ledger_lines())

    # run 3: DRIFTED data through score mode -> retraining recommended
    drifted = labeled_frame.assign(spend=labeled_frame["spend"] * 4 + 500)
    m3 = run_pipeline(drifted, "credit-union", autonomy_level=1,
                      registry=registry, raise_on_error=True)
    drift_rep = m3.data["reports"]["drift"]
    assert "spend" in drift_rep["alerts"]
    assert any("retraining recommended" in line for line in m3.ledger_lines())


def test_train_mode_with_datetime_column(labeled_frame, registry):
    """Regression (LilliaCare demo): a datetime column becomes a '<col>_days'
    training feature - the score stage must replay that derivation."""
    with_date = labeled_frame.assign(
        InvoiceDate=pd.Series(pd.date_range("2023-01-01",
                                            periods=len(labeled_frame)))
        .dt.strftime("%Y-%m-%d"))
    m = run_pipeline(with_date, "dated-source", autonomy_level=1,
                     registry=registry, raise_on_error=True)
    assert m.status == rm.RUN_COMPLETED
    assert m.data["stages"]["score"]["status"] == rm.DONE
    # derived features must not be falsely reported as 'missing' drift
    assert m.data["reports"]["drift"]["alerts"] == []


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


def test_force_train_ignores_existing_artifacts(labeled_frame, registry):
    """A session that never trained a source must get fresh models even when
    artifacts exist on disk (another session's, or leftovers from a crash)."""
    m1 = run_pipeline(labeled_frame, "retrain-me", autonomy_level=1,
                      registry=registry, raise_on_error=True)
    assert m1.data["reports"]["mode_decision"]["mode"] == "train"

    m2 = run_pipeline(labeled_frame, "retrain-me", autonomy_level=1,
                      registry=registry, raise_on_error=True)
    assert m2.data["reports"]["mode_decision"]["mode"] == "score"
    assert m2.data["stages"]["train"]["status"] == rm.SKIPPED

    m3 = run_pipeline(labeled_frame, "retrain-me", autonomy_level=1,
                      registry=registry, raise_on_error=True, force_train=True)
    assert m3.data["reports"]["mode_decision"]["mode"] == "train"
    assert m3.data["stages"]["train"]["status"] == rm.DONE
    assert any("fresh training requested" in r
               for r in m3.data["reports"]["mode_decision"]["reasons"])
