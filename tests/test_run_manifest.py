"""Tests for Code/phase_2/run_manifest.py - the state machine must never lie."""
import json

import pytest

from phase_1 import config
from phase_2 import run_manifest as rm


@pytest.fixture(autouse=True)
def isolated_runs_dir(tmp_path, monkeypatch):
    """Every test writes manifests into its own temp dir, never runs/."""
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)


def test_happy_path_lifecycle():
    m = rm.RunManifest("bank-crm", autonomy_level=1)
    with m.stage("ingest"):
        m.record("ingest", rows_in=100, rows_out=100)
        m.add_note("ingest", "read 100 rows from upload")
    with m.stage("profile"):
        pass
    m.finish()

    assert m.status == rm.RUN_COMPLETED
    stages = m.data["stages"]
    assert stages["ingest"]["status"] == rm.DONE
    assert stages["ingest"]["rows_in"] == 100
    assert stages["ingest"]["elapsed_s"] is not None
    assert stages["profile"]["status"] == rm.DONE
    assert stages["map"]["status"] == rm.PENDING          # untouched stays pending
    assert m.data["finished_at"] is not None


def test_failure_is_recorded_and_propagates():
    m = rm.RunManifest("bank-crm")
    with pytest.raises(ValueError, match="boom"):
        with m.stage("map"):
            raise ValueError("boom")
    assert m.data["stages"]["map"]["status"] == rm.FAILED
    assert "ValueError: boom" in m.data["stages"]["map"]["error"]
    assert m.status == rm.RUN_FAILED
    # the failure was persisted before the exception escaped
    reloaded = rm.RunManifest.load(m.run_id)
    assert reloaded.data["stages"]["map"]["status"] == rm.FAILED


def test_pause_for_confirmation_gate():
    m = rm.RunManifest("bank-crm", autonomy_level=0)
    with m.stage("map"):
        m.pause("map", "mapping needs human confirmation (Level 0)")
    assert m.data["stages"]["map"]["status"] == rm.PAUSED   # not overwritten to DONE
    assert m.status == rm.RUN_PAUSED


def test_skip_records_reason():
    m = rm.RunManifest("bank-crm")
    m.skip("score", "no artifacts trained for this schema yet")
    assert m.data["stages"]["score"]["status"] == rm.SKIPPED
    assert "no artifacts" in m.data["stages"]["score"]["notes"][0]


def test_unknown_stage_is_a_hard_error():
    m = rm.RunManifest("bank-crm")
    with pytest.raises(KeyError):
        m.record("scorre", rows_in=1)          # typo must not create a stage
    with pytest.raises(ValueError):
        m.record("score", notes=["not like this"])


def test_save_load_roundtrip_and_valid_json():
    m = rm.RunManifest("bank-crm")
    m.attach_report("mapping", {"engine": "fuzzy", "renamed": {"a": "b"}})
    m.add_artifact("eda_report", "runs/x/eda.png")
    m.finish()
    raw = json.loads(m.path.read_text())        # valid JSON on disk
    assert raw["reports"]["mapping"]["engine"] == "fuzzy"
    again = rm.RunManifest.load(m.run_id)
    assert again.data == m.data


def test_views_for_process_screen():
    m = rm.RunManifest("bank-crm")
    with m.stage("ingest"):
        m.add_note("ingest", "read 50 rows")
    with m.stage("clean"):
        m.add_note("clean", "TotalCharges: 3 blanks set to 0")
    frame = m.summary_frame()
    assert list(frame["stage"]) == list(rm.STAGES)
    assert frame.loc[frame["stage"] == "ingest", "status"].iloc[0] == rm.DONE
    lines = m.ledger_lines()
    assert lines == ["[ingest] read 50 rows", "[clean] TotalCharges: 3 blanks set to 0"]


def test_run_ids_are_unique_and_slugged():
    a = rm.new_run_id("Bank CRM #1")
    b = rm.new_run_id("Bank CRM #1")
    assert a != b
    assert "bank-crm" in a
