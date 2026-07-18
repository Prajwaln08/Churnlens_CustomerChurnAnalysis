"""Tests for Code/phase_2/mapping.py - proposer, sanitizer, fallback, executor."""
import pandas as pd
import pytest

from phase_2 import mapping, profiler


# ---------------------------------------------------------------- fuzzy

def test_fuzzy_maps_obvious_columns(bank_frame):
    prof = profiler.profile_table(bank_frame)
    proposal = mapping.fuzzy_propose(prof)
    assert proposal["engine"] == "fuzzy"
    targets = {m["source"]: m["target"] for m in proposal["mappings"]}
    assert targets["cust_ref"] == "customerID"
    assert targets["monthly_fee"] == "MonthlyCharges"
    assert targets["one_value"] is None or targets["one_value"] != "Churn"


def test_fallback_when_ollama_offline(bank_frame, monkeypatch):
    monkeypatch.setattr(mapping, "ollama_available", lambda *a, **k: False)
    proposal = mapping.propose_mapping(bank_frame)
    assert proposal["engine"] == "fuzzy"
    assert proposal["label_candidates"][0] == "churned_flag"
    assert any(q["source"] == "close_reason" for q in proposal["quarantine"])


def test_fallback_when_ollama_errors(bank_frame, monkeypatch):
    monkeypatch.setattr(mapping, "ollama_available", lambda *a, **k: True)
    def boom(*a, **k):
        raise mapping.MappingError("garbage")
    monkeypatch.setattr(mapping, "ollama_propose", boom)
    proposal = mapping.propose_mapping(bank_frame)
    assert proposal["engine"] == "fuzzy"


# ---------------------------------------------------------------- sanitizer

def test_sanitize_rejects_garbage(bank_frame):
    prof = profiler.profile_table(bank_frame)
    raw = {"mappings": [
        {"source": "cust_ref", "target": "customerID", "confidence": 4.2},   # clamp
        {"source": "invented_col", "target": "tenure", "confidence": 0.9},   # fake source
        {"source": "months_active", "target": "NotAField", "confidence": 0.9},  # fake target
        {"source": "agreement_type", "target": "Contract", "confidence": 0.9,
         "value_map": {"monthly": "Month-to-month", "annual": "One year",
                       "two-year": "Bogus level"}},                          # bad level
    ]}
    out = mapping._sanitize(raw, prof)
    by_src = {m["source"]: m for m in out["mappings"]}
    assert by_src["cust_ref"]["confidence"] == 1.0
    assert "invented_col" not in by_src
    assert by_src["months_active"]["target"] is None
    vm = by_src["agreement_type"]["value_map"]
    assert vm == {"monthly": "Month-to-month", "annual": "One year"}
    assert set(by_src) == set(prof["columns"])          # every real column present


def test_sanitize_raises_on_zero_mappings(bank_frame):
    prof = profiler.profile_table(bank_frame)
    with pytest.raises(mapping.MappingError):
        mapping._sanitize({"mappings": []}, prof)


# ---------------------------------------------------------------- executor

def _proposal(**overrides):
    base = {
        "engine": "test",
        "mappings": [
            {"source": "cust_ref", "target": "customerID", "confidence": 0.95, "value_map": None},
            {"source": "agreement_type", "target": "Contract", "confidence": 0.9,
             "value_map": {"monthly": "Month-to-month", "annual": "One year",
                           "two-year": "Two year"}},
            {"source": "months_active", "target": "tenure", "confidence": 0.4, "value_map": None},
            {"source": "close_reason", "target": "Churn", "confidence": 0.99, "value_map": None},
        ],
        "quarantine": [{"source": "close_reason", "reason": "post-outcome"}],
    }
    base.update(overrides)
    return base


def test_apply_mapping_renames_translates_and_guards(bank_frame):
    out, report = mapping.apply_mapping(bank_frame, _proposal(), min_confidence=0.6)
    assert "customerID" in out.columns and "cust_ref" not in out.columns
    assert set(out["Contract"].unique()) <= {"Month-to-month", "One year", "Two year"}
    assert report["values_translated"]["Contract"] == len(bank_frame)
    assert "months_active" in report["below_threshold"]      # 0.4 < 0.6 -> not renamed
    assert "months_active" in out.columns
    assert "close_reason" in report["quarantined"]            # never applied
    assert "Churn" not in out.columns
    assert bank_frame["agreement_type"].iloc[0] in {"monthly", "annual", "two-year"}  # input untouched


def test_apply_mapping_tolerates_absent_source_columns(bank_frame):
    """A reused mapping may reference the label column, which scoring data
    legitimately lacks - must skip and report, never crash."""
    no_label = bank_frame.drop(columns=["close_reason"])
    prop = _proposal(mappings=[
        {"source": "cust_ref", "target": "customerID", "confidence": 0.95, "value_map": None},
        {"source": "close_reason", "target": "Churn", "confidence": 0.99,
         "value_map": {"1": "Yes", "0": "No"}},
    ], quarantine=[])
    out, report = mapping.apply_mapping(no_label, prop)
    assert "customerID" in out.columns
    assert "Churn" not in out.columns
    assert report["sources_absent"] == ["close_reason"]


def test_apply_mapping_collision_keeps_higher_confidence(bank_frame):
    prop = _proposal(mappings=[
        {"source": "monthly_fee", "target": "MonthlyCharges", "confidence": 0.9, "value_map": None},
        {"source": "total_billed", "target": "MonthlyCharges", "confidence": 0.7, "value_map": None},
    ], quarantine=[])
    out, report = mapping.apply_mapping(bank_frame, prop)
    assert report["renamed"] == {"monthly_fee": "MonthlyCharges"}
    assert report["collisions_skipped"] == ["total_billed"]


def test_apply_mapping_when_target_already_in_frame():
    """Regression: a newer export may already carry a canonical column while
    the saved mapping still renames a source onto that same name. The rename
    must be skipped (reported), never producing duplicate columns."""
    frame = pd.DataFrame({
        "churn_flag": ["Yes", "No"],
        "Churn": ["No", "Yes"],            # canonical name already present
        "months": [3, 40],
    })
    proposal = {"engine": "test", "mappings": [
        {"source": "churn_flag", "target": "Churn", "confidence": 1.0,
         "value_map": None},
        {"source": "months", "target": "tenure", "confidence": 1.0,
         "value_map": None},
    ]}
    mapped, report = mapping.apply_mapping(frame, proposal)
    assert list(mapped.columns).count("Churn") == 1
    assert "churn_flag" in report["collisions_skipped"]
    assert "churn_flag" in mapped.columns          # kept as an extra, not lost
    assert report["renamed"] == {"months": "tenure"}
