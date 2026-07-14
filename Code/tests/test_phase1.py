"""Phase 1 tests: profiler + mapping layer.

Run from Code/:  python -m pytest tests/ -q
Fixtures are synthetic (no datasets, no network); the Ollama path is exercised
with the server mocked offline - the live-LLM check is a separate demo script.
"""
import numpy as np
import pandas as pd
import pytest

from src import mapping, profiler


@pytest.fixture()
def bank_frame() -> pd.DataFrame:
    """A 'foreign' bank-style dataset: renamed columns, one post-outcome leak,
    numbers-as-text, a date column - the adversarial zoo in one frame."""
    rng = np.random.default_rng(7)
    n = 80
    return pd.DataFrame({
        "cust_ref": [f"C{i:05d}" for i in range(n)],
        "agreement_type": rng.choice(["monthly", "annual", "two-year"], n),
        "months_active": rng.integers(0, 72, n),
        "monthly_fee": rng.uniform(20, 110, n).round(2),
        "total_billed": [str(round(v, 2)) for v in rng.uniform(0, 5000, n)],  # text costume
        "signup_date": pd.Series(pd.date_range("2020-01-01", periods=n)).dt.strftime("%Y-%m-%d"),
        "notes": [f"long free text describing customer interaction history, case {i}"
                  for i in range(n)],
        "one_value": ["same"] * n,
        "close_reason": rng.choice(["", "price", "moved"], n),
        "churned_flag": rng.integers(0, 2, n),
    })


# ---------------------------------------------------------------- profiler

def test_profiler_roles(bank_frame):
    prof = profiler.profile_table(bank_frame)
    roles = {c: p["role"] for c, p in prof["columns"].items()}
    assert roles["cust_ref"] == profiler.ID
    assert roles["agreement_type"] == profiler.CATEGORICAL
    assert roles["months_active"] == profiler.NUMERIC
    assert roles["total_billed"] == profiler.NUMERIC      # parseable text -> numeric
    assert roles["signup_date"] == profiler.DATETIME
    assert roles["notes"] == profiler.TEXT
    assert roles["one_value"] == profiler.CONSTANT
    assert roles["churned_flag"] == profiler.BINARY


def test_label_and_leak_flags(bank_frame):
    prof = profiler.profile_table(bank_frame)
    assert profiler.label_candidates(prof)[0] == "churned_flag"
    assert prof["columns"]["close_reason"]["post_outcome_suspect"]
    assert not prof["columns"]["monthly_fee"]["post_outcome_suspect"]


def test_samples_are_masked():
    df = pd.DataFrame({"x": ["a" * 100] * 3})
    prof = profiler.profile_table(df)
    assert all(len(s) <= 17 for s in prof["columns"]["x"]["samples"])


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


def test_apply_mapping_collision_keeps_higher_confidence(bank_frame):
    prop = _proposal(mappings=[
        {"source": "monthly_fee", "target": "MonthlyCharges", "confidence": 0.9, "value_map": None},
        {"source": "total_billed", "target": "MonthlyCharges", "confidence": 0.7, "value_map": None},
    ], quarantine=[])
    out, report = mapping.apply_mapping(bank_frame, prop)
    assert report["renamed"] == {"monthly_fee": "MonthlyCharges"}
    assert report["collisions_skipped"] == ["total_billed"]
