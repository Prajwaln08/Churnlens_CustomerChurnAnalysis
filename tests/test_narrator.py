"""Tests for Code/phase_2/narration/narrator.py - words yes, numbers never."""
import pandas as pd
import pytest

from phase_2 import narrator


@pytest.fixture()
def manifest_data():
    return {
        "source": "credit-union",
        "status": "completed",
        "input": {"rows": 7043, "origin": "db:sqlite:///crm.db#members"},
        "reports": {
            "mode_decision": {"mode": "train", "dataset_id": "credit-union",
                              "reasons": ["label 'Churn' mapped, 7,043 rows, "
                                          "no artifacts - training fresh models"]},
            "mapping": {"engine": "ollama:qwen2.5:7b-instruct",
                        "renamed": {"member_no": "customerID", "left_us": "Churn"},
                        "quarantined": ["close_reason"]},
            "training": {"lenses": {"A": {"status": "promoted",
                                          "test_roc_auc": 0.833},
                                    "B": {"status": "promoted", "c_index": 0.84}},
                         "leakage": {"kept": list("abcdefg"),
                                     "dropped": {"customerID": "id-like"}}},
        },
    }


@pytest.fixture()
def scored():
    return pd.DataFrame({"risk_band": ["Low"] * 50 + ["High"] * 30 + ["Critical"] * 20,
                         "revenue_at_risk": [100.0] * 100})


def test_facts_are_computed_not_guessed(manifest_data, scored):
    facts = narrator.facts_from_run(manifest_data, scored)
    assert facts["rows"] == 7043
    assert facts["mode"] == "train"
    assert facts["columns_mapped"] == 2
    assert facts["training"]["A"]["test_roc_auc"] == 0.833
    assert facts["high_or_critical"] == 50
    assert facts["total_revenue_at_risk"] == 10000


def test_template_contains_the_key_numbers(manifest_data, scored):
    facts = narrator.facts_from_run(manifest_data, scored)
    text = narrator.template_summary(facts)
    assert "7,043" in text and "TRAIN" in text
    assert "0.833" in text and "promoted" in text
    assert "$10,000" in text


def test_audit_flags_invented_numbers(manifest_data):
    facts = narrator.facts_from_run(manifest_data)
    ok_text = "The run mapped 7,043 rows and lens A reached 0.833 (83.3%)."
    assert narrator.audit_numbers(ok_text, facts) == []
    bad_text = "The model reached 99.9% accuracy on 7,043 rows."
    assert "99.9" in narrator.audit_numbers(bad_text, facts)
    prose_text = "All three lenses ran in one pass."      # small numbers exempt
    assert narrator.audit_numbers(prose_text, facts) == []


def test_narrate_falls_back_when_ollama_offline(manifest_data, monkeypatch):
    monkeypatch.setattr(narrator.llm, "ollama_available", lambda *a, **k: False)
    out = narrator.narrate_run(manifest_data)
    assert out["engine"] == "template"
    assert "credit-union" in out["summary"]


def test_narrate_rejects_hallucinated_llm_output(manifest_data, monkeypatch):
    monkeypatch.setattr(narrator.llm, "ollama_available", lambda *a, **k: True)
    monkeypatch.setattr(narrator, "_ollama_summary",
                        lambda facts: "Amazing 99.9% accuracy on 12,345 customers!")
    out = narrator.narrate_run(manifest_data)
    assert out["engine"] == "template (audit rejected LLM)"
    assert "99.9" in out["audit_violations"]
    assert "99.9" not in out["summary"]                   # the lie never ships


def test_narrate_accepts_grounded_llm_output(manifest_data, monkeypatch):
    monkeypatch.setattr(narrator.llm, "ollama_available", lambda *a, **k: True)
    monkeypatch.setattr(narrator, "_ollama_summary",
                        lambda facts: "Trained on 7,043 rows; lens A promoted "
                                      "at 0.833 ROC-AUC.")
    out = narrator.narrate_run(manifest_data)
    assert out["engine"].startswith("ollama:")
    assert out["audit_violations"] == []
