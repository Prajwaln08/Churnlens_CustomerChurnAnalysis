"""Tests for Code/phase_2/ingestion/connectors.py - one shape out, always."""
import json
import sqlite3

import pytest

from phase_2 import connectors
from phase_2.orchestration.orchestrator import run_pipeline
from phase_2.orchestration.registry import Registry
from phase_1 import config


def test_from_dataframe_metadata(bank_frame):
    src = connectors.from_dataframe(bank_frame, name="bank")
    assert src.origin == "memory:bank"
    assert src.rows == len(bank_frame)
    assert len(src.content_hash) == 16
    # same data -> same hash (reproducibility anchor)
    assert src.content_hash == connectors.from_dataframe(bank_frame).content_hash


def test_from_file_csv_json_parquet(bank_frame, tmp_path):
    csv_p = tmp_path / "d.csv"
    bank_frame.to_csv(csv_p, index=False)
    assert connectors.from_file(csv_p).rows == len(bank_frame)

    pq_p = tmp_path / "d.parquet"
    bank_frame.to_parquet(pq_p)
    assert connectors.from_file(pq_p).origin == "file:d.parquet"

    json_p = tmp_path / "d.json"
    json_p.write_text(json.dumps([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]))
    src = connectors.from_file(json_p)
    assert list(src.frame.columns) == ["a", "b"] and src.rows == 2

    with pytest.raises(ValueError, match="unsupported file type"):
        connectors.from_file(tmp_path / "d.docx")


def test_from_database_sqlite_table_query_limit(bank_frame, tmp_path):
    db = tmp_path / "crm.sqlite"
    with sqlite3.connect(db) as con:
        bank_frame.to_sql("customers", con, index=False)
    url = f"sqlite:///{db}"

    full = connectors.from_database(url, table="customers")
    assert full.rows == len(bank_frame)
    assert full.origin.endswith("#customers")

    limited = connectors.from_database(url, table="customers", limit=10)
    assert limited.rows == 10

    q = connectors.from_database(
        url, query="SELECT cust_ref, monthly_fee FROM customers WHERE monthly_fee > 60")
    assert list(q.frame.columns) == ["cust_ref", "monthly_fee"]
    assert (q.frame["monthly_fee"] > 60).all()          # pushdown honored

    with pytest.raises(ValueError, match="exactly one"):
        connectors.from_database(url)


def test_watermark_incremental_fetch(bank_frame, tmp_path):
    """#5: only rows past the high-water mark travel; the new mark rides back."""
    db = tmp_path / "crm.sqlite"
    with sqlite3.connect(db) as con:
        bank_frame.to_sql("customers", con, index=False)
    url = f"sqlite:///{db}"

    first = connectors.from_database(url, table="customers",
                                     watermark_column="months_active", since=-1)
    assert first.rows == len(bank_frame)
    assert first.watermark == str(bank_frame["months_active"].max())

    incremental = connectors.from_database(url, table="customers",
                                           watermark_column="months_active",
                                           since=50)
    assert incremental.rows == int((bank_frame["months_active"] > 50).sum())
    assert (incremental.frame["months_active"] > 50).all()


def test_watermark_state_roundtrip(tmp_path):
    from phase_2.orchestration.registry import Registry

    reg = Registry(db_path=tmp_path / "r.sqlite")
    assert reg.get_watermark("crm") is None
    reg.set_watermark("crm", "updated_at", "2026-07-01", "2026-07-15T10:00:00Z")
    reg.set_watermark("crm", "updated_at", "2026-07-14", "2026-07-15T11:00:00Z")
    mark = reg.get_watermark("crm")
    assert mark["last_value"] == "2026-07-14"              # upserted, not duplicated


def test_credentials_are_redacted():
    assert "secret" not in connectors.redact_url(
        "postgresql://analyst:secret@db.bank.internal:5432/crm")
    assert "analyst" in connectors.redact_url(
        "postgresql://analyst:secret@db.bank.internal:5432/crm")


def test_from_api_records_path(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass
        def json(self):
            return {"data": {"items": [{"id": 1, "fee": 20.5},
                                       {"id": 2, "fee": 31.0}]},
                    "page": 1}
    captured = {}
    def fake_get(url, headers=None, params=None, timeout=None):
        captured.update(url=url, headers=headers)
        return FakeResp()
    monkeypatch.setattr(connectors.requests, "get", fake_get)

    src = connectors.from_api("https://api.example.com/v2/customers",
                              records_path="data.items",
                              headers={"Authorization": "Bearer T"})
    assert src.rows == 2 and list(src.frame.columns) == ["id", "fee"]
    assert captured["headers"]["Authorization"] == "Bearer T"
    assert src.origin == "api:https://api.example.com/v2/customers"

    with pytest.raises(ValueError, match="not found"):
        connectors.from_api("https://x", records_path="data.missing")
    with pytest.raises(ValueError, match="does not point to a list"):
        connectors.from_api("https://x", records_path="data")


def test_orchestrator_accepts_sourcedata(bank_frame, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    from phase_2 import mapping as mapping_mod
    monkeypatch.setattr(mapping_mod, "ollama_available", lambda *a, **k: False)

    src = connectors.from_dataframe(bank_frame, name="bank")
    m = run_pipeline(src, "bank-src", registry=Registry(db_path=tmp_path / "r.sqlite"),
                     raise_on_error=True)
    assert m.data["input"]["origin"] == "memory:bank"
    assert m.data["input"]["fetched_at"] is not None
    assert any("from memory:bank" in line for line in m.ledger_lines())
