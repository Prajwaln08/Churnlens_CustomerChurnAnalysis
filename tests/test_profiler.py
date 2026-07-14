"""Tests for Code/phase_2/profiler.py - facts only, roles inferred correctly."""
import pandas as pd

from phase_2 import profiler


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


def test_correlation_report_finds_collinear_pairs(bank_frame):
    """#4: near-duplicate numerics get named; independent ones stay quiet."""
    df = bank_frame.copy()
    df["fee_x2"] = df["monthly_fee"] * 2 + 0.01          # perfectly collinear
    pairs = profiler.correlation_report(df, threshold=0.9)
    assert any({p["a"], p["b"]} == {"monthly_fee", "fee_x2"} for p in pairs)
    assert all(p["corr"] >= 0.9 for p in pairs)
    assert profiler.correlation_report(bank_frame[["monthly_fee"]]) == []


def test_summary_frame_covers_every_column(bank_frame):
    prof = profiler.profile_table(bank_frame)
    frame = profiler.summary_frame(prof)
    assert set(frame["column"]) == set(bank_frame.columns)
    assert {"role", "null_%", "unique", "samples", "flags"} <= set(frame.columns)
