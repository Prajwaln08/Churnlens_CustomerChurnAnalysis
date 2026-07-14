"""Tests for Code/phase_2/training/leakage.py - the answer-key detector."""
import numpy as np
import pandas as pd

from phase_2 import leakage


def _frame(n=120):
    rng = np.random.default_rng(3)
    label = rng.integers(0, 2, n)
    return pd.DataFrame({
        "cust_id": [f"C{i:05d}" for i in range(n)],
        "constant_col": ["x"] * n,
        "signup_date": pd.Series(pd.date_range("2021-01-01", periods=n)).dt.strftime("%Y-%m-%d"),
        "notes": [f"free text about this customer, number {i}, quite long indeed" for i in range(n)],
        "close_reason": rng.choice(["", "price", "moved"], n),
        "label_copy": np.where(label == 1, "gone", "here"),      # the label in a costume
        "good_numeric": rng.normal(size=n),
        "good_cat": rng.choice(["a", "b", "c"], n),
        "Churn": label,
    }), label


def test_screen_drops_the_right_columns():
    df, _ = _frame()
    result = leakage.screen(df, "Churn")
    dropped = result["dropped"]

    assert "cust_id" in dropped and "id-like" in dropped["cust_id"]
    assert "constant_col" in dropped and "constant" in dropped["constant_col"]
    assert "notes" in dropped and "free text" in dropped["notes"]
    assert "close_reason" in dropped and "post-outcome" in dropped["close_reason"]
    assert "label_copy" in dropped and "answer key" in dropped["label_copy"]
    assert "signup_date" in dropped                       # replaced, not lost
    assert result["derived"] == {"signup_date_days": "signup_date"}

    assert set(result["features"]) == {"good_numeric", "good_cat"}
    assert "Churn" not in result["features"]              # never its own feature


def test_near_perfect_numeric_predictor_is_caught():
    df, label = _frame()
    df["leaky_score"] = label * 10.0 + 0.001              # numeric answer key
    result = leakage.screen(df, "Churn")
    assert "leaky_score" in result["dropped"]


def test_noisy_correlate_survives():
    rng = np.random.default_rng(5)
    df, label = _frame()
    df["decent_feature"] = label + rng.normal(0, 1.0, len(df))   # useful, not perfect
    result = leakage.screen(df, "Churn")
    assert "decent_feature" in result["features"]


def test_derive_datetime_features_is_relative_not_wall_clock():
    df, _ = _frame()
    result = leakage.screen(df, "Churn")
    out = leakage.derive_datetime_features(df, result["derived"])
    days = out["signup_date_days"]
    assert days.min() == 0.0                              # newest row = 0 days
    assert days.max() == float(len(df) - 1)               # daily range fixture
    assert out is not df                                  # input untouched
