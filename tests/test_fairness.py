"""Tests for Code/phase_2/analytics/fairness.py - neutral group auditing."""
import numpy as np
import pandas as pd

from phase_2.analytics import fairness


def _scored(n=600, biased=False, seed=2):
    rng = np.random.default_rng(seed)
    group = rng.choice(["a", "b"], n)
    p = rng.uniform(0.1, 0.6, n)
    if biased:
        p = np.where(group == "a", p + 0.3, p)
    return pd.DataFrame({
        "segment_col": group,
        "churn_probability": np.clip(p, 0, 1).round(4),
        "risk_band": np.where(p > 0.5, "High", "Low"),
    })


def test_balanced_groups_not_flagged():
    rep = fairness.audit_column(_scored(biased=False), "segment_col")
    assert len(rep["groups"]) == 2
    assert not rep["flagged"]


def test_biased_groups_flagged_with_gap():
    rep = fairness.audit_column(_scored(biased=True), "segment_col")
    assert rep["flagged"]
    assert rep["prob_gap"] >= 0.2


def test_small_groups_stay_unreported():
    df = _scored(40)
    df.loc[:35, "segment_col"] = "a"       # group b shrinks below MIN_GROUP
    rep = fairness.audit_column(df, "segment_col")
    assert all(g["group"] == "a" for g in rep["groups"])


def test_tpr_fpr_with_actuals():
    df = _scored(400, seed=6)
    actual = pd.Series((df["churn_probability"] > 0.45).astype(int))
    rep = fairness.audit_column(df, "segment_col", actual=actual,
                                threshold=0.45)
    for g in rep["groups"]:
        assert g["tpr"] == 1.0           # flags exactly match actuals here
        assert g["fpr"] == 0.0
