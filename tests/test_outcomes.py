"""Tests for Code/phase_2/monitoring/outcomes.py - delayed-label grading."""
import numpy as np
import pandas as pd

from phase_2.monitoring import outcomes


def _scored(n=400, seed=3):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    return pd.DataFrame({"customerID": [f"c{i}" for i in range(n)],
                         "churn_probability": p.round(4)})


def test_good_predictions_grade_healthy():
    scored = _scored()
    rng = np.random.default_rng(4)
    actual = (rng.random(len(scored)) < scored["churn_probability"]).astype(int)
    actuals = pd.DataFrame({"cust": scored["customerID"], "did_churn": actual})
    rep = outcomes.realized_performance(scored, actuals, "cust", "did_churn")
    assert rep["status"] == "ok"
    assert rep["matched"] == len(scored)
    assert rep["realized_auc"] > outcomes.GATE_AUC
    assert rep["healthy"]
    assert rep["precision_top_decile"] > rep["actual_churn_rate"]


def test_degraded_predictions_recommend_retraining():
    scored = _scored()
    rng = np.random.default_rng(5)
    actuals = pd.DataFrame({"cust": scored["customerID"],
                            "did_churn": rng.integers(0, 2, len(scored))})
    rep = outcomes.realized_performance(scored, actuals, "cust", "did_churn")
    assert rep["status"] == "ok"
    assert not rep["healthy"]
    assert "retrain" in rep["verdict"]


def test_insufficient_overlap_is_honest():
    scored = _scored(30)
    actuals = pd.DataFrame({"cust": ["nope1", "nope2"], "did_churn": [1, 0]})
    rep = outcomes.realized_performance(scored, actuals, "cust", "did_churn")
    assert rep["status"] == "insufficient"
    assert "matched" in rep["reason"]


def test_yes_no_actuals_are_understood():
    scored = _scored(200, seed=8)
    actual = np.where(scored["churn_probability"] > 0.5, "Yes", "No")
    actuals = pd.DataFrame({"id": scored["customerID"], "outcome": actual})
    rep = outcomes.realized_performance(scored, actuals, "id", "outcome")
    assert rep["status"] == "ok"
    assert rep["realized_auc"] > 0.95
