"""Tests for Code/phase_2/analytics/insights.py - the section-14 contract."""
import numpy as np
import pytest

from phase_1 import config
from phase_2 import auto_train
from phase_2.analytics import enrich_and_summarize


@pytest.fixture(autouse=True)
def isolated_models(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")


@pytest.fixture()
def trained_scored(labeled_frame):
    """Train on a frame whose spend column is mapped as MonthlyCharges so the
    revenue-at-risk path activates, then score it."""
    df = labeled_frame.rename(columns={"spend": "MonthlyCharges"})
    auto_train.train_for_dataset(df, "insight-source")
    scored = auto_train.score_with_dataset_model(df, "insight-source")
    return df, scored


def test_individual_level_columns(trained_scored):
    df, scored = trained_scored
    enriched, _ = enrich_and_summarize(scored, "insight-source")

    for col in ("churn_probability", "risk_band", "local_drivers", "segment",
                "expected_remaining_tenure", "clv_simple", "revenue_at_risk"):
        assert col in enriched.columns, col
    assert (enriched["revenue_at_risk"] >= 0).all()
    # local drivers are human-readable pushes, not numbers-only blobs
    driven = enriched.loc[enriched["local_drivers"] != "no single dominant driver",
                          "local_drivers"]
    assert len(driven) > 0 and driven.str.contains("pp").all()


def test_population_level_report(trained_scored):
    df, scored = trained_scored
    _, population = enrich_and_summarize(scored, "insight-source")

    assert population["actual_churn_rate"] > 0
    assert population["risk_bands"]
    drivers = population["global_drivers"]
    assert drivers and drivers[0]["mean_abs_effect_pp"] >= drivers[-1]["mean_abs_effect_pp"]
    assert "tenure" in {d["feature"] for d in drivers}     # the known signal
    assert population["top_decile_lift"] > 1.5             # model beats random
    assert population["segments"] and "share" in population["segments"][0]
    assert population["remaining_tenure"]["mean"] >= 0
    assert population["revenue_at_risk"]["total"] > 0
    assert 0 < population["revenue_at_risk"]["top_decile_share"] <= 1


def test_unlabeled_scoring_still_gets_drivers(trained_scored):
    """At scoring time there is often NO label: drivers and segments must not
    depend on it; lift is simply absent."""
    df, _ = trained_scored
    unlabeled = df.drop(columns=["Churn"])
    scored = auto_train.score_with_dataset_model(unlabeled, "insight-source")
    enriched, population = enrich_and_summarize(scored, "insight-source")

    assert "local_drivers" in enriched.columns
    assert population["global_drivers"]
    assert "lift_by_decile" not in population
    assert "actual_churn_rate" not in population


def test_revenue_at_risk_ships_a_confidence_interval(trained_scored):
    """Phase 1.3: the RaR headline carries a bootstrap 95% CI around it."""
    df, scored = trained_scored
    _, population = enrich_and_summarize(scored, "insight-source")
    rar = population["revenue_at_risk"]
    lo, hi = rar["total_ci95"]
    assert lo < hi
    assert lo <= rar["total"] <= hi
