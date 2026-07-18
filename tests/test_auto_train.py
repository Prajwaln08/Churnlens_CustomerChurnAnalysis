"""Tests for Code/phase_2/training/auto_train.py - gates over vibes."""
import numpy as np
import pandas as pd
import pytest

from phase_1 import config
from phase_2 import auto_train


@pytest.fixture(autouse=True)
def isolated_models(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")


def test_signal_data_trains_and_promotes(labeled_frame):
    report = auto_train.train_for_dataset(labeled_frame, "credit-union")

    lens_a = report["lenses"]["A"]
    assert lens_a["status"] == "promoted"
    assert lens_a["test_roc_auc"] >= auto_train.GATE_AUC
    assert (config.MODEL_DIR / "credit-union" / "churn_pipeline.joblib").exists()

    # leakage screen ran: id + post-outcome column excluded from features
    dropped = report["leakage"]["dropped"]
    assert "customerID" in dropped and "close_reason" in dropped
    assert "tenure" in report["leakage"]["kept"]

    # tenure present -> survival trained; >=2 features -> kmeans promoted
    assert report["lenses"]["B"]["status"] in ("promoted", "trained_not_promoted")
    assert report["lenses"]["D"]["status"] == "promoted"
    assert (config.MODEL_DIR / "credit-union" / "kmeans.joblib").exists()


def test_garbage_labels_fail_the_gate(labeled_frame):
    rng = np.random.default_rng(0)
    garbage = labeled_frame.copy()
    garbage["Churn"] = rng.integers(0, 2, len(garbage))   # pure noise label
    report = auto_train.train_for_dataset(garbage, "noise-source")

    assert report["lenses"]["A"]["status"] == "trained_not_promoted"
    assert "below gate" in report["lenses"]["A"]["reason"]
    assert not (config.MODEL_DIR / "noise-source" / "churn_pipeline.joblib").exists()


def test_cv_folds_option(labeled_frame):
    """#3: k-fold CV decides winner and gate; report says which evaluation ran."""
    report = auto_train.train_for_dataset(labeled_frame, "cv-source", cv_folds=5)
    lens_a = report["lenses"]["A"]
    assert lens_a["status"] == "promoted"
    assert lens_a["evaluation"] == "5-fold stratified cv"
    assert "cv_std" in lens_a and lens_a["cv_std"] < 0.15
    # default path still reports its evaluation honestly
    default = auto_train.train_for_dataset(labeled_frame, "holdout-source")
    assert default["lenses"]["A"]["evaluation"] == "single stratified holdout (80/20)"


def test_survival_skipped_without_duration(labeled_frame):
    no_tenure = labeled_frame.drop(columns=["tenure"])
    report = auto_train.train_for_dataset(no_tenure, "no-duration")
    assert report["lenses"]["B"]["status"] == "skipped"
    assert "tenure" in report["lenses"]["B"]["reason"]


def test_messy_real_world_data_trains_and_scores(labeled_frame):
    """The hardening regression: NaNs + a 60-level categorical, end to end."""
    rng = np.random.default_rng(2)
    messy = labeled_frame.copy()
    messy["city"] = rng.choice([f"city_{i}" for i in range(60)], len(messy))
    messy.loc[rng.choice(len(messy), 60, replace=False), "spend"] = np.nan
    messy.loc[rng.choice(len(messy), 60, replace=False), "plan"] = np.nan

    report = auto_train.train_for_dataset(messy, "messy-source")
    assert report["lenses"]["A"]["status"] == "promoted"
    assert report["encoding"]["frequency"] == ["city"]     # not one-hot exploded
    assert "plan" in report["encoding"]["one_hot"]
    assert report["lenses"]["B"]["status"] != "failed"     # Cox survives NaNs now

    scored = auto_train.score_with_dataset_model(messy, "messy-source")
    assert scored["churn_probability"].between(0, 1).all()


def test_score_with_dataset_model_roundtrip(labeled_frame):
    auto_train.train_for_dataset(labeled_frame, "credit-union")
    scored = auto_train.score_with_dataset_model(labeled_frame, "credit-union")
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["risk_band"]) <= {"Low", "Medium", "High", "Critical"}

    with pytest.raises(ValueError, match="needs columns"):
        auto_train.score_with_dataset_model(
            labeled_frame.drop(columns=["tenure"]), "credit-union")


def test_cox_survives_high_cardinality_columns():
    """Regression: a wide categorical (e.g. city, provider) must become one
    frequency column, never thousands of dummies - raw get_dummies in the Cox
    path is what OOM-killed training on real files."""
    rng = np.random.default_rng(11)
    n = 600
    charges = rng.uniform(20, 120, n).round(2)
    risk = (charges - 20) / 100                       # high bill = high hazard
    frame = pd.DataFrame({
        "tenure": (1 + rng.exponential(40 * (1.05 - risk))).astype(int),
        "MonthlyCharges": charges,
        "city": [f"city_{i}" for i in rng.integers(0, 400, n)],
        "plan": rng.choice(["basic", "plus", "max"], n),
    })
    frame["Churn"] = pd.Series(rng.random(n) < 0.35 + 0.55 * risk).map(
        {True: "Yes", False: "No"})
    report = auto_train.train_for_dataset(frame, "high-card-cox")
    assert report["lenses"]["B"]["status"] in ("promoted", "trained_not_promoted")
    from phase_1.modeling import load_model
    bundle = load_model("cox_bundle", dataset_id="high-card-cox")
    assert "city" in bundle["freq_maps"]              # frequency-encoded
    assert len(bundle["columns"]) < 30                # no dummy explosion
