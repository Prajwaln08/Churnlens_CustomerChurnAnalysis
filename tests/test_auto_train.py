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


def test_lens_a_probabilities_are_calibrated():
    """Phase 1.1: served probabilities are calibrated - '80%' means 80%.
    Brier and a bin table land in the report, computed on the holdout only."""
    rng = np.random.default_rng(5)
    n = 1600
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    p_true = 1 / (1 + np.exp(-(0.9 * x1 - 0.7 * x2)))
    frame = pd.DataFrame({
        "tenure": rng.integers(1, 60, n),
        "MonthlyCharges": (50 + 10 * x1).round(2),
        "signal_b": x2.round(3),
        "Churn": np.where(rng.random(n) < p_true, "Yes", "No"),
    })
    report = auto_train.train_for_dataset(frame, "calib-check")
    a = report["lenses"]["A"]
    assert a["calibration_method"] in ("sigmoid", "isotonic")
    assert 0 < a["brier"] < 0.3
    bins = [b for b in a["calibration_bins"] if b["n"] >= 30]
    assert bins, "expected populated calibration bins"
    mean_gap = sum(abs(b["predicted"] - b["observed"]) for b in bins) / len(bins)
    assert mean_gap < 0.15, f"calibration gap too wide: {mean_gap:.3f}"


def test_temporal_holdout_used_when_dates_exist():
    """Phase 1.2: with a datetime column, lens A trains on the past and tests
    on the most recent 20% - and says so. Without dates, the stratified
    random split remains."""
    rng = np.random.default_rng(9)
    n = 900
    start = pd.Timestamp("2025-01-01")
    signup = pd.to_datetime(start) + pd.to_timedelta(
        rng.integers(0, 540, n), unit="D")
    drift_p = 0.15 + 0.5 * (signup - start).days / 540      # churn grows over time
    frame = pd.DataFrame({
        "signup_date": signup,
        "tenure": rng.integers(1, 60, n),
        "MonthlyCharges": rng.uniform(20, 120, n).round(2),
        "plan": rng.choice(["a", "b", "c"], n),
        "Churn": np.where(rng.random(n) < drift_p, "Yes", "No"),
    })
    report = auto_train.train_for_dataset(frame, "temporal-check")
    a = report["lenses"]["A"]
    assert a["evaluation"].startswith("temporal holdout")
    assert "signup_date" in a["evaluation"]

    no_dates = frame.drop(columns=["signup_date"])
    report2 = auto_train.train_for_dataset(no_dates, "temporal-check-2")
    assert report2["lenses"]["A"]["evaluation"] == \
        "single stratified holdout (80/20)"


def test_auc_confidence_interval_brackets_the_point(labeled_frame):
    """Phase 1.3: the held-out AUC ships with a 95% bootstrap CI."""
    report = auto_train.train_for_dataset(labeled_frame, "ci-check")
    a = report["lenses"]["A"]
    lo, hi = a["auc_ci95"]
    assert lo < hi
    assert lo <= a["test_roc_auc"] <= hi


def test_cost_threshold_reported_and_used_for_bands(labeled_frame):
    """Phase 1.4: training picks a cost-optimal threshold, ships it in the
    bundle, and scoring bands pivot on it. Raising the miss cost never raises
    the threshold."""
    report = auto_train.train_for_dataset(labeled_frame, "cost-check")
    cost = report["lenses"]["A"]["cost"]
    curve = cost["cost_curve"]
    assert 0.05 <= cost["decision_threshold"] <= 0.95
    t_cheap = min(curve, key=lambda c: c["fn"] * 1 + c["fp"] * 1)["t"]
    t_dear = min(curve, key=lambda c: c["fn"] * 20 + c["fp"] * 1)["t"]
    assert t_dear <= t_cheap

    from phase_1.modeling import load_model
    bundle = load_model("churn_pipeline", dataset_id="cost-check")
    assert bundle["threshold"] == cost["decision_threshold"]

    scored = auto_train.score_with_dataset_model(labeled_frame, "cost-check")
    t = bundle["threshold"]
    just_below = scored[scored["churn_probability"] < t / 2]
    if len(just_below):
        assert (just_below["risk_band"] == "Low").all()


def test_served_artifact_refits_on_full_data(labeled_frame):
    """Phase 4.4: metrics come from the holdout, the served artifact from a
    refit on 100% of rows - and the report states it."""
    report = auto_train.train_for_dataset(labeled_frame, "refit-check")
    assert report["lenses"]["A"]["refit_on_full"] is True
    scored = auto_train.score_with_dataset_model(labeled_frame, "refit-check")
    assert scored["churn_probability"].between(0, 1).all()


def test_campaign_uplift_persists_and_reuses():
    """Phase 2.3: the campaign-specific model saves under the dataset
    namespace with its Qini score, and reuses instantly when features match."""
    rng = np.random.default_rng(13)
    n = 400
    risk = rng.uniform(0, 1, n)
    treated = rng.random(n) < 0.5
    frame = pd.DataFrame({
        "treatment_group": np.where(treated, "treatment", "control"),
        "conversion": (rng.random(n) < 0.25 + 0.3 * risk * treated).astype(int),
        "engagement": (risk * 100).round(1),
        "spend_m": rng.uniform(10, 90, n).round(2),
    })
    scored, rep = auto_train.train_uplift_for_campaign(
        frame, dataset_id="camp-src")
    assert scored is not None and "uplift_score" in scored.columns
    assert rep.get("persisted") is True
    assert "qini_auc" in rep

    scored2, rep2 = auto_train.score_campaign_with_saved(frame, "camp-src")
    assert scored2 is not None and "uplift_score" in scored2.columns
    assert "saved artifact" in rep2["engine"]

    renamed = frame.rename(columns={"engagement": "eng2"})
    s3, r3 = auto_train.score_campaign_with_saved(renamed, "camp-src")
    assert s3 is None and r3 is None


def test_lowercase_labels_train_fine():
    """Regression (visitor path): 'yes'/'no' labels in any casing must train,
    not collapse to a single class."""
    rng = np.random.default_rng(17)
    n = 500
    tenure = rng.integers(1, 60, n)
    frame = pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": rng.uniform(20, 110, n).round(2),
        "Churn": np.where((tenure < 15) & (rng.random(n) < 0.8), "yes", "no"),
    })
    report = auto_train.train_for_dataset(frame, "lowercase-check")
    assert report["lenses"]["A"]["status"] == "promoted"


def test_single_class_label_degrades_honestly():
    """A label with one value skips A and B with a reason - D still trains."""
    rng = np.random.default_rng(18)
    n = 300
    frame = pd.DataFrame({
        "tenure": rng.integers(1, 60, n),
        "MonthlyCharges": rng.uniform(20, 110, n).round(2),
        "plan": rng.choice(["a", "b"], n),
        "Churn": ["no"] * n,
    })
    report = auto_train.train_for_dataset(frame, "oneclass-check")
    assert report["lenses"]["A"]["status"] == "skipped"
    assert "single class" in report["lenses"]["A"]["reason"]
    assert report["lenses"]["D"]["status"] == "promoted"


def test_labels_survive_pandas_string_dtype():
    """Regression (pandas 3 cloud bug): labels stored with the modern string
    dtype (pandas 3 default) must normalize exactly like object dtype."""
    from phase_2.training import leakage
    obj = pd.Series(["yes", "no", "YES", "No"], dtype=object)
    modern = pd.Series(["yes", "no", "YES", "No"], dtype="string")
    assert list(leakage._numeric_label(obj)) == [1.0, 0.0, 1.0, 0.0]
    assert list(leakage._numeric_label(modern)) == [1.0, 0.0, 1.0, 0.0]
