"""Tests for Code/phase_2/monitoring/drift.py - the model-aging alarm."""
import numpy as np
import pandas as pd
import pytest

from phase_1 import config
from phase_1.modeling import save_model
from phase_2 import drift


@pytest.fixture(autouse=True)
def isolated_models(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")


@pytest.fixture()
def train_frame():
    rng = np.random.default_rng(4)
    return pd.DataFrame({
        "spend": rng.normal(100, 20, 1000),
        "plan": rng.choice(["basic", "plus", "premium"], 1000, p=[0.5, 0.3, 0.2]),
        "constant": ["x"] * 1000,
    })


def test_baseline_structure(train_frame):
    baseline = drift.build_baseline(train_frame)
    assert baseline["rows"] == 1000
    assert baseline["features"]["spend"]["kind"] == "numeric"
    assert len(baseline["features"]["spend"]["edges"]) == 11
    assert baseline["features"]["plan"]["kind"] == "categorical"
    assert "constant" not in baseline["features"]        # nothing can drift


def test_psi_zero_for_same_and_large_for_shifted(train_frame):
    baseline = drift.build_baseline(train_frame)
    same = drift.compare(train_frame, baseline)
    assert (same["status"] == "ok").all()
    assert same["psi"].max() < 0.01

    shifted = train_frame.copy()
    shifted["spend"] = shifted["spend"] * 2 + 100        # regime change
    table = drift.compare(shifted, baseline)
    spend_row = table[table["feature"] == "spend"].iloc[0]
    assert spend_row["status"] == "alert"
    assert spend_row["psi"] >= drift.ALERT_PSI


def test_categorical_shift_and_new_level(train_frame):
    baseline = drift.build_baseline(train_frame)
    moved = train_frame.copy()
    moved["plan"] = np.where(moved["plan"] == "basic", "enterprise", moved["plan"])
    row = drift.compare(moved, baseline).set_index("feature").loc["plan"]
    assert row["status"] == "alert"                      # half the mass moved


def test_missing_feature_is_drift_too(train_frame):
    baseline = drift.build_baseline(train_frame)
    table = drift.compare(train_frame.drop(columns=["spend"]), baseline)
    assert table.set_index("feature").loc["spend", "status"] == "missing"


def test_assess_none_without_baseline(train_frame):
    assert drift.assess(train_frame, "never-trained") is None


def test_assess_recommends_retraining_on_alert(train_frame):
    save_model(drift.build_baseline(train_frame), "drift_baseline",
               dataset_id="my-source")
    shifted = train_frame.assign(spend=train_frame["spend"] * 3)
    report = drift.assess(shifted, "my-source")
    assert "spend" in report["alerts"]
    assert "retraining recommended" in report["recommendation"]

    stable = drift.assess(train_frame, "my-source")
    assert stable["alerts"] == []
    assert "stable" in stable["recommendation"]
