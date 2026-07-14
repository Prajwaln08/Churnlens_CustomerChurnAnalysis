"""Tests for Code/phase_2/training/mode.py - the score/train/explore fork."""
import joblib
import pytest

from phase_1 import config
from phase_2 import mode


@pytest.fixture(autouse=True)
def isolated_models(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")


def _fake_artifact(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"fake": True}, path)


def test_train_when_label_mapped_and_no_artifacts(labeled_frame):
    decision = mode.decide(labeled_frame, "Credit Union #1")
    assert decision["mode"] == mode.TRAIN
    assert decision["dataset_id"] == "credit-union--1"
    assert any("training fresh models" in r for r in decision["reasons"])


def test_score_when_dataset_artifacts_exist(labeled_frame):
    _fake_artifact(config.MODEL_DIR / "credit-union--1" / "churn_pipeline.joblib")
    decision = mode.decide(labeled_frame, "Credit Union #1")
    assert decision["mode"] == mode.SCORE
    assert any("artifacts exist for dataset" in r for r in decision["reasons"])


def test_explore_when_no_label_but_candidates_are_named(bank_frame):
    decision = mode.decide(bank_frame, "bank-crm")      # churned_flag unmapped
    assert decision["mode"] == mode.EXPLORE
    assert any("churned_flag" in r for r in decision["reasons"])
    assert any("confirm one as 'Churn'" in r for r in decision["reasons"])


def test_too_few_rows_blocks_training(labeled_frame):
    decision = mode.decide(labeled_frame.head(40), "tiny-source")
    assert decision["mode"] == mode.EXPLORE
    assert any("too little to train honestly" in r for r in decision["reasons"])


def test_too_few_features_blocks_training(labeled_frame):
    slim = labeled_frame[["customerID", "Churn"]]
    decision = mode.decide(slim, "slim-source")
    assert decision["mode"] == mode.EXPLORE
    assert any("feature column" in r for r in decision["reasons"])
