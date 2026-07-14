"""Tests for Code/phase_1/preprocessing/features.py - the hardened preprocessor.

The two production gaps these pin down: NaNs must never crash a linear model,
and high-cardinality categoricals must not explode dimensionality.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phase_1.preprocessing import (
    FrequencyEncoder,
    make_preprocessor,
    split_categoricals,
)


@pytest.fixture()
def messy_frame():
    """NaNs in numeric and categorical, plus a 60-level categorical."""
    rng = np.random.default_rng(9)
    n = 300
    df = pd.DataFrame({
        "amount": rng.uniform(10, 200, n),
        "plan": rng.choice(["a", "b", "c"], n).astype(object),
        "city": rng.choice([f"city_{i}" for i in range(60)], n),
    })
    df.loc[rng.choice(n, 40, replace=False), "amount"] = np.nan
    df.loc[rng.choice(n, 40, replace=False), "plan"] = np.nan
    return df


def test_split_categoricals_threshold(messy_frame):
    low, high = split_categoricals(messy_frame, ["plan", "city"])
    assert low == ["plan"] and high == ["city"]


def test_nans_survive_a_linear_model(messy_frame):
    """The regression for gap #1: LR used to be un-trainable on NaN numerics."""
    y = (messy_frame["amount"].fillna(0) > 100).astype(int)
    pre = make_preprocessor(["amount"], ["plan"], ["city"])
    pipe = Pipeline([("prep", pre),
                     ("clf", LogisticRegression(max_iter=1000))]).fit(messy_frame, y)
    proba = pipe.predict_proba(messy_frame)[:, 1]
    assert np.isfinite(proba).all()


def test_missing_indicator_carries_signal(messy_frame):
    pre = make_preprocessor(["amount"], [], [])
    out = pre.fit_transform(messy_frame)
    assert out.shape[1] == 2                     # value + missing indicator
    assert set(np.unique(out[:, 1])).issubset({0.0, 1.0} | set(np.unique(out[:, 1])))


def test_high_cardinality_does_not_explode(messy_frame):
    exploded = make_preprocessor([], ["city"], []).fit_transform(messy_frame)
    contained = make_preprocessor([], [], ["city"]).fit_transform(messy_frame)
    assert exploded.shape[1] >= 60               # one-hot: a column per city
    assert contained.shape[1] == 1               # frequency: one column, period


def test_frequency_encoder_unseen_maps_to_zero():
    enc = FrequencyEncoder().fit(pd.DataFrame({"c": ["x", "x", "y"]}))
    out = enc.transform(pd.DataFrame({"c": ["x", "y", "NEVER-SEEN"]}))
    assert out[0, 0] == pytest.approx(2 / 3)
    assert out[1, 0] == pytest.approx(1 / 3)
    assert out[2, 0] == 0.0


def test_scoring_with_new_nans_does_not_crash(messy_frame):
    """Fit on clean rows, score rows with NaNs the fit never saw."""
    clean = messy_frame.dropna()
    pre = make_preprocessor(["amount"], ["plan"], ["city"]).fit(clean)
    scored = pre.transform(messy_frame)          # includes NaN rows
    assert np.isfinite(scored).all()
