"""Shared test setup: make Code/ importable + fixtures used across files."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))


@pytest.fixture()
def bank_frame() -> pd.DataFrame:
    """A 'foreign' bank-style dataset: renamed columns, one post-outcome leak,
    numbers-as-text, a date column - the adversarial zoo in one frame."""
    rng = np.random.default_rng(7)
    n = 80
    months = rng.integers(0, 72, n)
    return pd.DataFrame({
        "cust_ref": [f"C{i:05d}" for i in range(n)],
        "agreement_type": rng.choice(["monthly", "annual", "two-year"], n),
        "months_active": months,
        "monthly_fee": rng.uniform(20, 110, n).round(2),
        "total_billed": [str(round(v, 2)) for v in rng.uniform(0, 5000, n)],  # text costume
        "signup_date": pd.Series(pd.date_range("2020-01-01", periods=n)).dt.strftime("%Y-%m-%d"),
        "notes": [f"long free text describing customer interaction history, case {i}"
                  for i in range(n)],
        "one_value": ["same"] * n,
        "close_reason": rng.choice(["", "price", "moved"], n),
        # signal-bearing label: early-tenure customers mostly churn (15% noise)
        "churned_flag": ((months < 18).astype(int)
                         ^ (rng.random(n) < 0.15).astype(int)),
    })


@pytest.fixture()
def labeled_frame() -> pd.DataFrame:
    """A clean, mapped, labeled dataset (canonical names) with real signal -
    what a frame looks like entering the mode/train stages."""
    rng = np.random.default_rng(11)
    n = 400
    tenure = rng.integers(0, 72, n)
    plan = rng.choice(["basic", "plus", "premium"], n)
    logits = -0.08 * tenure + (plan == "basic") * 1.2 + rng.normal(0, 0.8, n) + 1.0
    return pd.DataFrame({
        "customerID": [f"K{i:05d}" for i in range(n)],
        "tenure": tenure,
        "plan": plan,
        "spend": rng.uniform(10, 200, n).round(2),
        "region": rng.choice(["north", "south", "east", "west"], n),
        "noise": rng.normal(size=n),
        "close_reason": rng.choice(["", "price"], n),
        "Churn": (logits > 0).astype(int),
    })
