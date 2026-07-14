"""Tests for Code/phase_2/analytics/journeys.py - churn as stopping showing up.

The anchor test encodes the user's own example: weekly-active for a month,
then silence -> churned 'after month 1' at monthly grain, 'in week 5' weekly.
"""
import numpy as np
import pandas as pd
import pytest

from phase_2.analytics import journeys as jn


def _events(user, dates):
    return pd.DataFrame({"user_id": user, "event_ts": pd.to_datetime(dates)})


@pytest.fixture()
def example_log():
    """u1: the example user - active weeks 1-4 then silent for 8+ weeks.
    u2: still active at data end (must be censored, never churned).
    u3: one-and-done in week 1."""
    u1 = _events("u1", ["2026-01-05", "2026-01-12", "2026-01-15",
                        "2026-01-19", "2026-01-26"])
    u2 = _events("u2", pd.date_range("2026-01-05", "2026-03-30", freq="7D"))
    u3 = _events("u3", ["2026-01-06"])
    return pd.concat([u1, u2, u3], ignore_index=True)


def test_the_users_own_example(example_log):
    weekly = jn.build_journeys(example_log, "user_id", "event_ts",
                               granularity="W", inactivity_window=2)
    u1_weekly = weekly.set_index("user").loc["u1"]
    assert u1_weekly["journey_status"] == jn.CHURNED
    assert u1_weekly["periods_survived"] == 4          # active weeks 1-4
    assert u1_weekly["churn_period"] == 5              # "churned in week 5"

    monthly = jn.build_journeys(example_log, "user_id", "event_ts",
                                granularity="M", inactivity_window=1)
    u1_monthly = monthly.set_index("user").loc["u1"]
    assert u1_monthly["periods_survived"] == 1         # one active month
    assert u1_monthly["churn_period"] == 2             # "churned after month 1"


def test_still_active_user_is_censored_not_churned(example_log):
    weekly = jn.build_journeys(example_log, "user_id", "event_ts",
                               granularity="W", inactivity_window=2)
    u2 = weekly.set_index("user").loc["u2"]
    assert u2["journey_status"] == jn.ACTIVE
    assert not u2["observed"]                          # censored for KM
    assert np.isnan(u2["churn_period"])


def test_retention_curve_and_timing(example_log):
    weekly = jn.build_journeys(example_log, "user_id", "event_ts",
                               granularity="W", inactivity_window=2)
    curve = jn.retention_curve(weekly)
    assert float(curve.iloc[0]) == 1.0                 # everyone starts alive
    assert curve.is_monotonic_decreasing
    timing = jn.churn_timing(weekly)
    assert timing.loc[5] == 1                          # u1 went silent week 5
    assert timing.loc[2] == 1                          # u3 after week 1


@pytest.fixture()
def segmented_journeys():
    """200 synthetic users, two plans with very different lifetimes."""
    rng = np.random.default_rng(6)
    rows = []
    for i in range(200):
        plan = "family" if i % 2 else "trial"
        lifetime = rng.geometric(0.08 if plan == "family" else 0.35)
        start = pd.Timestamp("2026-01-05") + pd.Timedelta(days=int(rng.integers(0, 28)))
        for w in range(int(lifetime)):
            rows.append({"user_id": f"u{i}", "plan": plan,
                         "event_ts": start + pd.Timedelta(weeks=w)})
    events = pd.DataFrame(rows)
    attrs = events[["user_id", "plan"]].drop_duplicates()
    return jn.build_journeys(events, "user_id", "event_ts", granularity="W",
                             inactivity_window=2, attributes=attrs)


def test_segment_curves_and_summary_separate_populations(segmented_journeys):
    curves = jn.retention_by_segment(segmented_journeys, "plan")
    assert set(curves) == {"family", "trial"}
    at8 = {seg: float(c[c.index <= 8].iloc[-1]) for seg, c in curves.items()}
    assert at8["family"] > at8["trial"] + 0.15          # family clearly stickier

    summary = jn.segment_summary(segmented_journeys, "plan")
    trial = summary.set_index("plan").loc["trial"]
    family = summary.set_index("plan").loc["family"]
    assert trial["churned_%"] >= family["churned_%"]


def test_individual_outlook_conditions_on_survival(segmented_journeys):
    outlook = jn.individual_outlook(segmented_journeys, segment_col="plan")
    alive = outlook[~outlook["churned"]]
    known = alive.dropna(subset=["expected_remaining_periods"])
    if len(known):
        assert (known["expected_remaining_periods"] >= 0).all()
        assert (known["estimated_churn_period"]
                > known["periods_survived"]).all()      # future, not past
    # users at the data horizon are honestly 'unknown', never falsely imminent
    horizon = alive[alive["expected_remaining_periods"].isna()]
    assert np.isnan(horizon["estimated_churn_period"]).all()
    gone = outlook[outlook["churned"]]
    assert (gone["expected_remaining_periods"] == 0).all()
    assert (gone["estimated_churn_period"] == gone["churn_period"]).all()
