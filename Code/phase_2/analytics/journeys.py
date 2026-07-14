"""Journey churn - engagement churn from raw EVENT data.

The other kind of churn: nobody cancels an app, they stop showing up. Churn
is defined relative to a time grain and an inactivity window:

    a user weekly-active for a month who then stops has churned
    "after month 1" at monthly grain, and "in week 5" at weekly grain.

Users still inside the inactivity window at the end of the data are CENSORED
(alive so far), never counted as churned - Kaplan-Meier handles that
correctly, which is why it is the engine here.

Population level: retention curves (overall and per segment: age band, plan,
condition...), churn-timing distributions, segment comparison tables.
Individual level: each user's journey status plus the expected remaining
periods, conditioned on how far into the journey they already survived -
E[T - t | T > t] read off their segment's survival curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE = "active"
AT_RISK = "at_risk"        # gap has started but window not yet exceeded
CHURNED = "churned"


# ---------------------------------------------------------------------------
# Journey table - one row per user, from raw events
# ---------------------------------------------------------------------------

def build_journeys(events: pd.DataFrame, user_col: str, ts_col: str,
                   granularity: str = "W", inactivity_window: int = 2,
                   attributes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Collapse an event log into one journey row per user.

    granularity: 'W' (weekly) or 'M' (monthly) - the grain churn is judged in.
    inactivity_window: churned = no activity for >= this many whole periods
    before the end of the observed data.

    Columns out: first_period, last_period, periods_survived (1-based age of
    the journey at last activity), active_periods, current_gap, journey_status,
    churn_period (period number in the user's journey where they went silent;
    NaN while alive), churned (bool), duration + observed (KM inputs).
    """
    ts = pd.to_datetime(events[ts_col], errors="coerce")
    frame = pd.DataFrame({
        "user": events[user_col],
        "period": ts.dt.to_period(granularity),
    }).dropna()
    data_end = frame["period"].max()

    grouped = frame.groupby("user")["period"]
    journeys = pd.DataFrame({
        "first_period": grouped.min(),
        "last_period": grouped.max(),
        "active_periods": grouped.nunique(),
    })
    # 1-based journey age at last activity: first period of life = 1
    journeys["periods_survived"] = (
        (journeys["last_period"] - journeys["first_period"])
        .apply(lambda d: d.n) + 1)
    journeys["current_gap"] = (data_end - journeys["last_period"]).apply(lambda d: d.n)

    journeys["churned"] = journeys["current_gap"] >= inactivity_window
    journeys["journey_status"] = np.select(
        [journeys["churned"], journeys["current_gap"] > 0],
        [CHURNED, AT_RISK], default=ACTIVE)
    # the period IN THEIR JOURNEY where silence began (your "week 5")
    journeys["churn_period"] = np.where(
        journeys["churned"], journeys["periods_survived"] + 1, np.nan)

    # Kaplan-Meier inputs: duration lived (so far), observed = actually died
    journeys["duration"] = journeys["periods_survived"]
    journeys["observed"] = journeys["churned"]

    if attributes is not None:
        journeys = journeys.join(
            attributes.set_index(user_col) if user_col in attributes.columns
            else attributes, how="left")
    return journeys.reset_index().rename(columns={"index": "user"})


# ---------------------------------------------------------------------------
# Population level
# ---------------------------------------------------------------------------

def retention_curve(journeys: pd.DataFrame) -> pd.Series:
    """P(still engaged at journey period t) - Kaplan-Meier, censoring-aware."""
    from lifelines import KaplanMeierFitter

    km = KaplanMeierFitter().fit(journeys["duration"], journeys["observed"])
    curve = km.survival_function_.iloc[:, 0]
    curve.index.name = "journey_period"
    return curve


def retention_by_segment(journeys: pd.DataFrame, segment_col: str,
                         min_users: int = 20) -> dict[str, pd.Series]:
    """One retention curve per segment value (age band, plan, condition...)."""
    curves = {}
    for value, group in journeys.groupby(segment_col, observed=True, dropna=True):
        if len(group) >= min_users:
            curves[str(value)] = retention_curve(group)
    return curves


def churn_timing(journeys: pd.DataFrame) -> pd.Series:
    """Among churned users: WHEN in the journey they went silent
    (index = journey period of churn, value = user count)."""
    timing = (journeys.loc[journeys["churned"], "churn_period"]
              .astype(int).value_counts().sort_index())
    timing.index.name = "churn_period"
    return timing


def segment_summary(journeys: pd.DataFrame, segment_col: str,
                    checkpoints: tuple[int, ...] = (4, 8, 12),
                    min_users: int = 20) -> pd.DataFrame:
    """The comparison table: per segment - size, churn rate, median lifetime
    (KM, censoring-aware) and retention at checkpoint periods."""
    from lifelines import KaplanMeierFitter

    rows = []
    for value, group in journeys.groupby(segment_col, observed=True, dropna=True):
        if len(group) < min_users:
            continue
        km = KaplanMeierFitter().fit(group["duration"], group["observed"])
        curve = km.survival_function_.iloc[:, 0]
        row = {segment_col: value, "users": len(group),
               "churned_%": round(100 * group["churned"].mean(), 1),
               "median_lifetime": (float(km.median_survival_time_)
                                   if np.isfinite(km.median_survival_time_)
                                   else np.nan)}
        for cp in checkpoints:
            reached = curve[curve.index <= cp]
            row[f"retention@{cp}"] = round(float(reached.iloc[-1]), 3) if len(reached) else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("churned_%", ascending=False)


# ---------------------------------------------------------------------------
# Individual level
# ---------------------------------------------------------------------------

def expected_remaining_periods(curve: pd.Series, survived: int) -> float:
    """E[T - t | T > t] read off a KM curve: how many more periods a user who
    already survived `survived` periods is expected to stay."""
    s_t = float(curve[curve.index <= survived].iloc[-1]) if (curve.index <= survived).any() else 1.0
    if s_t <= 0:
        return 0.0
    tail = curve[curve.index > survived]
    if tail.empty:
        # the user outlived the observed horizon - the honest answer is
        # 'unknown', never 'imminent' (the settled-bundler lesson, again)
        return float("nan")
    # integrate the conditional survival over the observed horizon
    steps = np.diff([survived] + list(tail.index))
    return float(np.sum(tail.values / s_t * steps))


def individual_outlook(journeys: pd.DataFrame,
                       segment_col: str | None = None) -> pd.DataFrame:
    """Per user: status + expected remaining periods from their (segment's)
    retention curve, conditioned on their current journey age. Answers
    'when will THIS user churn' with the population's evidence."""
    out = journeys.copy()
    if segment_col:
        curves = retention_by_segment(out, segment_col)
        overall = retention_curve(out)
        out["expected_remaining_periods"] = [
            round(expected_remaining_periods(
                curves.get(str(row[segment_col]), overall),
                int(row["periods_survived"])), 1)
            if not row["churned"] else 0.0
            for _, row in out.iterrows()]
    else:
        curve = retention_curve(out)
        out["expected_remaining_periods"] = [
            round(expected_remaining_periods(curve, int(s)), 1) if not c else 0.0
            for s, c in zip(out["periods_survived"], out["churned"])]
    out["estimated_churn_period"] = np.where(
        out["churned"], out["churn_period"],
        out["periods_survived"] + out["expected_remaining_periods"])
    return out
