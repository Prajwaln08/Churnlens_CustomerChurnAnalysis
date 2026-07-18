"""The leakage screen - what stands between "all columns in" and "model trained".

Origin: architecture.md section 12 (Phase 3 of the build order).

Given a mapped, labeled frame, decide which columns may become features.
Everything dropped or flagged is REPORTED with a reason - never silent.

Categories:
- dropped (mechanical): id-like, constant, the label itself, datetime originals
  (after deriving a numeric days-based feature), long free text;
- dropped (leakage): near-duplicates of the label, near-perfect single-column
  predictors, post-outcome name suspects (auto-pilot drops them; a Level 0 UI
  may resurface them for human override later).
"""
from __future__ import annotations

import pandas as pd

from ..profiling import profiler

# a single column this predictive of the label is an answer key, not a feature
NEAR_PERFECT = 0.98


def screen(df: pd.DataFrame, label_col: str) -> dict:
    """Returns {"features": [...], "derived": {new: from}, "dropped": {col: reason}}.

    `features` are the surviving column names IN df (plus derived ones the
    caller must add via derive_datetime_features before fitting).
    """
    profile = profiler.profile_table(df)
    y = _numeric_label(df[label_col])
    dropped: dict[str, str] = {}
    derived: dict[str, str] = {}
    features: list[str] = []

    for col, p in profile["columns"].items():
        if col == label_col:
            continue
        role = p["role"]
        if role == profiler.CONSTANT:
            dropped[col] = "constant - carries no information"
        elif role == profiler.ID:
            dropped[col] = "id-like - unique tokens would memorize customers"
        elif role == profiler.TEXT:
            dropped[col] = "long free text - out of scope for tabular training"
        elif p["post_outcome_suspect"]:
            dropped[col] = "post-outcome name pattern - likely written after churn"
        elif role == profiler.DATETIME:
            derived[f"{col}_days"] = col
            dropped[col] = f"datetime - replaced by numeric '{col}_days'"
        elif _near_perfect_predictor(df[col], y):
            dropped[col] = (f"predicts the label at >{NEAR_PERFECT:.0%} alone - "
                            "an answer key, not a feature")
        elif _missingness_mirrors_label(df[col], y):
            dropped[col] = ("filled on only one side of the outcome - its "
                            "missingness matches the label at "
                            f">{NEAR_PERFECT:.0%} (post-outcome answer key)")
        else:
            features.append(col)

    # perfect numeric duplicates: one copy carries all the information
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) > 1:
        corr = df[numeric].corr().abs()
        for i, a in enumerate(numeric):
            if a not in features:
                continue
            for b in numeric[i + 1:]:
                if b in features and corr.loc[a, b] >= 0.999:
                    features.remove(b)
                    dropped[b] = (f"linear duplicate of '{a}' "
                                  f"(|corr| {corr.loc[a, b]:.3f}) - one copy "
                                  "is enough")

    return {"features": features, "derived": derived, "dropped": dropped}


def derive_datetime_features(df: pd.DataFrame, derived: dict[str, str]) -> pd.DataFrame:
    """Materialize '<col>_days' = days before the column's own max date -
    a leakage-safe relative age (no wall clock involved)."""
    out = df.copy()
    for new_col, src in derived.items():
        parsed = pd.to_datetime(out[src], errors="coerce")
        out[new_col] = (parsed.max() - parsed).dt.days.astype(float)
    return out


def _numeric_label(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.map({"Yes": 1, "No": 0, "True": 1, "False": 0}).fillna(0).astype(float)
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(float)


def _missingness_mirrors_label(col: pd.Series, y: pd.Series) -> bool:
    """A column filled only for churners (or only for stayers) is written
    AFTER the outcome - its NaN pattern is the label. Imputers with missing
    indicators would hand the model that answer key (the AUC=1.0 bug)."""
    isna = col.isna()
    if isna.sum() < 20 or (~isna).sum() < 20:
        return False
    match = float((isna.astype(float) == y).mean())
    return max(match, 1 - match) >= NEAR_PERFECT


def _near_perfect_predictor(col: pd.Series, y: pd.Series) -> bool:
    """Can this single column alone reproduce the label almost exactly?"""
    non_null = col.notna()
    if non_null.sum() < 20:
        return False
    x, yy = col[non_null], y[non_null]
    if pd.api.types.is_numeric_dtype(x):
        corr = abs(pd.Series(x).corr(yy))
        return bool(corr >= NEAR_PERFECT)
    if x.nunique() <= 50:  # best level->majority-class mapping accuracy
        majority = yy.groupby(x.astype(str)).transform(
            lambda g: (g.mean() >= 0.5) * 1.0)
        return bool((majority == yy).mean() >= NEAR_PERFECT)
    return False
