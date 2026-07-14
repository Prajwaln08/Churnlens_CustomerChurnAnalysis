"""Drift monitoring - PSI between today's data and the training baseline.

Origin: architecture.md section 4 (Phase 6 of the build order).

At training time, auto_train captures a BASELINE: each feature's distribution
(quantile bins for numerics, level frequencies for categoricals). At scoring
time, this module measures how far the new data has moved using the
Population Stability Index, with the industry-standard reading:

    PSI < 0.10        ok       - stable
    0.10 <= PSI < .25 warn     - drifting, watch it
    PSI >= 0.25       alert    - the model was trained on different data

Crossing the alert threshold produces "retraining recommended" - a system
FLAG; the retrain itself stays a human decision (architecture section 19).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4
WARN_PSI = 0.10
ALERT_PSI = 0.25
NUM_BINS = 10
OTHER = "__other__"


# ---------------------------------------------------------------------------
# Baseline - captured once, at training time
# ---------------------------------------------------------------------------

def build_baseline(X: pd.DataFrame, features: list[str] | None = None) -> dict:
    """Per-feature reference distributions. Numeric -> quantile bin edges +
    proportions; categorical -> level frequencies; constants are skipped."""
    features = features if features is not None else list(X.columns)
    baseline: dict = {"rows": len(X), "features": {}}
    for col in features:
        series = X[col].dropna()
        if series.nunique() <= 1:
            continue                                   # nothing can drift
        if pd.api.types.is_numeric_dtype(series):
            edges = np.unique(np.quantile(series, np.linspace(0, 1, NUM_BINS + 1)))
            if len(edges) < 3:
                continue
            props = _numeric_props(series, edges)
            baseline["features"][col] = {"kind": "numeric",
                                         "edges": edges.tolist(),
                                         "props": props.tolist()}
        else:
            freq = series.astype(str).value_counts(normalize=True)
            baseline["features"][col] = {"kind": "categorical",
                                         "props": freq.to_dict()}
    return baseline


def _numeric_props(series: pd.Series, edges) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    clipped = series.clip(edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    return counts / max(counts.sum(), 1)


# ---------------------------------------------------------------------------
# PSI + comparison
# ---------------------------------------------------------------------------

def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(np.asarray(expected, dtype=float), EPS, None)
    actual = np.clip(np.asarray(actual, dtype=float), EPS, None)
    expected, actual = expected / expected.sum(), actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _status(value: float) -> str:
    return "alert" if value >= ALERT_PSI else "warn" if value >= WARN_PSI else "ok"


def compare(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """One row per baselined feature present in df: psi + status. Features
    missing from df are reported as status 'missing' (that is drift too)."""
    rows = []
    for col, spec in baseline["features"].items():
        if col not in df.columns:
            rows.append({"feature": col, "psi": None, "status": "missing"})
            continue
        series = df[col].dropna()
        if spec["kind"] == "numeric":
            series = pd.to_numeric(series, errors="coerce").dropna()
            value = psi(spec["props"], _numeric_props(series, spec["edges"]))
        else:
            expected_levels = list(spec["props"])
            actual_freq = series.astype(str).value_counts(normalize=True)
            expected = np.array([spec["props"][k] for k in expected_levels] + [EPS])
            actual = np.array([actual_freq.get(k, 0.0) for k in expected_levels]
                              + [actual_freq[~actual_freq.index.isin(expected_levels)].sum()
                                 if (~actual_freq.index.isin(expected_levels)).any()
                                 else 0.0])
            value = psi(expected, actual)
        rows.append({"feature": col, "psi": round(value, 4),
                     "status": _status(value)})
    return pd.DataFrame(rows)


def assess(df: pd.DataFrame, dataset_id: str) -> dict | None:
    """Load the dataset's baseline and judge today's data against it.
    None when no baseline exists (nothing to compare against - reported
    upstream as a skip, never a silent pass)."""
    from phase_1.serving.pipeline import artifact_exists
    from phase_1.modeling import load_model

    if not artifact_exists("drift_baseline", dataset_id=dataset_id):
        return None
    baseline = load_model("drift_baseline", dataset_id=dataset_id)
    # replay datetime-derived training features so they are not falsely
    # reported as 'missing' drift (lazy import avoids a package cycle)
    derived = {c: c[:-5] for c in baseline["features"]
               if c.endswith("_days") and c not in df.columns
               and c[:-5] in df.columns}
    if derived:
        from ..training import leakage
        df = leakage.derive_datetime_features(df, derived)
    table = compare(df, baseline)
    alerts = table.loc[table["status"].isin(("alert", "missing")), "feature"].tolist()
    warns = table.loc[table["status"] == "warn", "feature"].tolist()
    max_psi = float(table["psi"].dropna().max()) if table["psi"].notna().any() else 0.0
    if alerts:
        recommendation = ("retraining recommended - the model was trained on "
                          "different data: " + ", ".join(alerts))
    elif warns:
        recommendation = "drifting - watch these features: " + ", ".join(warns)
    else:
        recommendation = "stable - no material drift against the training baseline"
    return {"table": table.to_dict("records"), "max_psi": round(max_psi, 4),
            "alerts": alerts, "warns": warns, "recommendation": recommendation}
