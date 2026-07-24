"""Fairness audit - do risk scores treat groups differently?

Neutral reporting, no automatic verdicts: for each user-flagged sensitive
column this module reports, per group, how the model scores them and - once
realized outcomes exist - how often it is right about them (TPR/FPR). Gaps
beyond a stated threshold are flagged for a human to judge; correlation
here is not causation, and a gap is a question, not a conviction.
"""
from __future__ import annotations

import pandas as pd

MIN_GROUP = 30           # groups smaller than this stay unreported (noise)
GAP_FLAG = 0.10          # probability-gap between groups worth flagging


def audit_column(scored: pd.DataFrame, col: str,
                 actual: pd.Series | None = None,
                 threshold: float = 0.5) -> dict:
    """Per-group score statistics for one sensitive column.

    Returns {"column", "groups": [...], "prob_gap", "flagged", and, when
    actuals are supplied, per-group tpr/fpr}.
    """
    if col not in scored.columns or "churn_probability" not in scored.columns:
        return {"column": col, "groups": [], "prob_gap": None, "flagged": False}
    frame = pd.DataFrame({
        "group": scored[col].fillna("missing").astype(str),
        "p": pd.to_numeric(scored["churn_probability"], errors="coerce"),
    }).dropna(subset=["p"])
    if "risk_band" in scored.columns:
        frame["high"] = scored["risk_band"].isin(("High", "Critical")).values
    if actual is not None:
        frame["actual"] = actual.values
        frame["flag"] = frame["p"] >= threshold

    rows = []
    for value, g in frame.groupby("group"):
        if len(g) < MIN_GROUP:
            continue
        row = {"group": value, "customers": int(len(g)),
               "mean_probability": round(float(g["p"].mean()), 4)}
        if "high" in g.columns:
            row["high_or_critical_share"] = round(float(g["high"].mean()), 4)
        if "actual" in g.columns and g["actual"].nunique() >= 1:
            pos = g[g["actual"] == 1]
            neg = g[g["actual"] == 0]
            if len(pos) >= 10:
                row["tpr"] = round(float(pos["flag"].mean()), 4)
            if len(neg) >= 10:
                row["fpr"] = round(float(neg["flag"].mean()), 4)
        rows.append(row)
    if not rows:
        return {"column": col, "groups": [], "prob_gap": None, "flagged": False}
    probs = [r["mean_probability"] for r in rows]
    gap = round(max(probs) - min(probs), 4)
    return {"column": col, "groups": rows, "prob_gap": gap,
            "flagged": gap >= GAP_FLAG}
