"""Delayed-label monitoring - predictions are promises, this module grades
them when the true outcomes arrive weeks later.

Drift (PSI) can only say the INPUTS changed; realized performance says
whether the model actually held up. Falling below the training gate flips
the recommendation to retrain - the honest closing of the loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..training.auto_train import GATE_AUC

_TRUTHY = {"yes", "1", "true", "churned", "y"}


def _to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0) > 0).astype(int)
    return series.astype(str).str.strip().str.lower().isin(_TRUTHY).astype(int)


def realized_performance(scored: pd.DataFrame, actuals: pd.DataFrame,
                         id_col: str, actual_col: str,
                         scored_id_col: str = "customerID") -> dict:
    """Join actual outcomes to a run's scored output and grade the model.

    Returns {"status": "ok", "matched", "realized_auc", "precision_top_decile",
    "actual_churn_rate", "gate", "verdict"} or {"status": "insufficient"...}.
    """
    if scored_id_col not in scored.columns:
        return {"status": "insufficient",
                "reason": f"scored output has no '{scored_id_col}' column to "
                          "join on"}
    if "churn_probability" not in scored.columns:
        return {"status": "insufficient",
                "reason": "scored output carries no churn_probability"}

    truth = pd.DataFrame({
        "customer_id": actuals[id_col].astype(str).str.strip(),
        "actual": _to_binary(actuals[actual_col]),
    }).drop_duplicates("customer_id")
    left = pd.DataFrame({
        "customer_id": scored[scored_id_col].astype(str).str.strip(),
        "predicted": pd.to_numeric(scored["churn_probability"], errors="coerce"),
    }).drop_duplicates("customer_id")
    joined = left.merge(truth, on="customer_id", how="inner").dropna(
        subset=["predicted"])

    if len(joined) < 20:
        return {"status": "insufficient",
                "reason": f"only {len(joined)} customers matched between the "
                          "actuals file and this run's scored output"}
    if joined["actual"].nunique() < 2:
        return {"status": "insufficient",
                "reason": "matched actuals contain a single outcome value - "
                          "nothing to grade against"}

    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(joined["actual"], joined["predicted"]))
    top_n = max(1, len(joined) // 10)
    top = joined.nlargest(top_n, "predicted")
    precision_top = float(top["actual"].mean())
    rate = float(joined["actual"].mean())
    healthy = auc >= GATE_AUC
    return {
        "status": "ok",
        "matched": int(len(joined)),
        "realized_auc": round(auc, 4),
        "precision_top_decile": round(precision_top, 4),
        "actual_churn_rate": round(rate, 4),
        "gate": GATE_AUC,
        "verdict": ("model holding up - realized AUC above the serving gate"
                    if healthy else
                    "retrain recommended - realized AUC fell below the "
                    "serving gate"),
        "healthy": healthy,
        "frame": joined,
    }
