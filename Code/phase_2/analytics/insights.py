"""Two-altitude analytics for auto-trained datasets (architecture.md sec. 14).

POPULATION - "what should the business do": churn rate, risk-band mix, global
churn drivers, decile lift (when a label exists), segment profile cards,
remaining-tenure stats, total revenue at risk.

INDIVIDUAL - "what about THIS customer": probability + band (from scoring),
plus columns added here: segment membership, expected remaining tenure,
simplified CLV, revenue at risk, and the TOP LOCAL DRIVERS of each
customer's own risk.

Driver method: counterfactual deltas - set one feature to its typical value
(median/mode), re-predict, and the probability drop IS that feature's push on
this customer. Model-agnostic, no extra dependency, needs no label; the mean
absolute delta across customers doubles as the global importance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from phase_1.modeling import load_model
from phase_1.serving.pipeline import artifact_exists

MAX_DRIVER_FEATURES = 15     # counterfactual predicts are one pass per feature
LOCAL_TOP_K = 2
LOCAL_MIN_DELTA = 0.02       # pushes smaller than 2pp are noise, not drivers


def enrich_and_summarize(scored: pd.DataFrame, dataset_id: str,
                         label_col: str = "Churn") -> tuple[pd.DataFrame, dict]:
    """Returns (scored with individual-level columns added, population report).
    Never raises: each lens contributes only when its artifact exists."""
    out = scored.copy()
    population: dict = {}

    bundle = load_model("churn_pipeline", dataset_id=dataset_id)
    pipe, features = bundle["pipeline"], bundle["features"]

    # ---------------------------------------------------------- risk mix
    if "churn_probability" in out.columns:
        population["mean_churn_probability"] = round(
            float(out["churn_probability"].mean()), 4)
        population["risk_bands"] = out["risk_band"].value_counts().to_dict()
    if label_col in out.columns:
        y = _numeric_label(out[label_col])
        population["actual_churn_rate"] = round(float(y.mean()), 4)

    # ---------------------------------------------------------- drivers
    base, deltas = _driver_deltas(pipe, out, features)
    global_importance = deltas.abs().mean().sort_values(ascending=False)
    population["global_drivers"] = [
        {"feature": f, "mean_abs_effect_pp": round(float(v) * 100, 2)}
        for f, v in global_importance.head(10).items()]
    out["local_drivers"] = _local_driver_strings(deltas)

    # ---------------------------------------------------------- lift (labeled)
    if label_col in out.columns and "churn_probability" in out.columns:
        from phase_1.evaluation import lift_table

        lift = lift_table(_numeric_label(out[label_col]),
                          out["churn_probability"].values)
        population["lift_by_decile"] = lift[
            ["decile", "churn_rate", "lift", "cum_capture"]].to_dict("records")
        population["top_decile_lift"] = float(lift["lift"].iloc[0])

    # ---------------------------------------------------------- segments (lens D)
    if artifact_exists("kmeans", dataset_id=dataset_id):
        km = load_model("kmeans", dataset_id=dataset_id)
        out["segment"] = km["pipeline"].predict(out[km["features"]])
        profiles = []
        for seg, group in out.groupby("segment"):
            profile = {"segment": int(seg), "size": len(group),
                       "share": round(len(group) / len(out), 3)}
            if "churn_probability" in group:
                profile["mean_churn_probability"] = round(
                    float(group["churn_probability"].mean()), 4)
            top_num = (group[km["features"]].select_dtypes("number").mean()
                       .round(2).head(4).to_dict())
            profile["numeric_means"] = top_num
            profiles.append(profile)
        population["segments"] = profiles

    # ---------------------------------------------------------- survival (lens B)
    if artifact_exists("cox_bundle", dataset_id=dataset_id) and "tenure" in out.columns:
        remaining = _expected_remaining(out, features, dataset_id)
        if remaining is not None:
            out["expected_remaining_tenure"] = remaining.round(1)
            population["remaining_tenure"] = {
                "mean": round(float(remaining.mean()), 1),
                "median": round(float(remaining.median()), 1),
                "q25": round(float(remaining.quantile(0.25)), 1)}

    # ---------------------------------------------------------- revenue at risk
    if ("MonthlyCharges" in out.columns
            and "expected_remaining_tenure" in out.columns
            and "churn_probability" in out.columns):
        monthly = pd.to_numeric(out["MonthlyCharges"], errors="coerce").fillna(0)
        out["clv_simple"] = (monthly * out["expected_remaining_tenure"]).round(2)
        out["revenue_at_risk"] = (out["churn_probability"] * out["clv_simple"]).round(2)
        ranked = out["revenue_at_risk"].sort_values(ascending=False)
        total = float(ranked.sum())
        top_n = max(1, len(ranked) // 10)
        rng_rar = np.random.default_rng(7)
        vals = out["revenue_at_risk"].to_numpy()
        sums = [float(vals[rng_rar.integers(0, len(vals), len(vals))].sum())
                for _ in range(200)] if len(vals) else []
        population["revenue_at_risk"] = {
            "total": round(total, 2),
            **({"total_ci95": [round(float(np.percentile(sums, 2.5)), 2),
                               round(float(np.percentile(sums, 97.5)), 2)]}
               if sums else {}),
            "top_decile_share": round(float(ranked.head(top_n).sum())
                                      / total, 3) if total else 0.0}

    return out, population


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _numeric_label(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _driver_deltas(pipe, out: pd.DataFrame, features: list[str]):
    """delta[i, f] = p(customer i) - p(customer i with feature f set to its
    typical value). Positive = this feature's value pushes risk UP."""
    X = out[features]
    base = pipe.predict_proba(X)[:, 1]
    deltas = {}
    for f in features[:MAX_DRIVER_FEATURES]:
        counterfactual = X.copy()
        col = counterfactual[f]
        if pd.api.types.is_numeric_dtype(col):
            counterfactual[f] = col.median()
        else:
            mode = col.mode(dropna=True)
            counterfactual[f] = mode.iloc[0] if len(mode) else col.iloc[0]
        deltas[f] = base - pipe.predict_proba(counterfactual)[:, 1]
    return base, pd.DataFrame(deltas, index=out.index)


def _local_driver_strings(deltas: pd.DataFrame) -> list[str]:
    """Per customer: the top features pushing their risk up, human-readable."""
    values = deltas.values
    columns = np.asarray(deltas.columns)
    order = np.argsort(-values, axis=1)[:, :LOCAL_TOP_K]
    strings = []
    for row_idx in range(values.shape[0]):
        parts = [f"{columns[j]} (+{values[row_idx, j] * 100:.0f}pp)"
                 for j in order[row_idx]
                 if values[row_idx, j] >= LOCAL_MIN_DELTA]
        strings.append("; ".join(parts) if parts else "no single dominant driver")
    return strings


def _expected_remaining(out: pd.DataFrame, features: list[str],
                        dataset_id: str) -> pd.Series | None:
    try:
        from ..training.auto_train import prepare_cox_covariates

        bundle = load_model("cox_bundle", dataset_id=dataset_id)
        covariates = [c for c in features if c != "tenure"]
        dummies, _ = prepare_cox_covariates(
            out[covariates], bundle.get("freq_maps") or {})
        dummies = dummies.reindex(columns=bundle["columns"], fill_value=0.0)
        expected_total = bundle["model"].predict_expectation(dummies)
        tenure = pd.to_numeric(out["tenure"], errors="coerce").fillna(0)
        return (pd.Series(expected_total.values, index=out.index) - tenure).clip(lower=0)
    except Exception:
        return None                     # survival insight is optional, never fatal
