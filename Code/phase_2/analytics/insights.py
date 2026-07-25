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
    base, deltas, driver_method = _driver_deltas(pipe, out, features)
    population["driver_method"] = driver_method
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
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    return s.isin(("yes", "y", "true", "1", "churned")).astype(int)


def _driver_deltas(pipe, out: pd.DataFrame, features: list[str]):
    """Per-customer signed attributions per ORIGINAL column, in probability
    points. SHAP (exact, tree path) when the served model wraps HGB and the
    frame is small enough to stay interactive; otherwise the counterfactual
    median-swap. Returns (base_probs, deltas frame, method string)."""
    X = out[features]
    base = pipe.predict_proba(X)[:, 1]
    if len(X) <= 50_000:
        shap_deltas = _shap_deltas(pipe, X)
        if shap_deltas is not None:
            return base, shap_deltas, "shap (tree)"
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
    return base, pd.DataFrame(deltas, index=out.index), "counterfactual swap"


def _match_original(encoded: str, columns) -> str | None:
    """'plan_b' -> 'plan'; 'tenure' -> 'tenure'. Longest matching original."""
    if encoded in columns:
        return encoded
    hits = [c for c in columns if encoded.startswith(f"{c}_")]
    return max(hits, key=len) if hits else None


def _shap_deltas(pipe, X: pd.DataFrame) -> pd.DataFrame | None:
    """Exact tree-path SHAP for the underlying HGB, aggregated from encoded
    features back to original columns and expressed as probability points
    (sigmoid of the log-odds with and without each contribution). Returns
    None whenever the artifact shape does not support it - the caller falls
    back to counterfactuals, never fails."""
    try:
        import shap
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import Pipeline as SkPipeline

        est = pipe
        if isinstance(est, CalibratedClassifierCV):
            est = est.calibrated_classifiers_[0].estimator
        if not isinstance(est, SkPipeline):
            return None
        prep, clf = est.named_steps["prep"], est.named_steps["clf"]
        if clf.__class__.__name__ != "HistGradientBoostingClassifier":
            return None
        Xt = prep.transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        names = list(prep.get_feature_names_out())
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[-1]
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        groups: dict[str, list[int]] = {}
        for j, name in enumerate(names):
            orig = _match_original(name.split("__", 1)[-1], X.columns)
            if orig:
                groups.setdefault(orig, []).append(j)
        if not groups:
            return None
        # each row's log-odds = explainer baseline + sum of contributions
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = np.asarray(expected).ravel()[-1]
        row_logit = expected + sv.sum(axis=1)
        p_full = 1 / (1 + np.exp(-row_logit))
        deltas = {}
        for orig, ix in groups.items():
            contrib = sv[:, ix].sum(axis=1)
            p_without = 1 / (1 + np.exp(-(row_logit - contrib)))
            deltas[orig] = p_full - p_without
        return pd.DataFrame(deltas, index=X.index)
    except Exception:
        return None


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
