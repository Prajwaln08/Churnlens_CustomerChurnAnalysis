"""Dataset-adaptive training - fresh artifacts from whatever columns arrive.

Origin: architecture.md sections 0 (adaptive training), 12 (leakage),
13 (quality gates) - Phase 3 of the build order.

Ten columns train a ten-column model; fifty train a fifty-column model. The
sample datasets' schemas impose nothing. Every lens passes a QUALITY GATE:
an artifact is only PROMOTED (saved under models/<dataset_id>/) when it
clears its minimum bar - otherwise the report says "trained, not promoted"
and scoring stays off rather than pretending.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from phase_1 import config
from phase_1.modeling import fit_kmeans_pipeline, kmeans_scan, save_model
from phase_1.preprocessing import make_preprocessor, split_categoricals

from . import leakage

# quality gates (architecture.md section 13)
GATE_AUC = 0.65          # lens A: below this, the model is not worth serving
GATE_C_INDEX = 0.60      # lens B
KMEANS_KS = range(2, 7)


def train_for_dataset(df: pd.DataFrame, dataset_id: str,
                      label_col: str = "Churn", seed: int | None = None) -> dict:
    """Train lens A (+ B when a duration exists, + D always) on all surviving
    columns. Returns a per-lens report; promoted artifacts land in
    models/<dataset_id>/. Never raises for weak data - it reports."""
    seed = config.RANDOM_SEED if seed is None else seed
    report: dict = {"dataset_id": dataset_id, "lenses": {}}

    screened = leakage.screen(df, label_col)
    data = leakage.derive_datetime_features(df, screened["derived"])
    features = screened["features"] + list(screened["derived"])
    report["leakage"] = {"kept": features, "dropped": screened["dropped"]}

    X = data[features]
    y = leakage._numeric_label(data[label_col]).astype(int)
    numeric = X.select_dtypes("number").columns.tolist()
    categorical_all = [c for c in X.columns if c not in numeric]
    categorical, high_card = split_categoricals(X, categorical_all)
    report["encoding"] = {"one_hot": categorical, "frequency": high_card,
                          "numeric": numeric}

    # ------------------------------------------------------------ lens A
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=stratify, random_state=seed)
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed),
    }
    best_name, best_auc, best_pipe = None, 0.0, None
    for name, est in candidates.items():
        pipe = Pipeline([("prep", make_preprocessor(numeric, categorical,
                                                    high_card)),
                         ("clf", est)]).fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        if auc > best_auc:
            best_name, best_auc, best_pipe = name, auc, pipe
    promoted = best_auc >= GATE_AUC
    if promoted:
        save_model({"pipeline": best_pipe, "features": list(X.columns),
                    "metrics": {"test_roc_auc": round(best_auc, 4)},
                    "model": best_name}, "churn_pipeline", dataset_id=dataset_id)
        # drift baseline: the training data's distributions, for phase-6 checks
        from ..monitoring import drift as drift_mod
        save_model(drift_mod.build_baseline(X), "drift_baseline",
                   dataset_id=dataset_id)
        report["drift_baseline_saved"] = True
    report["lenses"]["A"] = {
        "status": "promoted" if promoted else "trained_not_promoted",
        "model": best_name, "test_roc_auc": round(best_auc, 4),
        "gate": GATE_AUC,
        "reason": (f"test ROC-AUC {best_auc:.3f} >= gate {GATE_AUC}" if promoted
                   else f"test ROC-AUC {best_auc:.3f} below gate {GATE_AUC} - "
                        "artifact not served"),
    }

    # ------------------------------------------------------------ lens B
    if "tenure" in X.columns:
        report["lenses"]["B"] = _train_survival(data, features, y, dataset_id)
    else:
        report["lenses"]["B"] = {"status": "skipped",
                                 "reason": "no duration column (tenure) mapped"}

    # ------------------------------------------------------------ lens D
    try:
        pre = make_preprocessor(numeric, categorical, high_card)
        scan = kmeans_scan(pre.fit_transform(X), ks=KMEANS_KS)
        k = int(scan.loc[scan["silhouette"].idxmax(), "k"])
        km = fit_kmeans_pipeline(
            X, make_preprocessor(numeric, categorical, high_card), k)
        save_model({"pipeline": km, "features": list(X.columns), "k": k},
                   "kmeans", dataset_id=dataset_id)
        report["lenses"]["D"] = {
            "status": "promoted", "k": k,
            "silhouette": round(float(scan["silhouette"].max()), 3),
            "reason": f"k={k} at silhouette peak"}
    except Exception as exc:
        report["lenses"]["D"] = {"status": "failed", "reason": str(exc)}

    return report


def _train_survival(data: pd.DataFrame, features: list[str], y: pd.Series,
                    dataset_id: str) -> dict:
    from lifelines import CoxPHFitter

    covariates = [c for c in features if c != "tenure"]
    frame = pd.get_dummies(data[covariates], drop_first=True).astype(float)
    frame["tenure"] = pd.to_numeric(data["tenure"], errors="coerce").clip(lower=0.5)
    frame["Churn"] = y.values
    frame = frame.dropna(subset=["tenure"])
    try:
        cox = CoxPHFitter(penalizer=0.1).fit(frame, duration_col="tenure",
                                             event_col="Churn")
        c_index = float(cox.concordance_index_)
        if c_index >= GATE_C_INDEX:
            cols = [c for c in frame.columns if c not in ("tenure", "Churn")]
            save_model({"model": cox, "columns": cols}, "cox_bundle",
                       dataset_id=dataset_id)
            return {"status": "promoted", "c_index": round(c_index, 4),
                    "reason": f"C-index {c_index:.3f} >= gate {GATE_C_INDEX}"}
        return {"status": "trained_not_promoted", "c_index": round(c_index, 4),
                "reason": f"C-index {c_index:.3f} below gate {GATE_C_INDEX}"}
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}


def score_for_source(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Score RAW records for a trained source: replays the source's saved
    (confirmed) mapping, cleaning, and derived features - the same recipe the
    training run used - before predicting. This is the entry point for the
    REST/MCP doors, where callers send unmapped data."""
    from phase_1.serving.pipeline import _coerce_telco

    from ..mapping import mapping as mapping_mod
    from ..orchestration.registry import Registry
    from . import leakage
    from . import mode as mode_mod

    dataset_id = mode_mod.dataset_id_for(source_name)
    saved = Registry().latest_mapping(source_name)
    frame = df
    if saved is not None:
        frame, _ = mapping_mod.apply_mapping(df, saved)
    frame = _coerce_telco(frame)
    from phase_1.modeling import load_model
    bundle = load_model("churn_pipeline", dataset_id=dataset_id)
    derived = {f: f[:-5] for f in bundle["features"]
               if f.endswith("_days") and f not in frame.columns
               and f[:-5] in frame.columns}
    if derived:
        frame = leakage.derive_datetime_features(frame, derived)
    return score_with_dataset_model(frame, dataset_id)


def score_with_dataset_model(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Score a frame with ITS OWN promoted artifacts (per-dataset namespace)."""
    from phase_1.modeling import load_model
    from phase_1.serving.pipeline import risk_band

    bundle = load_model("churn_pipeline", dataset_id=dataset_id)
    out = df.copy()
    missing = [c for c in bundle["features"] if c not in out.columns]
    if missing:
        raise ValueError(f"dataset model needs columns: {missing}")
    proba = bundle["pipeline"].predict_proba(out[bundle["features"]])[:, 1]
    out["churn_probability"] = proba.round(4)
    out["risk_band"] = [risk_band(p) for p in proba]
    return out
