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

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
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
                      label_col: str = "Churn", seed: int | None = None,
                      cv_folds: int | None = None) -> dict:
    """Train lens A (+ B when a duration exists, + D always) on all surviving
    columns. Returns a per-lens report; promoted artifacts land in
    models/<dataset_id>/. Never raises for weak data - it reports.

    cv_folds=None (default): fast single stratified 80/20 holdout.
    cv_folds=k: k-fold stratified CV decides the winner AND the gate metric
    (steadier estimate for small/noisy datasets); the winner refits on all rows.
    """
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
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed),
    }
    best_name, best_auc, best_pipe, cv_std = None, 0.0, None, None
    brier, calibration_bins, cal_method, auc_ci = None, None, None, None
    threshold, cost_report = None, None
    refit_on_full = False
    if cv_folds:
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
        for name, est in candidates.items():
            pipe = Pipeline([("prep", make_preprocessor(numeric, categorical,
                                                        high_card)),
                             ("clf", est)])
            scores = cross_val_score(pipe, X, y, scoring="roc_auc",
                                     cv=cv, n_jobs=-1)
            if scores.mean() > best_auc:
                best_name, best_auc = name, float(scores.mean())
                cv_std, best_pipe = float(scores.std()), pipe
        cal_method = "isotonic" if len(X) >= 10_000 else "sigmoid"
        try:                                    # winner refits on all rows,
            best_pipe = CalibratedClassifierCV(  # probability-calibrated
                Pipeline([("prep", make_preprocessor(numeric, categorical,
                                                     high_card)),
                          ("clf", clone(candidates[best_name]))]),
                method=cal_method, cv=cv_folds).fit(X, y)
        except Exception as exc:
            cal_method = f"skipped ({type(exc).__name__})"
            best_pipe.fit(X, y)
        evaluation = f"{cv_folds}-fold stratified cv"
        refit_on_full = True
    else:
        # deployment-honest split: when the data carries a date, train on the
        # past and test on the most recent 20% - random splits flatter churn
        # models because tomorrow is never a shuffle of yesterday
        temporal_col = None
        date_cols = [c for c in screened["derived"] if c in X.columns]
        if date_cols:
            cand = min(date_cols, key=lambda c: X[c].isna().mean())
            if X[cand].isna().mean() <= 0.5:
                temporal_col = cand
        X_tr = None
        if temporal_col is not None:
            order = (X[temporal_col].fillna(X[temporal_col].median())
                     .sort_values(ascending=False).index)   # oldest first
            cut = int(len(order) * 0.8)
            tr_ix, te_ix = order[:cut], order[cut:]
            if y.loc[tr_ix].nunique() == 2 and y.loc[te_ix].nunique() == 2:
                X_tr, X_te = X.loc[tr_ix], X.loc[te_ix]
                y_tr, y_te = y.loc[tr_ix], y.loc[te_ix]
                evaluation = (f"temporal holdout - most recent 20% by "
                              f"'{temporal_col[:-5]}'")
        if X_tr is None:
            stratify = (y if y.nunique() == 2
                        and y.value_counts().min() >= 2 else None)
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, stratify=stratify, random_state=seed)
            evaluation = "single stratified holdout (80/20)"
        for name, est in candidates.items():
            pipe = Pipeline([("prep", make_preprocessor(numeric, categorical,
                                                        high_card)),
                             ("clf", est)]).fit(X_tr, y_tr)
            auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
            if auc > best_auc:
                best_name, best_auc, best_pipe = name, auc, pipe
        # calibration: '80% probability' should mean 80% - fit the calibrator
        # on training data only, judge it on the untouched holdout
        cal_method = "isotonic" if len(X_tr) >= 10_000 else "sigmoid"
        try:
            calib = CalibratedClassifierCV(
                Pipeline([("prep", make_preprocessor(numeric, categorical,
                                                     high_card)),
                          ("clf", clone(candidates[best_name]))]),
                method=cal_method, cv=3).fit(X_tr, y_tr)
            p_te = calib.predict_proba(X_te)[:, 1]
            best_pipe = calib
            best_auc = float(roc_auc_score(y_te, p_te))
            brier = round(float(brier_score_loss(y_te, p_te)), 4)
            y_arr = y_te.to_numpy()
            bin_ix = np.clip((p_te * 10).astype(int), 0, 9)
            calibration_bins = [
                {"bin": f"{b_ / 10:.1f}-{(b_ + 1) / 10:.1f}",
                 "predicted": round(float(p_te[bin_ix == b_].mean()), 3),
                 "observed": round(float(y_arr[bin_ix == b_].mean()), 3),
                 "n": int((bin_ix == b_).sum())}
                for b_ in range(10) if (bin_ix == b_).any()]
        except Exception as exc:
            cal_method = f"skipped ({type(exc).__name__})"
        # 95% bootstrap CI: a point AUC on a small holdout deserves error bars
        try:
            probs_te = best_pipe.predict_proba(X_te)[:, 1]
            y_arr_ci = y_te.to_numpy()
            rng_ci = np.random.default_rng(seed)
            boots = []
            for _ in range(200):
                ix = rng_ci.integers(0, len(y_arr_ci), len(y_arr_ci))
                if y_arr_ci[ix].min() != y_arr_ci[ix].max():
                    boots.append(roc_auc_score(y_arr_ci[ix], probs_te[ix]))
            if len(boots) >= 50:
                auc_ci = [round(float(np.percentile(boots, 2.5)), 4),
                          round(float(np.percentile(boots, 97.5)), 4)]
        except Exception:
            pass
        # cost-based decision threshold: a missed churner is worth more than
        # one retention offer, so 0.5 is rarely the right cut. The full
        # FN/FP curve is reported so the UI can re-optimize for any costs.
        try:
            probs_th = best_pipe.predict_proba(X_te)[:, 1]
            y_th = y_te.to_numpy()
            curve = []
            for t_ in [round(0.05 * i, 2) for i in range(1, 20)]:
                pred = probs_th >= t_
                curve.append({"t": t_,
                              "fn": int(((~pred) & (y_th == 1)).sum()),
                              "fp": int((pred & (y_th == 0)).sum())})
            cost_fn, cost_fp = 5, 1        # default: missed churner = 5x offer
            threshold = min(curve,
                            key=lambda c: c["fn"] * cost_fn + c["fp"] * cost_fp)["t"]
            cost_report = {"decision_threshold": threshold,
                           "cost_assumption": "missed churner = 5x offer cost",
                           "cost_curve": curve}
        except Exception:
            threshold, cost_report = None, None
        # every metric above came from the untouched holdout; the SERVED
        # artifact now refits on all rows so no training data is wasted
        try:
            final_cfg = Pipeline([("prep", make_preprocessor(
                numeric, categorical, high_card)),
                ("clf", clone(candidates[best_name]))])
            if cal_method and not str(cal_method).startswith("skipped"):
                best_pipe = CalibratedClassifierCV(
                    final_cfg, method=cal_method, cv=3).fit(X, y)
            else:
                best_pipe = final_cfg.fit(X, y)
            refit_on_full = True
        except Exception:
            pass                    # keep the holdout-fitted model, say so
    promoted = best_auc >= GATE_AUC
    if promoted:
        save_model({"pipeline": best_pipe, "features": list(X.columns),
                    "metrics": {"test_roc_auc": round(best_auc, 4)},
                    "threshold": threshold,
                    "model": best_name}, "churn_pipeline", dataset_id=dataset_id)
        # drift baseline: the training data's distributions, for phase-6 checks
        from ..monitoring import drift as drift_mod
        save_model(drift_mod.build_baseline(X), "drift_baseline",
                   dataset_id=dataset_id)
        report["drift_baseline_saved"] = True
    report["lenses"]["A"] = {
        "status": "promoted" if promoted else "trained_not_promoted",
        "model": best_name, "test_roc_auc": round(best_auc, 4),
        "evaluation": evaluation,
        **({"cv_std": round(cv_std, 4)} if cv_std is not None else {}),
        **({"auc_ci95": auc_ci} if auc_ci else {}),
        **({"cost": cost_report} if cost_report else {}),
        "refit_on_full": refit_on_full,
        **({"brier": brier} if brier is not None else {}),
        **({"calibration_method": cal_method} if cal_method else {}),
        **({"calibration_bins": calibration_bins} if calibration_bins else {}),
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


MAX_COX_LEVELS = 20      # categoricals beyond this become one frequency column
MAX_COX_ROWS = 50_000    # Cox fit sample cap - ample statistically, safe in RAM


def prepare_cox_covariates(frame: pd.DataFrame,
                           freq_maps: dict | None = None
                           ) -> tuple[pd.DataFrame, dict]:
    """Impute + encode Cox covariates without column explosions. A categorical
    beyond MAX_COX_LEVELS becomes its level frequency (one numeric column)
    instead of thousands of dummies - the raw get_dummies path is what killed
    training on wide real-world files. The maps are saved with the bundle so
    scoring replays the training encoding exactly."""
    out = frame.copy()
    fitting = freq_maps is None
    maps = {} if fitting else dict(freq_maps)
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].median())
        else:
            out[col] = out[col].fillna("missing").astype(str)
            if col in maps:
                out[col] = out[col].map(maps[col]).fillna(0.0)
            elif out[col].nunique() > MAX_COX_LEVELS:
                m = out[col].value_counts(normalize=True).to_dict()
                if fitting:
                    maps[col] = m
                out[col] = out[col].map(m).fillna(0.0)
    return pd.get_dummies(out, drop_first=True).astype(float), maps


def _train_survival(data: pd.DataFrame, features: list[str], y: pd.Series,
                    dataset_id: str) -> dict:
    from lifelines import CoxPHFitter

    covariates = [c for c in features if c != "tenure"]
    frame, freq_maps = prepare_cox_covariates(data[covariates])
    frame["tenure"] = pd.to_numeric(data["tenure"], errors="coerce").clip(lower=0.5)
    frame["Churn"] = y.values
    frame = frame.dropna(subset=["tenure"])
    if len(frame) > MAX_COX_ROWS:      # the fit is O(rows x cols^2) in RAM
        frame = frame.sample(MAX_COX_ROWS, random_state=42)
    try:
        cox = CoxPHFitter(penalizer=0.1).fit(frame, duration_col="tenure",
                                             event_col="Churn")
        c_index = float(cox.concordance_index_)
        if c_index >= GATE_C_INDEX:
            cols = [c for c in frame.columns if c not in ("tenure", "Churn")]
            save_model({"model": cox, "columns": cols,
                        "freq_maps": freq_maps}, "cox_bundle",
                       dataset_id=dataset_id)
            return {"status": "promoted", "c_index": round(c_index, 4),
                    "reason": f"C-index {c_index:.3f} >= gate {GATE_C_INDEX}"}
        return {"status": "trained_not_promoted", "c_index": round(c_index, 4),
                "reason": f"C-index {c_index:.3f} below gate {GATE_C_INDEX}"}
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}


def train_uplift_for_campaign(df: pd.DataFrame,
                              treatment_col: str = "treatment_group",
                              outcome_col: str = "conversion",
                              dataset_id: str | None = None
                              ) -> tuple[pd.DataFrame | None, dict]:
    """Uplift on THIS campaign's own features: a cross-fitted T-learner.
    Every row is scored by models that never saw it (out-of-fold), so the
    scores are honest rather than self-graded. Returns (scored frame, report)
    or (None, {"reason": ...}) when the file cannot support a model."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline

    feats = [c for c in df.columns if c not in (treatment_col, outcome_col)]
    feats = [c for c in feats if df[c].nunique() > 1]
    if not feats:
        return None, {"reason": "no usable campaign features beyond "
                                "treatment and outcome"}
    t_raw = df[treatment_col].astype(str).str.strip().str.lower()
    treated = t_raw.isin(("treatment", "treated", "test", "1", "true", "yes"))
    y = pd.to_numeric(df[outcome_col], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        return None, {"reason": "conversion never varies - nothing to learn"}
    n_t, n_c = int(treated.sum()), int((~treated).sum())
    if min(n_t, n_c) < 50:
        return None, {"reason": f"needs at least 50 rows per arm "
                                f"(treatment {n_t:,}, control {n_c:,})"}

    X = df[feats]
    numeric = X.select_dtypes("number").columns.tolist()
    categorical, high_card = split_categoricals(
        X, [c for c in feats if c not in numeric])
    uplift = pd.Series(0.0, index=df.index)
    strata = y.astype(str) + "_" + treated.astype(str)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for tr_pos, va_pos in skf.split(X, strata):
        tr, va = df.index[tr_pos], df.index[va_pos]

        def arm_proba(arm_mask):
            rows = tr[arm_mask.loc[tr]]
            yy = y.loc[rows]
            if yy.nunique() < 2:       # an arm-fold with one outcome value
                return pd.Series(float(yy.iloc[0]), index=va)
            pipe = Pipeline([
                ("prep", make_preprocessor(numeric, categorical, high_card)),
                ("clf", HistGradientBoostingClassifier(random_state=42))])
            pipe.fit(X.loc[rows], yy)
            return pd.Series(pipe.predict_proba(X.loc[va])[:, 1], index=va)

        uplift.loc[va] = arm_proba(treated) - arm_proba(~treated)
    out = df.copy()
    out["uplift_score"] = uplift.round(4)
    report = {"engine": "campaign-specific T-learner (out-of-fold)",
              "features_used": len(feats), "treated": n_t, "control": n_c}
    try:            # targeting quality of the out-of-fold scores
        from sklift.metrics import qini_auc_score
        report["qini_auc"] = round(float(qini_auc_score(
            y, uplift, treated.astype(int))), 4)
    except Exception:
        pass
    if dataset_id:   # persist final arms fit on all rows, for instant reuse
        try:
            def _fit_full(mask):
                pipe = Pipeline([
                    ("prep", make_preprocessor(numeric, categorical,
                                               high_card)),
                    ("clf", HistGradientBoostingClassifier(random_state=42))])
                return pipe.fit(X[mask], y[mask])
            if y[treated].nunique() == 2 and y[~treated].nunique() == 2:
                save_model({"model_t": _fit_full(treated),
                            "model_c": _fit_full(~treated),
                            "features": feats,
                            "treatment_col": treatment_col,
                            "outcome_col": outcome_col,
                            "report": report},
                           "uplift_campaign", dataset_id=dataset_id)
                report["persisted"] = True
        except Exception:
            pass
    return out, report


def score_campaign_with_saved(df: pd.DataFrame, dataset_id: str
                              ) -> tuple[pd.DataFrame | None, dict | None]:
    """Score a campaign file with the saved campaign-specific arms, when the
    features line up. Returns (None, None) when there is nothing to reuse."""
    from phase_1.modeling import load_model
    from phase_1.serving.pipeline import artifact_exists

    if not artifact_exists("uplift_campaign", dataset_id=dataset_id):
        return None, None
    try:
        bundle = load_model("uplift_campaign", dataset_id=dataset_id)
    except Exception:
        return None, None
    if any(c not in df.columns for c in bundle["features"]):
        return None, None
    X = df[bundle["features"]]
    up = (bundle["model_t"].predict_proba(X)[:, 1]
          - bundle["model_c"].predict_proba(X)[:, 1])
    out = df.copy()
    out["uplift_score"] = np.round(up, 4)
    rep = dict(bundle.get("report") or {})
    rep["engine"] = "campaign-specific T-learner (saved artifact)"
    return out, rep


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
    return score_with_dataset_model(frame, dataset_id)


def score_with_dataset_model(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Score a frame with ITS OWN promoted artifacts (per-dataset namespace)."""
    from phase_1.modeling import load_model
    from phase_1.serving.pipeline import risk_band

    bundle = load_model("churn_pipeline", dataset_id=dataset_id)
    out = df.copy()
    # replay any datetime-derived training features ('<col>_days') so every
    # caller - orchestrator, REST, MCP - is derivation-safe
    derived = {f: f[:-5] for f in bundle["features"]
               if f.endswith("_days") and f not in out.columns
               and f[:-5] in out.columns}
    if derived:
        from . import leakage
        out = leakage.derive_datetime_features(out, derived)
    missing = [c for c in bundle["features"] if c not in out.columns]
    if missing:
        raise ValueError(f"dataset model needs columns: {missing}")
    proba = bundle["pipeline"].predict_proba(out[bundle["features"]])[:, 1]
    out["churn_probability"] = proba.round(4)
    out["risk_band"] = [risk_band(p, threshold=bundle.get("threshold"))
                        for p in proba]
    return out
