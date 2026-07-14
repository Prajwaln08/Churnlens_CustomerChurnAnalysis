"""Model training + artifact persistence for all five lenses.

Origin: Code/03_feature_engineering_modeling.ipynb.

Every trainer returns fitted objects AND saves nothing implicitly - the
notebooks decide what becomes an artifact via save_model(), so models/ only
ever contains deliberate, named products (they are what Stage 5 serves).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import config


# --------------------------------------------------------------------------
# Artifact + metrics persistence
# --------------------------------------------------------------------------

def _strip_lifetimes_lambdas(obj) -> None:
    """lifetimes fitters carry a `generate_new_data` lambda that pickle cannot
    serialize (their own save_model strips it too); it is only used for
    simulation, never for prediction, so dropping it is loss-free."""
    if isinstance(obj, dict):
        for value in obj.values():
            _strip_lifetimes_lambdas(value)
    elif hasattr(obj, "__dict__"):
        obj.__dict__.pop("generate_new_data", None)


def save_model(obj, name: str) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _strip_lifetimes_lambdas(obj)
    joblib.dump(obj, config.MODEL_DIR / f"{name}.joblib")


def load_model(name: str):
    return joblib.load(config.MODEL_DIR / f"{name}.joblib")


def update_metrics(section: str, payload: dict) -> None:
    """Merge a lens's metrics into models/metrics.json (the app's Model Info tab)."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if config.METRICS_FILE.exists():
        metrics = json.loads(config.METRICS_FILE.read_text())
    metrics[section] = payload
    config.METRICS_FILE.write_text(json.dumps(metrics, indent=2, default=str))


# --------------------------------------------------------------------------
# Lens A - churn classification
# --------------------------------------------------------------------------

def churn_model_zoo() -> dict[str, object]:
    """Baseline-first: interpretable LR sets the bar the ensembles must beat.
    class_weight handles the 73/27 imbalance without synthetic resampling."""
    seed = config.RANDOM_SEED
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed),
    }


def train_churn_models(
    X_train: pd.DataFrame, y_train: pd.Series, preprocessor
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Fit every zoo model in a leakage-safe Pipeline; report 5-fold CV ROC-AUC."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    fitted, rows = {}, []
    for name, est in churn_model_zoo().items():
        pipe = Pipeline([("prep", preprocessor), ("clf", est)])
        scores = cross_val_score(pipe, X_train, y_train,
                                 scoring="roc_auc", cv=cv, n_jobs=-1)
        fitted[name] = pipe.fit(X_train, y_train)
        rows.append({"model": name,
                     "cv_roc_auc_mean": scores.mean(),
                     "cv_roc_auc_std": scores.std()})
    table = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    return fitted, table.reset_index(drop=True)


def tune_hgb(
    X_train: pd.DataFrame, y_train: pd.Series, preprocessor, n_iter: int = 25
) -> tuple[Pipeline, pd.DataFrame]:
    """Time-boxed random search on the boosted model. This dataset plateaus
    quickly - n_iter=25 is deliberate; more buys ~nothing."""
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", HistGradientBoostingClassifier(
            class_weight="balanced", random_state=config.RANDOM_SEED)),
    ])
    space = {
        "clf__learning_rate": [0.03, 0.06, 0.1, 0.2],
        "clf__max_leaf_nodes": [15, 31, 63],
        "clf__min_samples_leaf": [10, 20, 40],
        "clf__l2_regularization": [0.0, 0.1, 1.0],
        "clf__max_iter": [100, 200, 400],
    }
    search = RandomizedSearchCV(
        pipe, space, n_iter=n_iter, scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_SEED),
        random_state=config.RANDOM_SEED, n_jobs=-1,
    ).fit(X_train, y_train)
    results = (
        pd.DataFrame(search.cv_results_)
        [["params", "mean_test_score", "std_test_score"]]
        .sort_values("mean_test_score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    return search.best_estimator_, results


# --------------------------------------------------------------------------
# Lens B - survival analysis
# --------------------------------------------------------------------------

def train_survival(sdf: pd.DataFrame) -> dict:
    """Kaplan-Meier (whole base), Cox PH and Weibull AFT on the encoded frame.
    sdf: output of features.encode_for_survival (numeric, tenure + Churn)."""
    from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter

    # Parametric AFT models reject non-positive durations; the 11 tenure-0
    # newcomers get half a month so every fitter sees the same data.
    sdf = sdf.assign(tenure=sdf["tenure"].clip(lower=0.5))
    km = KaplanMeierFitter().fit(sdf["tenure"], sdf["Churn"], label="all customers")
    cox = CoxPHFitter(penalizer=0.1).fit(sdf, duration_col="tenure", event_col="Churn")
    aft = WeibullAFTFitter(penalizer=0.01).fit(sdf, duration_col="tenure", event_col="Churn")
    return {
        "km": km, "cox": cox, "aft": aft,
        "c_index": {"cox": float(cox.concordance_index_),
                    "weibull_aft": float(aft.concordance_index_)},
    }


def expected_remaining_tenure(cox, sdf: pd.DataFrame) -> pd.Series:
    """E[total lifetime | covariates] - current tenure, floored at 0.
    A deliberate simplification (documented in notebook 03): good enough to
    rank customers and to price the CLV bridge, not a per-customer guarantee."""
    expected_total = cox.predict_expectation(sdf)
    return (expected_total - sdf["tenure"]).clip(lower=0).rename("expected_remaining_tenure")


# --------------------------------------------------------------------------
# Lens C - uplift
# --------------------------------------------------------------------------

def train_uplift(
    X_train: pd.DataFrame, y_train: pd.Series, t_train: pd.Series
) -> dict[str, object]:
    """Three meta-learner families, same data, judged on Qini in Stage 4.

    ClassTransformation formally assumes ~50/50 treatment propensity - it is
    included for the comparison, with that caveat attached to its results.
    """
    from sklift.models import ClassTransformation, SoloModel, TwoModels

    seed = config.RANDOM_SEED
    solo = SoloModel(HistGradientBoostingClassifier(random_state=seed))
    two = TwoModels(
        estimator_trmnt=HistGradientBoostingClassifier(random_state=seed),
        estimator_ctrl=HistGradientBoostingClassifier(random_state=seed),
        method="vanilla",
    )
    ct = ClassTransformation(Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=seed)),
    ]))
    models = {"solo_model": solo, "two_models": two, "class_transformation": ct}
    for model in models.values():
        model.fit(X_train, y_train, t_train)
    return models


# --------------------------------------------------------------------------
# Lens D - segmentation
# --------------------------------------------------------------------------

def kmeans_scan(X_matrix: np.ndarray, ks=range(2, 9)) -> pd.DataFrame:
    """Inertia (elbow) + silhouette per k. Silhouette on a 5k sample - it is
    O(n^2) and the ranking is stable at that size."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    sample = X_matrix
    if len(X_matrix) > 5000:
        sample_idx = rng.choice(len(X_matrix), 5000, replace=False)
        sample = X_matrix[sample_idx]
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_SEED).fit(X_matrix)
        sample_labels = km.predict(sample)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(sample, sample_labels)})
    return pd.DataFrame(rows)


def fit_kmeans_pipeline(X: pd.DataFrame, preprocessor, k: int) -> Pipeline:
    """Preprocessing + KMeans in ONE pipeline so the saved artifact scores raw
    dataframes (scaling is mandatory for a distance-based method)."""
    return Pipeline([
        ("prep", preprocessor),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_SEED)),
    ]).fit(X)


# --------------------------------------------------------------------------
# Lens E - CLV (BG/NBD + Gamma-Gamma)
# --------------------------------------------------------------------------

def train_clv(summary: pd.DataFrame, penalizer: float = 0.001) -> dict:
    """Fit BG/NBD on (frequency, recency, T); Gamma-Gamma on returning
    customers only (frequency>0, its mathematical domain), after checking the
    frequency-monetary independence assumption."""
    from lifetimes import BetaGeoFitter, GammaGammaFitter

    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    returning = summary[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)]
    corr = returning[["frequency", "monetary_value"]].corr().iloc[0, 1]
    if abs(corr) > 0.3:
        print(f"WARNING: frequency-monetary correlation {corr:.2f} strains the "
              "Gamma-Gamma independence assumption - interpret CLV with care.")
    ggf = GammaGammaFitter(penalizer_coef=penalizer)
    ggf.fit(returning["frequency"], returning["monetary_value"])
    return {"bgf": bgf, "ggf": ggf, "freq_monetary_corr": float(corr)}


def clv_per_customer(
    clv_models: dict, summary: pd.DataFrame,
    months: int = 12, discount_rate: float = 0.01,
) -> pd.Series:
    """12-month discounted CLV per customer (0 for one-time buyers, whom
    Gamma-Gamma cannot price)."""
    returning = summary[(summary["frequency"] > 0) & (summary["monetary_value"] > 0)]
    clv = clv_models["ggf"].customer_lifetime_value(
        clv_models["bgf"],
        returning["frequency"], returning["recency"], returning["T"],
        returning["monetary_value"],
        time=months, discount_rate=discount_rate, freq="D",
    )
    return clv.reindex(summary.index).fillna(0.0).rename("clv")
