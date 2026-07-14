"""First-boot bootstrap for cloud deployments.

models/ and Dataset/ never enter git, so a fresh deployment starts empty. This
module rebuilds the demo artifacts (lenses A, B, D) from the public IBM Telco
dataset in about a minute, once, on first boot. Local installs with trained
artifacts are never touched: everything is behind an existence check.

Cloud scope is deliberate: uplift (C) and full CLV (E) need heavy/licensed
training data and stay local-only - the app's lens cards already explain that.
"""
from __future__ import annotations

import io

import pandas as pd

from .. import config
from ..loader.data_loader import clean_telco
from ..preprocessing import features
from . import models

# IBM's public mirror of the Telco churn dataset (same columns as the zip).
TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

CORE_ARTIFACTS = ("churn_pipeline", "cox_bundle", "kmeans_telco")


def artifacts_ready() -> bool:
    return all(
        (config.MODEL_DIR / f"{name}.joblib").exists() for name in CORE_ARTIFACTS
    )


def _download_telco() -> pd.DataFrame:
    # requests ships its own CA bundle (certifi) - urllib fails on macOS
    # framework Pythons without installed certificates. requests is already a
    # Streamlit dependency, so it is always present where this runs.
    import requests

    resp = requests.get(TELCO_URL, timeout=60)
    resp.raise_for_status()
    return clean_telco(pd.read_csv(io.BytesIO(resp.content)))


def ensure_artifacts(progress=print) -> bool:
    """Train and save the demo artifacts if any are missing.

    Returns True when a bootstrap run happened, False when artifacts already
    existed. `progress` receives short status strings (print, st.write, ...).
    """
    if artifacts_ready():
        return False

    progress("Downloading the public Telco dataset (~1MB)...")
    telco = _download_telco()

    progress("Training lens A - churn classification...")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline

    eng = features.engineer_telco(telco)
    X_train, _, y_train, _ = features.split_telco(eng)
    pre = features.make_preprocessor(*features.telco_feature_lists(eng))
    pipe = Pipeline([
        ("prep", pre),
        ("clf", HistGradientBoostingClassifier(
            class_weight="balanced", random_state=config.RANDOM_SEED)),
    ]).fit(X_train, y_train)
    models.save_model(
        {"pipeline": pipe, "features": list(X_train.columns)}, "churn_pipeline")

    progress("Training lens B - survival (Cox PH)...")
    sdf = features.encode_for_survival(telco)
    surv = models.train_survival(sdf)
    covariates = [c for c in sdf.columns if c not in ("tenure", "Churn")]
    models.save_model(
        {"model": surv["cox"], "columns": covariates}, "cox_bundle")

    progress("Training lens D - segmentation (K-Means, k=2)...")
    X_all = eng.drop(columns=["customerID", "Churn"])
    km_pipe = models.fit_kmeans_pipeline(
        X_all, features.make_preprocessor(*features.telco_feature_lists(eng)), 2)
    models.save_model(
        {"pipeline": km_pipe, "features": list(X_all.columns)}, "kmeans_telco")

    progress("Bootstrap complete - lenses A, B, D ready.")
    return True
