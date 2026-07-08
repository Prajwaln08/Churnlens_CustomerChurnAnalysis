"""Feature engineering and dataset splitting for all five lenses.

Origin: Code/03_feature_engineering_modeling.ipynb.

Leakage discipline, stated once and enforced everywhere:
split FIRST, then fit every transformer on the training side only - which is
why encoders/scalers live inside sklearn Pipelines rather than mutating frames.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .data_loader import TELCO_SERVICE_COLS

TELCO_ID = "customerID"
TELCO_TARGET = "Churn"


# --------------------------------------------------------------------------
# Telco - lenses A (classification), B (survival), D (segmentation)
# --------------------------------------------------------------------------

def engineer_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features. Each is a business hypothesis, not a fishing trip:

    - tenure_bucket: churn risk is front-loaded in the customer lifecycle
    - services_count: more services = more entanglement = harder to leave
    - avg_monthly_spend: TotalCharges/tenure, robust to the tenure-0 newcomers
    - is_month_to_month: THE contract effect, isolated as a flag
    - has_no_support_services: no OnlineSecurity and no TechSupport - the
      fiber-with-no-safety-net profile EDA flags as high risk
    """
    out = df.copy()
    out["tenure_bucket"] = pd.cut(
        out["tenure"], bins=[-1, 12, 24, 48, 200],
        labels=["0-12", "13-24", "25-48", "49+"],
    ).astype(str)
    out["services_count"] = (out[TELCO_SERVICE_COLS] == "Yes").sum(axis=1)
    out["avg_monthly_spend"] = out["TotalCharges"] / out["tenure"].clip(lower=1)
    out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
    out["has_no_support_services"] = (
        (out["OnlineSecurity"] == "No") & (out["TechSupport"] == "No")
    ).astype(int)
    return out


def telco_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(numeric, categorical) feature names - everything except id and target."""
    feats = df.drop(columns=[TELCO_ID, TELCO_TARGET], errors="ignore")
    numeric = feats.select_dtypes("number").columns.tolist()
    categorical = feats.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Scale numerics, one-hot categoricals. Dense output (HGB/KMeans need it);
    handle_unknown='ignore' keeps scoring robust to unseen categories."""
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                              sparse_output=False), categorical),
    ])


def split_telco(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split - done before ANY fitting (the leakage rule)."""
    X = df.drop(columns=[TELCO_ID, TELCO_TARGET], errors="ignore")
    y = df[TELCO_TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_SEED
    )


def encode_for_survival(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric frame for lifelines: duration=tenure, event=Churn.

    TotalCharges is dropped - it is ~tenure x MonthlyCharges, i.e. it encodes
    the duration itself, which a survival model must not see as a covariate.
    """
    keep = df.drop(columns=[TELCO_ID, "TotalCharges"], errors="ignore")
    encoded = pd.get_dummies(keep, drop_first=True)
    return encoded.astype(float)


# --------------------------------------------------------------------------
# Online Retail II - lenses D (RFM segmentation), E (CLV)
# --------------------------------------------------------------------------

def rfm_table(retail: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Recency/Frequency/Monetary per customer + 1-5 quantile scores + segment.

    rank(method='first') breaks the massive ties (many one-purchase customers)
    that would otherwise make qcut fail.
    """
    snapshot = snapshot or retail["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = retail.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
    )
    rfm["R"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["segment"] = rfm.apply(lambda r: _rfm_segment(r["R"], r["F"]), axis=1)
    return rfm


def _rfm_segment(r: int, f: int) -> str:
    if r >= 4 and f >= 4:
        return "Champions"
    if f >= 4:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "New / promising"
    if r <= 2 and f >= 3:
        return "At risk"          # used to buy often, gone quiet - retention target
    if r <= 2:
        return "Hibernating"
    return "Regular"


def clv_summary(retail: pd.DataFrame) -> pd.DataFrame:
    """Full-history (frequency, recency, T, monetary_value) per customer for
    fitting the production BG/NBD + Gamma-Gamma models."""
    from lifetimes.utils import summary_data_from_transaction_data

    return summary_data_from_transaction_data(
        retail, "CustomerID", "InvoiceDate",
        monetary_value_col="Revenue", freq="D",
    )


def clv_calibration_holdout(retail: pd.DataFrame, holdout_days: int = 180) -> pd.DataFrame:
    """Time-split CLV data: fit on the calibration window, validate on the
    holdout window - the step that makes Lens E falsifiable."""
    from lifetimes.utils import calibration_and_holdout_data

    end = retail["InvoiceDate"].max()
    cal_end = end - pd.Timedelta(days=holdout_days)
    return calibration_and_holdout_data(
        retail, "CustomerID", "InvoiceDate",
        calibration_period_end=cal_end, observation_period_end=end,
        monetary_value_col="Revenue", freq="D",
    )


# --------------------------------------------------------------------------
# MegaFon - lens C (uplift)
# --------------------------------------------------------------------------

def megafon_xty(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(X, treatment 0/1, y). The treatment flag is the causal switch -
    it is passed to uplift models separately and is NEVER a feature."""
    X = df[[c for c in df.columns if c.startswith("X_")]]
    t = df["treatment_group"]
    if t.dtype == object:
        t = t.map({"control": 0, "treatment": 1})
    return X, t.astype(int), df["conversion"].astype(int)


def split_megafon(
    X: pd.DataFrame, t: pd.Series, y: pd.Series, test_size: float = 0.3
):
    """Split stratified on treatment x outcome so both models of a T-learner
    see enough of every cell."""
    strata = t.astype(str) + "_" + y.astype(str)
    return train_test_split(
        X, t, y, test_size=test_size, stratify=strata,
        random_state=config.RANDOM_SEED,
    )
