"""ChurnLens pipeline: schema contracts, validation, lens detection, scoring.

Origin: Code/05_pipeline_tool.ipynb.

The tool's promise: hand it a DataFrame/file, it tells you exactly which of
the five lenses your data supports (and why not, when not), then scores every
supported lens using the artifacts trained in notebook 03. Failures are named
and human-readable; no silent column drops.

CLI:
    python -m phase_1.serving.pipeline score --input new_customers.csv --output scored.csv
    python -m phase_1.serving.pipeline train --data labeled_telco.csv
    python -m phase_1.serving.pipeline contracts
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..modeling import models
from ..preprocessing import features

# --------------------------------------------------------------------------
# Schema contracts - the tool's public data requirements
# --------------------------------------------------------------------------

TELCO_FEATURE_KINDS = {
    "gender": "categorical", "SeniorCitizen": "categorical", "Partner": "categorical",
    "Dependents": "categorical", "tenure": "numeric", "PhoneService": "categorical",
    "MultipleLines": "categorical", "InternetService": "categorical",
    "OnlineSecurity": "categorical", "OnlineBackup": "categorical",
    "DeviceProtection": "categorical", "TechSupport": "categorical",
    "StreamingTV": "categorical", "StreamingMovies": "categorical",
    "Contract": "categorical", "PaperlessBilling": "categorical",
    "PaymentMethod": "categorical", "MonthlyCharges": "numeric",
    "TotalCharges": "numeric",
}

CONTRACTS: dict[str, dict] = {
    "A": {
        "name": "Churn prediction",
        "requires": TELCO_FEATURE_KINDS,
        "optional": ["customerID", "Churn"],
        "artifact": "churn_pipeline",
    },
    "B": {
        "name": "Survival (expected remaining tenure)",
        "requires": TELCO_FEATURE_KINDS,   # duration col 'tenure' is inside
        "optional": ["customerID", "Churn"],
        "artifact": "cox_bundle",
    },
    "C": {
        "name": "Uplift targeting",
        "requires": {"treatment_group": "any", "conversion": "numeric"},
        "optional": [],
        "artifact": "uplift_model",
    },
    "D": {
        "name": "Segmentation",
        "requires": TELCO_FEATURE_KINDS,
        "optional": ["customerID"],
        "artifact": "kmeans_telco",
    },
    "E": {
        "name": "CLV (transaction history)",
        "requires": {"CustomerID": "any", "InvoiceDate": "datetime"},
        "optional": ["Revenue", "Quantity", "Price", "Invoice"],
        "artifact": "clv_models",
    },
}

RISK_BANDS = [(0.30, "Low"), (0.60, "Medium"), (0.80, "High"), (1.01, "Critical")]


def artifact_exists(name: str, dataset_id: str | None = None) -> bool:
    directory = config.MODEL_DIR / dataset_id if dataset_id else config.MODEL_DIR
    return (directory / f"{name}.joblib").exists()


# --------------------------------------------------------------------------
# Validation & lens detection
# --------------------------------------------------------------------------

def _column_ok(df: pd.DataFrame, col: str, kind: str) -> tuple[bool, str | None]:
    """(satisfied, coercion_note). A numeric column that arrives as text but
    is >=95% parseable counts as satisfied-with-coercion, not missing."""
    if col not in df.columns:
        return False, None
    s = df[col]
    if kind == "numeric" and not pd.api.types.is_numeric_dtype(s):
        parseable = pd.to_numeric(s, errors="coerce").notna().mean()
        if parseable >= 0.95:
            return True, f"{col}: text -> numeric ({parseable:.0%} parseable)"
        return False, None
    if kind == "datetime" and not pd.api.types.is_datetime64_any_dtype(s):
        parseable = pd.to_datetime(s, errors="coerce").notna().mean()
        if parseable >= 0.95:
            return True, f"{col}: text -> datetime ({parseable:.0%} parseable)"
        return False, None
    return True, None


def validate(df: pd.DataFrame) -> dict:
    """Per-lens report: ok / missing columns / coercions / artifact status,
    plus the list of uploaded columns no satisfied lens uses (never silent)."""
    report: dict = {"lenses": {}, "n_rows": len(df)}
    used_cols: set[str] = set()
    # Lens E can derive Revenue from Quantity x Price - handled as a special case.
    for lens, contract in CONTRACTS.items():
        missing, coercions = [], []
        for col, kind in contract["requires"].items():
            ok, note = _column_ok(df, col, kind)
            if not ok:
                missing.append(col)
            else:
                used_cols.add(col)
                if note:
                    coercions.append(note)
        if lens == "E" and not missing:
            if "Revenue" not in df.columns and not {"Quantity", "Price"} <= set(df.columns):
                missing.append("Revenue (or Quantity + Price)")
        has_artifact = artifact_exists(contract["artifact"])
        ok = not missing
        message = (
            f"ready ({contract['name']})" if ok and has_artifact else
            f"data ok, but artifact '{contract['artifact']}' is not trained yet"
            if ok else
            f"disabled - missing: {', '.join(missing)}"
        )
        report["lenses"][lens] = {
            "name": contract["name"], "ok": ok, "artifact_available": has_artifact,
            "missing": missing, "coercions": coercions, "message": message,
        }
        if ok:
            used_cols.update(c for c in contract["optional"] if c in df.columns)
    report["ignored_columns"] = sorted(set(df.columns) - used_cols)
    return report


def detect_lenses(df: pd.DataFrame) -> list[str]:
    """Lenses that are both satisfied by the data AND have a trained artifact."""
    rep = validate(df)
    return [k for k, v in rep["lenses"].items() if v["ok"] and v["artifact_available"]]


# --------------------------------------------------------------------------
# Scoring - customer-snapshot schema (lenses A, B, D + simplified E -> RaR)
# --------------------------------------------------------------------------

def _coerce_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same normalizations the training data got (data_loader.clean_telco),
    so raw exports score identically to cleaned ones."""
    from ..loader.data_loader import TELCO_SERVICE_COLS

    out = df.copy()
    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        if col in out.columns and not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "SeniorCitizen" in out.columns and pd.api.types.is_numeric_dtype(out["SeniorCitizen"]):
        out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"})
    service_cols = [c for c in TELCO_SERVICE_COLS if c in out.columns]
    if service_cols:
        out[service_cols] = out[service_cols].replace(
            {"No internet service": "No", "No phone service": "No"})
    return out


def risk_band(p: float) -> str:
    for cutoff, band in RISK_BANDS:
        if p < cutoff:
            return band
    return "Critical"


def recommend_action(row: pd.Series) -> str:
    """Rule-of-thumb next step per customer - a starting point for the
    retention team, not a causal claim (that is Lens C's job)."""
    p = row.get("churn_probability", 0)
    if p < 0.30:
        return "Monitor"
    if row.get("Contract") == "Month-to-month":
        return "Offer 1-year contract discount"
    if row.get("TechSupport") == "No" and row.get("InternetService") == "Fiber optic":
        return "Bundle tech support"
    return "Retention call"


def score(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Score a customer-snapshot frame with every available lens.

    Returns (enriched copy, report). Each lens is independent: one lens
    failing (e.g. missing artifact) never blocks the others.
    """
    report = validate(df)
    report["applied"], report["errors"] = [], {}
    out = _coerce_telco(df)

    telco_ready = report["lenses"]["A"]["ok"]
    eng = features.engineer_telco(out) if telco_ready else None

    # Lens A - churn probability
    if telco_ready and artifact_exists("churn_pipeline"):
        try:
            bundle = models.load_model("churn_pipeline")
            proba = bundle["pipeline"].predict_proba(eng[bundle["features"]])[:, 1]
            out["churn_probability"] = np.round(proba, 4)
            out["risk_band"] = [risk_band(p) for p in proba]
            report["applied"].append("A")
        except Exception as exc:  # named, never silent
            report["errors"]["A"] = str(exc)

    # Lens B - expected remaining tenure
    if telco_ready and artifact_exists("cox_bundle"):
        try:
            bundle = models.load_model("cox_bundle")
            dummies = pd.get_dummies(
                out.drop(columns=["customerID", "TotalCharges"], errors="ignore")
            ).astype(float, errors="ignore")
            covars = dummies.reindex(columns=bundle["columns"], fill_value=0.0)
            expected_total = bundle["model"].predict_expectation(covars)
            out["expected_remaining_tenure"] = (
                (expected_total - out["tenure"]).clip(lower=0).round(1).values
            )
            report["applied"].append("B")
        except Exception as exc:
            report["errors"]["B"] = str(exc)

    # Lens D - segment
    if telco_ready and artifact_exists("kmeans_telco"):
        try:
            bundle = models.load_model("kmeans_telco")
            out["segment"] = bundle["pipeline"].predict(eng[bundle["features"]])
            report["applied"].append("D")
        except Exception as exc:
            report["errors"]["D"] = str(exc)

    # Simplified CLV + the synthesis (needs A and B)
    if {"churn_probability", "expected_remaining_tenure"} <= set(out.columns):
        out["clv_simple"] = (out["MonthlyCharges"] * out["expected_remaining_tenure"]).round(2)
        out["revenue_at_risk"] = (out["churn_probability"] * out["clv_simple"]).round(2)
        out["recommended_action"] = out.apply(recommend_action, axis=1)
        report["applied"].append("E-simplified")

    # Lens C - uplift scores (campaign-style schema)
    if report["lenses"]["C"]["ok"] and artifact_exists("uplift_model"):
        try:
            bundle = models.load_model("uplift_model")
            X = df[bundle["features"]]
            out["uplift_score"] = np.round(bundle["model"].predict(X), 4)
            report["applied"].append("C")
        except Exception as exc:
            report["errors"]["C"] = str(exc)

    return out, report


def survival_curve(df: pd.DataFrame, max_months: int = 72) -> pd.DataFrame | None:
    """P(still a customer) per month for each row, from the Cox artifact.

    Returns a frame indexed by month (0..max_months), one column per input row.
    None when the data does not satisfy lens B or the artifact is missing.
    Horizon note: the training data observes at most 72 months of tenure -
    the curve is not extrapolated beyond that.
    """
    rep = validate(df)
    if not rep["lenses"]["B"]["ok"] or not artifact_exists("cox_bundle"):
        return None
    bundle = models.load_model("cox_bundle")
    out = _coerce_telco(df)
    dummies = pd.get_dummies(
        out.drop(columns=["customerID", "TotalCharges"], errors="ignore")
    ).astype(float, errors="ignore")
    covars = dummies.reindex(columns=bundle["columns"], fill_value=0.0)
    times = list(range(0, max_months + 1))
    return bundle["model"].predict_survival_function(covars, times=times)


# --------------------------------------------------------------------------
# Scoring - transaction schema (lens E, full BG/NBD + Gamma-Gamma)
# --------------------------------------------------------------------------

def score_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Customer-level CLV + RFM segment from a raw transaction log."""
    report = validate(df)
    report["applied"], report["errors"] = [], {}
    tx = df.copy()
    tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"], errors="coerce")
    if "Revenue" not in tx.columns:
        tx["Revenue"] = tx["Quantity"] * tx["Price"]
    if "Invoice" not in tx.columns:
        tx["Invoice"] = tx["InvoiceDate"].astype(str)  # fallback order id
    tx = tx.dropna(subset=["CustomerID", "InvoiceDate"])
    tx = tx[tx["Revenue"] > 0]

    out = features.rfm_table(tx)
    if artifact_exists("clv_models"):
        try:
            clv_models = models.load_model("clv_models")
            summary = features.clv_summary(tx)
            out = out.join(summary, how="left")
            out["clv"] = models.clv_per_customer(clv_models, summary)
            report["applied"].append("E")
        except Exception as exc:
            report["errors"]["E"] = str(exc)
    return out.reset_index(), report


def score_any(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Route by schema: transaction log -> lens E; otherwise snapshot scoring."""
    rep = validate(df)
    if rep["lenses"]["E"]["ok"] and not rep["lenses"]["A"]["ok"]:
        return score_transactions(df)
    return score(df)


# --------------------------------------------------------------------------
# Retraining + contracts on disk
# --------------------------------------------------------------------------

def train_from_file(path: str | Path) -> dict:
    """Retrain Lens A on a labeled snapshot file; artifacts are timestamped
    and the default pointer is updated, so the tool improves instead of aging."""
    from sklearn.metrics import roc_auc_score

    from ..loader.data_loader import clean_telco

    df = _read_any(Path(path))
    if not pd.api.types.is_numeric_dtype(df.get("Churn", pd.Series(dtype=object))):
        df = clean_telco(df)
    eng = features.engineer_telco(df)
    X_train, X_test, y_train, y_test = features.split_telco(eng)
    numeric, categorical = features.telco_feature_lists(eng)
    pre = features.make_preprocessor(numeric, categorical)
    pipe, _ = models.tune_hgb(X_train, y_train, pre, n_iter=15)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle = {"pipeline": pipe, "features": list(X_train.columns), "trained": stamp}
    models.save_model(bundle, f"churn_pipeline_{stamp}")
    models.save_model(bundle, "churn_pipeline")  # default pointer
    models.update_metrics("lens_A_retrain", {"roc_auc_test": auc, "trained": stamp,
                                             "n_rows": len(df)})
    return {"roc_auc_test": auc, "artifact": f"churn_pipeline_{stamp}"}


def write_contracts() -> Path:
    config.CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CONTRACTS_DIR / "contracts.json"
    path.write_text(json.dumps(CONTRACTS, indent=2))
    return path


def _read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix} (use csv/xlsx/parquet)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="churn-pipeline",
        description="ChurnLens: validate, score or retrain on a customer data file.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="validate + score a file with every available lens")
    p_score.add_argument("--input", required=True)
    p_score.add_argument("--output", default=None)

    p_train = sub.add_parser("train", help="retrain Lens A on a labeled snapshot file")
    p_train.add_argument("--data", required=True)

    sub.add_parser("contracts", help="write schema contracts to models/contracts/")

    args = parser.parse_args(argv)
    if args.command == "score":
        df = _read_any(Path(args.input))
        scored, report = score_any(df)
        for lens, info in report["lenses"].items():
            print(f"  Lens {lens}: {info['message']}")
        if report["ignored_columns"]:
            print(f"  Ignored columns: {', '.join(report['ignored_columns'])}")
        print(f"  Applied: {report['applied'] or 'nothing - no lens available'}")
        for lens, err in report.get("errors", {}).items():
            print(f"  ERROR in lens {lens}: {err}")
        out_path = Path(args.output or Path(args.input).with_suffix("")
                        .as_posix() + "_scored.csv")
        scored.to_csv(out_path, index=False)
        print(f"  Wrote {len(scored):,} rows -> {out_path}")
    elif args.command == "train":
        result = train_from_file(args.data)
        print(f"  Retrained Lens A: test ROC-AUC {result['roc_auc_test']:.4f} "
              f"-> models/{result['artifact']}.joblib")
    elif args.command == "contracts":
        print(f"  Wrote {write_contracts()}")


if __name__ == "__main__":
    main()
