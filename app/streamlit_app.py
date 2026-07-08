"""ChurnLens - Streamlit UI over src/pipeline.py.

Run from the project root:
    .venv/bin/streamlit run app/streamlit_app.py

The app is a thin view layer: every decision (validation, lens detection,
scoring) lives in Code/src/pipeline.py, so CLI, notebooks and UI can never
disagree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from src import config, pipeline  # noqa: E402

st.set_page_config(page_title="ChurnLens", layout="wide")

LENS_LABELS = {
    "A": "A - Who churns", "B": "B - When", "C": "C - Who to target",
    "D": "D - Segments", "E": "E - Revenue at stake",
}


def lens_chips(report: dict) -> None:
    """The product identity: bold [on] chip = this data supports that lens."""
    cols = st.columns(len(LENS_LABELS))
    for col, (lens, label) in zip(cols, LENS_LABELS.items()):
        info = report["lenses"][lens]
        on = info["ok"] and info["artifact_available"]
        col.markdown(f"**{label} [on]**" if on else f"{label} [off]")
        if not on:
            col.caption(info["message"])


def whatif_form() -> pd.DataFrame | None:
    """Single-customer form: 'customer belongs to demographic X, has features Y'."""
    with st.form("whatif"):
        st.subheader("Demographics")
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gender", ["Female", "Male"])
        senior = c2.selectbox("Senior citizen", ["No", "Yes"])
        partner = c3.selectbox("Partner", ["No", "Yes"])
        dependents = c4.selectbox("Dependents", ["No", "Yes"])

        st.subheader("Account & contract")
        c1, c2, c3, c4 = st.columns(4)
        tenure = c1.slider("Tenure (months)", 0, 72, 6)
        contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = c3.selectbox("Paperless billing", ["Yes", "No"])
        payment = c4.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])

        st.subheader("Services")
        c1, c2, c3, c4 = st.columns(4)
        phone = c1.selectbox("Phone service", ["Yes", "No"])
        multiple = c2.selectbox("Multiple lines", ["No", "Yes"])
        internet = c3.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        security = c4.selectbox("Online security", ["No", "Yes"])
        c1, c2, c3, c4 = st.columns(4)
        backup = c1.selectbox("Online backup", ["No", "Yes"])
        protection = c2.selectbox("Device protection", ["No", "Yes"])
        support = c3.selectbox("Tech support", ["No", "Yes"])
        tv = c4.selectbox("Streaming TV", ["No", "Yes"])
        c1, c2, c3 = st.columns(3)
        movies = c1.selectbox("Streaming movies", ["No", "Yes"])
        monthly = c2.number_input("Monthly charges ($)", 15.0, 150.0, 80.0, step=5.0)
        total = c3.number_input("Total charges ($)", 0.0, 10000.0,
                                float(monthly * max(tenure, 1)), step=50.0)

        if not st.form_submit_button("Score this customer", type="primary"):
            return None
    return pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }])


def show_single_result(scored: pd.DataFrame) -> None:
    row = scored.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    if "churn_probability" in row:
        c1.metric("Churn probability", f"{row['churn_probability']:.1%}",
                  row.get("risk_band", ""))
        c1.progress(min(float(row["churn_probability"]), 1.0))
    if "expected_remaining_tenure" in row:
        c2.metric("Expected remaining tenure", f"{row['expected_remaining_tenure']:.0f} mo")
    if "clv_simple" in row:
        c3.metric("Simplified CLV", f"${row['clv_simple']:,.0f}")
    if "revenue_at_risk" in row:
        c4.metric("Revenue at risk", f"${row['revenue_at_risk']:,.0f}")
    if "segment" in row:
        st.caption(f"Segment: cluster {row['segment']}")
    if "recommended_action" in row:
        st.info(f"**Recommended action:** {row['recommended_action']}")


def show_batch_result(scored: pd.DataFrame, report: dict) -> None:
    st.success(f"Scored {len(scored):,} rows - lenses applied: "
               f"{', '.join(report['applied']) or 'none'}")
    if report["ignored_columns"]:
        st.caption("Ignored columns (no lens uses them): "
                   + ", ".join(report["ignored_columns"]))
    for lens, err in report.get("errors", {}).items():
        st.error(f"Lens {lens} failed: {err}")

    st.dataframe(scored.head(100), use_container_width=True)
    st.download_button("Download scored CSV",
                       scored.to_csv(index=False).encode(),
                       "scored.csv", "text/csv")
    c1, c2 = st.columns(2)
    if "risk_band" in scored.columns:
        order = ["Low", "Medium", "High", "Critical"]
        counts = scored["risk_band"].value_counts().reindex(order).fillna(0)
        c1.subheader("Customers per risk band")
        c1.bar_chart(counts)
    if "revenue_at_risk" in scored.columns:
        c2.subheader("Money on the table")
        c2.metric("Total revenue at risk",
                  f"${scored['revenue_at_risk'].sum():,.0f}")
        top = scored.nlargest(10, "revenue_at_risk")
        cols = [c for c in ("customerID", "churn_probability", "risk_band",
                            "clv_simple", "revenue_at_risk",
                            "recommended_action") if c in top.columns]
        c2.write("Top 10 customers to act on:")
        c2.dataframe(top[cols], use_container_width=True)


# ---------------------------------------------------------------------------

st.title("ChurnLens")
st.caption("Who leaves - when - who can be saved - which segment - what it costs. "
           "Upload customer data - your columns decide which lenses light up.")

mode = st.sidebar.radio("Data source", ["Single customer (what-if)", "Batch file upload"])
tab_score, tab_target, tab_info = st.tabs(["Score", "Targeting (uplift)", "Model info"])

with tab_score:
    if mode == "Single customer (what-if)":
        df = whatif_form()
        if df is not None:
            lens_chips(pipeline.validate(df))
            scored, report = pipeline.score(df)
            show_single_result(scored)
    else:
        upload = st.file_uploader("Customer snapshot or transaction log",
                                  type=["csv", "xlsx", "parquet"])
        if upload is not None:
            suffix = Path(upload.name).suffix.lower()
            df = (pd.read_csv(upload) if suffix == ".csv"
                  else pd.read_excel(upload) if suffix in (".xlsx", ".xls")
                  else pd.read_parquet(upload))
            report = pipeline.validate(df)
            lens_chips(report)
            with st.expander("Validation detail"):
                for lens, info in report["lenses"].items():
                    st.write(f"**Lens {lens} - {info['name']}:** {info['message']}")
                    if info["coercions"]:
                        st.caption("Coercions: " + "; ".join(info["coercions"]))
            scored, report = pipeline.score_any(df)
            show_batch_result(scored, report)

with tab_target:
    st.markdown(
        "Uplift needs **campaign data**: a `treatment_group` column (who got the "
        "offer) and a `conversion` outcome. Classification finds *likely churners*; "
        "uplift finds *persuadable* ones - the customers where the offer changes "
        "the outcome, which is what the budget should chase.")
    if not pipeline.artifact_exists("uplift_model"):
        st.warning("Uplift artifact not trained yet - run notebook 03.")
    st.caption("Upload a file with treatment_group + conversion + the campaign "
               "features in the Score tab; uplift scores appear automatically.")

with tab_info:
    if config.METRICS_FILE.exists():
        metrics = json.loads(config.METRICS_FILE.read_text())
        for section, payload in metrics.items():
            st.subheader(section)
            st.json(payload)
    else:
        st.info("No metrics recorded yet - run notebooks 03 and 04.")
    st.caption("Retrain Lens A on new labeled data: "
               "`python -m src.pipeline train --data your_file.csv` (run from Code/)")
