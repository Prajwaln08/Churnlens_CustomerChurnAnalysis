"""ChurnLens - Streamlit UI over src/pipeline.py.

Run from the project root:
    streamlit run app/streamlit_app.py

The app is a thin view layer: every decision (validation, lens detection,
scoring) lives in Code/src/pipeline.py, so CLI, notebooks and UI can never
disagree. Charts follow the project chart conventions (see src/eda.py):
single hue for single-measure bars, direct value labels, labeled status
zones, no dual axes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from src import bootstrap, pipeline  # noqa: E402
from src.eda import CONTEXT_COLOR, PALETTE  # noqa: E402

st.set_page_config(page_title="ChurnLens", layout="wide")

# Fresh deployments have no artifacts (models/ never enters git) - build the
# demo models once from the public Telco dataset. No-op when artifacts exist.
if not bootstrap.artifacts_ready():
    with st.status("First boot: training demo models from the public Telco "
                   "dataset (about a minute)...", expanded=True) as _status:
        bootstrap.ensure_artifacts(progress=st.write)
        _status.update(label="Demo models ready.", state="complete")

LENS_INFO = {
    "A": ("Who churns", "churn probability per customer"),
    "B": ("When", "expected remaining tenure"),
    "C": ("Who to target", "uplift - who an offer can save"),
    "D": ("Segments", "which customer group they belong to"),
    "E": ("Revenue at stake", "customer lifetime value"),
}

# ---------------------------------------------------------------------------
# Lens chips
# ---------------------------------------------------------------------------

def lens_chips(report: dict) -> None:
    """One card per lens: green = this data supports it, grey = it does not."""
    cols = st.columns(len(LENS_INFO))
    for col, (lens, (title, sub)) in zip(cols, LENS_INFO.items()):
        info = report["lenses"][lens]
        on = info["ok"] and info["artifact_available"]
        if on:
            style = "background:#e8f5ee;border:1.5px solid #029E73;color:#0b3d2e;"
            status = "Available"
        else:
            style = "background:#f1f1f1;border:1.5px solid #cfcfcf;color:#9a9a9a;"
            status = "Not available in this data"
        col.markdown(
            f'<div style="border-radius:10px;padding:10px 12px;min-height:108px;{style}">'
            f'<div style="font-weight:700;font-size:0.88rem;">Lens {lens} - {title}</div>'
            f'<div style="font-size:0.74rem;margin-top:2px;opacity:0.85;">{sub}</div>'
            f'<div style="font-size:0.74rem;margin-top:8px;font-weight:600;">{status}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Charts (matplotlib, project palette)
# ---------------------------------------------------------------------------

def survival_chart(curve: pd.Series, tenure: float):
    """Projected retention curve for one profile, with the 50/50 month marked."""
    fig, ax = plt.subplots(figsize=(7, 3.1))
    ax.plot(curve.index, curve.values, color=PALETTE[0], linewidth=2)
    ax.fill_between(curve.index, curve.values, alpha=0.12, color=PALETTE[0])
    below = curve[curve <= 0.5]
    if len(below):
        m = below.index[0]
        ax.axvline(m, color=CONTEXT_COLOR, linestyle="--", linewidth=1)
        ax.text(m + 1.2, 0.53, f"50/50 point: ~{m} mo", fontsize=8, color="#666666")
    if tenure and tenure > 0:
        ax.axvline(tenure, color=PALETTE[1], linestyle=":", linewidth=1.5)
        ax.text(tenure + 1.2, 0.92, f"today (month {tenure:.0f})",
                fontsize=8, color=PALETTE[1])
    ax.set_xlabel("Months from signup")
    ax.set_ylabel("P(still a customer)")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_title("Projected retention curve for this profile",
                 loc="left", fontweight="bold", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def counts_barh(series: pd.Series, title: str, value_fmt: str = "{:,.0f}"):
    """Horizontal bar of counts/amounts: single hue, direct labels."""
    fig, ax = plt.subplots(figsize=(6, 0.55 + 0.42 * len(series)))
    labels = [str(i) for i in series.index]
    ax.barh(labels, series.values, color=PALETTE[0], height=0.6)
    for i, v in enumerate(series.values):
        ax.text(v, i, "  " + value_fmt.format(v), va="center", fontsize=9,
                color="#333333")
    ax.invert_yaxis()
    ax.set_xlim(0, max(series.values) * 1.3 if len(series) else 1)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def prob_histogram(proba: pd.Series):
    """Distribution of churn probabilities across the uploaded base."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(proba, bins=25, color=PALETTE[0])
    mean = float(proba.mean())
    ax.axvline(mean, color=CONTEXT_COLOR, linestyle="--", linewidth=1.2)
    ax.text(mean, ax.get_ylim()[1] * 0.95, f" mean {mean:.0%}", fontsize=8,
            color="#666666")
    ax.set_xlabel("Churn probability")
    ax.set_ylabel("Customers")
    ax.set_title("How risk is spread across this base",
                 loc="left", fontweight="bold", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# What-if form
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Result views
# ---------------------------------------------------------------------------

def show_single_result(df: pd.DataFrame, scored: pd.DataFrame) -> None:
    row = scored.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    if "churn_probability" in row:
        c1.metric("Churn probability", f"{row['churn_probability']:.1%}",
                  row.get("risk_band", ""), delta_color="off")
        c1.caption("Lens A - how likely they are to leave")
    if "expected_remaining_tenure" in row:
        c2.metric("Expected remaining tenure",
                  f"{row['expected_remaining_tenure']:.0f} months")
        c2.caption("Lens B - how long before they go")
    if "clv_simple" in row:
        c3.metric("Simplified CLV", f"${row['clv_simple']:,.0f}")
        c3.caption("monthly bill x months remaining")
    if "revenue_at_risk" in row:
        c4.metric("Revenue at risk", f"${row['revenue_at_risk']:,.0f}")
        c4.caption("probability x value - the headline")

    if "segment" in row:
        st.caption(f"Segment: cluster {row['segment']} (Lens D)")
    if "recommended_action" in row:
        st.info(f"**Recommended action:** {row['recommended_action']}")

    # retention curve last, and never allowed to break the page above it
    curves = None
    if hasattr(pipeline, "survival_curve"):
        try:
            curves = pipeline.survival_curve(df)
        except Exception:
            curves = None
    if curves is not None:
        chart_col, _ = st.columns([3, 2])
        chart_col.pyplot(survival_chart(curves.iloc[:, 0],
                                        float(row.get("tenure", 0))))
        chart_col.caption("Chance this profile is still a customer, month by "
                          "month. Read where the curve crosses 50%.")


def show_batch_result(scored: pd.DataFrame, report: dict) -> None:
    st.success(f"Scored {len(scored):,} rows - lenses applied: "
               f"{', '.join(report['applied']) or 'none'}")
    if report["ignored_columns"]:
        st.caption("Ignored columns (no lens uses them): "
                   + ", ".join(report["ignored_columns"]))
    for lens, err in report.get("errors", {}).items():
        st.error(f"Lens {lens} failed: {err}")

    # -- transaction-log view (lens E route)
    if "E" in report["applied"] and "clv" in scored.columns:
        c1, c2 = st.columns(2)
        if "segment" in scored.columns:
            seg = scored["segment"].value_counts()
            c1.pyplot(counts_barh(seg, "Customers per RFM segment"))
        clv_pos = scored.loc[scored["clv"] > 0, "clv"]
        if len(clv_pos):
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(clv_pos, bins=40, color=PALETTE[0], log=True)
            ax.set_xlabel("12-month CLV")
            ax.set_ylabel("Customers (log scale)")
            ax.set_title("Customer value distribution - note the whale tail",
                         loc="left", fontweight="bold", fontsize=10)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            fig.tight_layout()
            c2.pyplot(fig)
            c2.metric("Total 12-month CLV", f"£{scored['clv'].sum():,.0f}")
    # -- customer-snapshot view (lenses A/B/D route)
    else:
        c1, c2 = st.columns(2)
        if "risk_band" in scored.columns:
            order = ["Low", "Medium", "High", "Critical"]
            counts = (scored["risk_band"].value_counts()
                      .reindex(order).dropna().astype(int))
            c1.pyplot(counts_barh(counts, "Customers per risk band"))
        if "churn_probability" in scored.columns:
            c2.pyplot(prob_histogram(scored["churn_probability"]))
        if "revenue_at_risk" in scored.columns:
            st.metric("Total revenue at risk",
                      f"${scored['revenue_at_risk'].sum():,.0f}")
            top = scored.nlargest(10, "revenue_at_risk")
            names = (top["customerID"] if "customerID" in top.columns
                     else top.index.astype(str))
            rar = pd.Series(top["revenue_at_risk"].values, index=names)
            st.pyplot(counts_barh(rar, "Top 10 customers by revenue at risk",
                                  value_fmt="${:,.0f}"))
            st.caption("These are the first phone calls to make.")

    st.dataframe(scored.head(100), use_container_width=True)
    st.download_button("Download scored CSV",
                       scored.to_csv(index=False).encode(),
                       "scored.csv", "text/csv")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("ChurnLens")
st.caption("Who leaves - when - who can be saved - which segment - what it costs. "
           "Upload customer data; the columns you provide decide which lenses can run.")

with st.sidebar:
    mode = st.radio("Data source",
                    ["Single customer (what-if)", "Batch file upload"])
    with st.expander("What are the five lenses?"):
        st.markdown(
            "- **A - Who churns:** probability each customer leaves\n"
            "- **B - When:** expected remaining time as a customer\n"
            "- **C - Who to target:** whose churn an offer can prevent "
            "(needs campaign data: treatment_group + conversion)\n"
            "- **D - Segments:** natural customer groups\n"
            "- **E - Revenue at stake:** customer lifetime value "
            "(needs a transaction log: CustomerID + InvoiceDate + amounts)\n\n"
            "A grey card means the provided data is missing the columns that "
            "lens needs - an explicit status, not an error.")

tab_score, tab_target = st.tabs(["Score", "Targeting (uplift)"])

with tab_score:
    if mode == "Single customer (what-if)":
        df = whatif_form()
        if df is not None:
            lens_chips(pipeline.validate(df))
            st.caption("Grey cards: this single-customer form has no campaign or "
                       "purchase-history columns, so lenses C and E cannot run on it.")
            scored, report = pipeline.score(df)
            show_single_result(df, scored)
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
            with st.expander("Validation detail - what was checked and why"):
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

st.divider()
st.markdown("##### About the data behind these predictions")
st.markdown(
    "All scores in this application are produced by models trained on **public "
    "demonstration datasets**:\n\n"
    "| Lenses | Training dataset | Description |\n"
    "|---|---|---|\n"
    "| A - Who churns, B - When, D - Segments | IBM Telco Customer Churn | "
    "7,043 telecom customers, one point-in-time snapshot |\n"
    "| C - Who to target | MegaFon uplift experiment | ~600,000 customers from a "
    "randomized marketing campaign, anonymized features |\n"
    "| E - Revenue at stake | UCI Online Retail II | 1M+ purchases from a UK "
    "online gift retailer, Dec 2009 - Dec 2011 |\n")
st.markdown(
    "**Scope and intended use:**\n"
    "- Model outputs are calibrated to the customer populations, prices and "
    "time periods of the datasets listed above.\n"
    "- All figures - churn probability, remaining tenure, CLV, revenue at "
    "risk - are illustrative and intended to demonstrate the workflow, not to "
    "support decisions about real customers.\n"
    "- For production use, retrain the models on business-specific labeled "
    "data: `python -m src.pipeline train --data <file>.csv`. The application "
    "loads the updated model automatically.")
