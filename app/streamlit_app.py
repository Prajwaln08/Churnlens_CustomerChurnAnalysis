"""ChurnLens - Streamlit UI over phase_1/pipeline.py.

Run from the project root:
    streamlit run app/streamlit_app.py

The app is a thin view layer: every decision (validation, lens detection,
scoring) lives in Code/phase_1/pipeline.py, so CLI, notebooks and UI can never
disagree. Charts follow the project chart conventions (see phase_1/eda.py):
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

from phase_1 import bootstrap, pipeline  # noqa: E402
from phase_1.visualization import CONTEXT_COLOR, PALETTE  # noqa: E402
from phase_2 import connectors, profiler  # noqa: E402
from phase_2.mapping import CANONICAL  # noqa: E402
from phase_2.orchestration import Registry, RunManifest, run_pipeline  # noqa: E402

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

    st.dataframe(scored.head(100), width="stretch")
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

tab_score, tab_integrate, tab_process, tab_target = st.tabs(
    ["Score", "Integrate (new data)", "Process (runs)", "Targeting (uplift)"])

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

with tab_integrate:
    st.markdown("Bring **any customer dataset** - file, database or API. The pipeline "
                "profiles it, proposes a mapping to the canonical schema, and runs the "
                "staged flow. Confirmed mappings are remembered per source.")
    src_kind = st.radio("Source type", ["File upload", "Database", "REST API"],
                        horizontal=True, key="integrate_kind")
    payload = None          # SourceData or DataFrame handed to the pipeline
    df_i = None             # the frame for profiling / editing
    default_name = ""

    if src_kind == "File upload":
        upload_i = st.file_uploader("Dataset (csv / xlsx / parquet)",
                                    type=["csv", "xlsx", "parquet"],
                                    key="integrate_upload")
        if upload_i is not None:
            suffix = Path(upload_i.name).suffix.lower()
            df_i = (pd.read_csv(upload_i) if suffix == ".csv"
                    else pd.read_excel(upload_i) if suffix in (".xlsx", ".xls")
                    else pd.read_parquet(upload_i))
            payload = connectors.from_dataframe(df_i, name=upload_i.name)
            default_name = Path(upload_i.name).stem

    elif src_kind == "Database":
        db_url = st.text_input(
            "Connection string",
            placeholder="postgresql://user:pass@host:5432/db   or   sqlite:///crm.db",
            help="Credentials are used for this fetch only - never saved. "
                 "Prefer keeping them in .env.")
        c1, c2, c3 = st.columns([2, 2, 1])
        db_table = c1.text_input("Table")
        db_query = c2.text_input("...or SQL query (overrides table)")
        db_limit = c3.number_input("Row limit (0 = all)", 0, value=0)
        if st.button("Fetch from database",
                     disabled=not (db_url and (db_table or db_query))):
            try:
                st.session_state["integ_src"] = connectors.from_database(
                    db_url, table=(None if db_query else db_table or None),
                    query=db_query or None, limit=db_limit or None)
            except Exception as exc:
                st.error(f"Fetch failed: {exc}")
        fetched = st.session_state.get("integ_src")
        if fetched is not None and fetched.origin.startswith("db:"):
            st.success(f"Fetched {fetched.rows:,} rows from {fetched.origin}")
            payload, df_i = fetched, fetched.frame
            default_name = (db_table or "db-query").replace(" ", "-")

    else:  # REST API
        api_url = st.text_input("Endpoint URL",
                                placeholder="https://api.example.com/v2/customers")
        c1, c2, c3 = st.columns(3)
        rec_path = c1.text_input("Records path", placeholder="data.items",
                                 help="dot-path to the list of records; empty = root")
        hdr_name = c2.text_input("Auth header name", placeholder="Authorization")
        hdr_value = c3.text_input("Auth header value", type="password",
                                  placeholder="Bearer ...")
        if st.button("Fetch from API", disabled=not api_url):
            try:
                headers = {hdr_name: hdr_value} if hdr_name and hdr_value else None
                st.session_state["integ_src"] = connectors.from_api(
                    api_url, records_path=rec_path, headers=headers)
            except Exception as exc:
                st.error(f"Fetch failed: {exc}")
        fetched = st.session_state.get("integ_src")
        if fetched is not None and fetched.origin.startswith("api:"):
            st.success(f"Fetched {fetched.rows:,} rows from {fetched.origin}")
            payload, df_i = fetched, fetched.frame
            default_name = api_url.rstrip("/").split("/")[-1] if api_url else "api"

    if df_i is not None:
        c1, c2 = st.columns(2)
        source_i = c1.text_input("Source name (mappings are saved per source)",
                                 value=default_name)
        level_i = c2.radio("Autonomy", [1, 0], horizontal=True,
                           format_func=lambda v: "Level 1 - auto-pilot" if v == 1
                           else "Level 0 - review mapping first")
        known = Registry().latest_mapping(source_i) if source_i else None
        if known is not None:
            st.caption(f"Confirmed mapping v{known['version']} on file for "
                       f"'{source_i}' - runs will reuse it without asking.")
        with st.expander("Column profile (what the tool sees)"):
            st.dataframe(profiler.summary_frame(profiler.profile_table(df_i)),
                         width="stretch")

        if st.button("Run pipeline", type="primary", disabled=not source_i):
            st.session_state.pop("integ_pending", None)
            with st.spinner("Running the staged pipeline (a new schema may take "
                            "~1 min while the local LLM proposes the mapping)..."):
                m = run_pipeline(payload if payload is not None else df_i,
                                 source_i, autonomy_level=level_i)
            if m.status == "paused":
                st.session_state["integ_pending"] = {
                    "proposal": m.data["reports"]["mapping_proposal"],
                    "source": source_i, "run_id": m.run_id}
            else:
                st.session_state["integ_done"] = m.run_id

        pending = st.session_state.get("integ_pending")
        if pending and pending["source"] == source_i:
            st.warning("Level 0 gate: review the proposed mapping, edit any target, "
                       "then confirm. Quarantined columns stay excluded either way.")
            prop = pending["proposal"]
            edit_frame = pd.DataFrame(
                [{"source": mp["source"], "target": mp["target"] or "",
                  "confidence": mp["confidence"]} for mp in prop["mappings"]])
            edited = st.data_editor(
                edit_frame, hide_index=True, width="stretch",
                disabled=["source", "confidence"],
                column_config={"target": st.column_config.SelectboxColumn(
                    "target (canonical)", options=[""] + sorted(CANONICAL))})
            if prop.get("quarantine"):
                st.caption("Quarantined (never applied): " + ", ".join(
                    q["source"] for q in prop["quarantine"]))
            if st.button("Confirm mapping & run", type="primary"):
                by_source = {mp["source"]: mp for mp in prop["mappings"]}
                confirmed_mappings = []
                for _, row in edited.iterrows():
                    orig = by_source[row["source"]]
                    target = row["target"] or None
                    kept = target == orig["target"]
                    confirmed_mappings.append({
                        "source": row["source"], "target": target,
                        "confidence": orig["confidence"] if kept else 0.99,
                        "value_map": orig["value_map"] if kept else None})
                override = {"engine": "human-confirmed",
                            "mappings": confirmed_mappings,
                            "quarantine": prop.get("quarantine", [])}
                with st.spinner("Running with the confirmed mapping..."):
                    m = run_pipeline(payload if payload is not None else df_i,
                                     source_i, autonomy_level=0,
                                     proposal_override=override)
                st.session_state.pop("integ_pending", None)
                st.session_state["integ_done"] = m.run_id

        done_id = st.session_state.get("integ_done")
        if done_id:
            try:
                m_done = RunManifest.load(done_id)
                status = m_done.status
                (st.success if status == "completed" else st.error)(
                    f"Run {done_id}: {status} - full detail in the Process tab.")
                st.dataframe(m_done.summary_frame()[
                    ["stage", "status", "elapsed_s", "rows_out", "notes"]],
                    width="stretch")
            except FileNotFoundError:
                st.session_state.pop("integ_done", None)

with tab_process:
    reg = Registry()
    runs_frame = reg.all_runs()
    if runs_frame.empty:
        st.info("No runs yet - bring a dataset through the Integrate tab.")
    else:
        st.dataframe(runs_frame[["run_id", "source_name", "status", "rows",
                                 "created_at", "finished_at"]],
                     width="stretch")
        run_pick = st.selectbox("Inspect run", runs_frame["run_id"])
        try:
            m_sel = RunManifest.load(run_pick)
        except FileNotFoundError:
            st.error("Manifest file missing for this run (runs/ was cleaned?).")
            m_sel = None
        if m_sel is not None:
            st.markdown(f"**Status:** {m_sel.status} - source `{m_sel.data['source']}` "
                        f"- autonomy level {m_sel.data['autonomy_level']}")
            narrative = m_sel.data.get("reports", {}).get("narrative")
            if narrative:
                st.info(narrative["summary"])
                st.caption(f"summary by {narrative['engine']} - every figure "
                           "audited against computed facts")
            drift_rep = m_sel.data.get("reports", {}).get("drift")
            if drift_rep and drift_rep.get("alerts"):
                st.warning(f"Drift alert: {drift_rep['recommendation']}")
            st.dataframe(m_sel.summary_frame(), width="stretch")
            with st.expander("Cleaning report (the decision ledger)", expanded=True):
                for line in m_sel.ledger_lines():
                    st.text(line)
            for name, payload in m_sel.data.get("reports", {}).items():
                with st.expander(f"Report: {name}"):
                    st.json(payload)
            eda_files = [p for p in m_sel.data["stages"]["eda_report"]["artifacts"]
                         if Path(p).exists()]
            if eda_files:
                st.markdown("**Data review charts**")
                for chunk_start in range(0, len(eda_files), 2):
                    cols = st.columns(2)
                    for col, path in zip(cols, eda_files[chunk_start:chunk_start + 2]):
                        col.image(path)
            out_file = next((m_sel.run_dir / n for n in ("scored.parquet", "mapped.parquet")
                             if (m_sel.run_dir / n).exists()), None)
            if out_file is not None:
                st.download_button(
                    f"Download {out_file.stem} output (CSV)",
                    pd.read_parquet(out_file).to_csv(index=False).encode(),
                    f"{run_pick}_{out_file.stem}.csv", "text/csv")

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
    "data: `python -m phase_1.serving.pipeline train --data <file>.csv`. The application "
    "loads the updated model automatically.")
