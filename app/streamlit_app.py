"""ChurnLens - Streamlit UI over the phase_1 demo suite and phase_2 pipeline.

Run from the project root:
    streamlit run app/streamlit_app.py

Layout follows the ChurnLens design refresh: a top frame (logo, data source,
five lens chips), four numbered tabs (01 What-if, 02 Add your data,
03 Process, 04 Dashboard) and a lens-card drill navigation on the Dashboard.
The app stays a thin view layer: every decision (validation, mapping,
training, scoring) lives in Code/, so CLI, notebooks and UI cannot disagree.
"""
from __future__ import annotations

import base64
import dataclasses
import io
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from phase_1 import bootstrap, pipeline  # noqa: E402
from phase_2 import connectors, profiler  # noqa: E402
from phase_2.analytics import insights as ins  # noqa: E402
from phase_2.analytics import journeys as jn  # noqa: E402
from phase_2.mapping import CANONICAL  # noqa: E402
from phase_2.monitoring import outcomes as outcomes_mod  # noqa: E402
from phase_2.training import mode as train_mode  # noqa: E402
from phase_2.training import auto_train  # noqa: E402
from phase_2.orchestration import Registry, RunManifest, run_pipeline  # noqa: E402

st.set_page_config(page_title="ChurnLens - Customer Churn Analytics Suite",
                   layout="wide")

# ---------------------------------------------------------------------------
# Design tokens (ChurnLens.dc.html)
# ---------------------------------------------------------------------------

INK = "#16181d"
MUTED = "#6b6f76"
FAINT = "#9a9ea6"
PAPER = "#f4f2ec"
CARD_LINE = "#e6e3db"
BLUE = "#0173B2"
GREEN = "#029E73"
AMBER = "#DE8F05"
RED = "#D55E00"
CRIT = "#a83208"
PURPLE = "#CC78BC"
LENS_COLOR = {"A": RED, "B": BLUE, "C": GREEN, "D": PURPLE, "E": AMBER}
BAND_COLOR = {"Low": GREEN, "Medium": AMBER, "High": RED, "Critical": CRIT}
SEG_PALETTE = [BLUE, AMBER, RED, GREEN, PURPLE, "#949494"]

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<style>
.stApp { background: #f4f2ec; }
html, body, [class*="css"], .stMarkdown, p, span, label {
    font-family: 'IBM Plex Sans', system-ui, sans-serif; color: #16181d; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em; }
.block-container { padding-top: 1.6rem; max-width: 100%; padding-left: .8rem; padding-right: .8rem; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { display: none; }
.stTabs [role="tablist"] {
    margin-top: 18px; display: flex; gap: 4px; background: #ecege2; border: 1px solid #e0ddd4;
    border-radius: 12px; padding: 4px; width: 100%; }
.stTabs [role="tablist"] [data-testid="stTab"] {
    flex: 1; display: flex; align-items: center; justify-content: center; }
.stTabs div[data-orientation="horizontal"] { border-bottom: 0 !important; }
.stTabs [data-testid="stTab"] {
    padding: 8px 18px; border-radius: 9px; border: 1px solid transparent;
    background: transparent; cursor: pointer; transition: all .15s; }
.stTabs [data-testid="stTab"] p {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 14.5px; font-weight: 600;
    color: #8a8e95; margin: 0; line-height: 1.3; }
.stTabs [data-testid="stTab"]:hover { background: #f4f2ec; }
.stTabs [data-testid="stTab"]:hover p { color: #16181d; }
.stTabs [data-testid="stTab"][aria-selected="true"] {
    background: #ffffff; border: 1px solid #d9d5cc;
    box-shadow: 0 1px 3px rgba(20,24,29,.10); }
.stTabs [data-testid="stTab"][aria-selected="true"] p { color: #16181d; }
.stTabs .react-aria-SelectionIndicator { display: none; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff; border: 1px solid #e6e3db !important;
    border-radius: 16px; padding: 6px 10px; }
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button,
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
    background: #16181d; color: #fff; border: none; border-radius: 10px;
    font-weight: 600; padding: 0.55rem 1.5rem; }
.stButton button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:hover,
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-primaryFormSubmit"]:hover {
    background: #2c2f36; color: #fff; }
[data-testid="stFormSubmitButton"] button p, .stButton button[kind="primary"] p,
[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-primaryFormSubmit"] p { color: #fff; }
.stButton button[kind="secondary"] {
    border: 1px solid #16181d; border-radius: 9px; background: none; font-weight: 600; }
.stDownloadButton button {
    border: 1px solid #16181d; border-radius: 9px; background: none; font-weight: 600; }
div[data-testid="stSegmentedControl"] button,
div[data-testid="stPills"] button { border-radius: 16px; font-weight: 600; }
div[data-testid="stDataFrame"] { border: 1px solid #eeebe3; border-radius: 10px; }
.cl-kicker { font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    letter-spacing: .1em; text-transform: uppercase; color: #9a9ea6; font-weight: 600; }
.cl-mono { font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# First boot
# ---------------------------------------------------------------------------

if not bootstrap.artifacts_ready():
    with st.status("First boot: training demo models from the public Telco "
                   "dataset (about a minute)...", expanded=True) as _status:
        bootstrap.ensure_artifacts(progress=st.write)
        _status.update(label="Demo models ready.", state="complete")

# ---------------------------------------------------------------------------
# HTML helpers - the design-system components
# ---------------------------------------------------------------------------

def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def tile(kicker: str, value: str, label: str = "", accent: str = CARD_LINE,
         tag: str = "", tag_bg: str = RED, value_color: str = INK) -> str:
    tag_html = ""
    if tag:
        tag_html = (f'<span style="display:inline-block;margin-top:9px;font-size:11px;'
                    f'font-weight:700;color:#fff;background:{tag_bg};padding:3px 9px;'
                    f'border-radius:20px;font-family:\'IBM Plex Mono\',monospace;">{tag}</span>')
    label_html = (f'<div style="font-size:12px;color:{MUTED};line-height:1.4;">{label}</div>'
                  if label else "")
    return (f'<div style="background:#fbfaf7;border:1px solid #eeebe3;border-radius:12px;'
            f'padding:13px 15px;border-top:3px solid {accent};height:100%;">'
            f'<div class="cl-kicker">{kicker}</div>'
            f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:26px;'
            f'font-weight:600;letter-spacing:-.02em;margin:6px 0 2px;color:{value_color};'
            f'line-height:1.1;">{value}</div>{label_html}{tag_html}</div>')


def tile_row(tiles: list[str]) -> None:
    n = len(tiles)
    html(f'<div style="display:grid;grid-template-columns:repeat({n},1fr);gap:12px;'
         f'margin-bottom:16px;">' + "".join(tiles) + "</div>")


CASUAL_STAGE = {
    "ingest": "Reading your data...",
    "profile": "Getting to know the columns...",
    "map": "Matching columns to the canonical schema...",
    "validate": "Checking which lenses this data can power...",
    "clean": "Tidying things up...",
    "eda_report": "Sketching the first charts...",
    "mode": "Deciding whether to train fresh models...",
    "train": "Training models - the longest step...",
    "score": "Scoring every customer...",
    "drift": "Running the stability check...",
    "persist": "Saving everything...",
    "narrate": "Writing the plain-English summary...",
}
_STAGE_ORDER = list(CASUAL_STAGE)


def stage_streamer(live):
    """Feeds run_pipeline(progress=...): a casual headline for everyone, the
    technical stage lines tucked inside for whoever expands the loader."""
    def write(name, entry):
        el = entry.get("elapsed_s")
        line = f"{name} - {entry['status']} ({el:.1f}s)" if el is not None \
            else f"{name} - {entry['status']}"
        notes = entry.get("notes") or []
        if notes:
            line += " - " + str(notes[0])[:110]
        live.write(line)
        try:
            nxt = _STAGE_ORDER[_STAGE_ORDER.index(name) + 1]
            live.update(label=":duck: " + CASUAL_STAGE[nxt])
        except (ValueError, IndexError):
            pass
    return write


def banner(kind: str, title: str, body: str, credit: str = "") -> None:
    styles = {
        "info": ("#eef5fb", "#cfe2f0", BLUE, "#0b5573"),
        "warn": ("#faf6e8", "#ece2c2", AMBER, "#7a6410"),
        "ok": ("#eaf6ef", "#bfe3cf", GREEN, "#0a6b4c"),
        "err": ("#fbeae4", "#f0cabb", RED, "#8a2f08"),
    }
    bg, line, edge, head = styles[kind]
    credit_html = (f'<div style="font-size:11px;color:#7a7e85;margin-top:8px;'
                   f'font-family:\'IBM Plex Mono\',monospace;">{credit}</div>' if credit else "")
    html(f'<div style="background:{bg};border:1px solid {line};border-left:4px solid {edge};'
         f'border-radius:11px;padding:13px 17px;margin-bottom:16px;">'
         f'<span style="font-size:10.5px;font-weight:700;letter-spacing:.08em;'
         f'text-transform:uppercase;color:{head};font-family:\'IBM Plex Mono\',monospace;">'
         f'{title}</span>'
         f'<p style="margin:4px 0 0;font-size:13.5px;line-height:1.6;color:{INK};">{body}</p>'
         f'{credit_html}</div>')


def rec_action(body: str, impact: str | None = None) -> None:
    if impact:
        body += ('<br><span style="font-size:12.5px;opacity:.85;">'
                 '<b>Why it pays:</b> ' + impact + '</span>')
    banner("info", "Recommended action", body)


def section_head(lens: str, name: str, note: str) -> None:
    html(f'<div style="display:flex;align-items:baseline;justify-content:space-between;'
         f'gap:12px;margin:2px 0 12px;flex-wrap:wrap;">'
         f'<div style="display:flex;align-items:baseline;gap:12px;">'
         f'<span class="cl-mono" style="font-size:12px;font-weight:600;'
         f'color:{LENS_COLOR.get(lens, INK)};">LENS {lens}</span>'
         f'<span style="font-family:\'Space Grotesk\',sans-serif;font-size:21px;'
         f'font-weight:600;">{name}</span></div>'
         f'<span class="cl-mono" style="font-size:11.5px;color:{FAINT};">{note}</span></div>')


def lineage(powered: str, unlock: str) -> None:
    html(f'<div style="border-top:1px dashed {CARD_LINE};padding-top:12px;margin-top:4px;'
         f'display:grid;grid-template-columns:1fr 1fr;gap:14px;">'
         f'<div style="font-size:11.5px;color:{MUTED};line-height:1.6;">'
         f'<b class="cl-mono" style="color:#0a6b4c;font-size:10.5px;letter-spacing:.06em;">'
         f'POWERED BY YOUR COLUMNS</b><br><span class="cl-mono">{powered}</span></div>'
         f'<div style="font-size:11.5px;color:#9a8f6e;line-height:1.6;">'
         f'<b class="cl-mono" style="color:#b57d09;font-size:10.5px;letter-spacing:.06em;">'
         f'UNLOCK MORE</b><br>{unlock}</div></div>')


def status_table(frame: pd.DataFrame, status_col: str = "status") -> None:
    """Design-styled table with a colored status dot per row."""
    dot = {"completed": GREEN, "done": GREEN, "paused": AMBER, "running": BLUE,
           "failed": RED, "skipped": FAINT, "pending": "#d2cec4"}
    heads = "".join(
        f'<span style="flex:{2.2 if c == "notes" else (1.6 if c == "run_id" else 1)};">{c}</span>'
        for c in frame.columns)
    rows = []
    for _, r in frame.iterrows():
        cells = []
        for c in frame.columns:
            v = "-" if pd.isna(r[c]) else str(r[c])
            if c == status_col:
                color = dot.get(v, FAINT)
                cells.append(
                    f'<span style="flex:1;display:inline-flex;align-items:center;gap:6px;"'
                    f' class="cl-mono"><span style="width:7px;height:7px;border-radius:50%;'
                    f'background:{color};"></span><span style="font-size:11.5px;font-weight:600;'
                    f'color:{color};">{v}</span></span>')
            else:
                flex = 2.2 if c == "notes" else (1.6 if c == "run_id" else 1)
                style = ("font-size:11.5px;color:#6b6f76;line-height:1.4;" if c == "notes"
                         else f'font-size:12px;color:{"#16181d" if c in ("stage", "run_id") else MUTED};')
                mono = "" if c in ("notes", "source_name") else ' class="cl-mono"'
                cells.append(f'<span style="flex:{flex};{style}overflow:hidden;'
                             f'text-overflow:ellipsis;white-space:nowrap;padding-right:8px;"'
                             f'{mono} title="{v}">{v}</span>')
        rows.append('<div style="display:flex;padding:9px 16px;border-top:1px solid #f1efe8;'
                    'align-items:center;">' + "".join(cells) + "</div>")
    html('<div style="background:#fff;border:1px solid #e6e3db;border-radius:14px;'
         'overflow:hidden;margin-bottom:14px;">'
         '<div style="display:flex;background:#f7f5ef;padding:10px 16px;" class="cl-kicker">'
         + heads + "</div>" + "".join(rows) + "</div>")


def dark_ledger(lines: list[str]) -> None:
    body = "".join(
        f'<div class="cl-mono" style="font-size:12px;color:#c9cdd4;line-height:1.7;">{ln}</div>'
        for ln in lines)
    html(f'<div style="background:{INK};border-radius:14px;padding:16px 20px;margin-bottom:16px;">'
         f'<div class="cl-kicker" style="color:#8a9098;margin-bottom:10px;">'
         f'Cleaning report - decision ledger</div>{body}</div>')


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Charts - matplotlib, design palette
# ---------------------------------------------------------------------------

def _style_ax(ax, title: str = ""):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d9d5cc")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10, color=INK)


def _fig(w=6.0, h=3.0):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    return fig, ax


def survival_chart(curve: pd.Series, tenure: float):
    fig, ax = _fig(7, 3.1)
    ax.plot(curve.index, curve.values, color=BLUE, linewidth=2.4)
    ax.fill_between(curve.index, curve.values, alpha=0.10, color=BLUE)
    below = curve[curve <= 0.5]
    if len(below):
        m = below.index[0]
        ax.axvline(m, color="#b3b0a7", linestyle="--", linewidth=1)
        ax.text(m + 1.2, 0.53, f"50/50 point: ~{m} mo", fontsize=8, color=MUTED)
    if tenure and tenure > 0:
        ax.axvline(tenure, color=AMBER, linestyle=":", linewidth=1.5)
        ax.text(tenure + 1.2, 0.92, f"today (month {tenure:.0f})", fontsize=8, color="#b57d09")
    ax.set_xlabel("Months from signup")
    ax.set_ylabel("P(still a customer)")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    _style_ax(ax)
    fig.tight_layout()
    return fig


def population_hist(base: pd.Series, this_p: float | None = None):
    fig, ax = _fig(6, 3)
    ax.hist(base, bins=25, color=BLUE, alpha=0.88)
    mean = float(base.mean())
    ax.axvline(mean, color="#949494", linestyle="--", linewidth=1.2)
    ax.text(mean + 0.01, ax.get_ylim()[1] * 0.93, f"mean {mean:.1%}", fontsize=8, color=MUTED)
    if this_p is not None:
        ax.axvline(this_p, color=RED, linestyle="--", linewidth=1.8)
        ax.text(this_p - 0.01, ax.get_ylim()[1] * 0.80, f"this customer {this_p:.1%}",
                fontsize=8, color=RED, ha="right", fontweight="bold")
    ax.set_xlabel("Churn probability")
    ax.set_ylabel("Customers")
    _style_ax(ax)
    fig.tight_layout()
    return fig


def counts_barh(series: pd.Series, value_fmt: str = "{:,.0f}",
                color: str | list = BLUE):
    fig, ax = _fig(6, 0.55 + 0.42 * len(series))
    labels = [str(i) for i in series.index]
    ax.barh(labels, series.values, color=color, height=0.6)
    for i, v in enumerate(series.values):
        ax.text(v, i, "  " + value_fmt.format(v), va="center", fontsize=9, color="#333333")
    ax.invert_yaxis()
    ax.set_xlim(0, max(series.values) * 1.3 if len(series) else 1)
    _style_ax(ax)
    ax.set_xticks([])
    fig.tight_layout()
    return fig


def band_donut(counts: dict[str, int]):
    fig, ax = _fig(3.4, 3.4)
    order = [b for b in ("Low", "Medium", "High", "Critical") if b in counts]
    vals = [counts[b] for b in order]
    colors = [BAND_COLOR[b] for b in order]
    ax.pie(vals, colors=colors, startangle=90, counterclock=False,
           wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 1.5})
    total = sum(vals)
    ax.text(0, 0.06, f"{total:,}", ha="center", fontsize=15, fontweight="bold",
            color=INK, family="sans-serif")
    ax.text(0, -0.16, "customers", ha="center", fontsize=7.5, color=FAINT)
    ax.legend(order, frameon=False, fontsize=8, loc="center left",
              bbox_to_anchor=(0.98, 0.5), labelcolor=MUTED)
    fig.tight_layout()
    return fig


def seg_pie(sizes: list[int], labels: list[str]):
    fig, ax = _fig(4.4, 3.2)
    ax.pie(sizes, colors=SEG_PALETTE[:len(sizes)], startangle=90, counterclock=False,
           wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    total = sum(sizes)
    legend = [f"{l} - {s / total:.0%}" for l, s in zip(labels, sizes)]
    ax.legend(legend, frameon=False, fontsize=7.5, loc="center left",
              bbox_to_anchor=(0.96, 0.5), labelcolor=MUTED)
    fig.tight_layout()
    return fig


def km_chart(curves: dict[str, pd.Series], grain_label: str = "periods"):
    fig, ax = _fig(7, 3.6)
    palette = [INK] + [BLUE, AMBER, GREEN, PURPLE]
    for (name, curve), color in zip(curves.items(), palette):
        ax.step(curve.index, curve.values, where="post", color=color,
                linewidth=2.4 if color == INK else 1.9, label=name)
    ax.set_xlabel(f"Journey period ({grain_label})")
    ax.set_ylabel("Still engaged")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    _style_ax(ax)
    fig.tight_layout()
    return fig


def timing_chart(timing: pd.Series, grain_label: str = "period"):
    fig, ax = _fig(7, 3.6)
    peak = timing.idxmax() if len(timing) else None
    colors = [RED if peak is not None and abs(int(i) - int(peak)) <= 1 else AMBER
              for i in timing.index]
    ax.bar(timing.index, timing.values, color=colors, width=0.8)
    ax.set_xlabel(f"Journey {grain_label} of churn (when they went silent)")
    ax.set_ylabel("Customers")
    _style_ax(ax)
    fig.tight_layout()
    return fig


def uplift_hist(up: pd.Series):
    fig, ax = _fig(6.4, 3.2)
    counts, edges, patches = ax.hist(up, bins=36)
    for edge, patch in zip(edges[:-1], patches):
        center = edge + (edges[1] - edges[0]) / 2
        patch.set_facecolor(RED if center < -0.01 else ("#949494" if center <= 0.01 else GREEN))
        patch.set_alpha(0.88)
    ax.axvline(0, color=INK, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Predicted uplift (change in conversion probability if contacted)")
    ax.set_ylabel("Customers")
    _style_ax(ax)
    fig.tight_layout()
    return fig


def trend_chart(series: pd.Series):
    fig, ax = _fig(6.4, 3.2)
    ax.plot(range(len(series)), series.values, color=AMBER, linewidth=2.4, zorder=2)
    ax.scatter(range(len(series)), series.values, color=AMBER, s=32, zorder=3)
    for i, v in enumerate(series.values):
        ax.annotate(f"${v / 1e6:.2f}M" if v >= 1e6 else f"${v / 1e3:.0f}K",
                    (i, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="#3a3e45", fontweight="bold")
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([str(i)[:16] for i in series.index], fontsize=7.5)
    ax.margins(y=0.25)
    ax.set_ylabel("Revenue at risk")
    _style_ax(ax)
    fig.tight_layout()
    return fig


def spark_b64(series: pd.Series) -> str:
    fig, ax = _fig(2.6, 0.9)
    ax.plot(range(len(series)), series.values, color="#f0b35f", linewidth=2)
    ax.scatter([len(series) - 1], [series.values[-1]], color="#f0b35f", s=22)
    ax.axis("off")
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _churn_bool(col: pd.Series) -> pd.Series:
    as_num = pd.to_numeric(col, errors="coerce")
    if as_num.notna().mean() > 0.9:
        return as_num.fillna(0) > 0
    return col.astype(str).str.strip().str.lower().isin(
        ["yes", "true", "churned", "attrited", "1", "closed"])


@st.cache_data(show_spinner=False)
def load_run_bundle(run_id: str) -> dict | None:
    try:
        m = RunManifest.load(run_id)
    except FileNotFoundError:
        return None
    scored_path = m.run_dir / "scored.parquet"
    scored = pd.read_parquet(scored_path) if scored_path.exists() else None
    return {"data": m.data, "run_dir": str(m.run_dir), "scored": scored,
            "stage_frame": m.summary_frame(), "ledger": m.ledger_lines()}


@st.cache_data(show_spinner=False)
def rar_history(source: str, allowed: tuple[str, ...]) -> pd.Series:
    """Total revenue at risk per completed run of this source, oldest first.
    Only runs made in this browser session count - nothing leaks across users."""
    runs = Registry().all_runs()
    runs = runs[(runs["source_name"] == source) & (runs["status"] == "completed")
                & runs["run_id"].isin(allowed)]
    totals = {}
    for _, r in runs.iloc[::-1].iterrows():
        try:
            p = RunManifest.load(r["run_id"]).run_dir / "scored.parquet"
            if p.exists():
                rar = pd.read_parquet(p, columns=["revenue_at_risk"])
                totals[str(r["created_at"])[:10]] = float(rar["revenue_at_risk"].sum())
        except Exception:
            continue
    return pd.Series(totals)


@st.cache_data(show_spinner=False)
def telco_base_probabilities() -> pd.Series | None:
    """Churn probabilities of the demo training base, for 'where this customer
    sits in the population'. Skipped gracefully when the file is absent."""
    base_path = PROJECT_ROOT / "Dataset" / "processed" / "telco_clean.parquet"
    if not base_path.exists():
        return None
    try:
        base = pd.read_parquet(base_path)
        scored, _ = pipeline.score(base.drop(columns=["Churn"], errors="ignore"))
        return scored["churn_probability"]
    except Exception:
        return None


def completed_runs_with_scores() -> pd.DataFrame:
    runs = Registry().all_runs()
    if runs.empty:
        return runs
    return runs[runs["status"] == "completed"].reset_index(drop=True)


def snapshot_journeys(scored: pd.DataFrame, slice_col: str | None) -> pd.DataFrame | None:
    """Journey frame from the customer snapshot: duration = tenure, observed =
    churn label. Reuses the phase_2 journeys engine (KM, censoring-aware)."""
    if "tenure" not in scored.columns or "Churn" not in scored.columns:
        return None
    frame = pd.DataFrame({
        "user": scored.get("customerID", pd.Series(scored.index.astype(str))).astype(str),
        "duration": pd.to_numeric(scored["tenure"], errors="coerce").fillna(0).clip(lower=1).astype(int),
        "churned": _churn_bool(scored["Churn"]),
    })
    frame["observed"] = frame["churned"]
    frame["periods_survived"] = frame["duration"]
    frame["churn_period"] = np.where(frame["churned"], frame["periods_survived"] + 1, np.nan)
    frame["journey_status"] = np.where(frame["churned"], jn.CHURNED, jn.ACTIVE)
    if slice_col and slice_col in scored.columns:
        frame[slice_col] = scored[slice_col].astype(str).values
    return frame


def slice_options(scored: pd.DataFrame) -> list[str]:
    skip = {"customerID", "Churn", "churn_probability", "risk_band", "segment",
            "local_drivers", "expected_remaining_tenure", "clv_simple", "revenue_at_risk"}
    return [c for c in scored.columns
            if c not in skip and not pd.api.types.is_numeric_dtype(scored[c])
            and 1 < scored[c].nunique() <= 12]


def money(v: float) -> str:
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:,.0f}K"
    return f"${v:,.0f}"


# ---------------------------------------------------------------------------
# Top frame: logo, data source, lens chips, lens explainer
# ---------------------------------------------------------------------------

st.session_state.setdefault("session_runs", [])
st.session_state.setdefault("session_trained", set())
_mine = st.session_state["session_runs"]
runs_all = Registry().all_runs()
if not runs_all.empty:
    runs_all = runs_all[runs_all["run_id"].isin(_mine)].reset_index(drop=True)
completed = completed_runs_with_scores()
if not completed.empty:
    completed = completed[completed["run_id"].isin(_mine)].reset_index(drop=True)
has_completed = not completed.empty

html(f'<div style="display:flex;align-items:center;gap:11px;margin-top:2px;">'
     f'<div style="width:18px;height:18px;border-radius:50%;border:3.5px solid {BLUE};'
     f'position:relative;"><div style="position:absolute;width:6px;height:6px;'
     f'background:{RED};border-radius:50%;right:-4px;bottom:-4px;"></div></div>'
     f'<div style="display:flex;align-items:baseline;gap:10px;">'
     f'<span style="font-family:\'Space Grotesk\',sans-serif;font-weight:700;'
     f'font-size:34px;letter-spacing:-.02em;">ChurnLens</span>'
     f'<span style="font-family:\'Space Grotesk\',sans-serif;font-weight:500;'
     f'font-size:21px;letter-spacing:-.01em;color:{MUTED};">- Customer Churn '
     f'Analytics Suite</span></div></div>'
     f'<p style="font-size:14px;color:{MUTED};line-height:1.65;'
     f'margin:12px 0 2px;">ChurnLens is a data-agnostic churn analytics tool: it turns '
     f'any customer dataset into retention decisions. Bring data from a file, database '
     f'or API - the pipeline profiles it, proposes a mapping to a canonical schema '
     f'(reviewable at a human gate), screens out leakage, trains models suited to the '
     f'columns available, and opens a dashboard across five lenses:</p>'
     f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin:10px auto 12px;'
     f'width:84%;justify-content:center;">'
     + "".join(
         f'<span style="display:inline-flex;align-items:center;justify-content:center;'
         f'gap:8px;flex:1;min-width:150px;'
         f'background:#fbfaf7;border:1px solid #e6e3db;border-radius:18px;'
         f'padding:6px 14px;">'
         f'<b class="cl-mono" style="color:{LENS_COLOR[k]};font-size:12.5px;">'
         f'LENS {k}</b>'
         f'<span style="font-size:12.5px;color:{INK};font-weight:600;">{label}'
         f'</span></span>'
         for k, label in [("A", "who leaves"), ("B", "when"),
                          ("C", "who can be saved"), ("D", "which segment"),
                          ("E", "what it costs")])
     + f'</div>'
     f'<p style="font-size:14px;color:{MUTED};line-height:1.65;margin:0 0 2px;">'
     'Every step is auditable - cleaning decisions, quality gates and drift checks '
     'all land on a ledger - and the analytics run at both population and individual '
     'level.</p>')

_lock_p = "" if not runs_all.empty else " :lock:"
_lock_d = "" if has_completed else " :lock:"
tab_whatif, tab_newdata, tab_process, tab_dash = st.tabs(
    ["01  What-if", "02  Add your data", f"03  Process{_lock_p}",
     f"04  Dashboard{_lock_d}"])

# ---------------------------------------------------------------------------
# 01 WHAT-IF
# ---------------------------------------------------------------------------

def whatif_form() -> pd.DataFrame | None:
    with st.form("whatif"):
        html('<div class="cl-kicker" style="margin:2px 0 10px;">Account &amp; contract</div>')
        c1, c2, c3, c4 = st.columns(4)
        tenure = c1.slider("Tenure (months)", 0, 72, 6)
        contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = c3.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        monthly = c4.slider("Monthly charges ($)", 15, 150, 80, step=5)

        html('<div class="cl-kicker" style="margin:10px 0 6px;">Demographics &amp; services</div>')
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        senior = t1.toggle("Senior citizen", value=False)
        partner = t2.toggle("Partner", value=False)
        dependents = t3.toggle("Dependents", value=False)
        phone = t4.toggle("Phone service", value=True)
        security = t5.toggle("Online security", value=False)
        support = t6.toggle("Tech support", value=False)

        with st.expander("More profile fields (defaults are sensible)"):
            c1, c2, c3, c4 = st.columns(4)
            gender = c1.selectbox("Gender", ["Female", "Male"])
            paperless = c2.selectbox("Paperless billing", ["Yes", "No"])
            payment = c3.selectbox("Payment method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"])
            multiple = c4.selectbox("Multiple lines", ["No", "Yes"])
            c1, c2, c3, c4 = st.columns(4)
            backup = c1.selectbox("Online backup", ["No", "Yes"])
            protection = c2.selectbox("Device protection", ["No", "Yes"])
            tv = c3.selectbox("Streaming TV", ["No", "Yes"])
            movies = c4.selectbox("Streaming movies", ["No", "Yes"])
            total = st.number_input("Total charges ($)", 0, 12000,
                                    int(monthly * max(tenure, 1)), step=50)

        if not st.form_submit_button("Score this customer", type="primary"):
            return None
    yn = lambda b: "Yes" if b else "No"
    return pd.DataFrame([{
        "gender": gender, "SeniorCitizen": yn(senior), "Partner": yn(partner),
        "Dependents": yn(dependents), "tenure": tenure, "PhoneService": yn(phone),
        "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": yn(security), "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": yn(support),
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }])


def show_single_result(df: pd.DataFrame, scored: pd.DataFrame) -> None:
    row = scored.iloc[0]
    tiles = []
    if "churn_probability" in row:
        p = float(row["churn_probability"])
        band = str(row.get("risk_band", ""))
        tiles.append(tile("Lens A - churn probability", f"{p:.1%}",
                          "how likely they are to leave", accent=RED,
                          value_color=RED if p >= 0.5 else INK,
                          tag=band, tag_bg=BAND_COLOR.get(band, RED)))
    if "expected_remaining_tenure" in row:
        tiles.append(tile("Lens B - remaining tenure",
                          f"{row['expected_remaining_tenure']:.0f} mo",
                          "expected time still a customer", accent=BLUE))
    if "clv_simple" in row:
        tiles.append(tile("Simplified CLV", f"${row['clv_simple']:,.0f}",
                          "monthly bill x months remaining", accent=PURPLE))
    if "revenue_at_risk" in row:
        tiles.append(tile("Headline - revenue at risk", f"${row['revenue_at_risk']:,.0f}",
                          "probability x value", accent=INK))
    if tiles:
        tile_row(tiles)

    seg_pill = ""
    if "segment" in row:
        seg_pill = (f'<span style="font-size:12.5px;font-weight:600;background:#f0eef8;'
                    f'color:#8a5fb0;border:1px solid #e2daf0;padding:6px 13px;'
                    f'border-radius:20px;">Segment - cluster {row["segment"]} (Lens D)</span>')
    html(f'<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;'
         f'margin-bottom:14px;">{seg_pill}<span style="font-size:12.5px;color:{MUTED};">'
         f'This form has no campaign or purchase-history columns, so lenses C and E '
         f'cannot run on it.</span></div>')

    if "recommended_action" in row and pd.notna(row["recommended_action"]):
        rec_action(str(row["recommended_action"]))

    curves = None
    if hasattr(pipeline, "survival_curve"):
        try:
            curves = pipeline.survival_curve(df)
        except Exception:
            curves = None
    c1, c2 = st.columns([1.2, 1])
    if curves is not None:
        with c1, st.container(border=True):
            html('<div style="font-size:14px;font-weight:700;">Chance this profile is '
                 'still a customer, month by month</div>'
                 f'<div style="font-size:12px;color:{MUTED};margin-bottom:6px;">Read where '
                 'the curve crosses 50%. Lens B - expected remaining tenure.</div>')
            st.pyplot(survival_chart(curves.iloc[:, 0], float(row.get("tenure", 0))))
    base = telco_base_probabilities()
    if base is not None and "churn_probability" in row:
        p = float(row["churn_probability"])
        pct = float((base < p).mean())
        with c2, st.container(border=True):
            html('<div style="font-size:14px;font-weight:700;">Where this customer sits '
                 'in the population</div>'
                 f'<div style="font-size:12px;color:{MUTED};margin-bottom:6px;">Risk '
                 f'distribution of the {len(base):,} training customers.</div>')
            st.pyplot(population_hist(base, p))
            html(f'<div style="font-size:11.5px;color:{MUTED};line-height:1.5;">Population '
                 f'mean churn probability is {base.mean():.1%}. At {p:.1%} this profile is '
                 f'riskier than roughly {pct:.0%} of the base.</div>')



with tab_whatif:
    with st.container(border=True):
        html('<div style="display:flex;justify-content:space-between;align-items:center;">'
             '<span style="font-family:\'Space Grotesk\',sans-serif;font-size:20px;'
             'font-weight:600;">Single customer - what-if</span>'
             f'<span class="cl-mono" style="font-size:11px;font-weight:700;'
             f'letter-spacing:.05em;color:#7a6410;background:#faf6e8;'
             f'border:1.5px solid #e3d5a8;border-radius:16px;padding:6px 14px;">'
             'DEMO SANDBOX - NOT YOUR DATA</span></div>'
             f'<p style="margin:2px 0 10px;font-size:13px;color:{MUTED};">'
             '<b>Everything on this tab runs on demo models trained on the public '
             'IBM Telco dataset - none of it reflects your data.</b> Adjust a '
             'profile and score it to try the tool; bring real data through '
             '02 Add your data, and its analytics appear in 04 Dashboard.</p>')
        df_w = whatif_form()
    if df_w is not None:
        scored_w, _ = pipeline.score(df_w)
        show_single_result(df_w, scored_w)

    # provenance footer - lives only on this tab
    html('<div style="font-size:16px;font-weight:700;margin:26px 0 12px;">About the data '
         'behind these predictions</div>')
    prov = [("Lens A / B / D", "IBM Telco Customer Churn",
             "7,043 customers, contract + services + charges."),
            ("Lens C", "MegaFon Uplift Experiment",
             "Randomized treatment/control campaign log."),
            ("Lens E", "UCI Online Retail II",
             "Transaction log for CLV and RFM segments.")]
    cells = "".join(
        f'<div style="background:#fbfaf7;border:1px solid #eeebe3;border-radius:11px;'
        f'padding:13px 15px;"><div class="cl-mono" style="font-size:11px;color:{FAINT};'
        f'font-weight:600;">{l}</div><div style="font-size:13px;font-weight:600;'
        f'margin:3px 0 2px;">{n}</div><div style="font-size:11.5px;color:{MUTED};">{d}'
        f'</div></div>' for l, n, d in prov)
    html(f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;'
         f'margin-bottom:14px;">{cells}</div>')
    html(f'<p style="font-size:11.5px;color:{FAINT};line-height:1.6;max-width:840px;">'
         'Figures on this tab are illustrative, computed on public training datasets. '
         'Model outputs are calibrated to the populations, prices and time periods of '
         'those datasets and are intended to demonstrate the workflow, not to support '
         'decisions about real customers. Bring business data through Add your data to '
         'train fresh models - the Dashboard then reports on that data instead.</p>')

# ---------------------------------------------------------------------------
# 02 ADD YOUR DATA
# ---------------------------------------------------------------------------

with tab_newdata:
    col_src, col_main = st.columns([1, 2.1], gap="medium")

    with col_src, st.container(border=True):
        html('<span style="font-family:\'Space Grotesk\',sans-serif;font-size:19px;'
             'font-weight:600;">Add your data</span>'
             f'<p style="margin:2px 0 12px;font-size:12.5px;color:{MUTED};line-height:1.5;">'
             'Bring any customer dataset - file, database or API. The pipeline profiles '
             'it, the local LLM proposes a mapping, models train - then the Dashboard '
             'unlocks. Confirmed mappings are remembered per source.</p>')
        src_kind = st.segmented_control("Source type", ["File", "Database", "REST API"],
                                        default="File", key="integrate_kind")
        payload = None
        df_i = None
        default_name = ""

        if src_kind == "File":
            upload_i = st.file_uploader("Dataset (csv / xlsx / parquet)",
                                        type=["csv", "xlsx", "parquet"],
                                        key="integrate_upload")
            if upload_i is not None:
                suffix = Path(upload_i.name).suffix.lower()
                try:
                    df_i = (pd.read_csv(upload_i) if suffix == ".csv"
                            else pd.read_excel(upload_i) if suffix in (".xlsx", ".xls")
                            else pd.read_parquet(upload_i))
                except Exception as exc:
                    df_i = None
                    banner("err", "Could not read this file",
                           f"{type(exc).__name__}: {exc}")
                if df_i is not None:
                    payload = connectors.from_dataframe(df_i, name=upload_i.name)
                    default_name = Path(upload_i.name).stem

        elif src_kind == "Database":
            db_url = st.text_input(
                "Connection string",
                placeholder="postgresql://user:pass@host:5432/db  or  sqlite:///crm.db",
                help="Credentials are used for this fetch only - never saved.")
            db_table = st.text_input("Table")
            db_query = st.text_input("...or SQL query (overrides table)")
            db_limit = st.number_input("Row limit (0 = all)", 0, value=0)
            html(f'<div style="font-size:11px;color:#9a8f6e;background:#faf6e8;'
                 f'border:1px solid #ece2c2;border-radius:7px;padding:7px 9px;'
                 f'margin-bottom:10px;">Credentials are used for this fetch only - '
                 f'never saved.</div>')
            if st.button("Fetch from database", type="primary",
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

        else:
            api_url = st.text_input("Endpoint URL",
                                    placeholder="https://api.example.com/v2/customers")
            rec_path = st.text_input("Records path", placeholder="data.items",
                                     help="dot-path to the list of records; empty = root")
            h1, h2 = st.columns(2)
            hdr_name = h1.text_input("Auth header name", placeholder="Authorization")
            hdr_value = h2.text_input("Auth header value", type="password",
                                      placeholder="Bearer ...")
            if st.button("Fetch from API", type="primary", disabled=not api_url):
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

        MAX_TEST_ROWS = 100_000
        if df_i is not None and len(df_i) > MAX_TEST_ROWS:
            _orig_rows = len(df_i)
            df_i = df_i.sample(MAX_TEST_ROWS, random_state=42).reset_index(drop=True)
            if payload is not None:
                payload = dataclasses.replace(payload, frame=df_i)
            banner("warn", "Large file - sampled for this test run",
                   f"{_orig_rows:,} rows exceeds the test portal budget; running on "
                   f"a random {MAX_TEST_ROWS:,}-row sample to stay fast and stable.")
        if df_i is not None:
            html(f'<div style="border-top:1px solid #eeebe3;margin:6px 0 10px;"></div>'
                 f'<div style="font-size:12px;font-weight:600;color:#0a6b4c;'
                 f'margin-bottom:8px;">Fetched {len(df_i):,} rows x {df_i.shape[1]} '
                 f'columns.</div>')
            source_i = st.text_input("Source name (mappings are saved per source)",
                                     value=default_name)
            level_i = st.radio(
                "Autonomy", [1, 0],
                format_func=lambda v: ("Level 1 - auto-pilot: accept the LLM mapping "
                                       "and run end to end") if v == 1
                else "Level 0 - review mapping first: pause at a gate to edit targets",
                key="integrate_autonomy")
            known = Registry().latest_mapping(source_i) if source_i else None
            if known is not None:
                st.caption(f"Confirmed mapping v{known['version']} on file for "
                           f"'{source_i}' - runs will reuse it without asking.")
            if st.button("Run pipeline", type="primary", disabled=not source_i,
                         key="run_pipeline_btn"):
                st.session_state.pop("integ_pending", None)
                m = None
                try:
                    with st.status(":duck: Reading your data... (expand for "
                                   "the technical stage log)",
                                   expanded=False) as live:
                        m = run_pipeline(
                            payload if payload is not None else df_i,
                            source_i, autonomy_level=level_i,
                            force_train=(source_i not in
                                         st.session_state["session_trained"]),
                            progress=stage_streamer(live))
                        live.update(label="All done - dashboards ready",
                                    state="complete")
                except Exception as exc:
                    banner("err", "Run failed", f"{type(exc).__name__}: {exc}")
                if m is not None:
                    st.session_state["session_runs"].append(m.run_id)
                    if m.status == "completed":
                        st.session_state["session_trained"].add(source_i)
                    load_run_bundle.clear()
                    rar_history.clear()
                    if m.status == "paused":
                        st.session_state["integ_pending"] = {
                            "proposal": m.data["reports"]["mapping_proposal"],
                            "source": source_i, "run_id": m.run_id}
                    else:
                        st.session_state["integ_done"] = m.run_id
                    st.rerun()

        with st.expander("Companion files - optional (event log / campaign)"):
            html(f'<div style="font-size:12px;color:{MUTED};margin-bottom:8px;">'
                 'All uploads live here - the Dashboard lenses read these files '
                 'and never ask for uploads themselves. The event log powers the '
                 'Lens B engagement view; the campaign file powers Lens C.</div>')
            up_ev = st.file_uploader(
                "Event log (csv / parquet) - one row per user activity",
                type=["csv", "parquet"], key="companion_events_up")
            if up_ev is not None:
                try:
                    ev_df = (pd.read_csv(up_ev) if up_ev.name.endswith(".csv")
                             else pd.read_parquet(up_ev))
                    st.session_state["companion_events"] = ev_df
                    st.caption(f"Event log connected: {len(ev_df):,} rows")
                except Exception as exc:
                    banner("err", "Could not read the event log",
                           f"{type(exc).__name__}: {exc}")
            else:
                st.session_state.pop("companion_events", None)
            up_cp = st.file_uploader(
                "Campaign file (treatment_group + conversion + features)",
                type=["csv", "parquet"], key="companion_campaign_up")
            if up_cp is not None:
                try:
                    cp_df = (pd.read_csv(up_cp) if up_cp.name.endswith(".csv")
                             else pd.read_parquet(up_cp))
                    st.session_state["companion_campaign"] = cp_df
                    st.caption(f"Campaign file connected: {len(cp_df):,} rows")
                except Exception as exc:
                    banner("err", "Could not read the campaign file",
                           f"{type(exc).__name__}: {exc}")
            else:
                st.session_state.pop("companion_campaign", None)

    if st.session_state["session_runs"]:
        if st.button("Remove connected data - reset this session",
                     key="reset_session"):
            for _k in ("integ_pending", "integ_done", "integ_src",
                       "companion_events", "companion_campaign",
                       "campaign_scored", "campaign_fit",
                       "session_campaign_trained"):
                st.session_state.pop(_k, None)
            st.session_state["session_runs"] = []
            st.session_state["session_trained"] = set()
            load_run_bundle.clear()
            rar_history.clear()
            st.rerun()

    with col_main:
        pending = st.session_state.get("integ_pending")
        done_id = st.session_state.get("integ_done")

        if df_i is None and not pending and not done_id:
            with st.container(border=True):
                html('<div style="font-size:14px;font-weight:700;">Column profile - what '
                     'the tool sees</div>'
                     f'<div style="text-align:center;padding:48px 20px;color:{FAINT};'
                     f'font-size:13px;">Choose a source on the left and fetch data to '
                     f'begin.</div>')

        if df_i is not None and not pending and not done_id:
            with st.container(border=True):
                html('<div style="font-size:14px;font-weight:700;">Column profile - what '
                     'the tool sees</div>'
                     f'<div style="font-size:12px;color:{MUTED};margin-bottom:8px;">Loaded '
                     'automatically on fetch. Roles drive which lenses can run.</div>')
                st.dataframe(profiler.summary_frame(profiler.profile_table(df_i)),
                             width="stretch")

        if pending:
            html(f'<div style="display:flex;align-items:center;gap:14px;background:#fdfaf0;'
                 f'border:1px solid #ece2c2;border-radius:13px;padding:14px 18px;'
                 f'margin-bottom:14px;">'
                 f'<span style="width:10px;height:10px;border-radius:50%;background:{AMBER};">'
                 f'</span><div style="flex:1;"><div style="font-size:13.5px;font-weight:700;'
                 f'color:#7a6410;">Paused at Level 0 gate</div>'
                 f'<div class="cl-mono" style="font-size:11.5px;color:{MUTED};margin-top:2px;">'
                 f'awaiting mapping confirmation - source {pending["source"]}</div></div></div>')
            banner("warn", "Level 0 gate",
                   "Review the LLM-proposed mapping, edit any target, then confirm. "
                   "Quarantined columns stay excluded either way.")
            prop = pending["proposal"]
            edit_frame = pd.DataFrame(
                [{"source": mp["source"], "target": mp["target"] or "",
                  "confidence": mp["confidence"]} for mp in prop["mappings"]])
            edited = st.data_editor(
                edit_frame, hide_index=True, width="stretch",
                disabled=["source", "confidence"],
                column_config={
                    "target": st.column_config.SelectboxColumn(
                        "target (canonical)", options=[""] + sorted(CANONICAL)),
                    "confidence": st.column_config.ProgressColumn(
                        "confidence", min_value=0.0, max_value=1.0, format="%.2f")})
            if prop.get("quarantine"):
                st.caption("Quarantined (never applied): " + ", ".join(
                    q["source"] for q in prop["quarantine"]))
            if st.button("Confirm mapping & run", type="primary"):
                by_source = {mp["source"]: mp for mp in prop["mappings"]}
                confirmed_mappings = []
                for _, r in edited.iterrows():
                    orig = by_source[r["source"]]
                    target = r["target"] or None
                    kept = target == orig["target"]
                    confirmed_mappings.append({
                        "source": r["source"], "target": target,
                        "confidence": orig["confidence"] if kept else 0.99,
                        "value_map": orig["value_map"] if kept else None})
                override = {"engine": "human-confirmed",
                            "mappings": confirmed_mappings,
                            "quarantine": prop.get("quarantine", [])}
                m = None
                try:
                    with st.status(":duck: Running with your confirmed "
                                   "mapping... (expand for the stage log)",
                                   expanded=False) as live:
                        m = run_pipeline(
                            payload if payload is not None else df_i,
                            pending["source"], autonomy_level=0,
                            proposal_override=override,
                            force_train=(pending["source"] not in
                                         st.session_state["session_trained"]),
                            progress=stage_streamer(live))
                        live.update(label="All done - dashboards ready",
                                    state="complete")
                except Exception as exc:
                    banner("err", "Run failed", f"{type(exc).__name__}: {exc}")
                if m is not None:
                    st.session_state.pop("integ_pending", None)
                    st.session_state["session_runs"].append(m.run_id)
                    if m.status == "completed":
                        st.session_state["session_trained"].add(pending["source"])
                    st.session_state["integ_done"] = m.run_id
                    load_run_bundle.clear()
                    rar_history.clear()
                    st.rerun()

        if done_id and not pending:
            bundle = load_run_bundle(done_id)
            if bundle is None:
                st.session_state.pop("integ_done", None)
            else:
                data = bundle["data"]
                stage_frame = bundle["stage_frame"]
                n_done = int((stage_frame["status"] == "done").sum())
                status = data.get("status", "completed")
                ok = status == "completed"
                head_bg, head_line, head_dot, head_c = (
                    ("#eaf6ef", "#bfe3cf", GREEN, "#0a6b4c") if ok
                    else ("#fbeae4", "#f0cabb", RED, "#8a2f08"))
                title = ("Pipeline complete" if ok else f"Pipeline {status}")
                sub = (f"{n_done}/12 stages done - models persisted"
                       if ok else "the failing stage is detailed in 03 Process")
                html(f'<div style="display:flex;align-items:center;gap:14px;background:{head_bg};'
                     f'border:1px solid {head_line};border-radius:13px;padding:14px 18px;'
                     f'margin-bottom:14px;">'
                     f'<span style="width:10px;height:10px;border-radius:50%;'
                     f'background:{head_dot};"></span>'
                     f'<div style="flex:1;"><div style="font-size:13.5px;font-weight:700;'
                     f'color:{head_c};">{title}</div>'
                     f'<div class="cl-mono" style="font-size:11.5px;color:{MUTED};'
                     f'margin-top:2px;">run {done_id}</div></div>'
                     f'<span class="cl-mono" style="font-size:12px;color:{MUTED};">'
                     f'{n_done} / 12 stages</span></div>')

                if ok:
                    mapping_rep = data.get("reports", {}).get("mapping", {}) or {}
                    training_rep = data.get("reports", {}).get("training", {}) or {}
                    renamed = mapping_rep.get("renamed", {})
                    dropped = (training_rep.get("leakage", {}) or {}).get("dropped", {})
                    campaign_c = {s: t for s, t in renamed.items()
                                  if t in ("treatment_group", "conversion")}
                    trans_c = {s: t for s, t in renamed.items()
                               if t in ("InvoiceDate", "Amount", "Quantity", "Price",
                                        "CustomerID", "Invoice")}
                    label_c = {s: t for s, t in renamed.items() if t == "Churn"}
                    ident_c = {s: t for s, t in renamed.items() if t == "customerID"}
                    core_c = {s: t for s, t in renamed.items()
                              if s not in campaign_c and s not in trans_c
                              and s not in label_c and s not in ident_c}
                    quarantined = (mapping_rep.get("quarantined", [])
                                   + mapping_rep.get("unmapped", []))
                    buckets = [
                        ("Lens A / B / D - core features",
                         ", ".join(f"{s} -> {t}" for s, t in core_c.items()) or None,
                         "#bfe3cf", "#eaf6ef", "#0a6b4c"),
                        ("Lens C - campaign log",
                         ", ".join(f"{s} -> {t}" for s, t in campaign_c.items()) or None,
                         "#bfe3cf", "#eaf6ef", "#0a6b4c"),
                        ("Lens E - transaction log",
                         ", ".join(f"{s} -> {t}" for s, t in trans_c.items()) or None,
                         "#bfe3cf", "#eaf6ef", "#0a6b4c"),
                        ("Label", ", ".join(f"{s} -> {t}" for s, t in label_c.items()) or None,
                         "#cfe2f0", "#eef5fb", "#0b5573"),
                        ("Identity",
                         (", ".join(f"{s} -> {t}" for s, t in ident_c.items())
                          + (" (excluded from training)" if any(
                              "customerID" in k for k in dropped) else "")) if ident_c else None,
                         CARD_LINE, "#fbfaf7", MUTED),
                        ("Quarantined / unmapped",
                         ", ".join(str(q) for q in quarantined) + " - kept out of every lens"
                         if quarantined else None,
                         "#ece2c2", "#faf6e8", "#9a8f6e"),
                    ]
                    cells = "".join(
                        f'<div style="border:1px solid {line};background:{bg};'
                        f'border-radius:10px;padding:11px 13px;">'
                        f'<div class="cl-mono" style="font-size:10.5px;font-weight:700;'
                        f'letter-spacing:.06em;text-transform:uppercase;color:{head};'
                        f'margin-bottom:6px;">{title_}</div>'
                        f'<div class="cl-mono" style="font-size:11.5px;color:#3a3e45;'
                        f'line-height:1.7;">{cols}</div></div>'
                        for title_, cols, line, bg, head in buckets if cols)
                    with st.container(border=True):
                        html('<div style="font-size:13.5px;font-weight:700;">How the '
                             'pipeline segregated the columns</div>'
                             f'<div style="font-size:12px;color:{MUTED};margin-bottom:12px;">'
                             'Every column was categorized - nothing dropped silently.</div>'
                             f'<div style="display:grid;grid-template-columns:repeat(3,1fr);'
                             f'gap:10px;">{cells}</div>')

                if ok:
                    banner("ok", "03 Process and 04 Dashboard are now unlocked",
                           "The full stage ledger, decision log and reports for this "
                           "run live in 03 Process; population and individual "
                           "analytics live in 04 Dashboard.")
                else:
                    banner("err", "Run failed - open 03 Process",
                           "The stage ledger there shows exactly which stage stopped "
                           "and why - nothing fails silently.")

# ---------------------------------------------------------------------------
# 03 PROCESS
# ---------------------------------------------------------------------------

with tab_process:
    if runs_all.empty:
        st.info("No runs yet - bring a dataset through Add your data.")
    else:
        html('<div style="font-size:18px;font-weight:700;margin-bottom:2px;">Runs registry'
             f'</div><div style="font-size:12px;color:{MUTED};margin-bottom:10px;">Every '
             'dataset that came through Add your data - completed, paused at a gate, or '
             'failed, all shown plainly.</div>')
        status_table(runs_all[["run_id", "source_name", "status", "rows",
                               "created_at", "finished_at"]].head(12))
        run_pick = st.selectbox("Inspect run", runs_all["run_id"])
        bundle = load_run_bundle(run_pick)
        if bundle is None:
            st.error("Manifest file missing for this run (runs/ was cleaned?).")
        else:
            data = bundle["data"]
            html(f'<div style="display:flex;align-items:center;gap:10px;margin:8px 0 12px;'
                 f'flex-wrap:wrap;"><span class="cl-kicker">Inspecting</span>'
                 f'<span class="cl-mono" style="font-size:13px;font-weight:600;">{run_pick}'
                 f'</span><span style="font-size:12.5px;color:{MUTED};">- source '
                 f'{data["source"]} - autonomy level {data["autonomy_level"]} - status '
                 f'{data["status"]}</span></div>')
            narrative = data.get("reports", {}).get("narrative")
            if narrative:
                banner("info", "Narrative", narrative["summary"],
                       credit=f"summary by {narrative['engine']} - every figure audited "
                              "against computed facts")
            drift_rep = data.get("reports", {}).get("drift")
            if drift_rep and drift_rep.get("alerts"):
                banner("warn", "Drift alert", str(drift_rep.get("recommendation", "")))
            html('<div style="font-size:13.5px;font-weight:700;margin:4px 0 8px;">'
                 'Stage ledger - loading, mapping, preprocessing, analysis</div>')
            status_table(bundle["stage_frame"][["stage", "status", "elapsed_s",
                                                "rows_out", "notes"]].fillna("-"))
            dark_ledger(bundle["ledger"])
            with st.expander("Actuals check - grade past predictions"):
                html(f'<div style="font-size:12.5px;color:{MUTED};'
                     f'margin-bottom:8px;">Predictions are promises. When the '
                     'real outcomes arrive later, upload them here (a customer '
                     'id plus what actually happened) and this run gets graded '
                     'on realized performance - the honest closing of the '
                     'loop.</div>')
                scored_run = bundle.get("scored")
                if (scored_run is None
                        or "churn_probability" not in scored_run.columns):
                    st.info("This run has no scored probabilities to grade.")
                else:
                    up_act = st.file_uploader(
                        "Actual outcomes (csv / parquet)",
                        type=["csv", "parquet"], key=f"actuals_{run_pick}")
                    if up_act is not None:
                        try:
                            act_df = (pd.read_csv(up_act)
                                      if up_act.name.endswith(".csv")
                                      else pd.read_parquet(up_act))
                        except Exception as exc:
                            act_df = None
                            banner("err", "Could not read the actuals file",
                                   f"{type(exc).__name__}: {exc}")
                        if act_df is not None:
                            cols_a = list(act_df.columns)
                            g_id = next((c for c in cols_a
                                         if "id" in c.lower()), cols_a[0])
                            g_out = next(
                                (c for c in cols_a if any(
                                    k in c.lower() for k in
                                    ("churn", "actual", "outcome", "label"))),
                                cols_a[-1])
                            ac1, ac2 = st.columns(2)
                            id_c = ac1.selectbox(
                                "Customer id column", cols_a,
                                index=cols_a.index(g_id),
                                key=f"act_id_{run_pick}")
                            out_c = ac2.selectbox(
                                "Actual outcome column", cols_a,
                                index=cols_a.index(g_out),
                                key=f"act_out_{run_pick}")
                            rep_o = outcomes_mod.realized_performance(
                                scored_run, act_df, id_c, out_c)
                            if rep_o["status"] != "ok":
                                banner("warn", "Cannot grade yet",
                                       rep_o["reason"])
                            else:
                                Registry().record_outcomes(
                                    run_pick, rep_o.pop("frame"))
                                trained_a = ((data.get("reports", {})
                                              .get("training", {})
                                              .get("lenses", {}) or {})
                                             .get("A", {}) or {}).get(
                                                 "test_roc_auc")
                                tile_row([
                                    tile("Customers matched",
                                         f"{rep_o['matched']:,}", accent=BLUE),
                                    tile("Realized AUC",
                                         f"{rep_o['realized_auc']:.3f}",
                                         (f"vs {trained_a:.3f} at training"
                                          if trained_a else "held-out grade"),
                                         accent=GREEN if rep_o["healthy"]
                                         else RED,
                                         value_color=GREEN if rep_o["healthy"]
                                         else RED),
                                    tile("Top-decile precision",
                                         f"{rep_o['precision_top_decile']:.0%}",
                                         "actual churners in the top 10%",
                                         accent=BLUE),
                                    tile("Actual churn rate",
                                         f"{rep_o['actual_churn_rate']:.1%}",
                                         accent=INK)])
                                banner("ok" if rep_o["healthy"] else "warn",
                                       "Model holding up" if rep_o["healthy"]
                                       else "Retrain recommended",
                                       rep_o["verdict"] + " (gate "
                                       f"{rep_o['gate']}). Outcomes recorded "
                                       "to the registry.")
            for name, payload_rep in data.get("reports", {}).items():
                with st.expander(f"Report: {name}"):
                    st.json(payload_rep)
            eda_files = [p for p in data["stages"]["eda_report"]["artifacts"]
                         if Path(p).exists()]
            if eda_files:
                html('<div style="font-size:13.5px;font-weight:700;margin:4px 0 2px;">'
                     'Data review charts</div>'
                     f'<div style="font-size:12px;color:{MUTED};margin-bottom:8px;">'
                     'Generated automatically at the eda_report stage, before any '
                     'training: label balance, churn rate per category, and numeric '
                     'distributions - a quick visual sanity check of what came in.</div>')
                for chunk_start in range(0, len(eda_files), 2):
                    cols = st.columns(2)
                    for col, path in zip(cols, eda_files[chunk_start:chunk_start + 2]):
                        col.image(path)
            run_dir = Path(bundle["run_dir"])
            out_file = next((run_dir / n for n in ("scored.parquet", "mapped.parquet")
                             if (run_dir / n).exists()), None)
            if out_file is not None:
                st.download_button(
                    f"Download {out_file.stem} output (CSV)",
                    pd.read_parquet(out_file).to_csv(index=False).encode(),
                    f"{run_pick}_{out_file.stem}.csv", "text/csv")

# ---------------------------------------------------------------------------
# 04 DASHBOARD
# ---------------------------------------------------------------------------

with tab_dash:
    if not has_completed:
        chain = "".join(
            f'<span style="font-size:12px;font-weight:600;color:{FAINT};background:#f3f1ea;'
            f'border:1px solid {CARD_LINE};padding:6px 13px;border-radius:20px;">{lbl}</span>'
            + ('<span style="color:#c9c5ba;font-size:12px;margin:0 6px;">-&gt;</span>'
               if i < 4 else "")
            for i, lbl in enumerate(["A - Who leaves", "B - When", "C - Who can be saved",
                                     "D - Segments", "E - What it costs"]))
        html(f'<div style="background:#fff;border:1px solid {CARD_LINE};border-radius:16px;'
             f'padding:56px 24px;text-align:center;">'
             f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:19px;'
             f'font-weight:600;margin-bottom:8px;">The Dashboard activates once '
             f'integration is done</div>'
             f'<div style="font-size:13px;color:{MUTED};max-width:520px;margin:0 auto 22px;'
             f'line-height:1.6;">Bring a dataset through <b>Add your data</b>. The '
             f'pipeline loads, maps and trains - then this tab tells the full story '
             f'across all five lenses.</div>'
             f'<div style="display:flex;justify-content:center;align-items:center;'
             f'flex-wrap:wrap;gap:2px;">{chain}</div></div>')
    else:
        # one entry per SOURCE - its latest completed run; run history lives in
        # the Process tab, never repeated here
        latest_per_source = completed.drop_duplicates("source_name")
        pick_labels = {str(r["source_name"]): r["run_id"]
                       for _, r in latest_per_source.iterrows()}
        if len(pick_labels) > 1:
            pick = st.selectbox("Dataset", list(pick_labels.keys()))
        else:
            pick = next(iter(pick_labels))
        bundle = load_run_bundle(pick_labels[pick])
        scored = bundle["scored"] if bundle else None
        if scored is None:
            st.warning("This run has no scored output on disk - pick another run.")
        else:
            data = bundle["data"]
            reports = data.get("reports", {})
            analytics = reports.get("analytics", {}) or {}
            training = reports.get("training", {}) or {}
            narrative = reports.get("narrative", {}) or {}
            source = data["source"]
            created = str(data.get("created_at", ""))[:10]
            n_cust = len(scored)

            has_rar = "revenue_at_risk" in scored.columns
            total_rar = float(scored["revenue_at_risk"].sum()) if has_rar else None
            history = (rar_history(source, tuple(_mine))
                       if has_rar else pd.Series(dtype=float))
            delta_chip = ""
            if len(history) >= 2 and history.values[-2] > 0:
                delta = history.values[-1] / history.values[-2] - 1
                arrow = "+" if delta >= 0 else ""
                delta_chip = (f'<span class="cl-mono" style="font-size:12px;font-weight:700;'
                              f'color:#f0b35f;background:rgba(240,179,95,.12);border:1px solid '
                              f'rgba(240,179,95,.3);padding:4px 11px;border-radius:16px;">'
                              f'{arrow}{delta:.1%} vs previous run</span>')
            spark_html = ""
            if len(history) >= 2:
                spark_html = (f'<div style="min-width:230px;"><div class="cl-kicker" '
                              f'style="color:#9aa0aa;margin-bottom:6px;">Revenue at risk - '
                              f'trend</div><img src="data:image/png;base64,'
                              f'{spark_b64(history)}" style="width:230px;"/></div>')
            bands = (scored["risk_band"].value_counts()
                     if "risk_band" in scored.columns else pd.Series(dtype=int))
            hi_crit = int(bands.get("High", 0) + bands.get("Critical", 0))

            # ---- KPI strip: deterministic facts, never LLM text
            def hero_tile(label, value, sub="", warn=False):
                line = "rgba(240,179,95,.45)" if warn else "rgba(255,255,255,.14)"
                sub_c = "#f0b35f" if warn else "#9aa0aa"
                sub_html = (f'<div style="font-size:10.5px;color:{sub_c};'
                            f'margin-top:2px;">{sub}</div>' if sub else "")
                return (f'<div style="background:rgba(255,255,255,.05);border:1px '
                        f'solid {line};border-radius:11px;padding:10px 14px;'
                        f'flex:1;min-width:118px;">'
                        f'<div class="cl-kicker" style="color:#9aa0aa;">{label}</div>'
                        f'<div style="font-family:\'Space Grotesk\',sans-serif;'
                        f'font-size:19px;font-weight:600;color:#fff;margin-top:3px;">'
                        f'{value}</div>{sub_html}</div>')

            tiles = [hero_tile("Customers", f"{n_cust:,}", "scored this run")]
            run_mode = (reports.get("mode_decision") or {}).get("mode", "")
            if run_mode:
                tiles.append(hero_tile("Mode", run_mode.upper(),
                                       "fresh models" if run_mode == "train"
                                       else "reused models"))
            lens_a = (training.get("lenses") or {}).get("A") or {}
            if lens_a.get("test_roc_auc") is not None:
                auc = float(lens_a["test_roc_auc"])
                sub_a = "test ROC-AUC"
                if lens_a.get("auc_ci95"):
                    lo_ci, hi_ci = lens_a["auc_ci95"]
                    sub_a += f" [{lo_ci:.2f} - {hi_ci:.2f}]"
                if lens_a.get("brier") is not None:
                    sub_a += f" - Brier {lens_a['brier']:.3f} (calibrated)"
                tiles.append(hero_tile(
                    "Model quality", f"{auc:.3f}",
                    "suspiciously perfect - check for label leakage"
                    if auc >= 0.99 else sub_a, warn=auc >= 0.99))
            lens_b = (training.get("lenses") or {}).get("B") or {}
            if lens_b.get("c_index") is not None:
                tiles.append(hero_tile("Survival", f"{float(lens_b['c_index']):.3f}",
                                       "Cox C-index"))
            if len(bands):
                tiles.append(hero_tile("High / critical", f"{hi_crit:,}",
                                       "customers to act on"))
            drift_rep = reports.get("drift") or {}
            if "alerts" in drift_rep:
                n_alerts = len(drift_rep["alerts"])
                tiles.append(hero_tile(
                    "Drift", "stable" if not n_alerts else f"{n_alerts} alerts",
                    "vs training baseline", warn=bool(n_alerts)))
            kpi_html = ('<div style="display:flex;gap:10px;flex-wrap:wrap;'
                        'margin-top:18px;">' + "".join(tiles) + "</div>")

            # ---- narrative: escaped, numbers lifted into white
            narr_html = ""
            if narrative.get("summary"):
                safe = (str(narrative["summary"]).replace("&", "&amp;")
                        .replace("<", "&lt;").replace(">", "&gt;"))
                lit = re.sub(r"(\$?\d[\d,]*\.?\d*%?)",
                             r'<b style="color:#fff;font-weight:600;">\1</b>', safe)
                credit = narrative.get("engine", "")
                narr_html = (
                    f'<div style="margin:16px 0 0;padding:2px 0 2px 16px;'
                    f'border-left:3px solid {BLUE};">'
                    f'<p style="margin:0;font-size:13.5px;color:#c3c7cf;'
                    f'line-height:1.7;text-align:justify;">{lit}</p>'
                    + (f'<div class="cl-kicker" style="color:#6f7480;margin-top:8px;">'
                       f'summary by {credit} - every figure audited against computed '
                       f'facts</div>' if credit else "") + "</div>")

            headline = money(total_rar) if total_rar is not None else f"{n_cust:,}"
            headline_sub = "revenue at risk" if total_rar is not None else "customers scored"
            rar_ci = (analytics.get("revenue_at_risk") or {}).get("total_ci95")
            if total_rar is not None and rar_ci:
                headline_sub += (f" - 95% CI {money(rar_ci[0])} to "
                                 f"{money(rar_ci[1])}")
            html(f'<div style="background:#1c1f26;color:#fff;border-radius:18px;'
                 f'padding:24px 28px;margin:10px 0 14px;">'
                 f'<div class="cl-kicker" style="color:#9aa0aa;">Executive summary - '
                 f'{source} - run {created}</div>'
                 f'<div style="display:flex;align-items:center;gap:14px;margin-top:12px;'
                 f'flex-wrap:wrap;">'
                 f'<span style="font-family:\'Space Grotesk\',sans-serif;font-size:46px;'
                 f'font-weight:700;letter-spacing:-.02em;line-height:1;color:#fff;">'
                 f'{headline}</span>'
                 f'<span style="font-size:14px;color:#c3c7cf;">{headline_sub}</span>'
                 f'{delta_chip}'
                 f'<span style="margin-left:auto;">{spark_html}</span></div>'
                 f'{kpi_html}{narr_html}</div>')

            with st.expander("Churn over time - trend and cohorts"):
                _date_cands = {}
                for c_t in scored.columns:
                    if (str(scored[c_t].dtype).startswith("datetime")
                            or "date" in c_t.lower() or "time" in c_t.lower()):
                        parsed_t = pd.to_datetime(scored[c_t], errors="coerce")
                        if parsed_t.notna().mean() > 0.8:
                            _date_cands[c_t] = parsed_t
                if not _date_cands:
                    st.info("This dataset carries no usable date column - the "
                            "over-time view needs one (a signup or period "
                            "date). Population analytics per lens are "
                            "unaffected below.")
                else:
                    dc_t = max(_date_cands,
                               key=lambda c: _date_cands[c].notna().mean())
                    month_t = _date_cands[dc_t].dt.to_period("M").astype(str)
                    has_label_t = "Churn" in scored.columns
                    val_t = (scored["Churn"].astype(str).str.lower()
                             .isin(("yes", "1", "true")).astype(float)
                             if has_label_t
                             else pd.to_numeric(scored["churn_probability"],
                                                errors="coerce"))
                    grp = val_t.groupby(month_t)
                    trend_t = grp.mean()[grp.size() >= 30]
                    ct1, ct2 = st.columns(2)
                    with ct1:
                        html('<div style="font-size:13px;font-weight:700;'
                             'margin-bottom:6px;">'
                             + ("Actual" if has_label_t else "Predicted")
                             + f' churn rate by {dc_t} month</div>')
                        if len(trend_t) < 2:
                            st.info("Fewer than two months with 30+ customers "
                                    "- not enough for a trend.")
                        else:
                            fig_t, ax_t = _fig(7, 3.2)
                            ax_t.plot(range(len(trend_t)), trend_t.values,
                                      color=BLUE, linewidth=2, marker="o",
                                      markersize=3)
                            ax_t.set_xticks(range(len(trend_t)))
                            ax_t.set_xticklabels(trend_t.index, rotation=45,
                                                 fontsize=7, ha="right")
                            ax_t.yaxis.set_major_formatter(
                                lambda y_, _: f"{y_:.0%}")
                            _style_ax(ax_t)
                            fig_t.tight_layout()
                            st.pyplot(fig_t)
                    with ct2:
                        if has_label_t and "tenure" in scored.columns:
                            html('<div style="font-size:13px;font-weight:700;'
                                 'margin-bottom:6px;">Share still active - '
                                 'cohort month x tenure bucket</div>')
                            buckets_t = pd.cut(
                                pd.to_numeric(scored["tenure"],
                                              errors="coerce"),
                                [0, 3, 6, 12, 24, 999],
                                labels=["0-3", "4-6", "7-12", "13-24", "25+"])
                            pivot_t = (pd.DataFrame({
                                "cohort": month_t, "bucket": buckets_t,
                                "churn": val_t})
                                .pivot_table(index="cohort", columns="bucket",
                                             values="churn", aggfunc="mean",
                                             observed=True).tail(12))
                            if pivot_t.empty:
                                st.info("Not enough labeled cohorts for the "
                                        "heatmap.")
                            else:
                                surv_t = 1 - pivot_t
                                fig_h, ax_h = _fig(7, 3.2)
                                im = ax_h.imshow(surv_t.values, cmap="RdYlGn",
                                                 vmin=0, vmax=1,
                                                 aspect="auto")
                                ax_h.set_xticks(range(len(surv_t.columns)))
                                ax_h.set_xticklabels(surv_t.columns,
                                                     fontsize=8)
                                ax_h.set_yticks(range(len(surv_t.index)))
                                ax_h.set_yticklabels(surv_t.index, fontsize=7)
                                ax_h.set_xlabel("tenure (months)")
                                for i_h in range(surv_t.shape[0]):
                                    for j_h in range(surv_t.shape[1]):
                                        v_h = surv_t.values[i_h, j_h]
                                        if pd.notna(v_h):
                                            ax_h.text(j_h, i_h, f"{v_h:.0%}",
                                                      ha="center", va="center",
                                                      fontsize=7)
                                fig_h.tight_layout()
                                st.pyplot(fig_h)
                        else:
                            st.info("Cohort heatmap needs a churn label and a "
                                    "tenure column.")

            # ---- lens cards + navigation
            jframe_all = snapshot_journeys(scored, None)
            timing_all = jn.churn_timing(jframe_all) if jframe_all is not None and jframe_all["churned"].any() else pd.Series(dtype=int)
            exit_peak = f"mo {int(timing_all.idxmax())}" if len(timing_all) else "-"
            campaign = st.session_state.get("campaign_scored")
            seg_profiles = analytics.get("segments", [])
            worst_seg = max(seg_profiles, key=lambda s: s["mean_churn_probability"]) if seg_profiles else None

            card_defs = [
                ("A", "Who leaves", f"{hi_crit:,}" if len(bands) else "--",
                 "high / critical customers", "churn_probability" in scored.columns),
                ("B", "When", exit_peak if len(timing_all) else "--",
                 "exit peak in the journey", "tenure" in scored.columns),
                ("C", "Who can be saved",
                 f"{campaign['persuadable']:,}" if campaign else "--",
                 "persuadable customers" if campaign else "add campaign data to unlock",
                 bool(campaign)),
                ("D", "Segments", str(len(seg_profiles)) if seg_profiles else "--",
                 (f"worst: cluster {worst_seg['segment']} - "
                  f"{worst_seg['mean_churn_probability']:.0%} churn") if worst_seg
                 else "no segment model", bool(seg_profiles)),
                ("E", "What it costs", money(total_rar) if total_rar is not None else "--",
                 (delta_chip and "trend vs previous run") or "prob x CLV summed",
                 has_rar),
            ]
            lens_pick = st.segmented_control(
                "View lens", [f"{k} - {t}" for k, t, *_ in card_defs] + ["What-if"],
                default="A - Who leaves", key="lens_view")
            sel_key = (lens_pick or "A").split(" ")[0]
            cells = []
            for k, title_, kpi, sub, on in card_defs:
                selected = k == sel_key
                line = INK if selected else CARD_LINE
                shadow = "0 10px 24px -12px rgba(20,24,29,.4)" if selected else "none"
                foot = "Viewing" if selected else "View details"
                foot_c = GREEN if selected else FAINT
                dim = "opacity:.6;" if not on else ""
                cells.append(
                    f'<div style="background:#fff;border:1px solid {line};border-top:3px '
                    f'solid {LENS_COLOR[k]};border-radius:13px;padding:14px 15px;'
                    f'box-shadow:{shadow};{dim}">'
                    f'<div class="cl-mono" style="font-size:11px;font-weight:600;'
                    f'color:{LENS_COLOR[k]};">{k} - {title_}</div>'
                    f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:26px;'
                    f'font-weight:600;letter-spacing:-.02em;margin:8px 0 3px;">{kpi}</div>'
                    f'<div style="font-size:11.5px;color:{MUTED};line-height:1.4;">{sub}</div>'
                    f'<div class="cl-mono" style="font-size:11px;font-weight:600;'
                    f'color:{foot_c};margin-top:9px;">{foot}</div></div>')
            html('<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;'
                 'margin:4px 0 16px;">' + "".join(cells) + "</div>")

            powered_feats = ", ".join((training.get("leakage", {}) or {}).get("kept", [])[:8])

            # ---------------- LENS A ----------------
            if sel_key == "A" and "churn_probability" in scored.columns:
                with st.container(border=True):
                    auc = (training.get("lenses", {}).get("A", {}) or {}).get("test_roc_auc")
                    section_head("A", "Who leaves",
                                 f"binary churn classifier - test AUC {auc}" if auc
                                 else "binary churn classifier")
                    crit = int(bands.get("Critical", 0))
                    crit_rar = (float(scored.loc[scored["risk_band"] == "Critical",
                                                 "revenue_at_risk"].sum())
                                if has_rar and crit else 0.0)
                    if crit:
                        impact = ("Critical customers are the densest slice of the "
                                  "problem - highest probability of leaving, still "
                                  "reachable today.")
                        if crit_rar:
                            impact = (f"these {crit:,} customers hold "
                                      f"{money(crit_rar)} of revenue at risk. At a "
                                      f"conservative 18% save rate, working this "
                                      f"list protects about "
                                      f"{money(crit_rar * 0.18)} - and because it "
                                      f"is ranked, the first call of the day is "
                                      f"always the highest-stakes one.")
                        rec_action(f"Contact the {crit:,} Critical-band customers "
                                   "first, ordered by revenue at risk.", impact)
                    cost_rep = lens_a.get("cost")
                    if cost_rep:
                        with st.expander("Cost assumptions - decision threshold"):
                            html(f'<div style="font-size:12.5px;color:{MUTED};'
                                 f'margin-bottom:8px;">Bands pivot on a '
                                 f'cost-optimal threshold of '
                                 f'<b>{cost_rep["decision_threshold"]:.2f}</b> '
                                 f'({cost_rep["cost_assumption"]}), chosen on the '
                                 'held-out slice - not the 0.5 default. Change '
                                 'the costs to see where the cut should sit for '
                                 'this business.</div>')
                            cc1, cc2 = st.columns(2)
                            c_fn = cc1.number_input("Cost of missing a churner",
                                                    1, 100, 5, key="cost_fn_a")
                            c_fp = cc2.number_input("Cost of one retention offer",
                                                    1, 100, 1, key="cost_fp_a")
                            curve_c = cost_rep["cost_curve"]
                            t_star = min(curve_c, key=lambda c: c["fn"] * c_fn
                                         + c["fp"] * c_fp)["t"]
                            if abs(t_star - cost_rep["decision_threshold"]) < 1e-9:
                                st.caption(f"With these costs the trained "
                                           f"threshold {t_star:.2f} is already "
                                           "optimal - the bands above reflect it.")
                            else:
                                bands_new = (scored["churn_probability"].apply(
                                    pipeline.risk_band, threshold=t_star)
                                    .value_counts())
                                order_b = ["Low", "Medium", "High", "Critical"]
                                summary_b = " - ".join(
                                    f"{b}: {int(bands_new.get(b, 0)):,}"
                                    for b in order_b)
                                st.caption(f"With these costs the optimal "
                                           f"threshold moves to {t_star:.2f}; "
                                           f"bands would become {summary_b}. "
                                           "Retrain (or rescore) to adopt it.")
                    mean_p = float(scored["churn_probability"].mean())
                    tiles = [tile("High + Critical", f"{hi_crit:,}", accent=RED),
                             tile("Mean churn probability", f"{mean_p:.0%}", accent=BLUE)]
                    if auc:
                        tiles.append(tile("Model quality - AUC", f"{auc}", accent=GREEN))
                    tile_row(tiles)

                    c1, c2 = st.columns([1.3, 1])
                    with c1:
                        html('<div style="font-size:13px;font-weight:700;margin-bottom:6px;">'
                             'Customers per risk band - pick a band to see its customers</div>')
                        order = [b for b in ("Low", "Medium", "High", "Critical")
                                 if b in bands.index]
                        band_sel = st.segmented_control(
                            "Risk band", order,
                            default="Critical" if "Critical" in order else order[-1],
                            key="band_sel") or order[-1]
                        band_counts = bands.reindex(order).dropna().astype(int)
                        st.pyplot(counts_barh(
                            band_counts,
                            color=[BAND_COLOR[b] for b in band_counts.index]))
                    with c2:
                        html('<div style="font-size:13px;font-weight:700;margin-bottom:8px;">'
                             'Share of base by band</div>')
                        st.pyplot(band_donut(bands.to_dict()))

                    in_band = scored[scored["risk_band"] == band_sel]
                    show_cols = [c for c in ("customerID", "churn_probability", "tenure",
                                             "clv_simple", "revenue_at_risk")
                                 if c in scored.columns]
                    html(f'<div style="font-size:12.5px;font-weight:700;margin:6px 0 6px;">'
                         f'Customers in the {band_sel} band ({len(in_band):,})</div>')
                    table = (in_band.sort_values("revenue_at_risk", ascending=False)
                             if has_rar else in_band)
                    st.dataframe(table[show_cols].head(50), width="stretch", hide_index=True)
                    top_driver = (analytics.get("global_drivers") or [{}])[0]
                    if top_driver.get("feature"):
                        html(f'<p style="margin:8px 0 14px;font-size:12.5px;color:{MUTED};'
                             f'line-height:1.6;">Risk is not uniform: {hi_crit:,} of '
                             f'{n_cust:,} customers need attention, and the strongest '
                             f'global driver is <b>{top_driver["feature"]}</b> '
                             f'({top_driver["mean_abs_effect_pp"]}pp mean effect).</p>')
                    lineage(powered_feats,
                            "Add support-ticket or product-usage columns to sharpen "
                            "individual predictions.")

            # ---------------- LENS B ----------------
            elif sel_key == "B":
                with st.container(border=True):
                    section_head("B", "When they leave - the journey view",
                                 "censoring-aware survival analysis")
                    jopts = slice_options(scored)
                    if len(timing_all):
                        peak_b = int(timing_all.idxmax())
                        share_by_peak = (float(
                            (timing_all[timing_all.index <= peak_b].sum())
                            / timing_all.sum()) if timing_all.sum() else 0)
                        rec_action(
                            f"The exit peak falls around period {peak_b} of the "
                            "journey. Front-load onboarding and schedule a "
                            "proactive check-in before the peak.",
                            f"{share_by_peak:.0%} of all churn has already "
                            f"happened by period {peak_b} - outreach scheduled "
                            "after the peak lands on customers who are already "
                            "gone. Moving the check-in one period earlier reaches "
                            "every eventual churner while the relationship is "
                            "still alive.")
                    tiles = []
                    if "expected_remaining_tenure" in scored.columns:
                        med = float(scored["expected_remaining_tenure"].median())
                        tiles.append(tile("Median remaining tenure", f"{med:.0f} mo",
                                          accent=BLUE))
                    if len(timing_all):
                        tiles.append(tile("Exit peak", f"period {int(timing_all.idxmax())}",
                                          accent=RED))
                    if jframe_all is not None:
                        censored = int((~jframe_all["churned"]).sum())
                        tiles.append(tile("Still active (censored)", f"{censored:,}",
                                          "never miscounted as churned", accent=GREEN))
                    if tiles:
                        tile_row(tiles)

                    if jframe_all is not None and jframe_all["churned"].any():
                        slice_sel = st.segmented_control(
                            "Slice by", ["(none)"] + jopts, default="(none)",
                            key="b_slice") or "(none)"
                        seg_col = None if slice_sel == "(none)" else slice_sel
                        jframe = snapshot_journeys(scored, seg_col)
                        curves = {"all customers": jn.retention_curve(jframe)}
                        if seg_col:
                            by_seg = jn.retention_by_segment(jframe, seg_col)
                            top_vals = jframe[seg_col].value_counts().head(4).index
                            for v in top_vals:
                                if str(v) in by_seg:
                                    curves[str(v)] = by_seg[str(v)]
                        c1, c2 = st.columns(2)
                        with c1:
                            html('<div style="font-size:13px;font-weight:700;'
                                 'margin-bottom:6px;">Retention by journey period</div>')
                            st.pyplot(km_chart(curves, "months of tenure"))
                        with c2:
                            html('<div style="font-size:13px;font-weight:700;'
                                 'margin-bottom:6px;">When in the journey customers churn</div>')
                            st.pyplot(timing_chart(jn.churn_timing(jframe), "month"))
                        if seg_col:
                            html(f'<div style="font-size:12.5px;font-weight:700;'
                                 f'margin:4px 0 6px;">Segment comparison - by {seg_col}</div>')
                            st.dataframe(jn.segment_summary(jframe, seg_col),
                                         width="stretch", hide_index=True)

                        html('<div style="font-size:12.5px;font-weight:700;margin:8px 0 4px;">'
                             'Individual journey outlook</div>')
                        outlook = jn.individual_outlook(jframe, segment_col=seg_col)
                        pick_u = st.selectbox("Inspect a user",
                                              outlook["user"].astype(str), key="b_user")
                        r = outlook[outlook["user"].astype(str) == pick_u].iloc[0]
                        status_color = {jn.CHURNED: RED, jn.AT_RISK: AMBER,
                                        jn.ACTIVE: GREEN}[r["journey_status"]]
                        exp_rem = r["expected_remaining_periods"]
                        est = r["estimated_churn_period"]
                        tile_row([
                            tile("Status", str(r["journey_status"]), accent=status_color,
                                 value_color=status_color),
                            tile("Journey age", f"{int(r['periods_survived'])} periods",
                                 accent=BLUE),
                            tile("Expected remaining",
                                 "beyond data horizon" if pd.isna(exp_rem)
                                 else f"~{exp_rem} periods", accent=GREEN),
                            tile("Estimated churn period",
                                 "unknown" if pd.isna(est) else f"period {est:.0f}",
                                 accent=INK)])
                        st.download_button("Download per-user outlook (CSV)",
                                           outlook.to_csv(index=False).encode(),
                                           "journey_outlook.csv", "text/csv")
                    elif "expected_remaining_tenure" in scored.columns:
                        st.caption("No churn label in this data - showing model-predicted "
                                   "remaining tenure instead of observed retention.")
                        fig, ax = _fig(7, 3)
                        ax.hist(scored["expected_remaining_tenure"].dropna(), bins=30,
                                color=BLUE, alpha=0.88)
                        ax.set_xlabel("Expected remaining tenure (months)")
                        ax.set_ylabel("Customers")
                        _style_ax(ax)
                        fig.tight_layout()
                        st.pyplot(fig)

                    with st.expander("Engagement-grain journeys - bring a raw event log"):
                        st.markdown("**Churn as stopping showing up.** One row per user "
                                    "activity; retention at a chosen grain with an "
                                    "inactivity window, sliced by any population filter.")
                        events_j = st.session_state.get("companion_events")
                        if events_j is None:
                            st.info("No event log connected - add one under "
                                    "'Companion files' in 02 Add your data to "
                                    "unlock this engagement view.")
                        else:
                            cols = list(events_j.columns)
                            guess_user = next((c for c in cols if "user" in c.lower()
                                               or "id" in c.lower()), cols[0])
                            guess_ts = next((c for c in cols if pd.to_datetime(
                                events_j[c].head(20), errors="coerce").notna().mean() > 0.8),
                                cols[-1])
                            c1, c2, c3, c4 = st.columns(4)
                            user_col_j = c1.selectbox("User column", cols,
                                                      index=cols.index(guess_user))
                            ts_col_j = c2.selectbox("Timestamp column", cols,
                                                    index=cols.index(guess_ts))
                            gran_j = c3.radio("Time grain", ["W", "M"], horizontal=True,
                                              format_func=lambda g: "Weekly" if g == "W"
                                              else "Monthly")
                            window_j = c4.number_input("Inactivity window (periods)",
                                                       1, 12, 2)
                            seg_opts = ["(none)"] + [
                                c for c in cols if c not in (user_col_j, ts_col_j)
                                and events_j[c].nunique() <= 12]
                            seg_j = st.selectbox("Slice by population filter", seg_opts)
                            seg_col_j = None if seg_j == "(none)" else seg_j
                            with st.spinner("Building journeys..."):
                                attrs = (events_j[[user_col_j, seg_col_j]]
                                         .drop_duplicates(user_col_j) if seg_col_j else None)
                                ev_frame = jn.build_journeys(
                                    events_j, user_col_j, ts_col_j, granularity=gran_j,
                                    inactivity_window=int(window_j), attributes=attrs)
                                ev_outlook = jn.individual_outlook(ev_frame,
                                                                   segment_col=seg_col_j)
                            grain_lbl = "weeks" if gran_j == "W" else "months"
                            ev_timing = jn.churn_timing(ev_frame)
                            tile_row([
                                tile("Users", f"{len(ev_frame):,}", accent=BLUE),
                                tile("Churned", f"{ev_frame['churned'].mean():.1%}",
                                     accent=RED),
                                tile("At risk (gap started)",
                                     f"{(ev_frame['journey_status'] == 'at_risk').mean():.1%}",
                                     accent=AMBER),
                                tile("Most common churn period",
                                     f"period {int(ev_timing.idxmax())}" if len(ev_timing)
                                     else "-", accent=INK)])
                            curves_e = {"all users": jn.retention_curve(ev_frame)}
                            if seg_col_j:
                                by_seg = jn.retention_by_segment(ev_frame, seg_col_j)
                                for v in (ev_frame[seg_col_j].value_counts()
                                          .head(4).index.astype(str)):
                                    if v in by_seg:
                                        curves_e[v] = by_seg[v]
                            c1, c2 = st.columns(2)
                            with c1:
                                st.pyplot(km_chart(curves_e, grain_lbl))
                            with c2:
                                if len(ev_timing):
                                    st.pyplot(timing_chart(
                                        ev_timing, "week" if gran_j == "W" else "month"))
                            if seg_col_j:
                                st.dataframe(jn.segment_summary(ev_frame, seg_col_j),
                                             width="stretch", hide_index=True)
                            pick_e = st.selectbox("Inspect a user",
                                                  ev_outlook["user"].astype(str),
                                                  key="ev_user")
                            re_ = ev_outlook[ev_outlook["user"].astype(str) == pick_e].iloc[0]
                            exp_rem = re_["expected_remaining_periods"]
                            est = re_["estimated_churn_period"]
                            tile_row([
                                tile("Status", str(re_["journey_status"]), accent=BLUE),
                                tile("Journey age",
                                     f"{int(re_['periods_survived'])} periods", accent=BLUE),
                                tile("Expected remaining",
                                     "beyond data horizon" if pd.isna(exp_rem)
                                     else f"~{exp_rem} periods", accent=GREEN),
                                tile("Estimated churn period",
                                     "unknown" if pd.isna(est) else f"period {est:.0f}",
                                     accent=INK)])
                            st.download_button("Download per-user outlook (CSV)",
                                               ev_outlook.to_csv(index=False).encode(),
                                               "journey_outlook.csv", "text/csv",
                                               key="ev_dl")
                    if jframe_all is not None and len(timing_all):
                        censored_n = int((~jframe_all["churned"]).sum())
                        html(f'<p style="margin:8px 0 14px;font-size:12.5px;'
                             f'color:{MUTED};line-height:1.6;">Attrition concentrates '
                             f'around period {int(timing_all.idxmax())} - customers who '
                             f'survive the early journey stay far longer. The '
                             f'{censored_n:,} still-active customers are censored, '
                             f'never miscounted as churned, and anyone who outlives the '
                             f'data horizon reports "beyond data horizon" rather than a '
                             f'fabricated estimate.</p>')
                    lineage("tenure, Churn (censoring flag)",
                            "A raw event log (user_id + timestamp) powers the "
                            "engagement-grain view above.")

            # ---------------- LENS C ----------------
            elif sel_key == "C":
                if (not pipeline.artifact_exists("uplift_model")
                        and st.session_state.get("companion_campaign") is None):
                    banner("warn", "Lens C",
                           "Connect a campaign file in 02 Add your data "
                           "(Companion files) - Lens C trains a campaign-specific "
                           "uplift model from it.")
                else:
                    html(f'<div style="background:#faf6e8;border:1px solid #ece2c2;'
                         f'border-radius:16px;padding:26px 28px;text-align:center;'
                         f'margin-bottom:14px;">'
                         f'<div class="cl-mono" style="font-size:12px;font-weight:600;'
                         f'color:#b57d09;margin-bottom:8px;">LENS C - NEEDS CAMPAIGN DATA'
                         f'</div><div style="font-family:\'Space Grotesk\',sans-serif;'
                         f'font-size:21px;font-weight:600;margin-bottom:8px;">Uplift '
                         f'targeting works on an experiment log</div>'
                         f'<p style="margin:0 auto 14px;font-size:13px;color:#7a6410;'
                         f'max-width:560px;line-height:1.6;">Classification finds likely '
                         f'churners; uplift finds persuadable ones - customers where the '
                         f'offer changes the outcome. Connect a campaign file in 02 Add your data (Companion files) with these '
                         f'columns:</p>'
                         f'<span class="cl-mono" style="font-size:12.5px;font-weight:600;'
                         f'background:#fff;border:1px solid #ece2c2;border-radius:8px;'
                         f'padding:7px 14px;margin-right:8px;">treatment_group</span>'
                         f'<span class="cl-mono" style="font-size:12.5px;font-weight:600;'
                         f'background:#fff;border:1px solid #ece2c2;border-radius:8px;'
                         f'padding:7px 14px;">conversion</span></div>')
                    df_t = st.session_state.get("companion_campaign")
                    if df_t is None:
                        st.info("No campaign file connected - add one under "
                                "'Companion files' in 02 Add your data to unlock "
                                "uplift targeting.")
                    else:
                        fit_rep = None
                        rep_t = pipeline.validate(df_t)
                        if not rep_t["lenses"]["C"]["ok"]:
                            banner("err", "Lens C", rep_t["lenses"]["C"]["message"])
                        else:
                            scored_t, run_rep = pipeline.score(df_t)
                            if "uplift_score" not in scored_t.columns:
                                err_c = str(run_rep.get("errors", {}).get(
                                    "C", "features do not match the trained model"))
                                if "do not match the trained" in err_c:
                                    _ck = (len(df_t), tuple(df_t.columns))
                                    _cached = st.session_state.get("campaign_fit")
                                    _ds = train_mode.dataset_id_for(source)
                                    st.session_state.setdefault(
                                        "session_campaign_trained", set())
                                    if _cached and _cached[0] == _ck:
                                        scored_t, fit_rep = _cached[1], _cached[2]
                                    else:
                                        scored_t, fit_rep = None, None
                                        if _ds in st.session_state[
                                                "session_campaign_trained"]:
                                            scored_t, fit_rep = (
                                                auto_train
                                                .score_campaign_with_saved(
                                                    df_t, _ds))
                                        if scored_t is None:
                                            with st.spinner(
                                                    "Training an uplift model "
                                                    "on this campaign's own "
                                                    "features..."):
                                                scored_t, fit_rep = (
                                                    auto_train
                                                    .train_uplift_for_campaign(
                                                        df_t, dataset_id=_ds))
                                            if scored_t is not None:
                                                st.session_state[
                                                    "session_campaign_trained"
                                                ].add(_ds)
                                        st.session_state["campaign_fit"] = (
                                            _ck, scored_t, fit_rep)
                                    if scored_t is None:
                                        banner("err", "Could not model this "
                                               "campaign file",
                                               "A campaign-specific uplift model "
                                               "was attempted, but: "
                                               + fit_rep.get("reason", "unknown")
                                               + ".")
                                    elif "saved artifact" in str(
                                            fit_rep.get("engine", "")):
                                        banner("info",
                                               "Campaign-specific uplift model "
                                               "(reused)",
                                               "Scoring with the campaign model "
                                               "this session trained earlier "
                                               "for this source - instant, no "
                                               "retraining.")
                                    else:
                                        banner("info",
                                               "Campaign-specific uplift model",
                                               "The demo artifact does not fit "
                                               "this file, so a T-learner was "
                                               "trained on this campaign's own "
                                               f"{fit_rep['features_used']} "
                                               "features (treatment "
                                               f"{fit_rep['treated']:,} / control "
                                               f"{fit_rep['control']:,}). Scores "
                                               "are out-of-fold - every customer "
                                               "is scored by models that never "
                                               "saw them.")
                                else:
                                    banner("err", "Uplift scoring failed", err_c)
                                    scored_t = None
                            if (scored_t is not None
                                    and "uplift_score" in scored_t.columns):
                                up = scored_t["uplift_score"]
                                top_n = max(1, len(up) // 10)
                                persuadable = scored_t.nlargest(top_n, "uplift_score")
                                sleeping = int((up < -0.01).sum())
                                fresh = {"persuadable": top_n, "sleeping": sleeping}
                                if st.session_state.get("campaign_scored") != fresh:
                                    st.session_state["campaign_scored"] = fresh
                                    st.rerun()
                                with st.container(border=True):
                                    section_head("C", "Who can be saved",
                                                 "uplift model - treatment vs control")
                                    extra_saves = int(round(
                                        max(float(persuadable["uplift_score"]
                                                  .mean()), 0) * top_n))
                                    rec_action(
                                        f"Direct campaign spend at the {top_n:,} "
                                        "top-decile persuadables. Exclude the "
                                        f"{sleeping:,} sleeping dogs - contacting "
                                        "them increases churn.",
                                        f"a top-decile uplift of "
                                        f"{persuadable['uplift_score'].mean():+.1%} "
                                        f"means roughly {extra_saves:,} extra "
                                        f"conversions from these {top_n:,} contacts "
                                        "that the same budget spent randomly would "
                                        "miss - and every skipped sleeping dog is "
                                        "a churn the campaign avoids causing.")
                                    tiles_c = [
                                        tile("Customers scored", f"{len(scored_t):,}",
                                             accent=BLUE),
                                        tile("Mean uplift", f"{up.mean():+.3f}",
                                             accent=GREEN, value_color=GREEN),
                                        tile("Top-decile uplift",
                                             f"{persuadable['uplift_score'].mean():+.3f}",
                                             accent=GREEN, value_color=GREEN),
                                        tile("Sleeping dogs - negative", f"{sleeping:,}",
                                             accent=RED, value_color=RED)]
                                    if fit_rep and fit_rep.get("qini_auc") is not None:
                                        tiles_c.append(tile(
                                            "Qini AUC",
                                            f"{fit_rep['qini_auc']:+.3f}",
                                            "out-of-fold targeting quality",
                                            accent=GREEN))
                                    tile_row(tiles_c)
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        html('<div style="font-size:13px;font-weight:700;'
                                             'margin-bottom:6px;">Who the offer actually '
                                             'changes</div>')
                                        st.pyplot(uplift_hist(up))
                                        html(f'<div style="font-size:11.5px;color:{MUTED};'
                                             f'line-height:1.5;">Right tail = persuadables '
                                             f'(spend here); around zero = sure things and '
                                             f'lost causes; left of zero = sleeping dogs '
                                             f'(contacting them CAUSES churn).</div>')
                                    with c2:
                                        html('<div style="font-size:13px;font-weight:700;'
                                             'margin-bottom:6px;">The persuadables - top '
                                             'decile, the campaign call list</div>')
                                        st.dataframe(persuadable.head(10)[
                                            [c for c in ("treatment_group", "conversion",
                                                         "uplift_score")
                                             if c in persuadable.columns]],
                                            width="stretch")
                                        st.download_button(
                                            "Download persuadables list (CSV)",
                                            persuadable.to_csv(index=False).encode(),
                                            "persuadables.csv", "text/csv")
                                    lineage("treatment_group, conversion + campaign features",
                                            "Add cost-per-contact to compute an ROI-optimal "
                                            "budget cutoff instead of a fixed decile.")

            # ---------------- LENS D ----------------
            elif sel_key == "D":
                with st.container(border=True):
                    k = (training.get("lenses", {}).get("D", {}) or {}).get("k")
                    section_head("D", "Which segment they belong to",
                                 f"k-means clustering - k={k} selected automatically"
                                 if k else "k-means clustering")
                    if not seg_profiles:
                        st.info("No segment model in this run.")
                    else:
                        base_rate = analytics.get("actual_churn_rate") or analytics.get(
                            "mean_churn_probability")
                        if worst_seg and base_rate:
                            ratio = worst_seg["mean_churn_probability"] / max(base_rate, 1e-9)
                            rec_action(
                                f"Cluster {worst_seg['segment']} churns at "
                                f"{worst_seg['mean_churn_probability']:.0%}, about "
                                f"{ratio:.1f}x the base rate - fix the root cause "
                                "there first.",
                                f"one shared fix beats thousands of individual "
                                f"saves: this cluster's {worst_seg['size']:,} "
                                "customers share a profile, so a single product or "
                                "pricing change aimed at it moves the company-wide "
                                "churn number more than any per-customer effort.")
                        sizes = [s["size"] for s in seg_profiles]
                        labels = [f"cluster {s['segment']}" for s in seg_profiles]
                        c1, c2 = st.columns([1, 1.3])
                        with c1:
                            html('<div style="font-size:13px;font-weight:700;'
                                 'margin-bottom:8px;">Base composition by segment</div>')
                            st.pyplot(seg_pie(sizes, labels))
                        with c2:
                            html('<div style="font-size:13px;font-weight:700;'
                                 'margin-bottom:6px;">Segments - pick one to see its '
                                 'customers</div>')
                            seg_table = pd.DataFrame([
                                {"segment": f"cluster {s['segment']}",
                                 "size": s["size"], "share": f"{s['share']:.0%}",
                                 "mean churn prob": f"{s['mean_churn_probability']:.0%}"}
                                for s in seg_profiles])
                            st.dataframe(seg_table, width="stretch", hide_index=True)
                            seg_sel = st.segmented_control(
                                "Segment", labels, default=labels[0], key="seg_sel")
                        seg_id = int((seg_sel or labels[0]).split(" ")[-1])
                        if "segment" in scored.columns:
                            in_seg = scored[scored["segment"] == seg_id]
                            show_cols = [c for c in ("customerID", "churn_probability",
                                                     "tenure", "revenue_at_risk")
                                         if c in scored.columns]
                            html(f'<div style="font-size:12.5px;font-weight:700;'
                                 f'margin:6px 0 6px;">Customers in cluster {seg_id} '
                                 f'({len(in_seg):,})</div>')
                            table = (in_seg.sort_values("revenue_at_risk", ascending=False)
                                     if has_rar else in_seg)
                            st.dataframe(table[show_cols].head(50), width="stretch",
                                         hide_index=True)
                        drivers = analytics.get("global_drivers", [])
                        if drivers:
                            html('<div style="font-size:13px;font-weight:700;'
                                 'margin:10px 0 6px;">Global churn drivers - mean effect, '
                                 'percentage points</div>')
                            dseries = pd.Series(
                                {d["feature"]: d["mean_abs_effect_pp"]
                                 for d in drivers[:8]})
                            st.pyplot(counts_barh(dseries, value_fmt="+{:.1f}pp",
                                                  color=RED))
                        html(f'<p style="margin:8px 0 14px;font-size:12.5px;color:{MUTED};'
                             f'line-height:1.6;">Churn is not one problem - each segment '
                             f'leaves for a different reason, and each needs a different '
                             f'intervention.</p>')
                        lineage(powered_feats or "all mapped feature columns",
                                "Add region or acquisition-channel columns for sharper, "
                                "more actionable clusters.")

            # ---------------- LENS E ----------------
            elif sel_key == "E":
                with st.container(border=True):
                    section_head("E", "What it costs",
                                 "revenue at risk = churn probability x CLV")
                    if not has_rar:
                        st.info("This run has no revenue-at-risk columns (needs tenure + "
                                "monthly charges, or a transaction log).")
                    else:
                        top_n = min(200, max(1, len(scored) // 10))
                        top_share = (float(scored.nlargest(top_n, "revenue_at_risk")
                                           ["revenue_at_risk"].sum()) / total_rar
                                     if total_rar else 0)
                        rec_action(
                            f"Route the top {top_n} accounts - {top_share:.0%} of "
                            "total exposure - to the retention desk first. "
                            "Automate offers for the long tail.",
                            f"concentration is leverage: {top_share:.0%} of "
                            f"everything at stake sits with just {top_n} accounts, "
                            "so a handful of successful saves protects a "
                            "disproportionate share of revenue while the automated "
                            "long tail costs almost nothing to run.")
                        tiles = [tile("Total revenue at risk", money(total_rar),
                                      accent=AMBER)]
                        if len(history) >= 2 and history.values[-2] > 0:
                            delta = history.values[-1] / history.values[-2] - 1
                            tiles.append(tile("vs previous run", f"{delta:+.1%}",
                                              accent=RED,
                                              value_color=RED if delta > 0 else GREEN))
                        tiles.append(tile(f"Top {top_n} share", f"{top_share:.0%}",
                                          accent=BLUE))
                        tile_row(tiles)
                        c1, c2 = st.columns(2)
                        with c1:
                            if len(history) >= 2:
                                html('<div style="font-size:13px;font-weight:700;'
                                     'margin-bottom:6px;">Revenue at risk - trend across '
                                     'runs</div>')
                                st.pyplot(trend_chart(history))
                            else:
                                html(f'<div style="font-size:12.5px;color:{MUTED};'
                                     f'padding:20px 4px;">One completed run so far for '
                                     f'this source - the trend chart appears from the '
                                     f'second run onward.</div>')
                        with c2:
                            html('<div style="font-size:13px;font-weight:700;">Top '
                                 'customers by revenue at risk</div>'
                                 f'<div style="font-size:12px;color:{MUTED};'
                                 f'margin-bottom:6px;">These are the first phone calls '
                                 f'to make.</div>')
                            top = scored.nlargest(8, "revenue_at_risk")
                            names = (top["customerID"] if "customerID" in top.columns
                                     else top.index.astype(str))
                            st.pyplot(counts_barh(
                                pd.Series(top["revenue_at_risk"].values, index=names),
                                value_fmt="${:,.0f}", color=RED))
                        trend_note = ""
                        if len(history) >= 2 and history.values[-2] > 0:
                            delta = history.values[-1] / history.values[-2] - 1
                            trend_note = (f"Exposure has "
                                          f"{'grown' if delta > 0 else 'fallen'} "
                                          f"{delta:+.1%} since the previous run. ")
                        html(f'<p style="margin:8px 0 14px;font-size:12.5px;'
                             f'color:{MUTED};line-height:1.6;">{trend_note}The exposure '
                             f'is concentrated: the top {top_n} accounts hold '
                             f'{top_share:.0%} of the total, so a focused retention '
                             f'effort moves the headline number materially.</p>')
                        lineage("tenure, MonthlyCharges + lens A probabilities",
                                "Add gross margin per customer to convert revenue at "
                                "risk into profit at risk.")

            # ---------------- WHAT-IF on this dataset's own model ----------
            elif sel_key == "What-if":
                with st.container(border=True):
                    html(f'<div style="display:flex;align-items:baseline;'
                         f'justify-content:space-between;gap:12px;margin:2px 0 12px;'
                         f'flex-wrap:wrap;"><div style="display:flex;align-items:baseline;'
                         f'gap:12px;"><span class="cl-mono" style="font-size:12px;'
                         f'font-weight:600;color:{INK};">WHAT-IF</span>'
                         f'<span style="font-family:\'Space Grotesk\',sans-serif;'
                         f'font-size:21px;font-weight:600;">Single customer</span></div>'
                         f'<span class="cl-mono" style="font-size:11.5px;color:{FAINT};">'
                         f'scored with the {source} model, not the demo</span></div>')
                    dataset_id = training.get("dataset_id") or (
                        reports.get("mode_decision", {}) or {}).get("dataset_id")
                    kept = [f for f in (training.get("leakage", {}) or {}).get("kept", [])
                            if f in scored.columns]
                    if not dataset_id or not kept:
                        st.info("This run has no trained per-dataset model to "
                                "score against.")
                    else:
                        html(f'<p style="margin:2px 0 10px;font-size:13px;color:{MUTED};">'
                             'Adjust a hypothetical customer built from the columns this '
                             'model was actually trained on. Ranges and options come from '
                             'the integrated data itself.</p>')
                        with st.form("dash_whatif"):
                            values = {}
                            fields = st.columns(min(4, max(1, len(kept))))
                            for i, feat in enumerate(kept):
                                col_w = fields[i % len(fields)]
                                series = scored[feat]
                                if pd.api.types.is_numeric_dtype(series):
                                    lo = int(np.floor(float(series.min())))
                                    hi = int(np.ceil(float(series.max())))
                                    if hi <= lo:
                                        hi = lo + 1
                                    med = min(max(int(round(float(
                                        series.median()))), lo), hi)
                                    values[feat] = col_w.slider(feat, lo, hi, med)
                                else:
                                    opts = series.astype(str).value_counts().index[:20]
                                    values[feat] = col_w.selectbox(feat, list(opts))
                            submitted = st.form_submit_button(
                                "Score this customer", type="primary")
                        if submitted:
                            row_df = pd.DataFrame([values])
                            scored_row = auto_train.score_with_dataset_model(
                                row_df, dataset_id)
                            try:
                                scored_row, _ = ins.enrich_and_summarize(
                                    scored_row, dataset_id)
                            except Exception:
                                pass
                            r = scored_row.iloc[0]
                            tiles = []
                            if "churn_probability" in r:
                                p = float(r["churn_probability"])
                                band = str(r.get("risk_band", ""))
                                tiles.append(tile(
                                    "Churn probability", f"{p:.1%}",
                                    "how likely this profile leaves", accent=RED,
                                    value_color=RED if p >= 0.5 else INK,
                                    tag=band, tag_bg=BAND_COLOR.get(band, RED)))
                            if "expected_remaining_tenure" in r and pd.notna(
                                    r["expected_remaining_tenure"]):
                                tiles.append(tile(
                                    "Expected remaining tenure",
                                    f"{r['expected_remaining_tenure']:.0f} periods",
                                    accent=BLUE))
                            if "clv_simple" in r and pd.notna(r["clv_simple"]):
                                tiles.append(tile("Simplified CLV",
                                                  f"${r['clv_simple']:,.0f}",
                                                  accent=PURPLE))
                            if "revenue_at_risk" in r and pd.notna(r["revenue_at_risk"]):
                                tiles.append(tile("Revenue at risk",
                                                  f"${r['revenue_at_risk']:,.0f}",
                                                  "probability x value", accent=INK))
                            if tiles:
                                tile_row(tiles)
                            if "segment" in r and pd.notna(r["segment"]):
                                html(f'<span style="font-size:12.5px;font-weight:600;'
                                     f'background:#f0eef8;color:#8a5fb0;border:1px solid '
                                     f'#e2daf0;padding:6px 13px;border-radius:20px;">'
                                     f'Segment - cluster {r["segment"]} (Lens D)</span>')
                            drivers_local = r.get("local_drivers")
                            if isinstance(drivers_local, (list, np.ndarray)) and len(
                                    drivers_local):
                                html('<div style="font-size:13px;font-weight:700;'
                                     'margin:12px 0 6px;">What drives THIS profile</div>')
                                items = list(drivers_local)[:5]
                                rows_html = "".join(
                                    f'<div class="cl-mono" style="font-size:12px;'
                                    f'color:{MUTED};line-height:1.8;">{d.get("feature", d)}'
                                    f' - {d.get("effect_pp", "")}pp</div>'
                                    if isinstance(d, dict) else
                                    f'<div class="cl-mono" style="font-size:12px;'
                                    f'color:{MUTED};line-height:1.8;">{d}</div>'
                                    for d in items)
                                html(rows_html)
                            html(f'<div style="font-size:11.5px;color:{FAINT};'
                                 f'margin-top:10px;">Lens C (who can be saved) is not '
                                 f'part of this profile: uplift needs campaign '
                                 f'experiment columns, which a what-if customer does '
                                 f'not have.</div>')
                        lineage(", ".join(kept),
                                "Every field here is a column the pipeline kept after "
                                "leakage screening.")

            # ---- closing play (built only from lenses that actually ran)
            parts = []
            if bands.get("Critical", 0):
                parts.append(f"call the <b>{int(bands['Critical']):,} Critical "
                             f"customers</b> sorted by revenue at risk (A x E)")
            if len(timing_all):
                parts.append(f"reach them <b>before period "
                             f"{int(timing_all.idxmax()) + 1}</b> of their journey (B)")
            if campaign:
                parts.append(f"spend campaign budget only on the "
                             f"<b>{campaign['persuadable']:,} persuadables</b> (C)")
            if worst_seg:
                parts.append(f"fix the root cause in <b>cluster "
                             f"{worst_seg['segment']}</b>, the segment churning at "
                             f"{worst_seg['mean_churn_probability']:.0%} (D)")
            if parts:
                step_rows = "".join(
                    f'<div style="display:flex;gap:14px;align-items:baseline;'
                    f'padding:9px 0;'
                    + ("border-bottom:1px solid #d6e6f2;" if i < len(parts) - 1
                       else "")
                    + f'"><span class="cl-mono" style="font-size:12.5px;'
                    f'font-weight:700;color:{BLUE};">0{i + 1}</span>'
                    f'<span style="font-size:14px;line-height:1.6;color:{INK};">'
                    f'{part[0].upper() + part[1:]}</span></div>'
                    for i, part in enumerate(parts))
                html(f'<div style="background:#eef5fb;border:1px solid #cfe2f0;'
                     f'border-left:4px solid {BLUE};border-radius:14px;'
                     f'padding:16px 22px;margin-top:14px;">'
                     f'<div class="cl-kicker" style="color:#0b5573;'
                     f'margin-bottom:4px;">The recommended play - A to E</div>'
                     f'{step_rows}</div>')
