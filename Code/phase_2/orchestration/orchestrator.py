"""The orchestrator - one call drives a dataset through the staged pipeline.

Origin: architecture.md sections 3 (orchestration), 16 (storage), 19 (autonomy),
Phase 2 of the build order.

    ingest -> profile -> map -> validate -> clean -> eda_report -> score -> persist

Every stage writes to the RunManifest (the ledger) and the Registry (the
queryable facts). Autonomy levels, implemented literally:

- Level 0 (assisted): a source with no CONFIRMED mapping pauses at the map
  stage; the proposal is saved unconfirmed for the UI to show. A later call
  with `proposal_override` (the human-confirmed mapping) runs through.
- Level 1 (auto-pilot): the proposal applies immediately at the confidence
  threshold and is saved as confirmed - the never-guess pauses stay elsewhere
  (label definition, quarantined columns are excluded either way).

Scope note: this is SCORE mode - artifacts must already exist. Train mode
(fresh artifacts per new dataset) is build-order phase 3 and slots in as one
more branch at the score stage.
"""
from __future__ import annotations

import hashlib

import matplotlib

matplotlib.use("Agg")  # headless-safe everywhere (tests, CLI, Streamlit)
import matplotlib.pyplot as plt
import pandas as pd

from phase_1.serving import pipeline
from phase_1.visualization import eda

from ..mapping import mapping as mapping_mod
from ..monitoring import drift as drift_mod
from ..narration import narrator
from ..profiling import profiler
from ..training import auto_train
from ..training import mode as mode_mod
from .registry import Registry
from .run_manifest import RunManifest


def _content_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


def _numeric_label(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.map({"Yes": 1, "No": 0, "True": 1, "False": 0}).fillna(0)
    return pd.to_numeric(series, errors="coerce").fillna(0)


def run_pipeline(
    df: pd.DataFrame,
    source_name: str,
    autonomy_level: int = 1,
    *,
    min_confidence: float = 0.6,
    use_llm: bool = True,
    registry: Registry | None = None,
    proposal_override: dict | None = None,
    raise_on_error: bool = False,
) -> RunManifest:
    """Run one dataset through the stage chain. Always returns the manifest -
    on failure its status is 'failed' with the error recorded (set
    raise_on_error=True to also propagate, e.g. in tests)."""
    registry = registry or Registry()
    m = RunManifest(source_name, autonomy_level=autonomy_level)
    try:
        _run(df, m, registry, autonomy_level, min_confidence,
             use_llm, proposal_override)
    except Exception:
        registry.record_run(m.data)
        if raise_on_error:
            raise
    return m


def _run(df, m, registry, autonomy_level, min_confidence, use_llm, proposal_override):
    # ---------------------------------------------------------------- ingest
    with m.stage("ingest"):
        origin, fetched_at = "dataframe", None
        if hasattr(df, "frame"):                     # a connectors.SourceData
            origin, fetched_at = df.origin, df.fetched_at
            df = df.frame
        m.record("ingest", rows_in=len(df), rows_out=len(df))
        m.data["input"] = {"rows": len(df), "columns": df.shape[1],
                           "content_hash": _content_hash(df),
                           "origin": origin, "fetched_at": fetched_at}
        m.add_note("ingest", f"received {len(df):,} rows x {df.shape[1]} columns "
                             f"from {origin} "
                             f"(hash {m.data['input']['content_hash']})")
    registry.record_run(m.data)

    # ---------------------------------------------------------------- profile
    with m.stage("profile"):
        profile = profiler.profile_table(df)
        roles = pd.Series([p["role"] for p in profile["columns"].values()])
        m.add_note("profile", "roles: " + ", ".join(
            f"{count} {role}" for role, count in roles.value_counts().items()))
        labels = profiler.label_candidates(profile)
        if labels:
            m.add_note("profile", f"label candidates: {', '.join(labels[:3])}")
        profiler.summary_frame(profile).to_csv(m.run_dir / "profile.csv", index=False)
        m.add_artifact("profile", m.run_dir / "profile.csv")
        m.record("profile", rows_in=len(df), rows_out=len(df))

    # ---------------------------------------------------------------- map
    with m.stage("map"):
        if proposal_override is not None:
            proposal = proposal_override
            m.add_note("map", "using human-confirmed mapping (override)")
            registry.save_mapping(m.data["source"], proposal, confirmed=True,
                                  created_at=m.data["created_at"])
        else:
            saved = registry.latest_mapping(m.data["source"])
            if saved is not None:
                proposal = saved
                m.add_note("map", f"reusing confirmed mapping v{saved['version']} "
                                  f"(engine: {saved.get('engine')})")
            else:
                proposal = mapping_mod.propose_mapping(df, use_llm=use_llm)
                m.add_note("map", f"new proposal from engine: {proposal['engine']}")
                if autonomy_level == 0:
                    registry.save_mapping(m.data["source"], proposal,
                                          confirmed=False,
                                          created_at=m.data["created_at"])
                    m.attach_report("mapping_proposal", {
                        k: v for k, v in proposal.items() if k != "profile"})
                    m.pause("map", "Level 0: mapping awaits human confirmation")
                    registry.record_run(m.data)
                    return
                registry.save_mapping(m.data["source"], proposal, confirmed=True,
                                      created_at=m.data["created_at"])
                m.add_note("map", "auto-pilot: proposal auto-confirmed at "
                                  f">= {min_confidence:.0%} confidence")
        mapped, map_report = mapping_mod.apply_mapping(df, proposal,
                                                       min_confidence=min_confidence)
        m.attach_report("mapping", map_report)
        m.add_note("map", f"renamed {len(map_report['renamed'])} columns; "
                          f"{len(map_report['unmapped'])} unmapped kept as extras")
        for col, n in map_report["values_translated"].items():
            m.add_note("map", f"{col}: {n:,} values translated to canonical levels")
        for col in map_report["quarantined"]:
            m.add_note("map", f"{col}: quarantined (post-outcome suspect) - excluded")
        m.record("map", rows_in=len(df), rows_out=len(mapped))

    # ---------------------------------------------------------------- validate
    with m.stage("validate"):
        lens_report = pipeline.validate(mapped)
        m.attach_report("lenses", {
            lens: info["message"] for lens, info in lens_report["lenses"].items()})
        ready = [lens for lens, info in lens_report["lenses"].items()
                 if info["ok"] and info["artifact_available"]]
        m.add_note("validate", f"lenses ready: {', '.join(ready) or 'none'}")
        if lens_report["ignored_columns"]:
            m.add_note("validate", "ignored columns (no lens uses them): "
                       + ", ".join(lens_report["ignored_columns"]))

    # ---------------------------------------------------------------- clean
    with m.stage("clean"):
        text_numerics = [c for c in ("tenure", "MonthlyCharges", "TotalCharges")
                         if c in mapped.columns and mapped[c].dtype == object]
        cleaned = pipeline._coerce_telco(mapped)
        for col in text_numerics:
            blanks = int(pd.to_numeric(mapped[col], errors="coerce").isna().sum())
            m.add_note("clean", f"{col}: text -> numeric"
                       + (f" ({blanks} unparseable values set to 0)" if blanks else ""))
        if not text_numerics:
            m.add_note("clean", "no coercions needed")
        m.record("clean", rows_in=len(mapped), rows_out=len(cleaned))

    # ---------------------------------------------------------------- eda_report
    with m.stage("eda_report"):
        charts = _eda_charts(cleaned, m.run_dir)
        for path in charts:
            m.add_artifact("eda_report", path)
        m.add_note("eda_report", f"{len(charts)} chart(s) attached to the run")

    # ---------------------------------------------------------------- mode
    with m.stage("mode"):
        decision = mode_mod.decide(cleaned, m.data["source"])
        m.attach_report("mode_decision", decision)
        m.add_note("mode", f"decision: {decision['mode'].upper()} - "
                           + "; ".join(decision["reasons"]))
    dataset_id = decision["dataset_id"]

    # ---------------------------------------------------------------- train
    train_report = None
    if decision["mode"] == mode_mod.TRAIN:
        with m.stage("train"):
            train_report = auto_train.train_for_dataset(cleaned, dataset_id)
            m.attach_report("training", train_report)
            for col, reason in train_report["leakage"]["dropped"].items():
                m.add_note("train", f"excluded {col}: {reason}")
            m.add_note("train", f"features used: "
                                f"{len(train_report['leakage']['kept'])}")
            high_card = train_report.get("encoding", {}).get("frequency", [])
            if high_card:
                m.add_note("train", "high-cardinality columns frequency-"
                                    "encoded: " + ", ".join(high_card))
            for lens, info in train_report["lenses"].items():
                m.add_note("train", f"lens {lens}: {info['status']} - "
                                    f"{info['reason']}")
            m.record("train", rows_in=len(cleaned), rows_out=len(cleaned))
    else:
        m.skip("train", f"mode is {decision['mode']} - no training needed")

    # ---------------------------------------------------------------- score
    scored = None
    if decision["mode"] == mode_mod.SCORE:
        with m.stage("score"):
            if pipeline.artifact_exists("churn_pipeline", dataset_id=dataset_id):
                scored = auto_train.score_with_dataset_model(cleaned, dataset_id)
                m.add_note("score", f"scored with dataset '{dataset_id}' model")
            else:
                scored, score_report = pipeline.score(cleaned)
                m.attach_report("scoring", {
                    "applied": score_report["applied"],
                    "errors": score_report.get("errors", {})})
                m.add_note("score", "lenses applied: "
                           + (", ".join(score_report["applied"]) or "none"))
                for lens, err in score_report.get("errors", {}).items():
                    m.add_note("score", f"lens {lens} error: {err}")
            m.record("score", rows_in=len(cleaned), rows_out=len(scored))
    elif (train_report is not None
          and train_report["lenses"]["A"]["status"] == "promoted"):
        with m.stage("score"):
            scored = auto_train.score_with_dataset_model(cleaned, dataset_id)
            auc = train_report["lenses"]["A"]["test_roc_auc"]
            m.add_note("score", f"scored with freshly trained model "
                                f"(test ROC-AUC {auc})")
            m.record("score", rows_in=len(cleaned), rows_out=len(scored))
    elif train_report is not None:
        m.skip("score", "trained model did not clear its quality gate - "
                        "scores would not be trustworthy")
    else:
        m.skip("score", "explore mode - no label, nothing to score with")

    # ---------------------------------------------------------------- drift
    if decision["mode"] in (mode_mod.SCORE, mode_mod.TRAIN):
        with m.stage("drift"):
            drift_report = drift_mod.assess(cleaned, dataset_id)
            if drift_report is None:
                m.add_note("drift", "no baseline for this dataset - "
                                    "drift check not possible (global-model "
                                    "scores are not baselined)")
            else:
                m.attach_report("drift", drift_report)
                if decision["mode"] == mode_mod.TRAIN:
                    m.add_note("drift", "baseline established this run "
                               f"(max PSI {drift_report['max_psi']})")
                else:
                    m.add_note("drift", drift_report["recommendation"]
                               + f" (max PSI {drift_report['max_psi']})")
    else:
        m.skip("drift", "explore mode - no model, nothing to drift against")

    # ---------------------------------------------------------------- persist
    with m.stage("persist"):
        out = scored if scored is not None else cleaned
        out_path = m.run_dir / ("scored.parquet" if scored is not None
                                else "mapped.parquet")
        out.to_parquet(out_path)
        m.add_artifact("persist", out_path)
        m.add_note("persist", f"{len(out):,} rows -> {out_path.name}")
        m.record("persist", rows_in=len(out), rows_out=len(out))

    # ---------------------------------------------------------------- narrate
    with m.stage("narrate"):
        # narrate is the final stage: every prior stage succeeded, so the
        # run's true status is 'completed' even though finish() runs just after
        narrative = narrator.narrate_run({**m.data, "status": "completed"},
                                         scored, use_llm=use_llm)
        m.attach_report("narrative", narrative)
        m.add_note("narrate", f"summary written by {narrative['engine']}")
        if narrative["audit_violations"]:
            m.add_note("narrate", "audit rejected LLM figures: "
                       + ", ".join(narrative["audit_violations"]))

    m.finish()
    registry.record_run(m.data)


def _eda_charts(df: pd.DataFrame, run_dir) -> list:
    """The Data Review charts: label balance + top categorical rates when a
    label exists, numeric distributions otherwise. Never raises for odd data."""
    paths = []
    has_label = "Churn" in df.columns
    if has_label:
        temp = df.assign(_label=_numeric_label(df["Churn"]))
        fig, ax = plt.subplots(figsize=(6, 2.2))
        rate = temp["_label"].mean()
        ax.barh(["stayed", "churned"],
                [(1 - rate) * len(df), rate * len(df)], color=eda.PALETTE[:2])
        ax.set_title(f"Label balance - {rate:.1%} churn", loc="left",
                     fontweight="bold", fontsize=10)
        paths.append(_save(fig, run_dir / "eda_label_balance.png"))
        cats = [c for c in temp.select_dtypes(exclude="number").columns
                if c != "Churn" and 2 <= temp[c].nunique() <= 6][:2]
        for col in cats:
            ax = eda.plot_rate_by_category(temp, col, target_col="_label")
            paths.append(_save(ax.figure, run_dir / f"eda_rate_{col}.png"))
    else:
        for col in df.select_dtypes("number").columns[:2]:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(df[col].dropna(), bins=30, color=eda.PALETTE[0])
            ax.set_title(f"{col} distribution", loc="left",
                         fontweight="bold", fontsize=10)
            paths.append(_save(fig, run_dir / f"eda_dist_{col}.png"))
    return paths


def _save(fig, path) -> str:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)
