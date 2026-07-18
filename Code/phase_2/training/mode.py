"""The mode decision - the fork of the whole system.

Origin: architecture.md section 0 (Phase 3 of the build order).

Given a MAPPED frame and its source, decide - deterministically, with reasons:

- SCORE:   trained artifacts exist for this data (per-dataset first, the
           global canonical-schema artifacts second) -> reuse them.
- TRAIN:   a canonical label is present but no usable artifacts -> train
           fresh artifacts on all surviving columns (the primary path for
           new datasets).
- EXPLORE: no label -> label-free lenses only. When unmapped label
           CANDIDATES exist, the reasons say so explicitly - defining the
           label is a human decision, never guessed.
"""
from __future__ import annotations

import pandas as pd

from phase_1.serving import pipeline

from ..profiling import profiler

SCORE = "score"
TRAIN = "train"
EXPLORE = "explore"

LABEL_COL = "Churn"
MIN_TRAIN_ROWS = 100
MIN_FEATURES = 3


def dataset_id_for(source_name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in source_name.lower()).strip("-")


def decide(df: pd.DataFrame, source_name: str,
           force_train: bool = False) -> dict:
    """Returns {"mode", "dataset_id", "reasons": [...]}. Pure function of the
    mapped frame + artifact presence - no side effects. force_train skips
    artifact reuse entirely: a label present means fresh training, even when
    older artifacts exist (e.g. left behind by another session or a crash)."""
    dataset_id = dataset_id_for(source_name)
    reasons: list[str] = []

    if force_train:
        reasons.append("fresh training requested - existing artifacts ignored")
    else:
        # 1) per-dataset artifacts win: this source was trained before
        if pipeline.artifact_exists("churn_pipeline", dataset_id=dataset_id):
            reasons.append(f"trained artifacts exist for dataset '{dataset_id}'")
            return {"mode": SCORE, "dataset_id": dataset_id, "reasons": reasons}

        # 2) global artifacts apply when the data satisfies the canonical contract
        lens_report = pipeline.validate(df)
        if (lens_report["lenses"]["A"]["ok"]
                and pipeline.artifact_exists("churn_pipeline")):
            reasons.append("data satisfies the canonical schema and global "
                           "artifacts are trained")
            return {"mode": SCORE, "dataset_id": dataset_id, "reasons": reasons}

    # 3) a mapped label -> train fresh artifacts for this dataset
    if LABEL_COL in df.columns:
        n_features = df.shape[1] - 1
        if len(df) < MIN_TRAIN_ROWS:
            reasons.append(f"label present but only {len(df)} rows "
                           f"(< {MIN_TRAIN_ROWS}) - too little to train honestly")
        elif n_features < MIN_FEATURES:
            reasons.append(f"label present but only {n_features} feature "
                           f"column(s) (< {MIN_FEATURES})")
        else:
            reasons.append(f"label '{LABEL_COL}' mapped, {len(df):,} rows, "
                           f"no artifacts for '{dataset_id}' - training fresh models")
            return {"mode": TRAIN, "dataset_id": dataset_id, "reasons": reasons}

    # 4) no usable label -> explore; name the candidates instead of guessing
    candidates = [c for c in profiler.label_candidates(profiler.profile_table(df))
                  if c != LABEL_COL]
    if candidates:
        reasons.append("no mapped label; possible label columns exist: "
                       + ", ".join(candidates[:3])
                       + " - confirm one as 'Churn' in the mapping to enable training")
    else:
        reasons.append("no label column - label-free lenses only")
    return {"mode": EXPLORE, "dataset_id": dataset_id, "reasons": reasons}
