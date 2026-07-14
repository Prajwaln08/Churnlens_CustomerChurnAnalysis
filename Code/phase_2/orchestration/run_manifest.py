"""Run manifest - the state machine and JSON ledger for one pipeline run.

Origin: architecture.md section 3 (Phase 2 of the build order).

Every stage records status, timing, row counts, notes and artifacts. The saved
manifest.json is the single source of truth behind the Process screen, the
cleaning report, debugging ("which stage dropped rows?") and reproducibility
(manifest + mapping + data hash = the full recipe of a run).

Design rules:
- the manifest never lies: a stage that raises is recorded FAILED with the
  error before the exception propagates;
- writes are atomic (tmp file + rename) so a crash never leaves half a ledger;
- stage names are fixed constants - a typo'd stage name is a hard error, not a
  silently new entry.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from phase_1 import config

# The stage chain, in execution order (architecture.md section 3).
STAGES = (
    "ingest",
    "profile",
    "map",
    "validate",
    "clean",
    "eda_report",
    "score",
    "persist",
)

# stage statuses
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"
PAUSED = "paused"          # Level 0: waiting for a human confirmation

# run statuses
RUN_RUNNING = "running"
RUN_COMPLETED = "completed"
RUN_FAILED = "failed"
RUN_PAUSED = "paused"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(source_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(c if c.isalnum() else "-" for c in source_name.lower())[:24]
    return f"run_{stamp}_{slug}_{uuid.uuid4().hex[:6]}"


class RunManifest:
    """One run's ledger. Use stage() as a context manager around real work:

        manifest = RunManifest("bank-crm", autonomy_level=1)
        with manifest.stage("profile"):
            ...work...
            manifest.record("profile", rows_in=100, rows_out=100)
        manifest.finish()
    """

    def __init__(self, source_name: str, autonomy_level: int = 1,
                 run_id: str | None = None):
        self.data: dict = {
            "run_id": run_id or new_run_id(source_name),
            "source": source_name,
            "autonomy_level": autonomy_level,
            "created_at": _utcnow(),
            "finished_at": None,
            "status": RUN_RUNNING,
            "input": {},          # rows, columns, content_hash, origin
            "stages": {
                name: {"status": PENDING, "started_at": None, "elapsed_s": None,
                       "rows_in": None, "rows_out": None, "error": None,
                       "notes": [], "artifacts": []}
                for name in STAGES
            },
            "reports": {},        # mapping report, cleaning report, lens report...
        }

    # ------------------------------------------------------------------ access

    @property
    def run_id(self) -> str:
        return self.data["run_id"]

    @property
    def run_dir(self) -> Path:
        d = config.RUNS_DIR / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def status(self) -> str:
        return self.data["status"]

    def _stage(self, name: str) -> dict:
        if name not in self.data["stages"]:
            raise KeyError(f"unknown stage '{name}' - stages are {STAGES}")
        return self.data["stages"][name]

    # ------------------------------------------------------------------ writing

    @contextmanager
    def stage(self, name: str):
        """Run a stage: RUNNING on entry; DONE on clean exit; FAILED with the
        error recorded when an exception escapes (which then propagates)."""
        entry = self._stage(name)
        entry["status"] = RUNNING
        entry["started_at"] = _utcnow()
        t0 = time.time()
        try:
            yield entry
        except Exception as exc:
            entry["status"] = FAILED
            entry["elapsed_s"] = round(time.time() - t0, 3)
            entry["error"] = f"{type(exc).__name__}: {exc}"
            self.finish(RUN_FAILED)
            raise
        else:
            entry["elapsed_s"] = round(time.time() - t0, 3)
            if entry["status"] == RUNNING:      # pause()/skip() may have changed it
                entry["status"] = DONE
            self.save()

    def record(self, stage: str, **fields) -> None:
        """Attach facts to a stage (rows_in/rows_out/artifacts/anything)."""
        entry = self._stage(stage)
        for key, value in fields.items():
            if key == "notes":
                raise ValueError("use add_note() for notes")
            entry[key] = value

    def add_note(self, stage: str, message: str) -> None:
        """One plain-English ledger line (these become the cleaning report)."""
        self._stage(stage)["notes"].append(message)

    def add_artifact(self, stage: str, path: str | Path) -> None:
        self._stage(stage)["artifacts"].append(str(path))

    def attach_report(self, name: str, payload) -> None:
        """Structured reports (mapping report, lens report...) - JSON-safe."""
        self.data["reports"][name] = payload

    def skip(self, stage: str, reason: str) -> None:
        entry = self._stage(stage)
        entry["status"] = SKIPPED
        entry["notes"].append(f"skipped: {reason}")

    def pause(self, stage: str, reason: str) -> None:
        """Level 0 confirmation gate: run stops here, resumable later."""
        entry = self._stage(stage)
        entry["status"] = PAUSED
        entry["notes"].append(f"paused: {reason}")
        self.data["status"] = RUN_PAUSED
        self.save()

    def finish(self, status: str = RUN_COMPLETED) -> None:
        self.data["status"] = status
        self.data["finished_at"] = _utcnow()
        self.save()

    # ------------------------------------------------------------------ persistence

    @property
    def path(self) -> Path:
        return self.run_dir / "manifest.json"

    def save(self) -> Path:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        tmp.replace(self.path)                  # atomic on POSIX
        return self.path

    @classmethod
    def load(cls, run_id: str) -> "RunManifest":
        path = config.RUNS_DIR / run_id / "manifest.json"
        manifest = cls.__new__(cls)
        manifest.data = json.loads(path.read_text())
        return manifest

    # ------------------------------------------------------------------ views

    def summary_frame(self) -> pd.DataFrame:
        """The Process screen's stage checklist, as a DataFrame."""
        rows = [
            {"stage": name, "status": s["status"], "elapsed_s": s["elapsed_s"],
             "rows_in": s["rows_in"], "rows_out": s["rows_out"],
             "notes": " | ".join(s["notes"]), "error": s["error"]}
            for name, s in self.data["stages"].items()
        ]
        return pd.DataFrame(rows)

    def ledger_lines(self) -> list[str]:
        """Every note across stages, in stage order - the cleaning report."""
        return [f"[{name}] {note}"
                for name in STAGES
                for note in self.data["stages"][name]["notes"]]
