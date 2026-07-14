"""SQLite registry - the queryable facts about runs, sources and mappings.

Origin: architecture.md section 16 (Phase 2 of the build order).
Rule of thumb implemented here: files for bulk, SQLite for facts about the
files. Customer data never enters this database - only metadata: which runs
happened, for which source, with which confirmed mapping.

Design rules:
- stdlib sqlite3, one file (models/registry.sqlite), zero administration;
- a fresh connection per operation - safe under Streamlit's threading;
- schema creation is idempotent - constructing a Registry never destroys data;
- confirmed mappings are versioned per source and reused forever (the LLM
  cost is paid once per new schema).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from phase_1 import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_name  TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    last_run_id  TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    source_name    TEXT NOT NULL,
    status         TEXT NOT NULL,
    autonomy_level INTEGER,
    created_at     TEXT,
    finished_at    TEXT,
    rows           INTEGER,
    manifest_path  TEXT,
    FOREIGN KEY (source_name) REFERENCES sources(source_name)
);
CREATE TABLE IF NOT EXISTS mappings (
    source_name  TEXT NOT NULL,
    version      INTEGER NOT NULL,
    engine       TEXT,
    confirmed    INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    mapping_json TEXT NOT NULL,
    PRIMARY KEY (source_name, version)
);
"""


class Registry:
    """All methods open a short-lived connection - no shared state to corrupt."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else config.REGISTRY_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    # ------------------------------------------------------------------ runs

    def record_run(self, manifest_data: dict) -> None:
        """Upsert one run from a RunManifest's data dict; keeps the source's
        last_run_id current. Call after every save-worthy transition."""
        now = manifest_data.get("created_at")
        with self._connect() as con:
            con.execute(
                "INSERT INTO sources (source_name, created_at, last_run_id) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(source_name) DO UPDATE SET last_run_id=excluded.last_run_id",
                (manifest_data["source"], now, manifest_data["run_id"]),
            )
            con.execute(
                "INSERT INTO runs (run_id, source_name, status, autonomy_level, "
                "created_at, finished_at, rows, manifest_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "status=excluded.status, finished_at=excluded.finished_at, "
                "rows=excluded.rows, manifest_path=excluded.manifest_path",
                (
                    manifest_data["run_id"],
                    manifest_data["source"],
                    manifest_data["status"],
                    manifest_data.get("autonomy_level"),
                    manifest_data.get("created_at"),
                    manifest_data.get("finished_at"),
                    (manifest_data.get("input") or {}).get("rows"),
                    str(config.RUNS_DIR / manifest_data["run_id"] / "manifest.json"),
                ),
            )

    def get_run(self, run_id: str) -> dict | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def runs_for_source(self, source_name: str) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query(
                "SELECT * FROM runs WHERE source_name=? ORDER BY created_at DESC",
                con, params=(source_name,),
            )

    def all_runs(self, limit: int = 50) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                con, params=(limit,),
            )

    # ------------------------------------------------------------------ sources

    def sources(self) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query(
                "SELECT * FROM sources ORDER BY source_name", con)

    # ------------------------------------------------------------------ mappings

    def save_mapping(self, source_name: str, proposal: dict,
                     confirmed: bool, created_at: str) -> int:
        """Store a mapping proposal (bulky profile stripped - it lives in the
        run manifest). Returns the new version number for this source."""
        payload = {k: v for k, v in proposal.items() if k != "profile"}
        with self._connect() as con:
            row = con.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM mappings "
                "WHERE source_name=?", (source_name,),
            ).fetchone()
            version = row["v"] + 1
            con.execute(
                "INSERT INTO mappings (source_name, version, engine, confirmed, "
                "created_at, mapping_json) VALUES (?, ?, ?, ?, ?, ?)",
                (source_name, version, payload.get("engine"),
                 int(confirmed), created_at, json.dumps(payload)),
            )
        return version

    def latest_mapping(self, source_name: str,
                       confirmed_only: bool = True) -> dict | None:
        """The mapping to reuse for a returning source: newest (confirmed)
        version, or None when this source has never been mapped."""
        query = ("SELECT mapping_json, version FROM mappings WHERE source_name=? "
                 + ("AND confirmed=1 " if confirmed_only else "")
                 + "ORDER BY version DESC LIMIT 1")
        with self._connect() as con:
            row = con.execute(query, (source_name,)).fetchone()
        if row is None:
            return None
        payload = json.loads(row["mapping_json"])
        payload["version"] = row["version"]
        return payload
