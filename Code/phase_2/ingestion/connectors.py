"""Connectors - files, databases, REST APIs, all ending at one shape.

Origin: architecture.md Layer 1 (Phase 4 of the build order).

Design rule implemented literally: every connector terminates at the same
internal object - a SourceData carrying the table plus source metadata
(origin, fetch timestamp, row count, content hash). Downstream code never
knows how data arrived.

Security rules:
- connection-string credentials are REDACTED in every stored origin string;
- API auth headers are never stored, only the URL;
- nothing here writes configs - credentials belong in .env / the keychain.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests


@dataclass
class SourceData:
    """The one shape every connector produces."""
    frame: pd.DataFrame
    origin: str
    fetched_at: str = field(default_factory=lambda: datetime.now(
        timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    @property
    def rows(self) -> int:
        return len(self.frame)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(
            pd.util.hash_pandas_object(self.frame, index=True).values.tobytes()
        ).hexdigest()[:16]


def redact_url(url: str) -> str:
    """postgres://user:secret@host/db -> postgres://user:***@host/db"""
    parts = urlsplit(url)
    if parts.password:
        netloc = parts.netloc.replace(f":{parts.password}@", ":***@")
        return urlunsplit((parts.scheme, netloc, parts.path,
                           parts.query, parts.fragment))
    return url


# ---------------------------------------------------------------------------
# Files (csv / xlsx / parquet / json)
# ---------------------------------------------------------------------------

def from_file(path: str | Path) -> SourceData:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        frame = pd.read_excel(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text())
        frame = pd.json_normalize(payload if isinstance(payload, list)
                                  else payload.get("data", payload))
    else:
        raise ValueError(f"unsupported file type: {suffix} "
                         "(use csv / xlsx / parquet / json)")
    return SourceData(frame=frame, origin=f"file:{path.name}")


def from_dataframe(frame: pd.DataFrame, name: str = "dataframe") -> SourceData:
    return SourceData(frame=frame, origin=f"memory:{name}")


# ---------------------------------------------------------------------------
# Databases - one integration, many engines
# ---------------------------------------------------------------------------

def from_database(url: str, table: str | None = None, query: str | None = None,
                  limit: int | None = None) -> SourceData:
    """Pull a table or query result. `url` is an SQLAlchemy-style connection
    string (postgresql://..., mysql://..., snowflake://..., sqlite:///...).
    Predicate pushdown: prefer `query` with WHERE clauses over full tables.
    """
    if (table is None) == (query is None):
        raise ValueError("provide exactly one of `table` or `query`")
    sql = query or f"SELECT * FROM {table}"          # nosec - local analyst tool
    if limit is not None:
        sql = f"SELECT * FROM ({sql}) AS churnlens_sub LIMIT {int(limit)}"

    if url.startswith("sqlite"):
        # stdlib path - zero extra deps for the most common local case
        db_file = url.split("///", 1)[-1]
        with sqlite3.connect(db_file) as con:
            frame = pd.read_sql_query(sql, con)
    else:
        try:
            from sqlalchemy import create_engine, text
        except ImportError as exc:                   # pragma: no cover
            raise ImportError("database connector needs sqlalchemy: "
                              "pip install sqlalchemy") from exc
        engine = create_engine(url)
        try:
            with engine.connect() as con:
                frame = pd.read_sql_query(text(sql), con)
        finally:
            engine.dispose()

    origin = f"db:{redact_url(url)}#{table or 'query'}"
    return SourceData(frame=frame, origin=origin)


# ---------------------------------------------------------------------------
# REST APIs - URL + auth header + path to the records array
# ---------------------------------------------------------------------------

def from_api(url: str, records_path: str = "", headers: dict | None = None,
             params: dict | None = None, timeout: int = 60) -> SourceData:
    """GET a JSON endpoint and normalize the records found at `records_path`
    (dot-notation into the payload, e.g. "data.items"; empty = the root)."""
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    node = payload
    for key in filter(None, records_path.split(".")):
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"records_path '{records_path}' not found in the "
                             f"response (failed at '{key}')")
        node = node[key]
    if not isinstance(node, list):
        raise ValueError(f"records_path '{records_path}' does not point to a "
                         f"list of records (got {type(node).__name__})")

    frame = pd.json_normalize(node)
    return SourceData(frame=frame, origin=f"api:{url}")
