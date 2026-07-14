"""Deterministic column profiling - the pipeline's first look at unknown data.

Origin: architecture.md section 2a (Phase 1 of the production build).

Facts only, no interpretation: every column gets the same battery of tests and
comes out with a ROLE (id / datetime / numeric / binary / categorical / text /
constant), null stats, cardinality and a few MASKED sample values. Meaning
(which canonical field a column corresponds to) is deliberately left to the
mapping stage - profiling can never be wrong, only incomplete.
"""
from __future__ import annotations

import re

import pandas as pd

# roles
ID = "id"
DATETIME = "datetime"
NUMERIC = "numeric"
BINARY = "binary"
CATEGORICAL = "categorical"
TEXT = "text"
CONSTANT = "constant"

_DATEISH = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")
_LABELISH_NAMES = re.compile(r"churn|conversion|converted|target|label|outcome|attrit", re.I)
_POST_OUTCOME_NAMES = re.compile(r"cancel|close[_ ]?reason|exit|termination|deactivat", re.I)
_ID_NAMES = re.compile(r"(^|[_ ])(id|uuid|guid|ref|key)([_ ]|$)|customer[_ ]?id|cust[_ ]?(id|ref)", re.I)


def _mask(value, width: int = 14) -> str:
    s = str(value)
    return s if len(s) <= width else s[:width] + "..."


def profile_column(s: pd.Series, n_rows: int) -> dict:
    non_null = s.dropna()
    nn = len(non_null)
    nunique = non_null.nunique()
    out: dict = {
        "dtype": str(s.dtype),
        "null_pct": round(100 * (1 - nn / max(n_rows, 1)), 2),
        "n_unique": int(nunique),
        "samples": [_mask(v) for v in non_null.unique()[:3]],
    }

    # role inference - order matters, most specific first
    name = str(s.name or "")
    if nunique <= 1:
        role = CONSTANT
    elif pd.api.types.is_datetime64_any_dtype(s):
        role = DATETIME
    elif s.dtype == object and nn > 0 and (
        non_null.astype(str).head(50).str.contains(_DATEISH).mean() >= 0.9
    ):
        role = DATETIME
    elif nunique == 2:
        role = BINARY
    elif s.dtype == object and nn > 0 and (
        (parsed := pd.to_numeric(non_null, errors="coerce")).notna().mean() >= 0.95
        and (parsed.dropna() % 1 != 0).any()
    ):
        role = NUMERIC  # decimal values in a text costume are measures, never ids
    elif nn > 50 and nunique / nn > 0.98 and (
        (s.dtype == object and non_null.astype(str).str.len().mean() <= 24)
        or pd.api.types.is_integer_dtype(s) or _ID_NAMES.search(name)
    ):
        role = ID  # identifiers are short unique tokens; long unique strings are text
    elif pd.api.types.is_numeric_dtype(s):
        role = NUMERIC
    elif s.dtype == object and pd.to_numeric(non_null, errors="coerce").notna().mean() >= 0.95:
        role = NUMERIC  # numbers wearing a text costume (the TotalCharges pattern)
    elif s.dtype == object and nn > 0 and non_null.astype(str).str.len().mean() > 40:
        role = TEXT
    else:
        role = CATEGORICAL

    out["role"] = role
    out["label_like"] = bool(role == BINARY and _LABELISH_NAMES.search(name)) or (
        role == BINARY and set(map(str, non_null.unique())) <= {"0", "1", "Yes", "No", "True", "False"}
        and bool(_LABELISH_NAMES.search(name))
    )
    out["post_outcome_suspect"] = bool(_POST_OUTCOME_NAMES.search(name))
    return out


def profile_table(df: pd.DataFrame) -> dict:
    """The full fingerprint: one profile per column + table-level facts."""
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "columns": {col: profile_column(df[col], len(df)) for col in df.columns},
    }


def label_candidates(profile: dict) -> list[str]:
    """Binary columns that could be the churn/conversion label, best first."""
    named = [c for c, p in profile["columns"].items() if p.get("label_like")]
    other_binary = [c for c, p in profile["columns"].items()
                    if p["role"] == BINARY and c not in named]
    return named + other_binary


def correlation_report(df: pd.DataFrame, threshold: float = 0.9,
                       max_cols: int = 60) -> list[dict]:
    """Highly correlated numeric pairs (|Pearson| >= threshold), strongest
    first. Informational, not a gate: trees shrug at collinearity, but linear
    coefficients and human interpretation deserve the warning. Capped at
    max_cols columns to stay O(k^2)-cheap on wide data."""
    numeric = df.select_dtypes("number").iloc[:, :max_cols]
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr().abs()
    cols = corr.columns
    pairs = [
        {"a": cols[i], "b": cols[j], "corr": round(float(corr.iloc[i, j]), 3)}
        for i in range(len(cols)) for j in range(i + 1, len(cols))
        if pd.notna(corr.iloc[i, j]) and corr.iloc[i, j] >= threshold
    ]
    return sorted(pairs, key=lambda p: -p["corr"])


def summary_frame(profile: dict) -> pd.DataFrame:
    """Human-readable view of a profile (the Data Review table)."""
    rows = [
        {"column": col, "role": p["role"], "dtype": p["dtype"],
         "null_%": p["null_pct"], "unique": p["n_unique"],
         "samples": ", ".join(p["samples"]),
         "flags": " ".join(f for f, on in
                           [("label?", p["label_like"]),
                            ("post-outcome?", p["post_outcome_suspect"])] if on)}
        for col, p in profile["columns"].items()
    ]
    return pd.DataFrame(rows)
