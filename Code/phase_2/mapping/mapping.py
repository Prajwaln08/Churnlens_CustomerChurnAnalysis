"""Semantic mapping: source columns -> the canonical churn-analytics schema.

Origin: architecture.md section 2b (Phase 1 of the production build).

The three production-safety rules, implemented literally:
1. LLM proposes, deterministic code disposes - the LLM (local Ollama) only ever
   returns a mapping CONFIG; apply_mapping() is plain pandas.
2. Confidence gates - proposals below threshold are surfaced, never auto-applied.
3. Graceful degradation - if Ollama is unreachable or returns garbage, a fuzzy
   name-matcher produces the proposal instead; the pipeline is never blocked.
"""
from __future__ import annotations

import difflib
import json
import os
import re

import pandas as pd

from ..profiling import profiler

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# ---------------------------------------------------------------------------
# Canonical schema - every field the lenses know, with meaning and levels
# ---------------------------------------------------------------------------

CANONICAL: dict[str, dict] = {
    # customer snapshot (lenses A, B, D)
    "customerID": {"desc": "unique customer identifier", "roles": ("id", "text", "numeric")},
    "gender": {"desc": "customer gender", "roles": ("categorical", "binary"),
               "levels": ["Female", "Male"]},
    "SeniorCitizen": {"desc": "is the customer a senior citizen", "roles": ("binary", "categorical"),
                      "levels": ["Yes", "No"]},
    "Partner": {"desc": "has a partner/spouse", "roles": ("binary", "categorical"),
                "levels": ["Yes", "No"]},
    "Dependents": {"desc": "has dependents/children", "roles": ("binary", "categorical"),
                   "levels": ["Yes", "No"]},
    "tenure": {"desc": "months as a customer so far", "roles": ("numeric",)},
    "PhoneService": {"desc": "has phone service", "roles": ("binary", "categorical"),
                     "levels": ["Yes", "No"]},
    "MultipleLines": {"desc": "has multiple phone lines", "roles": ("binary", "categorical"),
                      "levels": ["Yes", "No"]},
    "InternetService": {"desc": "internet service type", "roles": ("categorical",),
                        "levels": ["DSL", "Fiber optic", "No"]},
    "OnlineSecurity": {"desc": "has online security add-on", "roles": ("binary", "categorical"),
                       "levels": ["Yes", "No"]},
    "OnlineBackup": {"desc": "has online backup add-on", "roles": ("binary", "categorical"),
                     "levels": ["Yes", "No"]},
    "DeviceProtection": {"desc": "has device protection add-on", "roles": ("binary", "categorical"),
                         "levels": ["Yes", "No"]},
    "TechSupport": {"desc": "has tech support add-on", "roles": ("binary", "categorical"),
                    "levels": ["Yes", "No"]},
    "StreamingTV": {"desc": "has streaming TV add-on", "roles": ("binary", "categorical"),
                    "levels": ["Yes", "No"]},
    "StreamingMovies": {"desc": "has streaming movies add-on", "roles": ("binary", "categorical"),
                        "levels": ["Yes", "No"]},
    "Contract": {"desc": "contract commitment length", "roles": ("categorical",),
                 "levels": ["Month-to-month", "One year", "Two year"]},
    "PaperlessBilling": {"desc": "uses paperless billing", "roles": ("binary", "categorical"),
                         "levels": ["Yes", "No"]},
    "PaymentMethod": {"desc": "how the customer pays", "roles": ("categorical",),
                      "levels": ["Electronic check", "Mailed check",
                                 "Bank transfer (automatic)", "Credit card (automatic)"]},
    "MonthlyCharges": {"desc": "current monthly bill amount", "roles": ("numeric",)},
    "TotalCharges": {"desc": "lifetime billed amount", "roles": ("numeric",)},
    "Churn": {"desc": "TARGET: did the customer churn/leave", "roles": ("binary",),
              "levels": ["Yes", "No"], "is_label": True},
    # transaction log (lenses D, E)
    "CustomerID": {"desc": "customer identifier on a transaction row", "roles": ("id", "numeric", "text")},
    "InvoiceDate": {"desc": "date/time of the transaction", "roles": ("datetime",)},
    "Invoice": {"desc": "order/invoice number", "roles": ("id", "text", "numeric")},
    "Quantity": {"desc": "units purchased in the row", "roles": ("numeric",)},
    "Price": {"desc": "unit price", "roles": ("numeric",)},
    "Revenue": {"desc": "row amount = quantity x price", "roles": ("numeric",)},
    # campaign / experiment (lens C)
    "treatment_group": {"desc": "who received the campaign (treatment/control)",
                        "roles": ("binary", "categorical"),
                        "levels": ["control", "treatment"]},
    "conversion": {"desc": "TARGET: campaign outcome (converted or not)",
                   "roles": ("binary",), "is_label": True},
}

_SYNONYMS = {
    "cust": "customer", "acct": "account", "acc": "account", "dt": "date",
    "amt": "amount", "num": "number", "mth": "month", "mo": "month",
    "chrg": "charge", "chgs": "charges", "svc": "service", "pmt": "payment",
    "ref": "id", "key": "id", "flag": "", "ind": "",
}


class MappingError(RuntimeError):
    """Raised when the LLM path cannot produce a usable proposal."""


# ---------------------------------------------------------------------------
# Fuzzy fallback - stdlib only, never blocked
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"[_\-./]+", " ", s.lower()).strip()
    words = [(_SYNONYMS.get(w, w)) for w in s.split()]
    return " ".join(w for w in words if w)


def _role_compatible(source_role: str, target: str) -> bool:
    return source_role in CANONICAL[target]["roles"]


def fuzzy_propose(profile: dict) -> dict:
    """Name-similarity proposal. Confidence = similarity, minus a penalty when
    the profiled role disagrees with what the canonical field expects."""
    mappings = []
    for source, col in profile["columns"].items():
        best_target, best_score = None, 0.0
        for target, spec in CANONICAL.items():
            score = difflib.SequenceMatcher(
                None, _normalize(source), _normalize(target)).ratio()
            desc_score = difflib.SequenceMatcher(
                None, _normalize(source), _normalize(spec["desc"])).ratio()
            score = max(score, 0.85 * desc_score)
            if not _role_compatible(col["role"], target):
                score *= 0.5
            if score > best_score:
                best_target, best_score = target, score
        mappings.append({
            "source": source,
            "target": best_target if best_score >= 0.55 else None,
            "confidence": round(best_score, 3),
            "value_map": None,
        })
    return {"engine": "fuzzy", "mappings": mappings}


# ---------------------------------------------------------------------------
# LLM proposer - local Ollama, JSON mode, temperature 0
# ---------------------------------------------------------------------------

def ollama_available(url: str = OLLAMA_URL) -> bool:
    import requests

    try:
        return requests.get(f"{url}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


def _build_prompt(profile: dict) -> str:
    canon_lines = [
        f"- {name}: {spec['desc']}"
        + (f" [levels: {', '.join(spec['levels'])}]" if "levels" in spec else "")
        for name, spec in CANONICAL.items()
    ]
    src_lines = [
        f"- {col} | role={p['role']} | examples: {', '.join(p['samples']) or '-'}"
        for col, p in profile["columns"].items()
    ]
    return (
        "Map source data columns to a canonical churn-analytics schema.\n\n"
        "CANONICAL FIELDS:\n" + "\n".join(canon_lines) + "\n\n"
        "SOURCE COLUMNS:\n" + "\n".join(src_lines) + "\n\n"
        "Rules:\n"
        "- Map a source column only when the MEANING matches a canonical field; "
        "otherwise target must be null.\n"
        "- confidence is a number 0..1.\n"
        "- When the source uses different level names than the canonical levels, "
        "add value_map from source value to canonical level (else null).\n"
        "- Every source column must appear exactly once.\n\n"
        'Return ONLY JSON, no prose: {"mappings": [{"source": str, '
        '"target": str or null, "confidence": number, "value_map": object or null}]}'
    )


def ollama_propose(profile: dict, model: str = OLLAMA_MODEL,
                   url: str = OLLAMA_URL, timeout: int = 300) -> dict:
    import requests

    resp = requests.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": _build_prompt(profile)}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    try:
        raw = json.loads(resp.json()["message"]["content"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise MappingError(f"Ollama returned unparseable output: {exc}") from exc

    proposal = _sanitize(raw, profile)
    proposal["engine"] = f"ollama:{model}"
    return proposal


def _sanitize(raw: dict, profile: dict) -> dict:
    """Trust nothing: keep only real source columns and real canonical targets,
    clamp confidences, fill in any columns the model forgot (as unmapped)."""
    source_cols = set(profile["columns"])
    seen: dict[str, dict] = {}
    for m in raw.get("mappings", []):
        src = m.get("source")
        if src not in source_cols or src in seen:
            continue
        target = m.get("target")
        if target is not None and target not in CANONICAL:
            target = None
        try:
            conf = min(1.0, max(0.0, float(m.get("confidence", 0))))
        except (TypeError, ValueError):
            conf = 0.0
        value_map = m.get("value_map")
        if not isinstance(value_map, dict) or target is None:
            value_map = None
        elif "levels" in CANONICAL.get(target, {}):
            allowed = set(CANONICAL[target]["levels"])
            value_map = {str(k): v for k, v in value_map.items()
                         if isinstance(v, str) and v in allowed} or None
        seen[src] = {"source": src, "target": target,
                     "confidence": round(conf, 3), "value_map": value_map}
    for src in source_cols - set(seen):
        seen[src] = {"source": src, "target": None, "confidence": 0.0,
                     "value_map": None}
    if not any(m["target"] for m in seen.values()):
        raise MappingError("Ollama mapped zero columns - falling back")
    return {"mappings": list(seen.values())}


# ---------------------------------------------------------------------------
# Orchestration + the deterministic executor
# ---------------------------------------------------------------------------

def propose_mapping(df: pd.DataFrame, use_llm: bool = True) -> dict:
    """Profile the frame, then propose a mapping: Ollama when reachable,
    fuzzy otherwise. Deterministic quarantine flags ride along either way."""
    profile = profiler.profile_table(df)
    proposal = None
    if use_llm and ollama_available():
        try:
            proposal = ollama_propose(profile)
        except Exception:
            proposal = None  # any LLM failure -> fallback, never a dead end
    if proposal is None:
        proposal = fuzzy_propose(profile)

    proposal["label_candidates"] = profiler.label_candidates(profile)
    proposal["quarantine"] = [
        {"source": c, "reason": "post-outcome suspect (name pattern)"}
        for c, p in profile["columns"].items() if p["post_outcome_suspect"]
    ]
    proposal["profile"] = profile
    return proposal


def apply_mapping(df: pd.DataFrame, proposal: dict,
                  min_confidence: float = 0.6) -> tuple[pd.DataFrame, dict]:
    """Execute a (confirmed) proposal with plain pandas. Returns
    (mapped frame, report). Nothing below min_confidence is applied; two
    sources claiming one target keeps the higher-confidence one."""
    quarantined = {q["source"] for q in proposal.get("quarantine", [])}
    # a reused mapping may reference columns this frame does not carry
    # (e.g. the label at scoring time) - absent sources are skipped, reported
    absent = [m["source"] for m in proposal["mappings"]
              if m["source"] not in df.columns]
    candidates = [m for m in proposal["mappings"]
                  if m["target"] and m["confidence"] >= min_confidence
                  and m["source"] in df.columns
                  and m["source"] not in quarantined]
    candidates.sort(key=lambda m: -m["confidence"])

    renames: dict[str, str] = {}
    claimed: set[str] = set()
    skipped_collisions = []
    for m in candidates:
        if m["target"] in claimed:
            skipped_collisions.append(m["source"])
            continue
        renames[m["source"]] = m["target"]
        claimed.add(m["target"])

    out = df.rename(columns=renames).copy()
    translated: dict[str, int] = {}
    for m in candidates:
        if m["source"] in renames and m["value_map"]:
            col = renames[m["source"]]
            mask = out[col].astype(str).isin(m["value_map"])
            if mask.any():
                out.loc[mask, col] = out.loc[mask, col].astype(str).map(m["value_map"])
                translated[col] = int(mask.sum())

    report = {
        "engine": proposal.get("engine", "unknown"),
        "renamed": renames,
        "values_translated": translated,
        "below_threshold": [m["source"] for m in proposal["mappings"]
                            if m["target"] and m["confidence"] < min_confidence],
        "unmapped": [m["source"] for m in proposal["mappings"] if not m["target"]],
        "quarantined": sorted(quarantined),
        "collisions_skipped": skipped_collisions,
        "sources_absent": absent,
    }
    return out, report
