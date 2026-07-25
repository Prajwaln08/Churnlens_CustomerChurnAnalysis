"""Run narration - computed facts in, plain-English summary out.

Origin: architecture.md sections 4 and 19 (Phase 5 of the build order).

The guardrail, implemented literally: **numbers are injected from computed
results; the LLM phrases, never calculates.** Enforcement is mechanical -
every multi-digit number in the LLM's text is audited against the fact sheet,
and any invented figure triggers an automatic fallback to the deterministic
template. The pipeline therefore never blocks on the LLM and never publishes
a hallucinated number.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from ..mapping import mapping as llm  # call-time indirection: one Ollama switch

# numbers with fewer digits than this are treated as prose ("three lenses",
# "one minute") and exempt from the audit
AUDIT_MIN_DIGITS = 2


# ---------------------------------------------------------------------------
# Facts - deterministic extraction, the only source of numbers
# ---------------------------------------------------------------------------

def facts_from_run(manifest_data: dict, scored: pd.DataFrame | None = None) -> dict:
    """Everything narratable about a run, computed - never guessed."""
    facts: dict = {
        "source": manifest_data.get("source"),
        "origin": (manifest_data.get("input") or {}).get("origin")
                  or "unknown origin",
        "rows": (manifest_data.get("input") or {}).get("rows"),
        "status": manifest_data.get("status"),
    }
    reports = manifest_data.get("reports", {})

    decision = reports.get("mode_decision")
    if decision:
        facts["mode"] = decision["mode"]
        facts["mode_reasons"] = decision["reasons"]

    mapping_rep = reports.get("mapping")
    if mapping_rep:
        facts["mapping_engine"] = mapping_rep.get("engine")
        facts["columns_mapped"] = len(mapping_rep.get("renamed", {}))
        facts["columns_quarantined"] = len(mapping_rep.get("quarantined", []))

    training = reports.get("training")
    if training:
        facts["training"] = {
            lens: {"status": info.get("status"),
                   **{k: info[k] for k in ("test_roc_auc", "c_index", "k")
                      if k in info}}
            for lens, info in training.get("lenses", {}).items()}
        facts["features_used"] = len(training.get("leakage", {}).get("kept", []))
        facts["features_excluded"] = len(training.get("leakage", {}).get("dropped", {}))

    scoring = reports.get("scoring")
    if scoring:
        facts["lenses_applied"] = scoring.get("applied", [])

    if scored is not None and "risk_band" in scored.columns:
        counts = scored["risk_band"].value_counts()
        facts["risk_bands"] = {band: int(counts.get(band, 0))
                               for band in ("Low", "Medium", "High", "Critical")}
        facts["high_or_critical"] = int(counts.get("High", 0)
                                        + counts.get("Critical", 0))
    if scored is not None and "revenue_at_risk" in scored.columns:
        facts["total_revenue_at_risk"] = int(scored["revenue_at_risk"].sum())

    drift = reports.get("drift")
    if drift:
        facts["drift"] = {"max_psi": drift.get("max_psi"),
                          "alerts": drift.get("alerts", []),
                          "recommendation": drift.get("recommendation")}
    return facts


# ---------------------------------------------------------------------------
# Template - the deterministic fallback (and the audit's safety net)
# ---------------------------------------------------------------------------

def template_summary(facts: dict) -> str:
    parts = [f"Run for source '{facts.get('source')}' "
             f"({facts.get('rows'):,} rows from {facts.get('origin')}) "
             f"finished with status: {facts.get('status')}."]
    if facts.get("mode"):
        parts.append(f"Mode decision: {facts['mode'].upper()} - "
                     + "; ".join(facts.get("mode_reasons", [])) + ".")
    if "columns_mapped" in facts:
        parts.append(f"{facts['columns_mapped']} columns were mapped to the "
                     f"canonical schema ({facts.get('mapping_engine')})"
                     + (f"; {facts['columns_quarantined']} quarantined as "
                        f"potential leaks" if facts.get("columns_quarantined")
                        else "") + ".")
    for lens, info in (facts.get("training") or {}).items():
        metric = info.get("test_roc_auc") or info.get("c_index") or info.get("k")
        parts.append(f"Lens {lens}: {info['status']}"
                     + (f" ({metric})" if metric is not None else "") + ".")
    if facts.get("lenses_applied"):
        parts.append("Lenses applied at scoring: "
                     + ", ".join(facts["lenses_applied"]) + ".")
    if facts.get("high_or_critical") is not None:
        parts.append(f"{facts['high_or_critical']:,} customers scored High or "
                     f"Critical risk.")
    if facts.get("total_revenue_at_risk") is not None:
        parts.append(f"Total revenue at risk: "
                     f"${facts['total_revenue_at_risk']:,}.")
    if facts.get("drift"):
        parts.append(f"Drift check: {facts['drift']['recommendation']} "
                     f"(max PSI {facts['drift']['max_psi']}).")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# The number audit - no invented figures survive
# ---------------------------------------------------------------------------

def _allowed_numbers(facts) -> set[float]:
    allowed: set[float] = set()

    def walk(node):
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            value = float(node)
            allowed.add(round(value, 4))
            allowed.add(round(value, 2))
            allowed.add(round(value, 1))
            allowed.add(float(int(value)))
            if 0 <= value <= 1:                      # 0.8402 may appear as 84.02%
                allowed.update({round(value * 100, 2), round(value * 100, 1),
                                round(value * 100), float(int(value * 100))})
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple, set)):
            for v in node:
                walk(v)
    walk(facts)
    return allowed


def audit_numbers(text: str, facts: dict) -> list[str]:
    """Returns the list of numbers in `text` that do NOT come from the facts
    (empty list = the text is grounded)."""
    allowed = _allowed_numbers(facts)
    violations = []
    for token in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        digits = token.replace(",", "")
        if len(digits.replace(".", "")) < AUDIT_MIN_DIGITS:
            continue
        value = float(digits)
        if not any(abs(value - a) < 1e-6 for a in allowed):
            violations.append(token)
    return violations


# ---------------------------------------------------------------------------
# Narration - Ollama phrases, the audit judges, the template backstops
# ---------------------------------------------------------------------------

def narrate_run(manifest_data: dict, scored: pd.DataFrame | None = None,
                use_llm: bool = True) -> dict:
    """Returns {"summary", "engine", "facts", "audit_violations"}. Never raises:
    any LLM failure or audit violation lands on the template."""
    facts = facts_from_run(manifest_data, scored)
    fallback = template_summary(facts)

    if use_llm and llm.llm_available():
        try:
            text = _ollama_summary(facts)
            # LLMs love typographic dashes and quotes - project style is ASCII
            for bad, good in (("\u2014", " - "), ("\u2013", "-"),
                              ("\u2018", "'"), ("\u2019", "'"),
                              ("\u201c", '"'), ("\u201d", '"')):
                text = text.replace(bad, good)
            violations = audit_numbers(text, facts)
            if not violations:
                return {"summary": text, "engine": llm.llm_engine_name(),
                        "facts": facts, "audit_violations": []}
            return {"summary": fallback, "engine": "template (audit rejected LLM)",
                    "facts": facts, "audit_violations": violations}
        except Exception:
            pass
    return {"summary": fallback, "engine": "template", "facts": facts,
            "audit_violations": []}


def _ollama_summary(facts: dict) -> str:
    prompt = (
        "Facts about one completed data-pipeline run, as JSON:\n"
        + json.dumps(facts, indent=1, default=str) +
        "\n\nWrite a 3-5 sentence executive summary of this run for a business "
        "audience. Rules: use ONLY numbers that appear in the facts above - do "
        "not compute, convert, or invent any figure; state the mode decision "
        "plainly; if models were trained, say whether they passed their quality "
        "gates; plain professional prose, no bullet points, no headings."
    )
    return llm.llm_chat(prompt, timeout=240).strip()
