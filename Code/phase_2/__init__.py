"""Phase 2 - the production pipeline: any dataset, staged runs, visibility.

Functional layout - one folder per pipeline stage, mirroring architecture.md:

    profiling/      deterministic column profiling + role inference
    mapping/        canonical schema, Ollama proposer, fuzzy fallback, executor
    orchestration/  run manifest state machine (+ registry, orchestrator next)

Flat aliases below keep call sites short
(`from phase_2 import profiler, mapping, run_manifest`).
"""
from .profiling import profiler
from .mapping import mapping
from .orchestration import orchestrator, registry, run_manifest
from .training import auto_train, leakage, mode

__all__ = ["profiler", "mapping", "orchestrator", "registry", "run_manifest",
           "auto_train", "leakage", "mode"]
