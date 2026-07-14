"""Run orchestration stage: manifest state machine + SQLite registry +
the orchestrator that drives a dataset through the staged pipeline."""
from . import orchestrator, registry, run_manifest
from .orchestrator import run_pipeline
from .registry import Registry
from .run_manifest import STAGES, RunManifest, new_run_id
