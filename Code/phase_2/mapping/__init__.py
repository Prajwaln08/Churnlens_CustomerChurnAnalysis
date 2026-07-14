"""Semantic mapping stage."""
from . import mapping
from .mapping import (
    CANONICAL,
    MappingError,
    apply_mapping,
    fuzzy_propose,
    ollama_available,
    ollama_propose,
    propose_mapping,
)
