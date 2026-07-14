"""Narration stage (build-order phase 5): the LLM writes the words, never
the numbers - grounded generation with a number audit."""
from . import narrator
from .narrator import facts_from_run, narrate_run, template_summary
