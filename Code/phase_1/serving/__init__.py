"""The scoring tool: contracts, validation, lens detection, CLI.

pipeline is exposed as a module (call sites use it as a namespace:
pipeline.score, pipeline.validate, ...). CLI entry:
    python -m phase_1.serving.pipeline score --input file.csv
"""
from . import pipeline
