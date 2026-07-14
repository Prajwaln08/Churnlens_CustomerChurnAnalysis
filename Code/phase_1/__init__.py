"""Phase 1 - the local-dataset analytics suite (the five lenses).

Functional layout - one folder per concern, each re-exporting its public API:

    config.py       paths, seeds and settings read from the project .env
    loader/         dataset loaders (Telco, Online Retail II, MegaFon)
    preprocessing/  feature engineering + leakage-safe splitting
    visualization/  palette-safe EDA plotting helpers
    modeling/       lens trainers, artifact persistence, deploy bootstrap
    evaluation/     metrics, lift/gains, qini, profiles, revenue-at-risk
    serving/        schema contracts, validation, scoring, CLI

Flat aliases below keep historical imports working
(`from phase_1 import eda, features, models, pipeline, ...`).
"""
from . import config
from .loader import data_loader
from .preprocessing import features
from .visualization import eda
from .modeling import bootstrap, models
from .evaluation import evaluation
from .serving import pipeline

__all__ = ["config", "data_loader", "features", "eda", "models", "bootstrap",
           "evaluation", "pipeline"]
