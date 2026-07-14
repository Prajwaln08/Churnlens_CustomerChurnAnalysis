"""Training stage (build-order phase 3): mode decision + dataset-adaptive
training with leakage screening and quality gates."""
from . import auto_train, leakage, mode
from .auto_train import score_with_dataset_model, train_for_dataset
from .leakage import derive_datetime_features, screen
from .mode import EXPLORE, SCORE, TRAIN, decide, dataset_id_for
