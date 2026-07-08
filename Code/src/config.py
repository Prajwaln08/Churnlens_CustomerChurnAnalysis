"""Project configuration.

Origin: Code/01_data_loading.ipynb.

Single source of truth for filesystem paths and the random seed. Values come
from the project-root ``.env`` (never committed); every entry has a default so
the code also runs on a fresh clone before ``.env`` exists.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Code/src/config.py -> parents[0]=src, [1]=Code, [2]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _path(env_key: str, default: str) -> Path:
    return PROJECT_ROOT / os.getenv(env_key, default)


TELCO_ZIP = _path("TELCO_ZIP", "Dataset/IBM_telco.zip")
TELCO_CSV = os.getenv("TELCO_CSV", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

RETAIL_ZIP = _path("RETAIL_ZIP", "Dataset/online_retail_2.zip")
RETAIL_XLSX = os.getenv("RETAIL_XLSX", "online_retail_II.xlsx")

MEGAFON_CACHE = _path("MEGAFON_CACHE", "Dataset/megafon/megafon.parquet")

PROCESSED_DIR = _path("PROCESSED_DATA_DIR", "Dataset/processed")
MODEL_DIR = _path("MODEL_DIR", "models")
CONTRACTS_DIR = MODEL_DIR / "contracts"
METRICS_FILE = MODEL_DIR / "metrics.json"

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
