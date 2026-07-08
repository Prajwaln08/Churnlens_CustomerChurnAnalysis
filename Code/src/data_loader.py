"""Loaders for the three project datasets.

Origin: Code/01_data_loading.ipynb.

Each loader returns an analysis-ready DataFrame. The raw-data landmines and
the reasoning behind every fix are documented inline, so the cleaning
rationale lives next to the code that applies it.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

from . import config

TELCO_SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

RETAIL_CACHE = config.PROCESSED_DIR / "retail_raw.parquet"


def peek(df: pd.DataFrame, name: str = "dataframe", n: int = 5) -> pd.DataFrame:
    """Standardized first look: prints shape and a dtype/null summary,
    returns ``df.head(n)`` so notebooks render it as a rich table."""
    print(f"=== {name} ===")
    print(f"shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isna().sum(),
        "null_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
    })
    print(summary.to_string())
    return df.head(n)


# --------------------------------------------------------------------------
# 1) IBM Telco
# --------------------------------------------------------------------------

def load_telco(clean: bool = True) -> pd.DataFrame:
    """IBM Telco Customer Churn (7,043 x 21) from Dataset/IBM_telco.zip."""
    with zipfile.ZipFile(config.TELCO_ZIP) as zf:
        with zf.open(config.TELCO_CSV) as f:
            df = pd.read_csv(f)
    return clean_telco(df) if clean else df


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges arrives as a string: 11 rows are blank, all with tenure=0.
    # Those are brand-new customers who have not been billed yet, so 0 is the
    # faithful value (dropping them would bias the model against newcomers).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    # SeniorCitizen is 0/1 while every other flag is Yes/No - make it consistent.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    # "No internet service"/"No phone service" duplicate what InternetService/
    # PhoneService already say; collapsing them removes 3-level noise.
    df[TELCO_SERVICE_COLS] = df[TELCO_SERVICE_COLS].replace(
        {"No internet service": "No", "No phone service": "No"}
    )
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


# --------------------------------------------------------------------------
# 2) Online Retail II
# --------------------------------------------------------------------------

def load_retail(
    clean: bool = True,
    use_cache: bool = True,
    nrows: int | None = None,
) -> pd.DataFrame:
    """UCI Online Retail II (~1.07M rows, two xlsx sheets, 2009-2011).

    The xlsx read takes minutes, so the combined raw frame is cached to
    parquet on first load. ``nrows`` (per sheet) bypasses the cache - it is
    for quick tests only.
    """
    if nrows is None and use_cache and RETAIL_CACHE.exists():
        raw = pd.read_parquet(RETAIL_CACHE)
    else:
        with zipfile.ZipFile(config.RETAIL_ZIP) as zf:
            data = io.BytesIO(zf.read(config.RETAIL_XLSX))
        sheets = pd.read_excel(data, sheet_name=None, nrows=nrows)
        raw = pd.concat(sheets.values(), ignore_index=True)
        # The two sheets overlap around Dec 2010 - exact duplicates are re-listed
        # rows, not real repeated purchases.
        raw = raw.drop_duplicates(ignore_index=True)
        raw = raw.rename(columns={"Customer ID": "CustomerID"})
        raw["Invoice"] = raw["Invoice"].astype(str)
        # Excel yields mixed types inside text columns (StockCode '79323P' vs
        # 85123 as int; Description likewise) - parquet cannot serialize a
        # mixed column, so normalize every object column to NaN-safe 'string'.
        for col in raw.columns[raw.dtypes == object]:
            raw[col] = raw[col].astype("string")
        # Credit notes (cancellations/returns) have invoice numbers starting "C".
        raw["is_cancellation"] = raw["Invoice"].str.startswith("C")
        if nrows is None:
            config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            raw.to_parquet(RETAIL_CACHE)
    return clean_retail(raw) if clean else raw


def clean_retail(raw: pd.DataFrame) -> pd.DataFrame:
    """Purchases-only view for customer-level work (RFM, CLV).

    Drops: cancellations, rows without a CustomerID (~25%, cannot be used at
    customer level), and non-positive quantities/prices (returns, bad rows,
    zero-price adjustments). Adds ``Revenue``.
    """
    df = raw.loc[
        ~raw["is_cancellation"]
        & raw["CustomerID"].notna()
        & (raw["Quantity"] > 0)
        & (raw["Price"] > 0)
    ].copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["Price"]
    return df


# --------------------------------------------------------------------------
# 3) MegaFon Uplift
# --------------------------------------------------------------------------

def load_megafon(cache: bool = True) -> pd.DataFrame:
    """MegaFon uplift dataset (~600K x 50 features + treatment_group + conversion).

    First call downloads ~100MB via scikit-uplift and caches to parquet;
    every later call reads the cache.
    """
    if cache and config.MEGAFON_CACHE.exists():
        return pd.read_parquet(config.MEGAFON_CACHE)
    from sklift.datasets import fetch_megafon  # deferred: needs network once

    bunch = fetch_megafon()
    df = bunch.data.copy()
    df["treatment_group"] = pd.Series(bunch.treatment).values
    df["conversion"] = pd.Series(bunch.target).astype(int).values
    if cache:
        config.MEGAFON_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.MEGAFON_CACHE)
    return df


# --------------------------------------------------------------------------
# Processed-data I/O
# --------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, name: str) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PROCESSED_DIR / f"{name}.parquet")


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED_DIR / f"{name}.parquet")
