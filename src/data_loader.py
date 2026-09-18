"""Dataset loader and inspector for multilingual customer support tickets.

Robustly locates, downloads, loads, and inspects the real Kaggle dataset:
'tobiasbueck/multilingual-customer-support-tickets' (version 8).
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.config import (
    KAGGLE_DATASET_HANDLE,
    RAW_DATA_DIR,
)
from src.utils import ensure_directories


def find_csv_in_directory(dir_path: Path) -> Path:
    """Recursively find the first CSV dataset file in a directory."""
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {dir_path}")

    csv_files = list(dir_path.glob("*.csv"))
    if not csv_files:
        csv_files = list(dir_path.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {dir_path}")

    return csv_files[0]


def load_csv_with_fallback(file_path: Path) -> pd.DataFrame:
    """Load CSV trying standard UTF-8 and fallback encodings (latin1 / ISO-8859-1)."""
    encodings = ["utf-8", "latin1", "utf-8-sig", "cp1252"]
    last_error: Optional[Exception] = None

    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            break

    raise RuntimeError(
        f"Failed to read CSV at {file_path} with supported encodings. "
        f"Last error: {last_error}"
    )


def download_or_locate_dataset() -> Tuple[pd.DataFrame, Path]:
    """Download via KaggleHub or locate existing cached/local raw dataset file.

    Returns:
        Tuple of (DataFrame, Path to CSV file).
    """
    ensure_directories()

    # Check if a local copy already exists in data/raw
    local_raw_csvs = list(RAW_DATA_DIR.glob("*.csv"))
    if local_raw_csvs:
        csv_path = local_raw_csvs[0]
        df = load_csv_with_fallback(csv_path)
        return df, csv_path

    # Otherwise download via kagglehub
    try:
        import kagglehub

        downloaded_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE))
        source_csv = find_csv_in_directory(downloaded_dir)

        # Cache a local copy in data/raw for local offline development
        dest_csv = RAW_DATA_DIR / source_csv.name
        shutil.copy2(source_csv, dest_csv)

        df = load_csv_with_fallback(dest_csv)
        return df, dest_csv

    except Exception as exc:
        raise RuntimeError(
            f"Failed to download or locate dataset '{KAGGLE_DATASET_HANDLE}': {exc}"
        ) from exc


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive data inspection on raw tickets dataframe."""
    inspection: Dict[str, Any] = {
        "shape": df.shape,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_texts": int(
            df.duplicated(subset=["subject", "body"]).sum()
            if "subject" in df.columns and "body" in df.columns
            else 0
        ),
        "categorical_distributions": {},
        "text_columns": [
            col
            for col in df.columns
            if df[col].dtype == "object"
            and df[col].dropna().apply(lambda x: len(str(x))).mean() > 20
        ],
        "label_columns": [
            col
            for col in ["queue", "priority", "type", "category"]
            if col in df.columns
        ],
    }

    # Inspect categorical distributions
    for col in df.columns:
        if df[col].nunique() <= 15:
            inspection["categorical_distributions"][col] = (
                df[col].value_counts(dropna=False).to_dict()
            )

    return inspection


def print_inspection_report(df: pd.DataFrame, file_path: Path) -> None:
    """Print readable terminal report of dataset inspection."""
    info = inspect_dataset(df)

    print("=" * 70)
    print("DATASET INSPECTION REPORT")
    print("=" * 70)
    print(f"Source file:       {file_path}")
    print(f"Total Rows:        {info['total_rows']:,}")
    print(f"Total Columns:     {info['total_columns']}")
    print(f"Exact Duplicates:  {info['duplicate_rows']}")
    print(f"Duplicate Texts:   {info['duplicate_texts']}")
    print("-" * 70)
    print("COLUMNS & MISSING VALUES:")
    for col, null_count in info["missing_values"].items():
        dtype = info["dtypes"][col]
        pct = (null_count / info["total_rows"]) * 100
        print(f"  - {col:18} | Dtype: {dtype:8} | Nulls: {null_count:5} ({pct:.1f}%)")

    print("-" * 70)
    print("CATEGORICAL DISTRIBUTIONS:")
    for col, dist in info["categorical_distributions"].items():
        print(f"\n  [{col.upper()}]:")
        for val, count in dist.items():
            pct = (count / info["total_rows"]) * 100
            print(f"    * {str(val):32}: {count:5} ({pct:.1f}%)")

    print("-" * 70)
    print(f"IDENTIFIED TEXT COLUMNS:  {info['text_columns']}")
    print(f"IDENTIFIED LABEL COLUMNS: {info['label_columns']}")
    print("=" * 70)


if __name__ == "__main__":
    df, csv_path = download_or_locate_dataset()
    print_inspection_report(df, csv_path)
