"""Dataset generation script.

Loads the raw dataset, preprocesses and cleans tickets, applies label mappings,
and saves the clean dataset to data/processed/tickets_processed.csv.
"""

from pathlib import Path
import pandas as pd

from src.config import PROCESSED_DATA_DIR, PROCESSED_DATA_FILE
from src.data_loader import download_or_locate_dataset
from src.preprocessor import preprocess_dataset
from src.utils import ensure_directories


def generate_processed_dataset() -> Path:
    """Execute end-to-end dataset preparation and write processed CSV."""
    ensure_directories()

    print("Step 1: Locating/Downloading raw dataset...")
    raw_df, raw_path = download_or_locate_dataset()
    print(f"Loaded raw dataset from: {raw_path} (Shape: {raw_df.shape})")

    print("\nStep 2: Preprocessing and mapping labels...")
    processed_df = preprocess_dataset(raw_df, drop_unusable=True)
    print(f"Processed dataset shape: {processed_df.shape}")

    # Select columns for clean storage
    columns_to_keep = [
        col
        for col in [
            "combined_text",
            "category",
            "priority",
            "language",
            "original_queue",
            "original_priority",
            "type",
            "business_type",
        ]
        if col in processed_df.columns
    ]

    clean_df = processed_df[columns_to_keep]

    print(f"\nStep 3: Saving processed data to {PROCESSED_DATA_FILE}...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_DATA_FILE, index=False, encoding="utf-8")
    print(f"Successfully saved {len(clean_df):,} records to {PROCESSED_DATA_FILE}!")

    print("\n--- Summary of Processed Dataset ---")
    print("Categories:\n", clean_df["category"].value_counts())
    print("\nPriorities:\n", clean_df["priority"].value_counts())
    print("\nLanguages:\n", clean_df["language"].value_counts())

    return PROCESSED_DATA_FILE


if __name__ == "__main__":
    generate_processed_dataset()
