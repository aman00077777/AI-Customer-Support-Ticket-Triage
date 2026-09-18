"""Unit tests for dataset loading and inspection functions."""

import pandas as pd
import pytest
from pathlib import Path

from src.data_loader import (
    find_csv_in_directory,
    load_csv_with_fallback,
    inspect_dataset,
)
from src.config import RAW_DATA_DIR, PROCESSED_DATA_FILE


def test_find_csv_in_directory(tmp_path: Path):
    """Test locating CSV file in a given directory."""
    test_csv = tmp_path / "sample.csv"
    test_csv.write_text("a,b\n1,2", encoding="utf-8")

    found = find_csv_in_directory(tmp_path)
    assert found == test_csv


def test_find_csv_not_found(tmp_path: Path):
    """Test error raised when no CSV file exists."""
    with pytest.raises(FileNotFoundError):
        find_csv_in_directory(tmp_path)


def test_load_csv_with_fallback(tmp_path: Path):
    """Test CSV loading with non-UTF8 characters."""
    latin_csv = tmp_path / "latin.csv"
    # Write Latin-1 content
    content = "subject,body\nPréstamo,Atención urgente\n"
    latin_csv.write_bytes(content.encode("latin1"))

    df = load_csv_with_fallback(latin_csv)
    assert len(df) == 1
    assert "Préstamo" in df["subject"].iloc[0]


def test_inspect_dataset():
    """Test dataset inspection metric extraction."""
    df = pd.DataFrame({
        "subject": ["Billing query", "Server crash", None],
        "body": ["I need a refund", "AWS outage", "Help"],
        "queue": ["Billing and Payments", "Technical Support", "Customer Service"],
        "priority": ["high", "medium", "low"],
        "language": ["en", "en", "es"],
    })

    report = inspect_dataset(df)
    assert report["total_rows"] == 3
    assert report["total_columns"] == 5
    assert "queue" in report["label_columns"]
    assert "priority" in report["label_columns"]
    assert report["missing_values"]["subject"] == 1
