"""Unit tests for utility functions."""

from pathlib import Path
import pytest

from src.utils import (
    ensure_directories,
    format_percentage,
    generate_batch_template_df,
    load_json_file,
    save_json_file,
)


def test_format_percentage():
    """Test percentage string formatting."""
    assert format_percentage(0.6543) == "65.4%"
    assert format_percentage(0.1234, decimals=2) == "12.34%"
    assert format_percentage(1.0) == "100.0%"
    assert format_percentage(0.0) == "0.0%"
    assert format_percentage(None) == "N/A"


def test_save_and_load_json(tmp_path: Path):
    """Test JSON saving and loading helpers."""
    test_data = {"name": "TicketTriage", "accuracy": 0.85, "active": True}
    json_path = tmp_path / "subdir" / "test.json"

    save_json_file(test_data, json_path)
    assert json_path.exists()

    loaded = load_json_file(json_path)
    assert loaded == test_data

    with pytest.raises(FileNotFoundError):
        load_json_file(tmp_path / "nonexistent.json")


def test_generate_batch_template_df():
    """Test generating sample batch dataframe template."""
    df = generate_batch_template_df()
    assert not df.empty
    assert "ticket_id" in df.columns
    assert "ticket_text" in df.columns
    assert len(df) > 3


def test_ensure_directories():
    """Test directory creation without error."""
    ensure_directories()
