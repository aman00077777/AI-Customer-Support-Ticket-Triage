"""Utility helper functions for AI Customer Support Ticket Triage."""

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SAMPLE_TICKETS,
)


class TicketTriageError(Exception):
    """Base exception for ticket triage system."""
    pass


class ModelArtifactNotFoundError(TicketTriageError):
    """Raised when required model or vectorizer artifact files are missing."""
    pass


class InvalidTicketDataError(TicketTriageError):
    """Raised when input ticket data is invalid or empty."""
    pass


def ensure_directories() -> None:
    """Ensure all required project directories exist."""
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def format_percentage(val: float, decimals: int = 1) -> str:
    """Format float (0.0 to 1.0) as percentage string."""
    if val is None:
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def save_json_file(data: Dict[str, Any], file_path: Path) -> None:
    """Save dictionary to a formatted JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON file safely."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_batch_template_df() -> pd.DataFrame:
    """Generate a sample dataframe for batch processing demonstration."""
    sample_records = [
        {
            "ticket_id": f"TCK-100{i+1}",
            "ticket_text": text,
            "expected_domain": domain,
        }
        for i, (domain, text) in enumerate(SAMPLE_TICKETS.items())
    ]
    return pd.DataFrame(sample_records)
