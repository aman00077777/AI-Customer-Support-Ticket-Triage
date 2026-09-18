"""Unit tests for ticket preprocessing, text cleaning, and leakage prevention."""

import pandas as pd
import pytest

from src.preprocessor import (
    clean_ticket_text,
    combine_ticket_fields,
    map_category_label,
    map_priority_label,
    preprocess_dataset,
)


def test_clean_ticket_text():
    """Test text cleaning and whitespace normalization."""
    raw = "  Hello &amp; welcome!\n\nI need an  urgent   refund.  "
    cleaned = clean_ticket_text(raw)
    assert cleaned == "Hello & welcome! I need an urgent refund."
    assert "refund" in cleaned
    assert "urgent" in cleaned


def test_clean_ticket_text_none_or_empty():
    """Test cleaning on empty or None input."""
    assert clean_ticket_text(None) == ""
    assert clean_ticket_text("") == ""
    assert clean_ticket_text("   \n\t  ") == ""


def test_combine_ticket_fields_string():
    """Test combining subject and body as strings."""
    subj = "Payment failed"
    body = "My card was declined multiple times."
    combined = combine_ticket_fields(subj, body)
    assert combined == "Payment failed My card was declined multiple times."

    # Subject missing
    assert combine_ticket_fields(None, body) == body
    # Body missing
    assert combine_ticket_fields(subj, None) == subj


def test_combine_ticket_fields_series():
    """Test combining subject and body pandas series."""
    subj = pd.Series(["Subj 1", None])
    body = pd.Series(["Body 1", "Body 2"])
    combined = combine_ticket_fields(subj, body)
    assert combined.iloc[0] == "Subj 1 Body 1"
    assert combined.iloc[1] == "Body 2"


def test_label_mappings():
    """Test mapping raw queue and priority values."""
    queues = pd.Series([
        "Technical Support",
        "IT Support",
        "Billing and Payments",
        "Product Support",
        "Customer Service",
        "Unknown Queue",
    ])
    mapped_cats = map_category_label(queues)
    assert mapped_cats.iloc[0] == "Technical"
    assert mapped_cats.iloc[1] == "Technical"
    assert mapped_cats.iloc[2] == "Billing"
    assert mapped_cats.iloc[3] == "Product"
    assert mapped_cats.iloc[4] == "Customer Service"
    assert mapped_cats.iloc[5] == "Other"

    priorities = pd.Series(["high", "medium", "low", "critical", "unknown"])
    mapped_pri = map_priority_label(priorities)
    assert mapped_pri.iloc[0] == "High"
    assert mapped_pri.iloc[1] == "Medium"
    assert mapped_pri.iloc[2] == "Low"
    assert mapped_pri.iloc[3] == "High"
    assert mapped_pri.iloc[4] == "Medium"


def test_preprocess_dataset_leakage_prevention():
    """Test that preprocessing does NOT include target labels in text features."""
    df = pd.DataFrame({
        "subject": ["Server down"],
        "body": ["AWS instance unreachable"],
        "answer": ["Resolved by restarting instance."],  # Leakage candidate
        "queue": ["Technical Support"],                 # Label
        "priority": ["high"],                           # Label
        "tag_1": ["Urgent Outage"],                     # Leakage candidate
    })

    processed = preprocess_dataset(df, drop_unusable=True)
    assert "combined_text" in processed.columns
    combined_val = processed["combined_text"].iloc[0]

    # Combined text MUST ONLY contain subject and body
    assert "Server down AWS instance unreachable" in combined_val
    assert "Resolved by restarting" not in combined_val
    assert "Urgent Outage" not in combined_val
    assert "Technical Support" not in combined_val
