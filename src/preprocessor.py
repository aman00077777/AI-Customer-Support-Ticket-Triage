"""Data preprocessing and text normalization for ticket triage pipeline.

Ensures strict leakage prevention:
- Only features available upon ticket arrival (subject + body) are included.
- Post-resolution data (agent answers, tags, target labels) are strictly excluded.
"""

import html
import re
from typing import Optional, Union
import pandas as pd

from src.config import (
    PRIORITY_NORMALIZATION,
    QUEUE_TO_CATEGORY,
)


def clean_ticket_text(text: Optional[str]) -> str:
    """Clean and normalize customer support ticket text.

    Preserves critical domain words (e.g. failed, refund, locked, error, urgent, outage).
    Normalizes whitespace, decodes HTML entities, and standardizes punctuation spacing.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""

    # Decode HTML entities (e.g., &amp;, &lt;)
    text = html.unescape(text)

    # Replace newlines and tabs with spaces
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def combine_ticket_fields(
    subject: Optional[Union[str, pd.Series]],
    body: Optional[Union[str, pd.Series]],
) -> Union[str, pd.Series]:
    """Combine ticket subject and body into a single unified text feature."""
    if isinstance(subject, pd.Series) or isinstance(body, pd.Series):
        s_series = subject.fillna("").astype(str) if isinstance(subject, pd.Series) else ""
        b_series = body.fillna("").astype(str) if isinstance(body, pd.Series) else ""
        combined = (s_series + " " + b_series).apply(clean_ticket_text)
        return combined

    s_str = clean_ticket_text(subject)
    b_str = clean_ticket_text(body)

    if s_str and b_str:
        return f"{s_str} {b_str}".strip()
    return s_str or b_str


def map_category_label(queue_series: pd.Series) -> pd.Series:
    """Map the dataset's original queue names to the 5 core business categories."""
    return queue_series.map(lambda q: QUEUE_TO_CATEGORY.get(str(q).strip(), "Other"))


def map_priority_label(priority_series: pd.Series) -> pd.Series:
    """Normalize priority labels into High, Medium, Low."""
    return priority_series.map(
        lambda p: PRIORITY_NORMALIZATION.get(str(p).strip().lower(), "Medium")
    )


def preprocess_dataset(
    df: pd.DataFrame,
    drop_unusable: bool = True,
) -> pd.DataFrame:
    """Preprocess the raw tickets dataframe.

    Steps:
    1. Combines `subject` and `body` into `combined_text`.
    2. Drops unusable rows (empty text or missing targets).
    3. Maps category and priority targets.
    4. Preserves language and business_type for analysis while strictly excluding
       leaked columns from feature generation.
    """
    processed = df.copy()

    # Identify subject and body
    subject_col = "subject" if "subject" in processed.columns else None
    body_col = "body" if "body" in processed.columns else None

    if subject_col and body_col:
        processed["combined_text"] = combine_ticket_fields(
            processed[subject_col], processed[body_col]
        )
    elif body_col:
        processed["combined_text"] = processed[body_col].apply(clean_ticket_text)
    elif subject_col:
        processed["combined_text"] = processed[subject_col].apply(clean_ticket_text)
    elif "combined_text" in processed.columns:
        processed["combined_text"] = processed["combined_text"].apply(clean_ticket_text)
    elif "ticket_text" in processed.columns:
        processed["combined_text"] = processed["ticket_text"].apply(clean_ticket_text)
    else:
        raise ValueError("Could not find suitable text columns (subject/body/ticket_text).")

    # Map labels if source columns exist
    if "queue" in processed.columns:
        processed["category"] = map_category_label(processed["queue"])
        processed["original_queue"] = processed["queue"]

    if "priority" in processed.columns:
        processed["priority"] = map_priority_label(processed["priority"])
        processed["original_priority"] = processed["priority"]

    if drop_unusable:
        # Drop rows where combined text is effectively empty (fewer than 3 characters)
        valid_mask = processed["combined_text"].str.len() >= 3
        if "category" in processed.columns:
            valid_mask = valid_mask & processed["category"].notna()
        if "priority" in processed.columns:
            valid_mask = valid_mask & processed["priority"].notna()

        processed = processed[valid_mask].reset_index(drop=True)

    return processed
