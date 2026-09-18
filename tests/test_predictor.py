"""Unit tests for TriagePredictor and human-in-the-loop triage engine."""

import pandas as pd
import pytest

from src.predictor import get_predictor
from src.utils import InvalidTicketDataError
from src.config import HUMAN_REVIEW_QUEUE, CATEGORY_TO_QUEUE


@pytest.fixture(scope="module")
def predictor():
    """Shared predictor instance using trained models."""
    return get_predictor()


def test_predict_ticket_keys(predictor):
    """Verify all required prediction fields are present."""
    text = "I need a refund for my billing statement from yesterday."
    res = predictor.predict_ticket(text, confidence_threshold=0.65)

    required_keys = [
        "ticket_text",
        "predicted_category",
        "category_confidence",
        "predicted_priority",
        "priority_confidence",
        "assigned_queue",
        "target_queue",
        "human_review_required",
        "routing_decision",
        "review_reason",
        "category_influential_terms",
        "priority_influential_terms",
    ]
    for k in required_keys:
        assert k in res, f"Missing expected key: {k}"

    assert 0.0 <= res["category_confidence"] <= 1.0
    assert 0.0 <= res["priority_confidence"] <= 1.0
    assert res["predicted_category"] in CATEGORY_TO_QUEUE


def test_confidence_threshold_logic(predictor):
    """Test human review triggered when threshold exceeds confidence."""
    text = "General question about service availability."

    # Very low threshold -> should auto route
    res_low = predictor.predict_ticket(text, confidence_threshold=0.01)
    assert not res_low["human_review_required"]
    assert res_low["routing_decision"] == "Automatically Routed"
    assert res_low["assigned_queue"] == res_low["target_queue"]
    assert res_low["review_reason"] is None

    # Very high threshold (99%) -> should require human review
    res_high = predictor.predict_ticket(text, confidence_threshold=0.99)
    assert res_high["human_review_required"]
    assert res_high["routing_decision"] == "Human Review Required"
    assert res_high["assigned_queue"] == HUMAN_REVIEW_QUEUE
    assert res_high["review_reason"] is not None


def test_invalid_ticket_error(predictor):
    """Test empty and short ticket inputs raise proper error."""
    with pytest.raises(InvalidTicketDataError):
        predictor.predict_ticket("")

    with pytest.raises(InvalidTicketDataError):
        predictor.predict_ticket("   \n ")

    with pytest.raises(InvalidTicketDataError):
        predictor.predict_ticket("ab")


def test_predict_batch_success(predictor):
    """Test batch prediction on DataFrame."""
    batch_df = pd.DataFrame({
        "ticket_id": ["T-1", "T-2", "T-3"],
        "ticket_text": [
            "Please cancel my annual recurring charge.",
            "Server keeps throwing out of memory errors.",
            "Inquiry regarding product specifications.",
        ],
    })

    results = predictor.predict_batch(batch_df, confidence_threshold=0.65)
    assert len(results) == 3
    assert "predicted_category" in results.columns
    assert "predicted_priority" in results.columns
    assert "assigned_queue" in results.columns
    assert "routing_decision" in results.columns


def test_predict_batch_empty_dataframe(predictor):
    """Test batch prediction rejects empty dataframe."""
    empty_df = pd.DataFrame()
    with pytest.raises(InvalidTicketDataError):
        predictor.predict_batch(empty_df)


def test_predict_batch_missing_text_column(predictor):
    """Test batch prediction rejects dataframe without text column."""
    invalid_df = pd.DataFrame({"age": [25, 30], "city": ["NY", "LA"]})
    with pytest.raises(InvalidTicketDataError):
        predictor.predict_batch(invalid_df)
