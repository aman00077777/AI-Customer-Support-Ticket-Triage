"""Inference and triage engine for customer support tickets.

Provides single-ticket and batch prediction, confidence calculation,
queue routing, human-in-the-loop triage decisioning, and model explainability.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from src.classifier import TriageClassifier
from src.config import (
    CATEGORY_MODEL_PATH,
    CATEGORY_TO_QUEUE,
    CATEGORY_VECTORIZER_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    HUMAN_REVIEW_QUEUE,
    PRIORITY_MODEL_PATH,
    PRIORITY_VECTORIZER_PATH,
)
from src.preprocessor import clean_ticket_text
from src.utils import InvalidTicketDataError, ModelArtifactNotFoundError


class TriagePredictor:
    """Production triage engine executing predictions and human-review logic."""

    def __init__(
        self,
        category_clf: Optional[TriageClassifier] = None,
        priority_clf: Optional[TriageClassifier] = None,
    ):
        if category_clf is not None and priority_clf is not None:
            self.category_clf = category_clf
            self.priority_clf = priority_clf
        else:
            self.category_clf = TriageClassifier(name="CategoryClassifier")
            self.priority_clf = TriageClassifier(name="PriorityClassifier")
            self.load_models()

    def load_models(self) -> None:
        """Load trained category and priority model artifacts."""
        self.category_clf.load(CATEGORY_MODEL_PATH, CATEGORY_VECTORIZER_PATH)
        self.priority_clf.load(PRIORITY_MODEL_PATH, PRIORITY_VECTORIZER_PATH)

    def route_queue(
        self,
        predicted_category: str,
        human_review_required: bool,
    ) -> Tuple[str, str]:
        """Assign support queue based on predicted category and review status."""
        nominal_queue = CATEGORY_TO_QUEUE.get(
            predicted_category, "General Operations Queue"
        )
        assigned_queue = HUMAN_REVIEW_QUEUE if human_review_required else nominal_queue
        return assigned_queue, nominal_queue

    def predict_ticket(
        self,
        ticket_text: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        explain: bool = True,
    ) -> Dict[str, Any]:
        """Triage a single customer support ticket.

        Returns:
            Dict containing predicted category, priority, model confidences,
            assigned queue, routing decision, human review reason, and influential terms.
        """
        cleaned_text = clean_ticket_text(ticket_text)
        if not cleaned_text or len(cleaned_text) < 3:
            raise InvalidTicketDataError(
                "Ticket text is empty or too short for classification."
            )

        # Predict Category
        pred_cat, cat_conf = self.category_clf.predict_single_with_confidence(cleaned_text)

        # Predict Priority
        pred_pri, pri_conf = self.priority_clf.predict_single_with_confidence(cleaned_text)

        # Confidence checks against threshold
        cat_low = cat_conf < confidence_threshold
        pri_low = pri_conf < confidence_threshold
        human_review = cat_low or pri_low

        # Build human review reason
        if cat_low and pri_low:
            review_reason = (
                f"Both category confidence ({cat_conf * 100:.1f}%) and priority confidence "
                f"({pri_conf * 100:.1f}%) fall below the threshold ({confidence_threshold * 100:.1f}%)."
            )
        elif cat_low:
            review_reason = (
                f"Low category confidence: {cat_conf * 100:.1f}% is below threshold "
                f"({confidence_threshold * 100:.1f}%)."
            )
        elif pri_low:
            review_reason = (
                f"Low priority confidence: {pri_conf * 100:.1f}% is below threshold "
                f"({confidence_threshold * 100:.1f}%)."
            )
        else:
            review_reason = None

        routing_decision = "Human Review Required" if human_review else "Automatically Routed"
        assigned_queue, nominal_queue = self.route_queue(pred_cat, human_review)

        # Model explainability features
        cat_terms = []
        pri_terms = []
        if explain:
            cat_terms = self.category_clf.explain_prediction(cleaned_text, target_class=pred_cat, top_n=5)
            pri_terms = self.priority_clf.explain_prediction(cleaned_text, target_class=pred_pri, top_n=5)

        return {
            "ticket_text": ticket_text,
            "cleaned_text": cleaned_text,
            "predicted_category": pred_cat,
            "category_confidence": round(cat_conf, 4),
            "predicted_priority": pred_pri,
            "priority_confidence": round(pri_conf, 4),
            "assigned_queue": assigned_queue,
            "target_queue": nominal_queue,
            "human_review_required": human_review,
            "routing_decision": routing_decision,
            "review_reason": review_reason,
            "confidence_threshold": confidence_threshold,
            "category_influential_terms": cat_terms,
            "priority_influential_terms": pri_terms,
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> pd.DataFrame:
        """Process a dataframe of tickets and return enriched results."""
        if df.empty:
            raise InvalidTicketDataError("Uploaded CSV is empty.")

        # Identify ticket text column
        text_col = None
        candidates = [
            "ticket_text",
            "combined_text",
            "body",
            "subject",
            "text",
            "description",
            "message",
            "issue",
            "content",
        ]
        for col in candidates:
            if col in df.columns:
                text_col = col
                break

        # If both subject and body exist, combine them
        if "subject" in df.columns and "body" in df.columns:
            series_to_use = (
                df["subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
            )
        elif text_col is not None:
            series_to_use = df[text_col].fillna("").astype(str)
        else:
            raise InvalidTicketDataError(
                "No suitable ticket text column found in CSV. Expected one of: "
                f"{', '.join(candidates)}"
            )

        results = []
        for idx, text in enumerate(series_to_use):
            ticket_id = df["ticket_id"].iloc[idx] if "ticket_id" in df.columns else f"TCK-{idx+1:04d}"
            try:
                pred = self.predict_ticket(text, confidence_threshold=confidence_threshold, explain=False)
                results.append({
                    "ticket_id": ticket_id,
                    "ticket_text": text[:150] + ("..." if len(text) > 150 else ""),
                    "predicted_category": pred["predicted_category"],
                    "category_confidence": f"{pred['category_confidence'] * 100:.1f}%",
                    "predicted_priority": pred["predicted_priority"],
                    "priority_confidence": f"{pred['priority_confidence'] * 100:.1f}%",
                    "assigned_queue": pred["assigned_queue"],
                    "target_queue": pred["target_queue"],
                    "human_review_required": pred["human_review_required"],
                    "routing_decision": pred["routing_decision"],
                    "review_reason": pred["review_reason"] or "Meets confidence criteria",
                })
            except InvalidTicketDataError:
                results.append({
                    "ticket_id": ticket_id,
                    "ticket_text": str(text)[:150],
                    "predicted_category": "Invalid",
                    "category_confidence": "0.0%",
                    "predicted_priority": "Invalid",
                    "priority_confidence": "0.0%",
                    "assigned_queue": HUMAN_REVIEW_QUEUE,
                    "target_queue": "N/A",
                    "human_review_required": True,
                    "routing_decision": "Human Review Required",
                    "review_reason": "Empty or invalid ticket text",
                })

        return pd.DataFrame(results)


_GLOBAL_PREDICTOR: Optional[TriagePredictor] = None


def get_predictor() -> TriagePredictor:
    """Obtain or initialize the global cached predictor instance."""
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = TriagePredictor()
    return _GLOBAL_PREDICTOR
