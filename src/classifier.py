"""Machine learning model classifier wrapper for ticket triage.

Combines TF-IDF vectorization and Logistic Regression with explainability
capabilities (extracting influential feature tokens for predictions).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import (
    RANDOM_STATE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)
from src.utils import ModelArtifactNotFoundError


class TriageClassifier:
    """Wrapper around TF-IDF Vectorizer and Logistic Regression Classifier."""

    def __init__(
        self,
        name: str = "TriageClassifier",
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
        C: float = 1.5,
        random_state: int = RANDOM_STATE,
    ):
        self.name = name
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            strip_accents=None,  # Preserve multi-language accents
            lowercase=True,
        )
        self.model = LogisticRegression(
            C=self.C,
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.is_fitted = False
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: Union[List[str], pd.Series], y: Union[List[str], pd.Series]) -> "TriageClassifier":
        """Fit vectorizer and logistic regression model."""
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self

    def predict(self, X: Union[List[str], pd.Series]) -> np.ndarray:
        """Predict class labels for given texts."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)

    def predict_proba(self, X: Union[List[str], pd.Series]) -> np.ndarray:
        """Predict class probabilities for given texts."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
        X_vec = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vec)

    def predict_single_with_confidence(self, text: str) -> Tuple[str, float]:
        """Predict class and return (predicted_class, confidence_score)."""
        probs = self.predict_proba([text])[0]
        pred_idx = int(np.argmax(probs))
        predicted_class = str(self.classes_[pred_idx])
        confidence = float(probs[pred_idx])
        return predicted_class, confidence

    def explain_prediction(
        self,
        text: str,
        target_class: Optional[str] = None,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Identify most influential TF-IDF terms in the text for the predicted class.

        Computes the linear contribution: weight_{class, feature} * tfidf_{feature}
        for all active tokens in the given text.
        """
        if not self.is_fitted or self.classes_ is None:
            return []

        # Default to highest probability class if none specified
        if target_class is None:
            target_class, _ = self.predict_single_with_confidence(text)

        if target_class not in self.classes_:
            return []

        class_idx = list(self.classes_).index(target_class)
        x_vec = self.vectorizer.transform([text])

        # Get non-zero feature indices for this text
        feature_indices = x_vec.nonzero()[1]
        if len(feature_indices) == 0:
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        weights = self.model.coef_[class_idx]

        contributions = []
        for idx in feature_indices:
            tfidf_val = x_vec[0, idx]
            weight = weights[idx]
            contribution = float(weight * tfidf_val)
            contributions.append({
                "feature": feature_names[idx],
                "tfidf": round(float(tfidf_val), 4),
                "weight": round(float(weight), 4),
                "contribution": round(contribution, 4),
            })

        # Sort by contribution descending (positive influence towards target_class)
        contributions.sort(key=lambda item: item["contribution"], reverse=True)
        return contributions[:top_n]

    def save(self, model_path: Path, vectorizer_path: Path) -> None:
        """Save fitted model and vectorizer to disk."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

    def load(self, model_path: Path, vectorizer_path: Path) -> "TriageClassifier":
        """Load fitted model and vectorizer from disk."""
        if not model_path.exists() or not vectorizer_path.exists():
            raise ModelArtifactNotFoundError(
                f"Model or vectorizer artifact missing:\n"
                f"  Model: {model_path} (exists: {model_path.exists()})\n"
                f"  Vectorizer: {vectorizer_path} (exists: {vectorizer_path.exists()})"
            )
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self
