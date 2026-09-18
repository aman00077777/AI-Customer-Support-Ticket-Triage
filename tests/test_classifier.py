"""Unit tests for TriageClassifier model wrapper."""

import numpy as np
import pytest
from pathlib import Path

from src.classifier import TriageClassifier


@pytest.fixture
def dummy_data():
    X = [
        "Refund my payment immediately please",
        "Invoice charged twice for annual subscription",
        "The server crashed with fatal exception",
        "Application throws 500 error when uploading",
        "Password reset email not received for account",
        "Login is locked out due to failed attempts",
    ]
    y = [
        "Billing",
        "Billing",
        "Technical",
        "Technical",
        "Customer Service",
        "Customer Service",
    ]
    return X, y


def test_classifier_fit_predict(dummy_data):
    """Test model training and prediction on small sample."""
    X, y = dummy_data
    clf = TriageClassifier(max_features=50, random_state=42)

    with pytest.raises(RuntimeError):
        clf.predict(["Unfitted test"])

    clf.fit(X, y)
    assert clf.is_fitted
    assert len(clf.classes_) == 3

    preds = clf.predict(X)
    assert len(preds) == len(X)

    probs = clf.predict_proba(X)
    assert probs.shape == (len(X), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_predict_single_with_confidence(dummy_data):
    """Test single prediction and probability calculation."""
    X, y = dummy_data
    clf = TriageClassifier(max_features=50, random_state=42)
    clf.fit(X, y)

    label, conf = clf.predict_single_with_confidence("Please process my refund for the subscription")
    assert label in clf.classes_
    assert 0.0 <= conf <= 1.0


def test_explain_prediction(dummy_data):
    """Test influential term extraction for explainability."""
    X, y = dummy_data
    clf = TriageClassifier(max_features=50, random_state=42)
    clf.fit(X, y)

    explanations = clf.explain_prediction(
        "Refund my payment immediately please", target_class="Billing", top_n=3
    )
    assert isinstance(explanations, list)
    if explanations:
        assert "feature" in explanations[0]
        assert "contribution" in explanations[0]


def test_save_and_load(dummy_data, tmp_path: Path):
    """Test saving model artifacts and loading them back."""
    X, y = dummy_data
    clf = TriageClassifier(max_features=50, random_state=42)
    clf.fit(X, y)

    m_path = tmp_path / "test_model.joblib"
    v_path = tmp_path / "test_vec.joblib"
    clf.save(m_path, v_path)

    new_clf = TriageClassifier()
    new_clf.load(m_path, v_path)
    assert new_clf.is_fitted

    pred1 = clf.predict(["Server crash"])
    pred2 = new_clf.predict(["Server crash"])
    assert pred1[0] == pred2[0]
