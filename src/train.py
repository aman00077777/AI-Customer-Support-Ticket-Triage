"""Training pipeline for AI Customer Support Ticket Triage.

Loads preprocessed dataset, splits into stratified train/test sets,
trains separate Category and Priority classifiers, evaluates them,
and persists artifacts to the models/ directory.
"""

from typing import Any, Dict
import pandas as pd
from sklearn.model_selection import train_test_split

from src.classifier import TriageClassifier
from src.config import (
    CATEGORY_CLASSES,
    CATEGORY_MODEL_PATH,
    CATEGORY_VECTORIZER_PATH,
    EVALUATION_RESULTS_PATH,
    MODELS_DIR,
    PRIORITY_CLASSES,
    PRIORITY_MODEL_PATH,
    PRIORITY_VECTORIZER_PATH,
    PROCESSED_DATA_FILE,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.evaluator import evaluate_classifier
from src.generate_dataset import generate_processed_dataset
from src.utils import ensure_directories, save_json_file


def run_training_pipeline() -> Dict[str, Any]:
    """Execute end-to-end model training, evaluation, and serialization."""
    ensure_directories()

    # Step 1: Ensure processed dataset exists
    if not PROCESSED_DATA_FILE.exists():
        print(f"Processed dataset not found at {PROCESSED_DATA_FILE}. Generating...")
        generate_processed_dataset()

    print(f"Loading processed tickets from {PROCESSED_DATA_FILE}...")
    df = pd.read_csv(PROCESSED_DATA_FILE)
    print(f"Loaded {len(df):,} records for training.")

    # Validate required columns
    required_cols = ["combined_text", "category", "priority"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from processed dataset.")

    # Drop any nulls in text or targets
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    X = df["combined_text"]
    y_cat = df["category"]
    y_pri = df["priority"]

    # -------------------------------------------------------------------------
    # Step 2: Train Category Model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING CATEGORY CLASSIFIER")
    print("=" * 60)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X,
        y_cat,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_cat,
    )
    print(f"Category Train Samples: {len(X_train_c):,} | Test Samples: {len(X_test_c):,}")

    category_clf = TriageClassifier(name="CategoryClassifier", random_state=RANDOM_STATE)
    category_clf.fit(X_train_c, y_train_c)

    y_pred_c = category_clf.predict(X_test_c)
    cat_eval = evaluate_classifier(
        y_true=list(y_test_c),
        y_pred=list(y_pred_c),
        labels=sorted(list(set(y_cat))),
        model_name="CategoryClassifier",
    )

    print(f"Category Model Test Accuracy: {cat_eval['accuracy'] * 100:.2f}%")
    print(f"Category Macro F1-Score:      {cat_eval['macro_f1']:.4f}")
    print(f"Category Weighted F1-Score:   {cat_eval['weighted_f1']:.4f}")

    # -------------------------------------------------------------------------
    # Step 3: Train Priority Model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING PRIORITY CLASSIFIER")
    print("=" * 60)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X,
        y_pri,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_pri,
    )
    print(f"Priority Train Samples: {len(X_train_p):,} | Test Samples: {len(X_test_p):,}")

    priority_clf = TriageClassifier(name="PriorityClassifier", random_state=RANDOM_STATE)
    priority_clf.fit(X_train_p, y_train_p)

    y_pred_p = priority_clf.predict(X_test_p)
    pri_eval = evaluate_classifier(
        y_true=list(y_test_p),
        y_pred=list(y_pred_p),
        labels=PRIORITY_CLASSES,
        model_name="PriorityClassifier",
    )

    print(f"Priority Model Test Accuracy: {pri_eval['accuracy'] * 100:.2f}%")
    print(f"Priority Macro F1-Score:      {pri_eval['macro_f1']:.4f}")
    print(f"Priority Weighted F1-Score:   {pri_eval['weighted_f1']:.4f}")

    # -------------------------------------------------------------------------
    # Step 4: Persist Artifacts and Metrics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 60)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    category_clf.save(CATEGORY_MODEL_PATH, CATEGORY_VECTORIZER_PATH)
    print(f"Saved Category Model to:      {CATEGORY_MODEL_PATH}")
    print(f"Saved Category Vectorizer to: {CATEGORY_VECTORIZER_PATH}")

    priority_clf.save(PRIORITY_MODEL_PATH, PRIORITY_VECTORIZER_PATH)
    print(f"Saved Priority Model to:      {PRIORITY_MODEL_PATH}")
    print(f"Saved Priority Vectorizer to: {PRIORITY_VECTORIZER_PATH}")

    evaluation_payload = {
        "dataset_name": "tobiasbueck/multilingual-customer-support-tickets",
        "dataset_version": "v8",
        "total_dataset_rows": len(df),
        "test_size_ratio": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "category_model": cat_eval,
        "priority_model": pri_eval,
    }

    save_json_file(evaluation_payload, EVALUATION_RESULTS_PATH)
    print(f"Saved Evaluation Results to:  {EVALUATION_RESULTS_PATH}")
    print("=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return evaluation_payload


if __name__ == "__main__":
    run_training_pipeline()
