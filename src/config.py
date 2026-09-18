"""Central configuration for AI Customer Support Ticket Triage.

Contains dataset parameters, directory paths, label mappings,
routing definitions, and model hyperparameters.
"""

from pathlib import Path
from typing import Dict, List

# -----------------------------------------------------------------------------
# Directory & File Paths (Relative to repository root for portable deployment)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "tickets_processed.csv"

MODELS_DIR = PROJECT_ROOT / "models"
CATEGORY_MODEL_PATH = MODELS_DIR / "category_model.joblib"
CATEGORY_VECTORIZER_PATH = MODELS_DIR / "category_vectorizer.joblib"
PRIORITY_MODEL_PATH = MODELS_DIR / "priority_model.joblib"
PRIORITY_VECTORIZER_PATH = MODELS_DIR / "priority_vectorizer.joblib"
EVALUATION_RESULTS_PATH = MODELS_DIR / "evaluation_results.json"

# Kaggle Dataset Reference
KAGGLE_DATASET_HANDLE = "tobiasbueck/multilingual-customer-support-tickets/versions/8"

# -----------------------------------------------------------------------------
# Model & Training Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 8000
TFIDF_NGRAM_RANGE = (1, 2)
DEFAULT_CONFIDENCE_THRESHOLD = 0.65

# -----------------------------------------------------------------------------
# Taxonomy & Label Mappings
# -----------------------------------------------------------------------------
# Mapping dataset original queues into 5 core business categories
QUEUE_TO_CATEGORY: Dict[str, str] = {
    "Technical Support": "Technical",
    "IT Support": "Technical",
    "Service Outages and Maintenance": "Technical",
    "Product Support": "Product",
    "Customer Service": "Customer Service",
    "General Inquiry": "Customer Service",
    "Billing and Payments": "Billing",
    "Returns and Exchanges": "Billing",
    "Sales and Pre-Sales": "Other",
    "Human Resources": "Other",
}

CATEGORY_CLASSES: List[str] = [
    "Billing",
    "Customer Service",
    "Other",
    "Product",
    "Technical",
]

# Priority labels (normalized)
PRIORITY_NORMALIZATION: Dict[str, str] = {
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "critical": "High",
    "urgent": "High",
}

PRIORITY_CLASSES: List[str] = ["High", "Medium", "Low"]

# -----------------------------------------------------------------------------
# Queue Routing Configuration
# -----------------------------------------------------------------------------
CATEGORY_TO_QUEUE: Dict[str, str] = {
    "Technical": "Technical Support Queue",
    "Product": "Product Operations Queue",
    "Customer Service": "Customer Relations Queue",
    "Billing": "Billing & Accounts Queue",
    "Other": "General Operations Queue",
}

HUMAN_REVIEW_QUEUE = "Human Review Queue"

# -----------------------------------------------------------------------------
# Realistic Sample Tickets for Testing & Interactive UI
# -----------------------------------------------------------------------------
SAMPLE_TICKETS = {
    "Billing": (
        "I was charged twice for my subscription and need help getting one of the "
        "charges refunded. My invoice number is INV-90214."
    ),
    "Technical": (
        "The application crashes with a fatal segmentation fault whenever I try to "
        "upload an attachment larger than 5MB on Windows 11."
    ),
    "Account / Customer Service": (
        "My account is locked due to multiple failed login attempts and the password "
        "reset email is not arriving in my inbox or spam folder."
    ),
    "Product": (
        "I would like to know whether this product supports two-way calendar sync "
        "with external software such as Google Calendar or Microsoft Outlook."
    ),
    "Urgent / Security": (
        "Urgent: I noticed unauthorized transactions on my company account and believe "
        "our administrative API keys and payment credentials may be compromised."
    ),
    "Spanish (Multilingual)": (
        "Problema crítico del servidor requiere atención inmediata. Es necesaria una "
        "investigación sobre la interrupción en el servicio de AWS."
    ),
    "German (Multilingual)": (
        "Sehr geehrter Kundenservice, ich habe Probleme mit der drahtlosen Verbindung "
        "zu meinem HP DeskJet Drucker und möchte den Rückgabeprozess starten."
    ),
}
