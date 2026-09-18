# AI Customer Support Ticket Triage

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Status: Production-Ready](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)]()
[![No API Keys](https://img.shields.io/badge/API%20Keys-Zero%20(100%25%20Local)-lightgrey.svg)]()

> **Automated, intelligent customer support ticket categorization, priority scoring, and queue routing using NLP and Machine Learning with human-in-the-loop safeguards.**
> 
> *Developed as part of the YHills AI/ML Internship.*

---

## Overview

Customer support organizations face massive volumes of inbound tickets across multiple communication channels. Manual triage introduces response bottlenecks, escalates operational expenses, and risks critical service-outage or billing inquiries lingering unaddressed.

**AI Customer Support Ticket Triage** solves this by automating the initial triage layer. By analyzing the raw textual content of inbound tickets upon arrival, the system simultaneously determines:
1. **Ticket Domain / Category** (`Billing`, `Technical`, `Product`, `Customer Service`, `Other`)
2. **Urgency / Priority Level** (`High`, `Medium`, `Low`)
3. **Statistical Model Confidence** for both predictions
4. **Target Operational Support Queue** (e.g., *Billing & Accounts Queue*, *Technical Support Queue*)
5. **Human Review Decisioning**: Tickets falling below a configurable confidence threshold (default: `65%`) are automatically flagged and routed to a dedicated *Human Review Queue* with the explicit trigger reason documented.

---

## Key Features

- **End-to-End NLP + ML Pipeline:** TF-IDF feature extraction ($n$-gram range 1–2, sublinear term-frequency scaling) coupled with dual calibrated Logistic Regression classifiers.
- **Strict Leakage Prevention:** Built exclusively on pre-resolution customer text (`subject` + `body`). Resolution answers, agent notes, tags, and target labels are rigorously excluded from model features.
- **Multilingual Data Grounding:** Trained on 4,000 real multilingual customer support records covering English, German, Spanish, French, and Portuguese.
- **Dual Independent Classifiers:** Separate models for category and priority triage, enabling granular probability and confidence estimations.
- **Human-in-the-Loop Safeguards:** Configurable confidence thresholding (default `0.65`) prevents unvetted automated routing when model uncertainty is detected.
- **Transparent Model Interpretability:** Extracts influential TF-IDF n-grams that contributed positively to each classification decision.
- **Batch Processing & Data Export:** Upload bulk CSV ticket batches, inspect triage decisions in interactive tables, and export results directly to CSV or JSON.
- **Zero API Key Architecture:** 100% local inference with zero paid external API dependencies (no OpenAI, Gemini, Groq, Anthropic, or Hugging Face Inference endpoints required).

---

## Architecture & Workflow

```
                        INCOMING TICKET
                    (Customer Subject + Body)
                               ↓
                   Data Cleaning & Normalization
            (HTML entity decoding, whitespace handling)
                               ↓
                       TF-IDF Vectorization
                    (Sublinear TF, n-grams 1-2)
                               ↓
              ┌────────────────────────────────┐
              │   Dual Logistic Regression     │
              ├────────────────────────────────┤
              │ 1. Category Classifier (5-way) │
              │ 2. Priority Classifier (3-way) │
              └────────────────┬───────────────┘
                               ↓
                     Confidence Calculation
                     (Class Probabilities)
                               ↓
                 Is Confidence >= Threshold?
                      (Default: 0.65)
                        /          \
                     YES            NO
                     /                \
           [Automatically Routed]    [Human Review Required]
           Target Support Queue      Human Review Queue
           (Billing, Tech, etc.)     (With Trigger Reason)
```

---

## Dataset Details

The system is trained and evaluated on the real-world public Kaggle benchmark:
- **Dataset:** [`tobiasbueck/multilingual-customer-support-tickets`](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)
- **Version:** `versions/8` (`dataset-tickets-multi-lang3-4k.csv`)
- **Total Records:** 4,000 tickets
- **Character Encoding:** Latin-1 (`latin1` / ISO-8859-1) for preserving multilingual diacritics
- **Duplicates:** 0 exact duplicate rows

### Languages Represented
| Language | Code | Ticket Count | Percentage |
|---|---|---|---|
| English | `en` | 1,391 | 34.8% |
| German | `de` | 848 | 21.2% |
| Spanish | `es` | 812 | 20.3% |
| French | `fr` | 476 | 11.9% |
| Portuguese | `pt` | 473 | 11.8% |

### Category Taxonomy & Label Mapping
The dataset's original 10 queues are mapped to 5 core business operations categories:
| Original Dataset Queue | Core Category | Volume |
|---|---|---|
| Technical Support, IT Support, Service Outages and Maintenance | **Technical** | 1,903 (47.6%) |
| Product Support | **Product** | 690 (17.2%) |
| Customer Service, General Inquiry | **Customer Service** | 682 (17.1%) |
| Billing and Payments, Returns and Exchanges | **Billing** | 535 (13.4%) |
| Sales and Pre-Sales, Human Resources | **Other** | 190 (4.8%) |

### Priority Distribution
- **High:** 1,649 tickets (41.2%)
- **Medium:** 1,603 tickets (40.1%)
- **Low:** 748 tickets (18.7%)

### Data Leakage Prevention
To guarantee that the machine learning models reflect genuine operational conditions, columns representing post-resolution data are strictly excluded from feature vectorization:
- Excluded: `answer` (agent response), `tag_1` through `tag_9` (post-triage tags), `type`, `queue`, `priority`.
- Input feature: Combined `subject` + `body` normalized into `combined_text`.

---

## Machine Learning & Evaluation

The system utilizes an 80/20 stratified train/test split with `random_state = 42` (3,200 training samples, 800 held-out evaluation samples).

### Held-Out Test Set Performance Metrics

```json
{
  "test_samples": 800,
  "category_model": {
    "accuracy": "61.50%",
    "macro_f1": 0.5711,
    "weighted_f1": 0.6248
  },
  "priority_model": {
    "accuracy": "58.50%",
    "macro_f1": 0.5665,
    "weighted_f1": 0.5872
  }
}
```

> **Evaluation Disclaimer:** The reported metrics are based on a held-out portion (20% stratified test set, 800 samples) of the selected public benchmark dataset and should not be interpreted as guaranteed real-world performance across arbitrary enterprise domains.

---

## Technology Stack

- **Core Language:** Python 3.10+
- **Machine Learning & NLP:** Scikit-Learn (`TfidfVectorizer`, `LogisticRegression`)
- **Data Manipulation:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Web Application & UI:** Streamlit
- **Visualizations:** Plotly Express & Plotly Graph Objects
- **Testing Framework:** PyTest
- **Data Ingestion:** KaggleHub

---

## Project Structure

```text
AI-Customer-Support-Ticket-Triage/
├── app.py                          # Multi-page Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Comprehensive documentation
├── .gitignore                      # Git exclusion rules
│
├── data/
│   ├── raw/                        # Cached raw dataset from KaggleHub
│   └── processed/
│       └── tickets_processed.csv   # Normalized clean dataset (4,000 rows)
│
├── models/
│   ├── category_model.joblib       # Trained category classifier
│   ├── category_vectorizer.joblib  # Fitted category TF-IDF vectorizer
│   ├── priority_model.joblib       # Trained priority classifier
│   ├── priority_vectorizer.joblib  # Fitted priority TF-IDF vectorizer
│   └── evaluation_results.json     # Test-set evaluation metrics & confusion matrices
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central paths, taxonomies, thresholds, samples
│   ├── data_loader.py              # Download, encoding fallback, and inspection
│   ├── preprocessor.py             # Normalization, leakage prevention, label mapping
│   ├── generate_dataset.py         # End-to-end dataset generation CLI
│   ├── evaluator.py                # Multi-class evaluation calculator
│   ├── classifier.py               # TriageClassifier & feature explainability
│   ├── train.py                    # End-to-end training pipeline
│   ├── predictor.py                # Triage engine, routing logic, batch inference
│   └── utils.py                    # Utilities, JSON serialization, exceptions
│
└── tests/
    ├── __init__.py
    ├── test_data_loader.py         # Loader and inspection tests
    ├── test_preprocessor.py        # Preprocessing & leakage tests
    ├── test_classifier.py          # Model training, inference, explainability tests
    ├── test_predictor.py           # Triage engine & threshold logic tests
    └── test_utils.py               # Utility tests
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AI-Customer-Support-Ticket-Triage
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset Preparation & Model Training

### Inspect Raw Dataset
To download (via KaggleHub) and generate an inspection report of the raw dataset:
```bash
python -m src.data_loader
```

### Generate Processed Dataset
To clean, normalize, map labels, and save `data/processed/tickets_processed.csv`:
```bash
python -m src.generate_dataset
```

### Train Machine Learning Models
To run stratified train/test split, train both classifiers, evaluate metrics, and save artifacts:
```bash
python -m src.train
```

---

## Running the Web Application

Launch the Streamlit interactive dashboard:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## Automated Testing

Run the automated test suite using PyTest:
```bash
python -m pytest -v
```
All 24 unit tests validate:
- Dataset discovery, encoding fallback, and column inspection.
- Text cleaning, normalization, and absence of target data leakage.
- Model training, probability calculation, and feature explainability.
- Single ticket inference, confidence threshold triggers, and queue routing.
- Batch CSV processing, empty inputs, and invalid column handling.

---

## Example Usage & Output

### Single Ticket Python API
```python
from src.predictor import get_predictor

predictor = get_predictor()
ticket = "I was charged twice for my subscription and need help getting a refund."

result = predictor.predict_ticket(ticket, confidence_threshold=0.65)
print(result)
```

### JSON Output Structure
```json
{
  "ticket_text": "I was charged twice for my subscription and need help getting a refund.",
  "predicted_category": "Billing",
  "category_confidence": 0.7231,
  "predicted_priority": "High",
  "priority_confidence": 0.4042,
  "assigned_queue": "Human Review Queue",
  "target_queue": "Billing & Accounts Queue",
  "human_review_required": true,
  "routing_decision": "Human Review Required",
  "review_reason": "Low priority confidence: 40.4% is below threshold (65.0%).",
  "category_influential_terms": [
    {"feature": "charges", "contribution": 1.4821},
    {"feature": "subscription", "contribution": 1.1205},
    {"feature": "refund", "contribution": 0.9842}
  ]
}
```

---

## Limitations

- **Multilingual Lexical Coverage:** While the TF-IDF vectorizer captures cross-lingual n-grams effectively for major classes, vocabulary overlap across 5 languages can lead to lower confidence for low-frequency technical jargon.
- **Priority Subjectivity:** In customer support data, priority annotations frequently contain subjective human variance (e.g., non-critical issues marked as high urgency).
- **Linear Decision Boundaries:** Logistic Regression models linear feature contributions; complex semantic relationships and multi-turn conversational context are better served by transformer models where compute permits.

---

## Future Improvements

- **Fine-Tuned Multilingual Transformers:** Explore quantized local models (e.g., `xlm-roberta` or small distilled models) for deeper contextual semantics.
- **Entity & Account Number Extraction:** Integrate regex or spaCy NER pipelines to extract invoice numbers, email addresses, and error codes automatically.
- **Customer Sentiment Analysis:** Incorporate sentiment polarity scoring as an additional signal for urgency escalation.
- **Webhook / CRM Integration:** Connect the routing engine to Zendesk, Freshdesk, or Jira Service Management APIs.

---

## Internship Attribution

Developed by **Ashish Prajapati** as part of the **YHills AI/ML Internship** (Project 3: AI Customer Support Ticket Triage).
