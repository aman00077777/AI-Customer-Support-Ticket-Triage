"""AI Customer Support Ticket Triage - Streamlit Application.

A production-grade, API-free dashboard for classifying, prioritizing,
and routing customer support tickets using TF-IDF and Logistic Regression.
Developed for the YHills AI/ML Internship.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    CATEGORY_MODEL_PATH,
    CATEGORY_TO_QUEUE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    EVALUATION_RESULTS_PATH,
    PRIORITY_MODEL_PATH,
    PROCESSED_DATA_FILE,
    SAMPLE_TICKETS,
)
from src.predictor import get_predictor
from src.utils import (
    InvalidTicketDataError,
    format_percentage,
    generate_batch_template_df,
    load_json_file,
)

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Customer Support Ticket Triage",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (Slate Modern Dark Theme, Clean Typography, Cards & Badges)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main Container Header */
.main-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
}
.main-header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-header p {
    font-size: 1.02rem;
    color: #94a3b8;
    margin-top: 6px;
    margin-bottom: 0;
}

/* Metric Cards */
.triage-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    margin-bottom: 12px;
}
.triage-card-title {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 6px;
}
.triage-card-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
}
.triage-card-sub {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 4px;
}

/* Badges */
.badge-routed {
    display: inline-block;
    background: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
    border-radius: 9999px;
    padding: 4px 14px;
    font-size: 0.85rem;
    font-weight: 600;
}
.badge-review {
    display: inline-block;
    background: #451a03;
    color: #fbbf24;
    border: 1px solid #d97706;
    border-radius: 9999px;
    padding: 4px 14px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Status Indicator */
.status-pill-ok {
    color: #10b981;
    font-size: 0.85rem;
    font-weight: 500;
}
.status-pill-err {
    color: #ef4444;
    font-size: 0.85rem;
    font-weight: 500;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading machine learning models...")
def load_triage_engine():
    """Load and cache the trained prediction engine."""
    return get_predictor()


@st.cache_data
def load_processed_data():
    """Load processed dataset for insights tab."""
    if PROCESSED_DATA_FILE.exists():
        return pd.read_csv(PROCESSED_DATA_FILE)
    return None


@st.cache_data
def load_evaluation_data():
    """Load precomputed test-set evaluation results."""
    if EVALUATION_RESULTS_PATH.exists():
        return load_json_file(EVALUATION_RESULTS_PATH)
    return None


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎫 Ticket Triage AI")
    st.caption("YHills AI/ML Internship Portfolio Project")
    st.markdown("---")

    nav_option = st.radio(
        "Navigation",
        [
            "Single Ticket Analysis",
            "Batch Ticket Processing",
            "Model Performance",
            "Dataset Insights",
            "About",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### System Configuration")

    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Tickets where category or priority confidence falls below this threshold require human review.",
    )

    st.markdown("---")
    st.markdown("#### System Status")

    cat_loaded = CATEGORY_MODEL_PATH.exists()
    pri_loaded = PRIORITY_MODEL_PATH.exists()
    data_loaded = PROCESSED_DATA_FILE.exists()

    if cat_loaded and pri_loaded:
        st.markdown("<span class='status-pill-ok'>● Models: Loaded & Active</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-pill-err'>● Models: Artifacts Missing</span>", unsafe_allow_html=True)

    if data_loaded:
        st.markdown("<span class='status-pill-ok'>● Dataset: 4,000 Verified Tickets</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-pill-err'>● Dataset: Not found</span>", unsafe_allow_html=True)

    st.caption("100% Local Inference · No API Keys Required")


# -----------------------------------------------------------------------------
# Header Display
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>AI Customer Support Ticket Triage</h1>
        <p>Automatically classify, prioritize, and route customer support tickets using NLP and machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# PAGE 1: SINGLE TICKET ANALYSIS
# =============================================================================
if nav_option == "Single Ticket Analysis":
    st.subheader("Single Ticket Analysis")
    st.write(
        "Enter incoming customer support ticket text below or select an example from real support scenarios to triage it."
    )

    # Example ticket selector
    st.markdown("**Quick Load Real-World Example:**")
    example_cols = st.columns(len(SAMPLE_TICKETS))
    selected_example_text = None

    for col, (label, sample_text) in zip(example_cols, SAMPLE_TICKETS.items()):
        if col.button(label, use_container_width=True):
            selected_example_text = sample_text
            st.session_state["ticket_input"] = sample_text

    # Default state if not set
    if "ticket_input" not in st.session_state:
        st.session_state["ticket_input"] = (
            "I was charged twice for my subscription and need help getting one of the charges refunded. "
            "My invoice number is INV-90214."
        )

    ticket_input = st.text_area(
        "Customer Support Ticket Text",
        value=st.session_state["ticket_input"],
        height=130,
        placeholder="Example: I was charged twice for my subscription and need a refund.",
        help="Provide the raw customer inquiry, issue description, or combined subject and body.",
    )

    analyze_clicked = st.button("⚡ Analyze Ticket", type="primary", use_container_width=False)

    if analyze_clicked or ticket_input:
        if not ticket_input.strip():
            st.warning("Please enter ticket text to perform triage.")
        else:
            try:
                predictor = load_triage_engine()
                res = predictor.predict_ticket(ticket_input, confidence_threshold=threshold, explain=True)

                st.markdown("---")
                st.markdown("### Triage & Routing Output")

                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)

                with m1:
                    st.markdown(
                        f"""
                        <div class="triage-card">
                            <div class="triage-card-title">Predicted Category</div>
                            <div class="triage-card-value">{res['predicted_category']}</div>
                            <div class="triage-card-sub">Model Confidence: {format_percentage(res['category_confidence'])}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with m2:
                    st.markdown(
                        f"""
                        <div class="triage-card">
                            <div class="triage-card-title">Predicted Urgency</div>
                            <div class="triage-card-value">{res['predicted_priority']}</div>
                            <div class="triage-card-sub">Model Confidence: {format_percentage(res['priority_confidence'])}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with m3:
                    st.markdown(
                        f"""
                        <div class="triage-card">
                            <div class="triage-card-title">Assigned Support Queue</div>
                            <div class="triage-card-value" style="font-size: 1.15rem;">{res['assigned_queue']}</div>
                            <div class="triage-card-sub">Target: {res['target_queue']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with m4:
                    badge_html = (
                        '<span class="badge-review">Human Review Required</span>'
                        if res["human_review_required"]
                        else '<span class="badge-routed">Automatically Routed</span>'
                    )
                    st.markdown(
                        f"""
                        <div class="triage-card">
                            <div class="triage-card-title">Routing Decision</div>
                            <div style="margin-top: 4px;">{badge_html}</div>
                            <div class="triage-card-sub">Threshold: {format_percentage(threshold)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Human review alert if triggered
                if res["human_review_required"]:
                    st.warning(f"⚠️ **Review Trigger:** {res['review_reason']}")

                # Model Interpretation Expander
                with st.expander("🔍 Model Interpretation (Influential Text Features)", expanded=True):
                    st.caption(
                        "Top TF-IDF terms in the incoming ticket that positively influenced the classifier decision. "
                        "These features provide an approximate, transparent interpretation of the model's prediction."
                    )
                    i1, i2 = st.columns(2)

                    with i1:
                        st.markdown(f"**Influential Terms for Category: `{res['predicted_category']}`**")
                        if res["category_influential_terms"]:
                            cat_terms_df = pd.DataFrame(res["category_influential_terms"])
                            fig_cat = px.bar(
                                cat_terms_df,
                                x="contribution",
                                y="feature",
                                orientation="h",
                                labels={"contribution": "Influence Weight", "feature": "Term"},
                                color="contribution",
                                color_continuous_scale="Viridis",
                            )
                            fig_cat.update_layout(
                                margin=dict(l=0, r=0, t=10, b=0),
                                height=200,
                                yaxis=dict(autorange="reversed"),
                                coloraxis_showscale=False,
                            )
                            st.plotly_chart(fig_cat, use_container_width=True)
                        else:
                            st.info("No prominent specific n-grams detected in vocabulary.")

                    with i2:
                        st.markdown(f"**Influential Terms for Urgency: `{res['predicted_priority']}`**")
                        if res["priority_influential_terms"]:
                            pri_terms_df = pd.DataFrame(res["priority_influential_terms"])
                            fig_pri = px.bar(
                                pri_terms_df,
                                x="contribution",
                                y="feature",
                                orientation="h",
                                labels={"contribution": "Influence Weight", "feature": "Term"},
                                color="contribution",
                                color_continuous_scale="Plasma",
                            )
                            fig_pri.update_layout(
                                margin=dict(l=0, r=0, t=10, b=0),
                                height=200,
                                yaxis=dict(autorange="reversed"),
                                coloraxis_showscale=False,
                            )
                            st.plotly_chart(fig_pri, use_container_width=True)
                        else:
                            st.info("No prominent specific n-grams detected in vocabulary.")

            except Exception as e:
                st.error(f"Error during ticket triage: {e}")


# =============================================================================
# PAGE 2: BATCH TICKET PROCESSING
# =============================================================================
elif nav_option == "Batch Ticket Processing":
    st.subheader("Batch Ticket Processing")
    st.write(
        "Upload a CSV file containing multiple customer support tickets to run high-throughput triage, routing, and confidence analysis."
    )

    # Template download
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        template_df = generate_batch_template_df()
        st.download_button(
            label="📥 Download Sample Batch CSV Template",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="ticket_triage_sample_batch.csv",
            mime="text/csv",
            help="Contains sample tickets with ticket_id and ticket_text columns.",
        )

    uploaded_file = st.file_uploader("Upload CSV Tickets File", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded CSV with **{len(input_df):,}** rows and columns: `{list(input_df.columns)}`")

            if st.button("🚀 Process Batch Triage", type="primary"):
                with st.spinner("Processing tickets and assigning queues..."):
                    predictor = load_triage_engine()
                    results_df = predictor.predict_batch(input_df, confidence_threshold=threshold)

                st.success(f"Successfully processed {len(results_df):,} tickets!")

                # Batch summary KPIs
                auto_count = (results_df["human_review_required"] == False).sum()
                review_count = (results_df["human_review_required"] == True).sum()
                total_count = len(results_df)

                k1, k2, k3 = st.columns(3)
                with k1:
                    st.metric("Total Tickets Processed", f"{total_count:,}")
                with k2:
                    pct_auto = (auto_count / total_count) if total_count > 0 else 0
                    st.metric("Automatically Routed", f"{auto_count:,} ({pct_auto*100:.1f}%)")
                with k3:
                    pct_rev = (review_count / total_count) if total_count > 0 else 0
                    st.metric("Human Review Required", f"{review_count:,} ({pct_rev*100:.1f}%)")

                st.markdown("---")
                st.markdown("### Batch Results Table")
                st.dataframe(results_df, use_container_width=True)

                # Export buttons
                d1, d2, _ = st.columns([1, 1, 2])
                with d1:
                    csv_export = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_export,
                        file_name="triage_batch_results.csv",
                        mime="text/csv",
                    )
                with d2:
                    json_export = results_df.to_json(orient="records", indent=2)
                    st.download_button(
                        label="📥 Download Results as JSON",
                        data=json_export,
                        file_name="triage_batch_results.json",
                        mime="application/json",
                    )

        except InvalidTicketDataError as e:
            st.error(f"Invalid input file: {e}")
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")


# =============================================================================
# PAGE 3: MODEL PERFORMANCE
# =============================================================================
elif nav_option == "Model Performance":
    st.subheader("Model Performance & Test Set Evaluation")
    st.info(
        "📌 **Evaluation Disclaimer:** The reported metrics are based on a held-out portion (20% stratified test set, "
        "800 samples) of the selected public dataset and should not be interpreted as guaranteed real-world performance."
    )

    eval_data = load_evaluation_data()

    if eval_data is None:
        st.warning("Evaluation metrics file not found. Please train models first via `python -m src.train`.")
    else:
        cat_metrics = eval_data["category_model"]
        pri_metrics = eval_data["priority_model"]

        tab_cat, tab_pri = st.tabs(["🏷️ Category Classifier", "⚡ Priority Classifier"])

        # Tab 1: Category
        with tab_cat:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{cat_metrics['accuracy'] * 100:.2f}%")
            c2.metric("Macro F1-Score", f"{cat_metrics['macro_f1']:.4f}")
            c3.metric("Weighted F1-Score", f"{cat_metrics['weighted_f1']:.4f}")
            c4.metric("Test Samples", f"{cat_metrics['test_samples']:,}")

            st.markdown("#### Confusion Matrix")
            cat_cm = cat_metrics["confusion_matrix"]
            cat_labels = cat_metrics["labels"]

            fig_cm_cat = go.Figure(
                data=go.Heatmap(
                    z=cat_cm,
                    x=cat_labels,
                    y=cat_labels,
                    colorscale="Blues",
                    text=cat_cm,
                    texttemplate="%{text}",
                    textfont={"size": 14},
                )
            )
            fig_cm_cat.update_layout(
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                margin=dict(l=0, r=0, t=30, b=0),
                height=380,
            )
            st.plotly_chart(fig_cm_cat, use_container_width=True)

            st.markdown("#### Detailed Classification Report")
            rep_df = pd.DataFrame(cat_metrics["classification_report"]).T
            st.dataframe(rep_df.style.format("{:.4f}"), use_container_width=True)

        # Tab 2: Priority
        with tab_pri:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Accuracy", f"{pri_metrics['accuracy'] * 100:.2f}%")
            p2.metric("Macro F1-Score", f"{pri_metrics['macro_f1']:.4f}")
            p3.metric("Weighted F1-Score", f"{pri_metrics['weighted_f1']:.4f}")
            p4.metric("Test Samples", f"{pri_metrics['test_samples']:,}")

            st.markdown("#### Confusion Matrix")
            pri_cm = pri_metrics["confusion_matrix"]
            pri_labels = pri_metrics["labels"]

            fig_cm_pri = go.Figure(
                data=go.Heatmap(
                    z=pri_cm,
                    x=pri_labels,
                    y=pri_labels,
                    colorscale="Greens",
                    text=pri_cm,
                    texttemplate="%{text}",
                    textfont={"size": 14},
                )
            )
            fig_cm_pri.update_layout(
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                margin=dict(l=0, r=0, t=30, b=0),
                height=350,
            )
            st.plotly_chart(fig_cm_pri, use_container_width=True)

            st.markdown("#### Detailed Classification Report")
            pri_rep_df = pd.DataFrame(pri_metrics["classification_report"]).T
            st.dataframe(pri_rep_df.style.format("{:.4f}"), use_container_width=True)


# =============================================================================
# PAGE 4: DATASET INSIGHTS
# =============================================================================
elif nav_option == "Dataset Insights":
    st.subheader("Dataset Insights & Exploratory Analysis")
    df = load_processed_data()

    if df is None:
        st.warning("Processed dataset file not found. Run `python -m src.generate_dataset`.")
    else:
        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Tickets", f"{len(df):,}")
        k2.metric("Languages Represented", str(df["language"].nunique() if "language" in df.columns else 5))
        k3.metric("Category Classes", str(df["category"].nunique()))
        k4.metric("Priority Tiers", str(df["priority"].nunique()))

        st.markdown("---")
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("#### Category Distribution")
            cat_counts = df["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig1 = px.bar(
                cat_counts,
                x="Category",
                y="Count",
                color="Category",
                text="Count",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig1.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        with ch2:
            st.markdown("#### Priority Distribution")
            pri_counts = df["priority"].value_counts().reset_index()
            pri_counts.columns = ["Priority", "Count"]
            fig2 = px.pie(
                pri_counts,
                names="Priority",
                values="Count",
                hole=0.4,
                color_discrete_sequence=["#ef4444", "#f59e0b", "#10b981"],
            )
            fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
            st.plotly_chart(fig2, use_container_width=True)

        ch3, ch4 = st.columns(2)

        with ch3:
            if "language" in df.columns:
                st.markdown("#### Language Distribution")
                lang_counts = df["language"].value_counts().reset_index()
                lang_counts.columns = ["Language", "Count"]
                fig3 = px.bar(
                    lang_counts,
                    x="Language",
                    y="Count",
                    color="Language",
                    text="Count",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

        with ch4:
            if "original_queue" in df.columns:
                st.markdown("#### Original Dataset Queues (Top 10)")
                q_counts = df["original_queue"].value_counts().reset_index()
                q_counts.columns = ["Original Queue", "Count"]
                fig4 = px.bar(
                    q_counts,
                    y="Original Queue",
                    x="Count",
                    orientation="h",
                    color="Count",
                    color_continuous_scale="Teal",
                )
                fig4.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=320,
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig4, use_container_width=True)

        st.markdown("#### Dataset Preview (First 10 Records)")
        st.dataframe(df.head(10), use_container_width=True)


# =============================================================================
# PAGE 5: ABOUT
# =============================================================================
elif nav_option == "About":
    st.subheader("About the Project")

    st.markdown(
        """
        ### 🎯 Project Overview
        **AI Customer Support Ticket Triage** is an automated, production-style machine learning system
        engineered to triage, categorize, prioritize, and route incoming customer service tickets.
        By replacing manual ticket sorting with intelligent text classification, support organizations
        can drastically lower response latency, optimize agent workload allocation, and prioritize urgent issues.

        ### ⚙️ How It Works (NLP + ML Pipeline)
        ```
        Raw Ticket (Subject + Body)
                    ↓
        Text Normalization & Cleaning (Whitespace, HTML unescaping)
                    ↓
        TF-IDF Vectorization (n-grams 1-2, Sublinear TF)
                    ↓
        Dual Logistic Regression Classifiers
            ├── Category Classification (Technical, Product, Billing, Customer Service, Other)
            └── Priority Classification (High, Medium, Low)
                    ↓
        Confidence Scoring & Human-in-the-Loop Decisioning
            ├── If both confidences >= Threshold → Automatically Routed to Queue
            └── If either confidence < Threshold  → Human Review Required (Flagged with Reason)
        ```

        ### 🛡️ Preventing Data Leakage
        A critical engineering discipline in this project is strict **data leakage prevention**.
        Support ticket resolution data, agent replies, post-triage tags, and target labels are **strictly excluded**
        from feature vectorization. The model only receives textual information that would realistically be available
        at the moment a ticket is first submitted by a customer.

        ### 📊 Dataset Attribution
        This project uses the publicly available Kaggle dataset:
        - **Dataset:** `tobiasbueck/multilingual-customer-support-tickets` (Version 8)
        - **Scope:** 4,000 real multilingual customer support records across English, German, Spanish, French, and Portuguese.
        - **Privacy:** Sourced exclusively from open benchmark data; contains no private personal identifiable information (PII).

        ### 🔒 Zero API Key Architecture
        This system runs **100% locally and open-source**:
        - No OpenAI, Anthropic, Gemini, Groq, or external LLM API dependencies.
        - Built using Scikit-Learn, Pandas, NumPy, Joblib, Streamlit, Plotly, and Pytest.
        - Fully compatible with offline environments and zero-cost Streamlit Community Cloud deployment.

        ### 🎓 Internship Attribution
        Developed as part of the **YHills AI/ML Internship**.
        """
    )
