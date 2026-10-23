"""
app.py
Streamlit Web Application for Sentiment Analysis of Customer Reviews.
Features:
- Live Sentiment Classification with confidence scores and probability breakdowns
- Animated sentiment gauge and probability visualization
- Batch analysis mode for CSV uploads
- One-click sample review testing with categorized prompts
- NLP Preprocessing pipeline inspector with token visualization
- Interactive EDA and Model Performance tabs
- Prediction history tracking within session
- Modern glassmorphism UI with custom styling and micro-animations
"""

import os
import sys
import time
import pandas as pd
import streamlit as st

# Configure page metadata and layout
st.set_page_config(
    page_title="SentimentScope | Customer Reviews NLP",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set path to import from src
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.predict import predict_sentiment, predict_with_probabilities, load_artifacts, batch_predict
from src.preprocess import clean_text

OUTPUTS_DIR = os.path.join(CURRENT_DIR, "outputs")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS – Premium Dark Glassmorphism Theme with Micro-Animations
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global Fonts ── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Main Container ── */
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* ── Glassmorphism Hero Card ── */
    .hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 50%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 60px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.05);
        color: white;
        position: relative;
        overflow: hidden;
    }
    .hero-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(96,165,250,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-card::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(167,139,250,0.06) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 40%, #F472B6 80%, #FB923C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-size: 200% auto;
        animation: shimmer 4s ease-in-out infinite;
        margin-bottom: 0.4rem;
        position: relative;
        z-index: 1;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 100% center; }
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 1rem;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }

    .badge-pill {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: rgba(96, 165, 250, 0.12);
        color: #93C5FD;
        border: 1px solid rgba(147, 197, 253, 0.2);
        margin-right: 0.5rem;
        margin-bottom: 0.4rem;
        transition: all 0.3s ease;
        position: relative;
        z-index: 1;
    }
    .badge-pill:hover {
        background: rgba(96, 165, 250, 0.25);
        transform: translateY(-1px);
    }

    /* ── Sentiment Result Badges ── */
    .result-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.8rem 1.6rem;
        border-radius: 14px;
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        animation: fadeSlideUp 0.5s ease-out;
    }
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .badge-positive {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(52, 211, 153, 0.08) 100%);
        color: #10B981;
        border: 1.5px solid rgba(16, 185, 129, 0.5);
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    .badge-neutral {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(251, 191, 36, 0.08) 100%);
        color: #F59E0B;
        border: 1.5px solid rgba(245, 158, 11, 0.5);
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.15);
    }
    .badge-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(248, 113, 113, 0.08) 100%);
        color: #EF4444;
        border: 1.5px solid rgba(239, 68, 68, 0.5);
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
    }

    /* ── Probability Cards ── */
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        border-radius: 14px 14px 0 0;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.2);
        border-color: rgba(255, 255, 255, 0.15);
    }
    .metric-card h4 {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 0.3rem 0;
    }
    .metric-card h2 {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        font-variant-numeric: tabular-nums;
    }
    .metric-card .prob-bar {
        height: 4px;
        border-radius: 2px;
        margin-top: 0.6rem;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
    }
    .metric-card .prob-bar-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Token Chips ── */
    .token-chip {
        display: inline-block;
        background: linear-gradient(135deg, #1E293B, #334155);
        color: #E2E8F0;
        padding: 0.3rem 0.7rem;
        border-radius: 8px;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 0.2rem;
        border: 1px solid #475569;
        transition: all 0.2s ease;
    }
    .token-chip:hover {
        background: linear-gradient(135deg, #334155, #475569);
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .negation-chip {
        background: linear-gradient(135deg, #7F1D1D, #991B1B) !important;
        color: #FCA5A5 !important;
        border-color: #DC2626 !important;
        font-weight: 600;
    }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.5rem;
    }
    .section-icon {
        font-size: 1.5rem;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(96, 165, 250, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(96, 165, 250, 0.2);
    }

    /* ── History Cards ── */
    .history-item {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: background 0.2s ease;
    }
    .history-item:hover {
        background: rgba(255,255,255,0.06);
    }
    .history-text {
        font-size: 0.85rem;
        color: #CBD5E1;
        max-width: 60%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .history-label {
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
    }
    .hl-positive { background: rgba(16,185,129,0.15); color: #10B981; }
    .hl-neutral { background: rgba(245,158,11,0.15); color: #F59E0B; }
    .hl-negative { background: rgba(239,68,68,0.15); color: #EF4444; }

    /* ── Gauge Ring ── */
    .gauge-container {
        display: flex;
        justify-content: center;
        margin: 0.5rem 0;
    }
    .gauge-ring {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }
    .gauge-inner {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        background: #0F172A;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 1;
    }
    .gauge-value {
        font-size: 1.4rem;
        font-weight: 800;
        font-variant-numeric: tabular-nums;
    }
    .gauge-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94A3B8;
        margin-top: 2px;
    }

    /* ── Sidebar Polish ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #F1F5F9;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* ── Batch results table ── */
    .batch-stats {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .batch-stat-card {
        flex: 1;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .batch-stat-card h3 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 800;
    }
    .batch-stat-card p {
        margin: 0.2rem 0 0 0;
        font-size: 0.78rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────
if "review_text" not in st.session_state:
    st.session_state["review_text"] = "This laptop is absolutely fantastic! Fast, reliable, and exceeded my expectations."
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []


def check_models_ready():
    """Verify if trained models exist."""
    m_path = os.path.join(MODELS_DIR, "best_model.joblib")
    v_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    return os.path.exists(m_path) and os.path.exists(v_path)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Configuration
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.5rem;">
        <div style="font-size:2rem;">🧠</div>
        <div>
            <div style="font-size:1.2rem; font-weight:800; color:#F1F5F9;">SentimentScope</div>
            <div style="font-size:0.75rem; color:#64748B; font-weight:500;">NLP Review Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("#### **System Status**")
    if check_models_ready():
        st.success("Model: **Trained & Loaded** ✅")
        try:
            model, vec = load_artifacts()
            model_type = type(model).__name__
            vocab_size = len(vec.vocabulary_) if hasattr(vec, 'vocabulary_') else 'N/A'
            st.info(f"**Architecture:** `{model_type}`\n\n**Vocab Size:** `{vocab_size:,} features`\n\n**N-grams:** `(1, 2)`")
        except Exception:
            pass
    else:
        st.warning("Model Status: **Not Found** ⚠️")
        st.info("Run `python src/train.py` to train models.")

    st.divider()
    st.markdown("#### **Quick Test Prompts**")
    st.caption("Click any sample to load it instantly:")

    sample_dict = {
        "🟢 Positive (Tech)": "This smartphone is phenomenal! Blazing performance, crisp OLED screen, and phenomenal battery life.",
        "🟢 Positive (Audio)": "Absolutely love these headphones! Sound quality is top tier and noise cancellation is magical.",
        "🟡 Neutral (Delivery)": "The product arrived in average packaging. Works as described, nothing particularly notable.",
        "🟡 Neutral (General)": "Fair purchase for the price point. Neither amazed nor dissatisfied with daily utility.",
        "🔴 Negative (Defect)": "Defective out of the box. Stopped working after 2 hours and customer service was unhelpful.",
        "🔴 Negative (Negation)": "I did not like this item at all. It is never reliable, totally cheap, and broken."
    }

    for label, text in sample_dict.items():
        if st.button(label, use_container_width=True):
            st.session_state["review_text"] = text

    st.divider()

    # Prediction History
    if st.session_state["prediction_history"]:
        st.markdown("#### **Recent Predictions**")
        for entry in reversed(st.session_state["prediction_history"][-5:]):
            hl_class = f"hl-{entry['label']}"
            truncated = entry['text'][:50] + "..." if len(entry['text']) > 50 else entry['text']
            st.markdown(f"""
            <div class="history-item">
                <span class="history-text">{truncated}</span>
                <span class="history-label {hl_class}">{entry['label'].upper()} {entry['conf']:.0%}</span>
            </div>
            """, unsafe_allow_html=True)
        st.divider()

    st.caption("CPU-Optimized Traditional ML • Negation-Aware NLP Pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Hero Section
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <div class="hero-title">SentimentScope • NLP Review Intelligence</div>
    <div class="hero-subtitle">Classify customer reviews into Positive, Neutral, or Negative sentiments using TF-IDF vectorization, negation-aware preprocessing, and tuned Machine Learning models.</div>
    <div>
        <span class="badge-pill">🔤 TF-IDF Vectorization</span>
        <span class="badge-pill">📐 N-Gram (1, 2)</span>
        <span class="badge-pill">🚫 Negation-Aware NLP</span>
        <span class="badge-pill">⚡ Scikit-Learn ML</span>
        <span class="badge-pill">📊 Batch Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs for Navigation
# ─────────────────────────────────────────────────────────────────────────────
tab_pred, tab_batch, tab_eda, tab_metrics, tab_docs = st.tabs([
    "🔍 Live Classifier",
    "📋 Batch Analysis",
    "📊 Exploratory Data Analysis",
    "🏆 Model Benchmarks",
    "📖 Pipeline Architecture"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Live Sentiment Classifier
# ─────────────────────────────────────────────────────────────────────────────
with tab_pred:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔍</div>
        <div>
            <h3 style="margin:0;">Review Classification Console</h3>
            <p style="margin:0; color:#94A3B8; font-size:0.85rem;">Type or paste any customer review to predict its sentiment and confidence level.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    user_input = st.text_area(
        label="Customer Review Text",
        value=st.session_state.get("review_text", ""),
        height=140,
        placeholder="e.g., Outstanding quality and super quick delivery! Highly recommend to everyone...",
        help="Input customer feedback or product review for sentiment analysis."
    )

    col_btn1, col_btn2, _ = st.columns([1.5, 1.2, 5])
    with col_btn1:
        predict_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

    if clear_clicked:
        st.session_state["review_text"] = ""
        st.rerun()

    if predict_clicked or (user_input and "review_text" in st.session_state):
        if not user_input.strip():
            st.warning("Please enter some text in the review box above.")
        elif not check_models_ready():
            st.error("Trained model files not detected. Please run `python src/train.py` first.")
        else:
            start_time = time.perf_counter()
            with st.spinner("Analyzing text with NLP pipeline..."):
                res = predict_with_probabilities(user_input)
                label = res['label']
                conf = res['confidence']
                probs = res['probabilities']
                cleaned = res['cleaned_text']
            inference_ms = (time.perf_counter() - start_time) * 1000

            # Track in history
            st.session_state["prediction_history"].append({
                'text': user_input[:100],
                'label': label,
                'conf': conf
            })
            # Keep only last 20
            if len(st.session_state["prediction_history"]) > 20:
                st.session_state["prediction_history"] = st.session_state["prediction_history"][-20:]

            st.markdown("---")

            # Determine styling
            badge_class = f"badge-{label.lower()}"
            icon_map = {'positive': '🟢 Positive', 'neutral': '🟡 Neutral', 'negative': '🔴 Negative'}
            color_map = {'positive': '#10B981', 'neutral': '#F59E0B', 'negative': '#EF4444'}
            display_label = icon_map.get(label.lower(), label.upper())
            accent_color = color_map.get(label.lower(), '#60A5FA')

            c1, c2 = st.columns([1.2, 2])

            with c1:
                st.markdown(f"""
                <div class="result-badge {badge_class}">
                    <span>{display_label}</span>
                </div>
                """, unsafe_allow_html=True)

                # Confidence Gauge Ring
                conf_pct = conf * 100
                gauge_gradient = f"conic-gradient({accent_color} {conf_pct * 3.6}deg, rgba(255,255,255,0.06) {conf_pct * 3.6}deg)"
                st.markdown(f"""
                <div class="gauge-container">
                    <div class="gauge-ring" style="background: {gauge_gradient};">
                        <div class="gauge-inner">
                            <div class="gauge-value" style="color: {accent_color};">{conf_pct:.1f}%</div>
                            <div class="gauge-label">Confidence</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.caption(f"⚡ Inference: {inference_ms:.0f}ms")

            with c2:
                st.markdown("**Class Probability Breakdown:**")
                prob_cols = st.columns(3)

                pos_prob = probs.get('positive', 0.0)
                neu_prob = probs.get('neutral', 0.0)
                neg_prob = probs.get('negative', 0.0)

                prob_data = [
                    ("Positive", pos_prob, "#10B981"),
                    ("Neutral", neu_prob, "#F59E0B"),
                    ("Negative", neg_prob, "#EF4444"),
                ]

                for col, (name, prob, color) in zip(prob_cols, prob_data):
                    with col:
                        st.markdown(f"""
                        <div class="metric-card" style="border-top: 3px solid {color};">
                            <h4 style="color:{color};">{name}</h4>
                            <h2 style="color:#F1F5F9;">{prob*100:.1f}%</h2>
                            <div class="prob-bar">
                                <div class="prob-bar-fill" style="width:{prob*100:.1f}%; background:{color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # NLP Preprocessing Inspection
            with st.expander("🔎 View NLP Preprocessed Tokens & Negation Inspection", expanded=False):
                st.markdown("**Cleaned & Lemmatized Representation (Fed into TF-IDF):**")
                negation_words = {'not', 'no', 'never', 'nor', 'neither'}
                if cleaned:
                    tokens = cleaned.split()
                    token_chips = " ".join([
                        f"<span class='token-chip negation-chip'>{tok}</span>"
                        if tok in negation_words
                        else f"<span class='token-chip'>{tok}</span>"
                        for tok in tokens
                    ])
                    st.markdown(token_chips, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="margin-top:0.8rem; padding:0.6rem 1rem; background:rgba(255,255,255,0.03);
                         border-radius:8px; border:1px solid rgba(255,255,255,0.06);">
                        <span style="color:#64748B; font-size:0.82rem;">
                            📊 <strong>{len(tokens)}</strong> tokens extracted &nbsp;•&nbsp;
                            <span style="color:#FCA5A5;">■</span> Red chips = negation tokens preserved for polarity detection
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.caption("No alphabetic tokens remained after stopword removal.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Batch Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📋</div>
        <div>
            <h3 style="margin:0;">Batch Sentiment Analysis</h3>
            <p style="margin:0; color:#94A3B8; font-size:0.85rem;">Upload a CSV file with a 'review' column, or paste multiple reviews (one per line).</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    batch_tab_csv, batch_tab_text = st.tabs(["📁 CSV Upload", "📝 Paste Reviews"])

    with batch_tab_csv:
        uploaded_file = st.file_uploader("Upload CSV with a 'review' column", type=['csv'])
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                # Try to find the review column
                review_col = None
                for candidate in ['review', 'Review', 'text', 'Text', 'comment', 'Comment', 'content']:
                    if candidate in df_upload.columns:
                        review_col = candidate
                        break
                if review_col is None:
                    st.error(f"Could not find a review column. Available columns: {list(df_upload.columns)}")
                else:
                    st.info(f"Found **{len(df_upload)}** reviews in column `{review_col}`")
                    if st.button("🚀 Analyze All Reviews", key="batch_csv"):
                        if not check_models_ready():
                            st.error("Trained model not found. Run `python src/train.py` first.")
                        else:
                            with st.spinner(f"Analyzing {len(df_upload)} reviews..."):
                                start = time.perf_counter()
                                texts = df_upload[review_col].astype(str).tolist()
                                results = batch_predict(texts)
                                elapsed = time.perf_counter() - start

                            df_upload['predicted_sentiment'] = [r['label'] for r in results]
                            df_upload['confidence'] = [r['confidence'] for r in results]

                            # Summary stats
                            counts = df_upload['predicted_sentiment'].value_counts()
                            total = len(df_upload)

                            st.markdown(f"""
                            <div class="batch-stats">
                                <div class="batch-stat-card">
                                    <h3 style="color:#10B981;">{counts.get('positive', 0)}</h3>
                                    <p>Positive ({counts.get('positive', 0)/total*100:.1f}%)</p>
                                </div>
                                <div class="batch-stat-card">
                                    <h3 style="color:#F59E0B;">{counts.get('neutral', 0)}</h3>
                                    <p>Neutral ({counts.get('neutral', 0)/total*100:.1f}%)</p>
                                </div>
                                <div class="batch-stat-card">
                                    <h3 style="color:#EF4444;">{counts.get('negative', 0)}</h3>
                                    <p>Negative ({counts.get('negative', 0)/total*100:.1f}%)</p>
                                </div>
                                <div class="batch-stat-card">
                                    <h3 style="color:#60A5FA;">{elapsed:.2f}s</h3>
                                    <p>Processing Time</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            st.dataframe(
                                df_upload[[review_col, 'predicted_sentiment', 'confidence']].style
                                .background_gradient(subset=['confidence'], cmap='YlGn')
                                .format({'confidence': '{:.2%}'}),
                                use_container_width=True,
                                height=400
                            )

                            # Download results
                            csv_data = df_upload.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Results as CSV",
                                csv_data,
                                "sentiment_results.csv",
                                "text/csv",
                                use_container_width=True
                            )
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    with batch_tab_text:
        multi_text = st.text_area(
            "Paste reviews (one per line):",
            height=200,
            placeholder="Great product, love it!\nTerrible quality, broke immediately.\nIt's okay, nothing special."
        )
        if st.button("🚀 Analyze All", key="batch_text"):
            lines = [l.strip() for l in multi_text.strip().split('\n') if l.strip()]
            if not lines:
                st.warning("Please enter at least one review.")
            elif not check_models_ready():
                st.error("Trained model not found. Run `python src/train.py` first.")
            else:
                with st.spinner(f"Analyzing {len(lines)} reviews..."):
                    start = time.perf_counter()
                    results = batch_predict(lines)
                    elapsed = time.perf_counter() - start

                df_results = pd.DataFrame([{
                    'Review': r['text'],
                    'Sentiment': r['label'].capitalize(),
                    'Confidence': r['confidence']
                } for r in results])

                counts = df_results['Sentiment'].value_counts()
                total = len(df_results)

                st.markdown(f"""
                <div class="batch-stats">
                    <div class="batch-stat-card">
                        <h3 style="color:#10B981;">{counts.get('Positive', 0)}</h3>
                        <p>Positive ({counts.get('Positive', 0)/total*100:.1f}%)</p>
                    </div>
                    <div class="batch-stat-card">
                        <h3 style="color:#F59E0B;">{counts.get('Neutral', 0)}</h3>
                        <p>Neutral ({counts.get('Neutral', 0)/total*100:.1f}%)</p>
                    </div>
                    <div class="batch-stat-card">
                        <h3 style="color:#EF4444;">{counts.get('Negative', 0)}</h3>
                        <p>Negative ({counts.get('Negative', 0)/total*100:.1f}%)</p>
                    </div>
                    <div class="batch-stat-card">
                        <h3 style="color:#60A5FA;">{elapsed:.2f}s</h3>
                        <p>Processing Time</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.dataframe(
                    df_results.style
                    .background_gradient(subset=['Confidence'], cmap='YlGn')
                    .format({'Confidence': '{:.2%}'}),
                    use_container_width=True
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Exploratory Data Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <div>
            <h3 style="margin:0;">Exploratory Data Analysis</h3>
            <p style="margin:0; color:#94A3B8; font-size:0.85rem;">Visualizations generated from the 3,000-row customer reviews dataset during pipeline execution.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    eda_p1 = os.path.join(OUTPUTS_DIR, "sentiment_distribution.png")
    eda_p2 = os.path.join(OUTPUTS_DIR, "review_length_distribution.png")
    eda_p3 = os.path.join(OUTPUTS_DIR, "top_words_per_class.png")
    eda_p4 = os.path.join(OUTPUTS_DIR, "wordclouds.png")

    c_eda1, c_eda2 = st.columns(2)
    with c_eda1:
        if os.path.exists(eda_p1):
            st.image(eda_p1, caption="Sentiment Class Balance (1,000 samples per class)", use_container_width=True)
        else:
            st.info("Sentiment distribution plot not yet generated. Run `python src/train.py`.")
    with c_eda2:
        if os.path.exists(eda_p2):
            st.image(eda_p2, caption="Review Word & Character Length Distribution", use_container_width=True)
        else:
            st.info("Review length distribution plot not yet generated.")

    st.markdown("---")
    st.markdown("#### **Top 15 Most Frequent Tokens per Sentiment Class**")
    if os.path.exists(eda_p3):
        st.image(eda_p3, caption="Top Keywords (Unigrams) After Lemmatization & Stopword Filtering", use_container_width=True)

    st.markdown("---")
    st.markdown("#### **Class-Specific Word Clouds**")
    if os.path.exists(eda_p4):
        st.image(eda_p4, caption="Visual Vocabulary of Positive, Neutral, and Negative Reviews", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Model Benchmarks & Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab_metrics:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🏆</div>
        <div>
            <h3 style="margin:0;">Model Benchmarks & Comparison</h3>
            <p style="margin:0; color:#94A3B8; font-size:0.85rem;">Performance evaluation of Multinomial NB, Logistic Regression, and Linear SVM on the stratified 20% test set.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    res_csv = os.path.join(OUTPUTS_DIR, "results.csv")
    if os.path.exists(res_csv):
        df_res = pd.read_csv(res_csv)

        # Styled metric highlights
        metric_cols = [c for c in ['Accuracy', 'Precision', 'Recall', 'F1'] if c in df_res.columns]
        styled_df = df_res.style.highlight_max(
            axis=0,
            subset=metric_cols,
            color='rgba(16, 185, 129, 0.25)'
        ).format({col: '{:.4f}' for col in metric_cols})

        st.dataframe(styled_df, use_container_width=True)

        # Best model callout
        if metric_cols:
            best_row = df_res.sort_values(by='F1', ascending=False).iloc[0]
            st.markdown(f"""
            <div style="background: rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.3);
                        border-radius:12px; padding:1rem 1.5rem; margin:1rem 0;">
                <span style="font-size:1.1rem; font-weight:700; color:#10B981;">
                    🏆 Best Model: {best_row['Model']}
                </span>
                <span style="color:#94A3B8; margin-left:1rem;">
                    Accuracy: <strong>{best_row.get('Accuracy', 0):.4f}</strong> •
                    F1: <strong>{best_row.get('F1', 0):.4f}</strong>
                </span>
            </div>
            """, unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    comp_plot = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    cm_plot = os.path.join(OUTPUTS_DIR, "confusion_matrices.png")

    with col_m1:
        if os.path.exists(comp_plot):
            st.image(comp_plot, caption="Model Performance Metric Comparison", use_container_width=True)
    with col_m2:
        if os.path.exists(cm_plot):
            st.image(cm_plot, caption="Confusion Matrix Heatmaps", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Pipeline Architecture
# ─────────────────────────────────────────────────────────────────────────────
with tab_docs:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📖</div>
        <div>
            <h3 style="margin:0;">Pipeline Architecture & Methodology</h3>
            <p style="margin:0; color:#94A3B8; font-size:0.85rem;">End-to-end system design from raw text to calibrated predictions.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline flow visualization
    st.markdown("""
    <div style="background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
                border-radius:14px; padding:1.5rem; margin:1rem 0; text-align:center;">
        <div style="display:flex; align-items:center; justify-content:center; gap:0.4rem; flex-wrap:wrap; font-size:0.85rem;">
            <span style="background:rgba(96,165,250,0.15); color:#60A5FA; padding:0.4rem 1rem; border-radius:8px; font-weight:600;">📥 Raw Text</span>
            <span style="color:#475569;">→</span>
            <span style="background:rgba(167,139,250,0.15); color:#A78BFA; padding:0.4rem 1rem; border-radius:8px; font-weight:600;">🧹 Clean & Normalize</span>
            <span style="color:#475569;">→</span>
            <span style="background:rgba(244,114,182,0.15); color:#F472B6; padding:0.4rem 1rem; border-radius:8px; font-weight:600;">🔤 Expand Contractions</span>
            <span style="color:#475569;">→</span>
            <span style="background:rgba(251,146,60,0.15); color:#FB923C; padding:0.4rem 1rem; border-radius:8px; font-weight:600;">📝 Lemmatize</span>
            <span style="color:#475569;">→</span>
            <span style="background:rgba(52,211,153,0.15); color:#34D399; padding:0.4rem 1rem; border-radius:8px; font-weight:600;">📊 TF-IDF</span>
            <span style="color:#475569;">→</span>
            <span style="background:rgba(16,185,129,0.15); color:#10B981; padding:0.4rem 1rem; border-radius:8px; font-weight:600; border:1px solid rgba(16,185,129,0.3);">🧠 ML Classify</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("""
        #### **1. Data Engineering & Cleaning**
        - Balanced 3,000-sample corpus (1,000 per class)
        - Null and duplicate record filtering
        - Automated dataset download with synthetic fallback

        #### **2. Negation-Aware NLP Preprocessing**
        - **Cleaning:** Strips URLs, HTML entities, numeric digits, special punctuation
        - **Contraction Handling:** `"didn't"` → `"did not"`, `"won't"` → `"will not"`
        - **Repeated Char Normalization:** `"sooooo"` → `"soo"`
        - **Stopword Filtering:** Removes stopwords while preserving `{'not', 'no', 'never', 'nor', 'neither'}`
        - **Lemmatization:** WordNet lemmatizer normalizes morphological variations
        """)

    with col_d2:
        st.markdown("""
        #### **3. Feature Extraction**
        - **TF-IDF Vectorizer:** Up to 5,000 n-gram features with `ngram_range=(1, 2)`
        - Captures bigrams: `"not good"`, `"highly recommend"`, `"never worked"`

        #### **4. ML & Hyperparameter Optimization**
        - **Stratified 80/20 Split:** Equal class representation guaranteed
        - **Models:** Multinomial NB, Logistic Regression, Linear SVM
        - **Tuning:** 5-fold Stratified CV GridSearchCV with `class_weight='balanced'`

        #### **5. Inference & Calibration**
        - Cached model/vectorizer loading for zero-latency repeat predictions
        - Full class probability distributions via calibrated classifiers
        - Batch prediction with single-pass vectorization for efficiency
        """)
