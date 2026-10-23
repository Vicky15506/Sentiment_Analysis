# Sentiment Analysis of Customer Reviews using NLP & Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

An end-to-end Natural Language Processing (NLP) and Machine Learning project that classifies customer reviews into **Positive**, **Neutral**, or **Negative** sentiment classes using CPU-optimized traditional machine learning algorithms.

---

## 📁 Project Directory Structure

```text
sentiment-analysis/
├── data/
│   └── customer_reviews.csv          # Balanced 3,000-row customer reviews dataset
├── notebooks/
│   └── sentiment_analysis.ipynb      # Step-by-step interactive Jupyter notebook
├── src/
│   ├── __init__.py
│   ├── preprocess.py                 # Negation-preserving text cleaning & lemmatization
│   ├── train.py                      # Data loading, EDA, training, tuning, and serialization
│   ├── evaluate.py                   # Metrics calculation, confusion matrices & comparison charts
│   └── predict.py                    # Inference function returning predicted class & confidence
├── models/
│   ├── best_model.joblib             # Serialized optimized classifier (Logistic Regression)
│   └── tfidf_vectorizer.joblib       # Serialized TF-IDF feature extractor (5,000 n-grams)
├── outputs/
│   ├── results.csv                   # Model comparison metrics table
│   ├── model_comparison.png          # Grouped bar chart comparing Accuracy, Precision, Recall, F1
│   ├── confusion_matrices.png        # Multi-panel confusion matrix heatmaps
│   ├── sentiment_distribution.png    # Class balance countplot
│   ├── review_length_distribution.png# Word & character length distribution histograms
│   ├── top_words_per_class.png       # Top 15 most frequent tokens per sentiment
│   └── wordclouds.png                # High-resolution word clouds for each sentiment class
├── app.py                            # Interactive Streamlit web application
├── streamlit.cmd                     # Windows Streamlit execution wrapper
├── requirements.txt                  # Python dependencies
└── README.md                         # Comprehensive project documentation
```

---

## ⚙️ Environment Setup & Installation

### 1. Prerequisites
- Python **3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.14 on Windows, Linux, and macOS)
- Standard CPU (no GPU required)

### 2. Install Dependencies
Clone or navigate to the project directory and install the required packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### 1. Train and Compare Models
Executes the full pipeline: loads the dataset, cleans missing/duplicate rows, generates EDA plots, preprocesses text, extracts TF-IDF features, trains baseline algorithms, tunes the best model via 5-fold Stratified GridSearchCV, and saves model artifacts:

```bash
python src/train.py
```

### 2. Run Inference on Test Samples
Tests the serialized model and vectorizer on 6 sample reviews (2 positive, 2 neutral, 2 negative) and prints predicted labels with confidence scores:

```bash
python src/predict.py
```

### 3. Launch the Streamlit Web Application
Starts the modern interactive dashboard with real-time sentiment prediction, confidence bars, token inspection, EDA viewer, and benchmark charts:

```bash
streamlit run app.py
```
*(On Windows systems where `streamlit.exe` is not in the system PATH, run `.\streamlit.cmd run app.py` or `python -m streamlit run app.py`).*

### 4. Interactive Jupyter Notebook
Open and execute the step-by-step notebook:

```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

---

## 📊 Final Model Comparison & Benchmark Results

The models were evaluated on an independent **stratified 20% test set (600 reviews)** with balanced class representation (200 positive, 200 neutral, 200 negative):

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 0.6600 | 0.6652 | 0.6600 | 0.6618 |
| **Linear SVM** | 0.6617 | 0.6601 | 0.6617 | 0.6607 |
| **Logistic Regression (Baseline)** | 0.6700 | 0.6734 | 0.6700 | 0.6712 |
| **Logistic Regression (Tuned + Balanced)** 🏆 | **0.6767** | **0.6787** | **0.6767** | **0.6772** |

### Hyperparameter Tuning Report:
- **Selected Model for Tuning:** Logistic Regression
- **Optimization Strategy:** 5-fold Stratified Cross-Validation (`GridSearchCV`)
- **Tuned Parameters:** `{'C': 0.5, 'class_weight': 'balanced', 'solver': 'lbfgs'}`
- **Baseline Accuracy:** `0.6700` (67.00%)
- **Tuned Accuracy:** `0.6767` (**67.67%**)
- **Improvement:** `+0.0067` (+0.67% absolute test accuracy gain)
- **Tuned Weighted F1:** `0.6772`

---

## 🔍 Stage-by-Stage Methodology Explanation

### 1. Data Ingestion & Class Balancing
- Review ratings are mapped as: **1–2 = negative**, **3 = neutral**, and **4–5 = positive**.
- The corpus is strictly balanced to **1,000 reviews per class** (3,000 reviews total) to prevent majority-class bias.
- Initial validation checks verify missing values, duplicate entries, and class distribution.

### 2. Data Cleaning & Diagnostics
- Null review texts and duplicate reviews are programmatically identified and removed.
- Validates that text lengths are non-zero before passing into NLP tokenization.

### 3. Exploratory Data Analysis (EDA)
Generates high-resolution visualizations saved to `outputs/`:
- **Sentiment Countplot** (`sentiment_distribution.png`): Confirms exact 1:1:1 class balance.
- **Review Length Distribution** (`review_length_distribution.png`): Displays word count and character count density functions across positive, neutral, and negative classes.
- **Top 15 Words Per Class** (`top_words_per_class.png`): Identifies high-frequency discriminative unigrams for each sentiment.
- **Word Clouds** (`wordclouds.png`): Visualizes sentiment-specific vocabularies using tailored colormaps.

### 4. Negation-Preserving NLP Preprocessing
Standard stopword removal often removes words like `"not"` or `"never"`, causing phrases like *"not good"* to collapse into *"good"*, completely flipping the sentiment. Our custom pipeline resolves this:
- **Lowercasing**: Standardizes case sensitivity.
- **URL & Noise Stripping**: Removes links, HTML tags, and numeric digits.
- **Contraction Expansion**: Expands `"didn't"` to `"did not"`, `"can't"` to `"can not"`, preserving negation terms.
- **Negation-Preserving Stopword Filtering**: Removes generic English stopwords while strictly retaining `{'not', 'no', 'never', 'nor', 'neither'}`.
- **Lemmatization**: Uses NLTK's `WordNetLemmatizer` to normalize words to their canonical base forms (e.g., *"broken"*, *"breaks"*, *"breaking"*).

### 5. Feature Extraction (TF-IDF Vectorization)
- `TfidfVectorizer` configured with `max_features=5000` and `ngram_range=(1, 2)`.
- Extracting bigrams alongside unigrams captures critical contextual phrases such as `"not recommend"`, `"never work"`, and `"highly impressed"`.

### 6. Stratified Train/Test Split
- An 80/20 train/test split is applied using `random_state=42` with `stratify=y`.
- Ensures exactly 800 training samples and 200 testing samples per sentiment class.

### 7. Model Benchmarking & Hyperparameter Tuning
- Compares three foundational ML classifiers: **Multinomial Naive Bayes**, **Logistic Regression**, and **Linear Support Vector Classifier (SVM)**.
- The highest-performing model (Logistic Regression) is fine-tuned using `GridSearchCV` with balanced class weights across regularization parameter $C \in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]$.

### 8. Model Serialization & Calibrated Inference
- The optimized classifier and fitted TF-IDF vectorizer are serialized to `models/` using `joblib`.
- In `src/predict.py`, `predict_sentiment(text)` returns the predicted class label along with a calibrated confidence score.
- Full class probability distributions ($P(\text{Positive}), P(\text{Neutral}), P(\text{Negative})$) are provided for downstream user interfaces.

### 9. Interactive Web Dashboard (`app.py`)
- Built with **Streamlit** featuring custom modern CSS, glassmorphism hero banner, and responsive cards.
- Supports instant review classification, quick-test sample prompts, class probability metrics, preprocessed token breakdown, and tabs for EDA and benchmark charts.

---

## 🧪 Sample Inference Verification (`src/predict.py`)

| # | Test Review | Expected Class | Predicted Class | Confidence |
| :-: | :--- | :---: | :---: | :---: |
| 1 | *"I love this product! It is the best purchase I have ever made, highly recommended and fantastic quality."* | Positive | **POSITIVE** | 73.12% |
| 2 | *"Absolutely brilliant purchase! Premium quality, easy setup, and great customer support."* | Positive | **POSITIVE** | 71.26% |
| 3 | *"The package arrived on time. The material is standard and works fine for everyday basic tasks."* | Neutral | **NEUTRAL** | 34.66% |
| 4 | *"It is okay for the price. Not particularly impressive, but it does what it claims to do."* | Neutral | **NEUTRAL** | 55.24% |
| 5 | *"Terrible product! Stopped working after only two days. Customer service refused to refund."* | Negative | **NEGATIVE** | 74.36% |
| 6 | *"Complete waste of money. Poor build quality, defective out of the box, and never worked."* | Negative | **NEGATIVE** | 73.76% |

---

## 📝 5-Line Internship Report Summary

> Developed an end-to-end sentiment classification system for customer reviews using natural language processing and CPU-efficient machine learning. Implemented a negation-aware text preprocessing and TF-IDF n-gram feature extraction pipeline to preserve sentiment polarity. Benchmarked Multinomial Naive Bayes, Linear SVM, and Logistic Regression across a balanced 3,000-review dataset with 3-class rating mapping. Optimized the top-performing Logistic Regression model via 5-fold Stratified GridSearchCV, achieving a peak test accuracy of 67.67% and weighted F1-score of 0.6772. Deployed the serialized models into an interactive Streamlit web dashboard providing real-time sentiment prediction, confidence metrics, and EDA diagnostics.
