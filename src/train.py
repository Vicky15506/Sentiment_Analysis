"""
train.py
End-to-end training and evaluation pipeline for Sentiment Analysis.

Tasks:
1. Load dataset (with automatic public download and synthetic fallback)
2. Clean missing and duplicate values
3. Generate and save EDA visualizations to outputs/
4. Preprocess review texts
5. Extract TF-IDF features (max_features=5000, ngram_range=(1,2))
6. Stratified 80/20 train/test split (random_state=42)
7. Train MultinomialNB, LogisticRegression, and LinearSVC
8. Evaluate baseline models, plot confusion matrices & comparison charts, save results.csv
9. Improve the best model with class_weight="balanced" and GridSearchCV, reporting before/after accuracy
10. Save the best model and TF-IDF vectorizer to models/ with joblib
"""

import os
import sys
import random
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Ensure project root is in sys.path for clean module imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text
from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrices,
    plot_model_comparison,
    generate_eda_visualizations
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "customer_reviews.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


def generate_synthetic_reviews(n_per_class=1000):
    """
    Fallback: Generate a realistic 3,000-row synthetic customer reviews dataset.
    """
    print("\n[INFO] Generating a realistic 3,000-row synthetic dataset (fallback)...")
    pos_templates = [
        "Absolutely loved this {product}! Exceeded all my expectations.",
        "High quality {product}, runs smoothly and arrived faster than expected.",
        "Very satisfied with my purchase. Would definitely recommend to anyone looking for a {product}.",
        "Outstanding performance, battery life and build. Worth every single penny!",
        "Five stars! The best {product} I have owned in years.",
        "Super easy to set up and works perfectly. Really happy with this order.",
        "Brilliant design and premium finish. Customer support was also helpful.",
        "Works like a charm! Crisp display and reliable for daily use."
    ]
    neu_templates = [
        "The {product} is okay, neither great nor terrible. Average build.",
        "Decent for the price point, but nothing extraordinary.",
        "It does what it says on the box. Standard performance and basic features.",
        "Fair product. It is acceptable for simple tasks, though delivery was just okay.",
        "Neutral impression so far. Meets basic expectations but has minor quirks.",
        "Neither impressed nor dissatisfied. An ordinary {product} for regular use.",
        "Middle of the road quality. Works as described, nothing special."
    ]
    neg_templates = [
        "Terrible experience. The {product} broke down within two days of use.",
        "Completely useless. Do not waste your money on this cheap {product}!",
        "Very poor build quality and arrived damaged. Customer service never replied.",
        "Extremely disappointed. It does not work as advertised and drains battery immediately.",
        "Horrible purchase! Defective out of the box and impossible to return.",
        "One star. Defective and unreliable. Avoid this {product} at all costs.",
        "Not worth the price. Constantly freezing and failing to start."
    ]
    products = [
        "laptop", "smartphone", "headphones", "wireless mouse", "smartwatch",
        "tablet", "coffee maker", "gaming keyboard", "portable speaker", "charger"
    ]

    records = []
    for _ in range(n_per_class):
        records.append({
            'review': random.choice(pos_templates).format(product=random.choice(products)),
            'sentiment': 'positive'
        })
        records.append({
            'review': random.choice(neu_templates).format(product=random.choice(products)),
            'sentiment': 'neutral'
        })
        records.append({
            'review': random.choice(neg_templates).format(product=random.choice(products)),
            'sentiment': 'negative'
        })

    df = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def load_dataset():
    """
    Load dataset from data/customer_reviews.csv.
    If absent, download real Amazon reviews from Hugging Face or fallback to synthetic.
    """
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    if os.path.exists(DATA_PATH):
        print(f"[1] Loading dataset from existing file: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("[1] Dataset file not found. Downloading public customer reviews dataset...")
        try:
            url = "https://huggingface.co/api/datasets/mteb/amazon_reviews_multi/parquet/en/train/0.parquet"
            raw_df = pd.read_parquet(url)

            def map_rating(label):
                if label in [0, 1]:
                    return 'negative'
                elif label == 2:
                    return 'neutral'
                elif label in [3, 4]:
                    return 'positive'
                return None

            raw_df['sentiment'] = raw_df['label'].apply(map_rating)
            raw_df = raw_df.rename(columns={'text': 'review'})[['review', 'sentiment']].dropna()

            # Downsample to 1000 per class (3000 rows total)
            subsets = [
                raw_df[raw_df['sentiment'] == s].sample(n=1000, random_state=42)
                for s in ['positive', 'negative', 'neutral']
            ]
            df = pd.concat(subsets, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
            df.to_csv(DATA_PATH, index=False)
            print(f"[SUCCESS] Downloaded and balanced 3,000 real customer reviews -> {DATA_PATH}")
        except Exception as err:
            print(f"[WARNING] Public dataset download failed ({err}).")
            df = generate_synthetic_reviews(n_per_class=1000)
            df.to_csv(DATA_PATH, index=False)
            print(f"[INFO] Notice: Using realistic 3,000-row synthetic dataset saved to {DATA_PATH}")

    # Step 1 inspection tasks
    print("\n--- Dataset Initial Summary ---")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Class counts:\n{df['sentiment'].value_counts()}\n")

    return df


def clean_dataset(df):
    """
    Drop missing values and duplicates.
    """
    initial_len = len(df)
    df_cleaned = df.dropna(subset=['review', 'sentiment']).drop_duplicates().reset_index(drop=True)
    dropped = initial_len - len(df_cleaned)
    print(f"[2] Cleaning complete. Dropped {dropped} null/duplicate rows. Remaining: {len(df_cleaned)}")
    return df_cleaned


def run_pipeline():
    """
    Execute full pipeline: Load -> Clean -> EDA -> Preprocess -> Vectorize -> Split -> Train -> Tune -> Save
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 1. Load data
    df = load_dataset()

    # 2. Clean data
    df = clean_dataset(df)

    # 3. Exploratory Data Analysis & Visualizations
    print("[3] Generating and saving EDA visualizations to outputs/...")
    generate_eda_visualizations(df, text_col='review', label_col='sentiment', output_dir=OUTPUTS_DIR)

    # 4. Preprocessing
    print("\n[4] Preprocessing reviews (lowercasing, punctuation/number removal, negations kept, lemmatization)...")
    df['cleaned_review'] = df['review'].apply(clean_text)

    # Filter any empty cleaned reviews
    df = df[df['cleaned_review'].str.strip() != ""].reset_index(drop=True)
    print(f"Preprocessing finished. Active rows: {len(df)}")

    # 5. TF-IDF Feature Extraction
    print("\n[5] Feature Extraction: TF-IDF (max_features=8000, ngram_range=(1,2), sublinear_tf=True)...")
    tfidf = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents='unicode'
    )

    # 6. Stratified 80/20 train/test split
    print("[6] Splitting dataset: Stratified 80% Train, 20% Test (random_state=42)...")
    X = df['cleaned_review']
    y = df['sentiment']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    X_test_tfidf = tfidf.transform(X_test_raw)

    print(f"Train vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"Train shape: {X_train_tfidf.shape}, Test shape: {X_test_tfidf.shape}")

    # 7. Train Baseline Models
    print("\n[7] Training Baseline Models (MultinomialNB, LogisticRegression, LinearSVC)...")
    baseline_models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000))
    }

    results = []
    eval_predictions = {}
    label_order = ['negative', 'neutral', 'positive']

    for name, model in baseline_models.items():
        print(f"  Training {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        metrics = calculate_metrics(y_test, y_pred)
        results.append({
            'Model': name,
            'Accuracy': round(metrics['accuracy'], 4),
            'Precision': round(metrics['precision'], 4),
            'Recall': round(metrics['recall'], 4),
            'F1': round(metrics['f1'], 4)
        })
        eval_predictions[name] = (y_test, y_pred)

        print(f"    {name} -> Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        print("    Classification Report:\n", metrics['report_str'])

    # 8. Evaluate and save baseline comparison
    print("\n[8] Saving baseline comparison table and confusion matrices...")
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(OUTPUTS_DIR, "results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved comparison table to: {results_csv_path}")

    plot_confusion_matrices(
        eval_predictions,
        labels=label_order,
        save_path=os.path.join(OUTPUTS_DIR, "confusion_matrices.png")
    )
    plot_model_comparison(
        results_df,
        save_path=os.path.join(OUTPUTS_DIR, "model_comparison.png")
    )

    # 9. Improve the best model with class_weight='balanced' and GridSearchCV
    best_baseline_row = results_df.sort_values(by='Accuracy', ascending=False).iloc[0]
    best_name = best_baseline_row['Model']
    baseline_acc = best_baseline_row['Accuracy']
    print(f"\n[9] Best Baseline Model: {best_name} with Accuracy = {baseline_acc:.4f}")
    print("Initiating Hyperparameter Optimization with class_weight='balanced' and GridSearchCV...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if 'Logistic Regression' in best_name:
        param_grid = {
            'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'solver': ['lbfgs', 'saga'],
            'class_weight': ['balanced'],
            'penalty': ['l2']
        }
        base_estimator = LogisticRegression(max_iter=2000, random_state=42)
    elif 'Linear SVM' in best_name:
        param_grid = {
            'C': [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            'class_weight': ['balanced'],
            'loss': ['hinge', 'squared_hinge']
        }
        base_estimator = LinearSVC(random_state=42, max_iter=3000)
    else:  # MultinomialNB
        param_grid = {
            'alpha': [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0],
            'fit_prior': [True, False]
        }
        base_estimator = MultinomialNB()

    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_tfidf, y_train)

    best_estimator = grid_search.best_estimator_
    print(f"  Best Parameters from GridSearch: {grid_search.best_params_}")

    # If best estimator is LinearSVC, wrap in CalibratedClassifierCV to enable predict_proba
    if isinstance(best_estimator, LinearSVC):
        final_model = CalibratedClassifierCV(best_estimator)
        final_model.fit(X_train_tfidf, y_train)
    else:
        final_model = best_estimator

    # Evaluate tuned model on test set
    y_tuned_pred = final_model.predict(X_test_tfidf)
    tuned_metrics = calculate_metrics(y_test, y_tuned_pred)
    tuned_acc = tuned_metrics['accuracy']

    print(f"\n==========================================")
    print(f"Model Optimization Report ({best_name}):")
    print(f"  Before Tuning Accuracy : {baseline_acc:.4f}")
    print(f"  After Tuning Accuracy  : {tuned_acc:.4f}")
    print(f"  Delta Accuracy         : {(tuned_acc - baseline_acc):+.4f}")
    print(f"  Tuned Weighted F1      : {tuned_metrics['f1']:.4f}")
    print("==========================================")

    # Update results table to include Tuned Best Model
    tuned_row = {
        'Model': f"{best_name} (Tuned + Balanced)",
        'Accuracy': round(tuned_acc, 4),
        'Precision': round(tuned_metrics['precision'], 4),
        'Recall': round(tuned_metrics['recall'], 4),
        'F1': round(tuned_metrics['f1'], 4)
    }
    updated_results_df = pd.concat([results_df, pd.DataFrame([tuned_row])], ignore_index=True)
    updated_results_df.to_csv(results_csv_path, index=False)
    print(f"Updated final comparison saved to: {results_csv_path}")

    # Re-plot comparison chart with tuned model included
    plot_model_comparison(
        updated_results_df,
        save_path=os.path.join(OUTPUTS_DIR, "model_comparison.png")
    )

    # 10. Save the best model and TF-IDF vectorizer
    model_save_path = os.path.join(MODELS_DIR, "best_model.joblib")
    vectorizer_save_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

    joblib.dump(final_model, model_save_path)
    joblib.dump(tfidf, vectorizer_save_path)
    print(f"\n[10] Saved best model to: {model_save_path}")
    print(f"[10] Saved TF-IDF vectorizer to: {vectorizer_save_path}")

    return updated_results_df, final_model, tfidf


if __name__ == "__main__":
    run_pipeline()
