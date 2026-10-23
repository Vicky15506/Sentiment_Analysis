"""
predict.py
Inference module for sentiment classification.

Provides:
- predict_sentiment(text): returns (predicted_label, confidence_score)
- predict_with_probabilities(text): returns predicted_label, confidence, and class probabilities
- batch_predict(texts): efficiently predicts sentiment for multiple texts at once
"""

import os
import sys
import joblib
import numpy as np

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.joblib")

_model = None
_vectorizer = None


def load_artifacts():
    """
    Load saved model and vectorizer from disk (cached).
    """
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Trained model artifacts not found. Please run 'python src/train.py' first.\n"
                f"Looked for:\n  - {MODEL_PATH}\n  - {VECTORIZER_PATH}"
            )
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


def _get_probabilities(model, features):
    """
    Extract class probabilities from a model, handling both predict_proba
    and decision_function interfaces.

    Returns:
        numpy.ndarray: Probability array of shape (n_samples, n_classes)
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(features)
        if decision.ndim == 1:
            decision = decision.reshape(1, -1)
        # Softmax normalization
        exp_d = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        return exp_d / np.sum(exp_d, axis=1, keepdims=True)
    else:
        n_classes = len(model.classes_) if hasattr(model, 'classes_') else 3
        return np.full((features.shape[0], n_classes), 1.0 / n_classes)


def predict_sentiment(text: str):
    """
    Predict sentiment and confidence score for a given raw review string.

    Parameters:
        text (str): Raw customer review text.

    Returns:
        tuple: (label: str, confidence: float)
            where label is 'positive', 'neutral', or 'negative'
            and confidence is a float between 0.0 and 1.0.
    """
    model, vectorizer = load_artifacts()

    cleaned = clean_text(text)
    if not cleaned.strip():
        # Fallback for empty or purely non-alphabetic inputs
        return "neutral", 0.3333

    features = vectorizer.transform([cleaned])
    predicted_label = model.predict(features)[0]

    probs = _get_probabilities(model, features)[0]
    confidence = float(np.max(probs))

    return str(predicted_label), round(confidence, 4)


def predict_with_probabilities(text: str):
    """
    Predict sentiment and return full probability distribution across classes.

    Returns:
        dict: {
            'label': str,
            'confidence': float,
            'cleaned_text': str,
            'probabilities': {class_name: prob}
        }
    """
    model, vectorizer = load_artifacts()
    cleaned = clean_text(text)

    classes = list(getattr(model, "classes_", ['negative', 'neutral', 'positive']))

    if not cleaned.strip():
        uniform = round(1.0 / len(classes), 4)
        return {
            'label': 'neutral',
            'confidence': uniform,
            'cleaned_text': '',
            'probabilities': {cls: uniform for cls in classes}
        }

    features = vectorizer.transform([cleaned])
    predicted_label = str(model.predict(features)[0])

    probs = _get_probabilities(model, features)[0]

    prob_dict = {str(cls): round(float(p), 4) for cls, p in zip(classes, probs)}
    confidence = float(np.max(probs))

    return {
        'label': predicted_label,
        'confidence': round(confidence, 4),
        'cleaned_text': cleaned,
        'probabilities': prob_dict
    }


def batch_predict(texts: list):
    """
    Efficiently predict sentiment for multiple texts in a single vectorization pass.

    Parameters:
        texts (list[str]): List of raw review texts.

    Returns:
        list[dict]: List of prediction results, each containing:
            - 'text': original text
            - 'label': predicted sentiment
            - 'confidence': prediction confidence
            - 'probabilities': class probability dict
    """
    model, vectorizer = load_artifacts()
    classes = list(getattr(model, "classes_", ['negative', 'neutral', 'positive']))

    # Batch clean
    cleaned_texts = [clean_text(t) for t in texts]

    results = []
    # Separate empty and non-empty for efficient vectorization
    non_empty_indices = [i for i, c in enumerate(cleaned_texts) if c.strip()]
    empty_indices = [i for i, c in enumerate(cleaned_texts) if not c.strip()]

    # Initialize results placeholder
    for _ in texts:
        results.append(None)

    # Handle empty texts
    uniform = round(1.0 / len(classes), 4)
    for idx in empty_indices:
        results[idx] = {
            'text': texts[idx],
            'label': 'neutral',
            'confidence': uniform,
            'probabilities': {cls: uniform for cls in classes}
        }

    # Batch predict non-empty texts
    if non_empty_indices:
        batch_cleaned = [cleaned_texts[i] for i in non_empty_indices]
        features = vectorizer.transform(batch_cleaned)
        predictions = model.predict(features)
        probs_all = _get_probabilities(model, features)

        for batch_idx, orig_idx in enumerate(non_empty_indices):
            probs = probs_all[batch_idx]
            prob_dict = {str(cls): round(float(p), 4) for cls, p in zip(classes, probs)}
            results[orig_idx] = {
                'text': texts[orig_idx],
                'label': str(predictions[batch_idx]),
                'confidence': round(float(np.max(probs)), 4),
                'probabilities': prob_dict
            }

    return results


if __name__ == "__main__":
    print("=============================================================")
    print("Testing Sentiment Inference on 6 Sample Reviews (2 per class)")
    print("=============================================================\n")

    sample_reviews = [
        # 2 Positive
        ("I love this product! It is the best purchase I have ever made, highly recommended and fantastic quality.", "positive"),
        ("Absolutely brilliant purchase! Premium quality, easy setup, and great customer support.", "positive"),
        # 2 Neutral
        ("The package arrived on time. The material is standard and works fine for everyday basic tasks.", "neutral"),
        ("It is okay for the price. Not particularly impressive, but it does what it claims to do.", "neutral"),
        # 2 Negative
        ("Terrible product! Stopped working after only two days. Customer service refused to refund.", "negative"),
        ("Complete waste of money. Poor build quality, defective out of the box, and never worked.", "negative")
    ]

    for idx, (rev, expected) in enumerate(sample_reviews, 1):
        label, conf = predict_sentiment(rev)
        result = predict_with_probabilities(rev)
        print(f"Review #{idx} [Expected: {expected.upper()}]:")
        print(f"  Text       : \"{rev}\"")
        print(f"  Cleaned    : \"{result['cleaned_text']}\"")
        print(f"  Predicted  : {label.upper()} (Confidence: {conf * 100:.2f}%)")
        print(f"  Probs      : {result['probabilities']}")
        print("-" * 60)
