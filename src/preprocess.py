"""
preprocess.py
Text preprocessing pipeline for Sentiment Analysis.

Performs:
- Lowercasing
- Removal of URLs, HTML tags, special symbols, and numeric digits
- Expansion of contractions with negation preservation (e.g. don't -> do not)
- Repeated character normalization (e.g. "sooooo" -> "soo")
- Tokenization
- Stopword removal with explicit preservation of negations ('not', 'no', 'never', 'nor', 'neither')
- WordNet lemmatization
"""

import re
import nltk
from functools import lru_cache
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are available
for resource in ['stopwords', 'wordnet', 'punkt', 'punkt_tab', 'omw-1.4']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# Initialize WordNet lemmatizer
_lemmatizer = WordNetLemmatizer()

# Define English stopwords while explicitly preserving negation words
_standard_stopwords = set(stopwords.words('english'))
_negation_words = {'not', 'no', 'never', 'nor', 'neither'}
CUSTOM_STOPWORDS = frozenset(_standard_stopwords - _negation_words)

# Contraction mappings to prevent loss of negation tokens
_CONTRACTIONS = {
    r"won\'t\b": "will not",
    r"can\'t\b": "can not",
    r"n\'t\b": " not",
    r"\'re\b": " are",
    r"\'s\b": " is",
    r"\'d\b": " would",
    r"\'ll\b": " will",
    r"\'ve\b": " have",
    r"\'m\b": " am"
}

# Pre-compile all regex patterns for performance (avoid re-compilation on every call)
_RE_URL = re.compile(r'https?://\S+|www\.\S+')
_RE_HTML_TAGS = re.compile(r'<.*?>')
_RE_HTML_ENTITIES = re.compile(r'&\w+;')
_RE_NON_ALPHA = re.compile(r'[^a-zA-Z\s]')
_RE_MULTI_SPACE = re.compile(r'\s+')
_RE_REPEATED_CHARS = re.compile(r'(.)\1{2,}')
_COMPILED_CONTRACTIONS = [(re.compile(p), r) for p, r in _CONTRACTIONS.items()]


@lru_cache(maxsize=4096)
def _lemmatize_word(word: str) -> str:
    """Cache-backed lemmatization for repeated tokens."""
    return _lemmatizer.lemmatize(word)


def clean_text(text: str) -> str:
    """
    Clean and preprocess a raw review text string.

    Parameters:
        text (str): Raw customer review text.

    Returns:
        str: Cleaned, lemmatized review string without stopwords (negations kept).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    cleaned = text.lower()

    # 2. Remove URLs and web links
    cleaned = _RE_URL.sub(' ', cleaned)

    # 3. Remove HTML tags and encoded entities
    cleaned = _RE_HTML_TAGS.sub(' ', cleaned)
    cleaned = _RE_HTML_ENTITIES.sub(' ', cleaned)

    # 4. Expand contractions to keep negations intact
    for pattern, replacement in _COMPILED_CONTRACTIONS:
        cleaned = pattern.sub(replacement, cleaned)

    # 5. Remove punctuation, symbols, and numeric digits (retain only alphabetical chars & spaces)
    cleaned = _RE_NON_ALPHA.sub(' ', cleaned)

    # 6. Normalize repeated characters (e.g. "sooooo" -> "soo", "amazinggg" -> "amazingg")
    cleaned = _RE_REPEATED_CHARS.sub(r'\1\1', cleaned)

    # 7. Collapse multiple spaces
    cleaned = _RE_MULTI_SPACE.sub(' ', cleaned).strip()

    # 8. Tokenize by whitespace
    tokens = cleaned.split()

    # 9. Remove stopwords (keeping negations) and lemmatize tokens
    processed_tokens = [
        _lemmatize_word(token)
        for token in tokens
        if token not in CUSTOM_STOPWORDS and len(token) > 1
    ]

    return " ".join(processed_tokens)


def preprocess_dataframe(df, text_col='review', out_col='cleaned_review'):
    """
    Apply clean_text to a pandas DataFrame column.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        text_col (str): Column containing raw text.
        out_col (str): Destination column name.

    Returns:
        pd.DataFrame: Dataframe with cleaned text column added.
    """
    df_copy = df.copy()
    df_copy[out_col] = df_copy[text_col].astype(str).apply(clean_text)
    return df_copy


if __name__ == "__main__":
    sample_text = (
        "I didn't like this product at all! 100% waste of money, never worked. "
        "Visit https://example.com for more info."
    )
    print("Original:", sample_text)
    print("Cleaned: ", clean_text(sample_text))
