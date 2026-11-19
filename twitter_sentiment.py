"""Twitter entity-sentiment classifier (from ``twitter.ipynb``).

Pipeline: clean tweet text (NLTK) -> Keras Tokenizer -> padded sequences ->
Embedding + LSTM network -> 4 classes (Negative, Positive, Neutral, Irrelevant).

Train and save the model:   python twitter_sentiment.py
Artifacts are written to ``models/`` and reused by ``app.py``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "twitter.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_lstm.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

LABEL_MAP = {"Negative": 0, "Positive": 1, "Neutral": 2, "Irrelevant": 3}
LABELS = list(LABEL_MAP)
MAX_VOCAB = 20000
RANDOM_STATE = 42


# ---------------------------------------------------------------- preprocessing
def ensure_nltk_data() -> None:
    import nltk

    for resource, path in [("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords"),
                           ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def _nlp_tools():
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    ensure_nltk_data()
    return set(stopwords.words("english")), WordNetLemmatizer()


def process_text(text: str) -> list[str]:
    """Clean a tweet into a list of unique lemmatised tokens (notebook logic)."""
    from nltk.tokenize import word_tokenize

    stop_words, lemm = _nlp_tools()
    # don't  don’t
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8", "ignore")

    text = text.lower()
    # Fixed: the notebook pattern was "https?//" (missing ':'), so URLs were kept.
    text = re.sub(r"https?://\S+|www\.\S+|\b\w+\.(com|ly|net|org|eg)\S*|\b\w+\.\w+\.(com|ly|net|org|eg)\S*", " ", text)
    text = re.sub(r"@\w+", " ", text)
    # Fixed: the notebook used "\1" (not raw) which replaced hashtags with '\x01'.
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    words = word_tokenize(text)
    words = [lemm.lemmatize(word) for word in words if word not in stop_words]
    words = [word for word in words if len(word) > 2]

    # keep the first occurrence of each word, in order
    return list(dict.fromkeys(words))


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna().drop_duplicates()
    df = df[df["labels"].isin(LABEL_MAP)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- model
def build_model(vocab_size: int, maxlen: int):
    from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
    from tensorflow.keras.models import Sequential

    return Sequential([
        Input(shape=(maxlen,)),
        # mask_zero=True: sequences are post-padded, so without masking the LSTM's
        # final state is computed over padding zeros and the model cannot learn.
        Embedding(vocab_size, 100, mask_zero=True),
        LSTM(100),
        Dense(32, activation="relu"),
        Dense(len(LABELS), activation="softmax"),
    ])


def train(epochs: int = 10) -> dict:
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    df = load_data()
    print(f"Cleaning {len(df)} tweets...")
    x = df["text"].apply(process_text)
    y = df["labels"].map(LABEL_MAP)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)

    tok = Tokenizer(num_words=MAX_VOCAB)
    tok.fit_on_texts(x_train)
    # Tokenizer(num_words=MAX_VOCAB) only emits indices < MAX_VOCAB
    vocab_size = min(len(tok.word_index) + 1, MAX_VOCAB)
    seq_train = tok.texts_to_sequences(x_train)
    seq_test = tok.texts_to_sequences(x_test)
    maxlen = max(len(s) for s in seq_train)
    x_train_pad = pad_sequences(seq_train, padding="post", truncating="post", maxlen=maxlen)
    x_test_pad = pad_sequences(seq_test, padding="post", truncating="post", maxlen=maxlen)

    model = build_model(vocab_size, maxlen)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        x_train_pad, y_train, epochs=epochs, validation_split=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
    )

    y_pred = model.predict(x_test_pad).argmax(axis=1)
    metadata = {
        "maxlen": int(maxlen),
        "vocab_size": int(vocab_size),
        "labels": LABELS,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, target_names=LABELS, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "label_counts": df["labels"].value_counts().to_dict(),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_PATH, include_optimizer=False)
    TOKENIZER_PATH.write_text(tok.to_json(), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Test accuracy: {metadata['test_accuracy']:.4f}. Saved artifacts to {MODEL_DIR}/")
    return metadata


# ---------------------------------------------------------------- inference
def load_artifacts():
    """Return ``(model, tokenizer, metadata)``; raises FileNotFoundError if not trained."""
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    for p in (MODEL_PATH, TOKENIZER_PATH, METADATA_PATH):
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found. Train first with: python twitter_sentiment.py")
    model = load_model(MODEL_PATH)
    tok = tokenizer_from_json(TOKENIZER_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, tok, metadata


def predict(texts: list[str], model, tok, maxlen: int) -> tuple[np.ndarray, list[list[str]]]:
    """Return class probabilities (n, 4) and the cleaned tokens for each text."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokens = [process_text(t) for t in texts]
    seqs = pad_sequences(tok.texts_to_sequences(tokens), padding="post", truncating="post", maxlen=maxlen)
    return model.predict(seqs, verbose=0), tokens


if __name__ == "__main__":
    train()
