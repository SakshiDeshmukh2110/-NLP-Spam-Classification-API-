"""
model.py - Training Logic for NLP Spam Classifier
Handles data loading, preprocessing, model training, and saving.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils import preprocess_text

# Paths
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "vectorizer.pkl")
DATA_PATH = os.path.join("data", "spam.csv")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and prepare the SMS spam dataset."""
    df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["label_enc"] = df["label"].map({"ham": 0, "spam": 1})
    df["clean_text"] = df["text"].apply(preprocess_text)
    return df


def train(data_path: str = DATA_PATH):
    """Train TF-IDF + Logistic Regression classifier and save artifacts."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("[*] Loading data...")
    df = load_data(data_path)
    print(f"    Total samples: {len(df)} | Spam: {df['label_enc'].sum()} | Ham: {(df['label_enc'] == 0).sum()}")

    X = df["clean_text"]
    y = df["label_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[*] Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{2,}",
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("[*] Training Logistic Regression model...")
    model = LogisticRegression(
        C=5.0,
        max_iter=300,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[✓] Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # Save artifacts
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"[✓] Model saved to: {MODEL_PATH}")
    print(f"[✓] Vectorizer saved to: {VECTORIZER_PATH}")
    return model, vectorizer


def load_model():
    """Load saved model and vectorizer from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Model artifacts not found. Run `python model.py` to train first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_single(text: str, model=None, vectorizer=None):
    """
    Predict spam/ham for a single text input.
    Returns (label, probability, vectorized_input).
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    label = "spam" if pred == 1 else "ham"
    confidence = float(prob[pred])

    return label, confidence, vec, vectorizer, model


if __name__ == "__main__":
    train()
