"""
explain.py - Explainability Module
Provides two explanation methods:
  1. Feature Importance  - top TF-IDF weights for predicted class
  2. LIME               - Local Interpretable Model-Agnostic Explanations
"""

import numpy as np
from typing import Optional
from utils import preprocess_text


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: Feature Importance (fast, no extra dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def explain_feature_importance(
    text: str,
    model,
    vectorizer,
    top_n: int = 10,
) -> dict:
    """
    Identify the top-N words that most influenced the model's decision
    by combining TF-IDF scores with Logistic Regression coefficients.

    Returns:
        dict with keys:
            method       - "feature_importance"
            top_words    - list of {word, tfidf_score, lr_coefficient, impact_score}
            description  - human-readable summary
    """
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])

    # Non-zero feature indices in this document
    feature_names = np.array(vectorizer.get_feature_names_out())
    nonzero_idx = vec.nonzero()[1]

    if len(nonzero_idx) == 0:
        return {
            "method": "feature_importance",
            "top_words": [],
            "description": "No recognizable tokens found in the input text.",
        }

    # For spam class (index 1)
    spam_class_idx = list(model.classes_).index(1)
    coefs = model.coef_[0]  # shape: (n_features,)

    tfidf_scores = np.array(vec[0, nonzero_idx].todense()).flatten()
    lr_coefs = coefs[nonzero_idx]

    # Impact = TF-IDF weight × LR coefficient (signed)
    impact_scores = tfidf_scores * lr_coefs

    # Sort by absolute impact, descending
    sorted_idx = np.argsort(np.abs(impact_scores))[::-1][:top_n]

    top_words = []
    for i in sorted_idx:
        word = feature_names[nonzero_idx[i]]
        top_words.append({
            "word": word,
            "tfidf_score": round(float(tfidf_scores[i]), 4),
            "lr_coefficient": round(float(lr_coefs[i]), 4),
            "impact_score": round(float(impact_scores[i]), 4),
            "pushes_toward": "spam" if lr_coefs[i] > 0 else "ham",
        })

    pred_label = "spam" if model.predict(vec)[0] == 1 else "ham"
    description = (
        f"Top {len(top_words)} words that most influenced the '{pred_label}' prediction. "
        f"Positive impact_score → pushes toward SPAM; negative → pushes toward HAM."
    )

    return {
        "method": "feature_importance",
        "top_words": top_words,
        "description": description,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: LIME (rich local explanation)
# ─────────────────────────────────────────────────────────────────────────────

def explain_lime(
    text: str,
    model,
    vectorizer,
    top_n: int = 10,
    num_samples: int = 500,
) -> dict:
    """
    Use LIME to explain a single prediction with local linear approximation.

    Falls back to feature_importance if lime is not installed.

    Returns:
        dict with keys:
            method          - "lime"
            top_features    - list of {word, weight, label}
            description     - human-readable summary
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        # Graceful fallback
        fallback = explain_feature_importance(text, model, vectorizer, top_n)
        fallback["description"] = (
            "[LIME not installed – falling back to Feature Importance] " +
            fallback["description"]
        )
        return fallback

    class_names = ["ham", "spam"]

    def predict_proba_fn(texts):
        """Wrapper so LIME can call the sklearn pipeline."""
        cleaned = [preprocess_text(t) for t in texts]
        vecs = vectorizer.transform(cleaned)
        return model.predict_proba(vecs)

    explainer = LimeTextExplainer(class_names=class_names, random_state=42)
    exp = explainer.explain_instance(
        text,
        predict_proba_fn,
        num_features=top_n,
        num_samples=num_samples,
    )

    pred_label = class_names[model.predict(vectorizer.transform([preprocess_text(text)]))[0]]
    label_idx = class_names.index(pred_label)

    top_features = [
        {
            "word": word,
            "weight": round(weight, 4),
            "label": pred_label,
            "pushes_toward": pred_label if weight > 0 else ("spam" if pred_label == "ham" else "ham"),
        }
        for word, weight in exp.as_list(label=label_idx)
    ]

    description = (
        f"LIME explanation for '{pred_label}' prediction. "
        f"Positive weight → supports '{pred_label}'; negative → opposes it."
    )

    return {
        "method": "lime",
        "top_features": top_features,
        "description": description,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_explanation(
    text: str,
    model,
    vectorizer,
    method: str = "feature_importance",
    top_n: int = 10,
) -> dict:
    """
    Dispatcher – choose explanation method at runtime.

    Args:
        text      - raw input text
        model     - trained sklearn classifier
        vectorizer- fitted TfidfVectorizer
        method    - "feature_importance" | "lime"
        top_n     - number of top features to return
    """
    method = method.lower().strip()

    if method == "lime":
        return explain_lime(text, model, vectorizer, top_n=top_n)
    else:
        return explain_feature_importance(text, model, vectorizer, top_n=top_n)
