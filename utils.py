"""
utils.py - Shared Utility Functions
Common helpers for text preprocessing and response formatting.
"""

import re
import string


def preprocess_text(text: str) -> str:
    """
    Clean and normalize raw SMS text for model input.
    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove phone numbers
        4. Remove punctuation & special chars
        5. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    # Remove phone numbers
    text = re.sub(r"\b\d{10,}\b", " phonenumber ", text)
    # Remove currency symbols then keep alphanumerics/spaces
    text = re.sub(r"[£$€¥]", " money ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_success_response(
    text: str,
    prediction: str,
    confidence: float,
    explanation: dict,
) -> dict:
    """Standardized JSON-serialisable success response."""
    return {
        "status": "success",
        "input_text": text,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "explanation": explanation,
    }


def build_error_response(message: str) -> dict:
    """Standardized JSON-serialisable error response."""
    return {
        "status": "error",
        "message": message,
    }
