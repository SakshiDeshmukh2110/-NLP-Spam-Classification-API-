"""
app.py - API Layer (Flask)
Exposes a /predict endpoint that accepts text and returns
spam/ham prediction + explainability output.
"""

from flask import Flask, request, jsonify
from model import load_model, predict_single
from explain import get_explanation
from utils import build_success_response, build_error_response

# ── Initialise app & load model once at startup ────────────────────────────
app = Flask(__name__)

print("[*] Loading model artifacts...")
MODEL, VECTORIZER = load_model()
print("[✓] Model ready.")


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Health check / welcome endpoint."""
    return jsonify({
        "service": "NLP Spam Classifier with Explainability",
        "status": "running",
        "endpoints": {
            "POST /predict": "Classify text as spam or ham with explanation",
            "GET  /health":  "Service health check",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    """Liveness probe (useful for Docker HEALTHCHECK)."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether a text message is spam or ham.

    Request body (JSON):
        {
            "text": "Congratulations! You won a free prize. Call now!",
            "method": "feature_importance"   // optional: "lime" | "feature_importance"
            "top_n": 10                       // optional: number of explanation features
        }

    Response (JSON):
        {
            "status": "success",
            "input_text": "...",
            "prediction": "spam",
            "confidence": 0.9821,
            "explanation": { ... }
        }
    """
    data = request.get_json(silent=True)

    # ── Input validation ────────────────────────────────────────────────────
    if not data:
        return jsonify(build_error_response("Request body must be valid JSON.")), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify(build_error_response("'text' field is required and must not be empty.")), 400

    if len(text) > 5000:
        return jsonify(build_error_response("'text' exceeds maximum length of 5000 characters.")), 400

    method = data.get("method", "feature_importance")
    if method not in ("feature_importance", "lime"):
        return jsonify(build_error_response(
            "Invalid 'method'. Choose 'feature_importance' or 'lime'."
        )), 400

    top_n = data.get("top_n", 10)
    if not isinstance(top_n, int) or not (1 <= top_n <= 30):
        top_n = 10  # silently reset to default

    # ── Prediction ──────────────────────────────────────────────────────────
    try:
        label, confidence, _, vectorizer, model = predict_single(
            text, model=MODEL, vectorizer=VECTORIZER
        )
    except Exception as e:
        return jsonify(build_error_response(f"Prediction error: {str(e)}")), 500

    # ── Explanation ─────────────────────────────────────────────────────────
    try:
        explanation = get_explanation(
            text,
            model=MODEL,
            vectorizer=VECTORIZER,
            method=method,
            top_n=top_n,
        )
    except Exception as e:
        explanation = {"error": f"Explanation failed: {str(e)}"}

    # ── Response ────────────────────────────────────────────────────────────
    response = build_success_response(text, label, confidence, explanation)
    return jsonify(response), 200


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
