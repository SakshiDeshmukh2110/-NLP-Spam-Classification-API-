# 📩 NLP Spam Classification API 

# 🚀 Overview

This project implements an end-to-end NLP system that classifies text messages as Spam or Ham (Not Spam) and provides human-understandable explanations for each prediction.

The system is deployed as a Flask REST API and fully containerized using Docker.

# 🧠 Features

Text classification using TF-IDF + Logistic Regression
REST API using Flask
Explainability using:
LIME (local explanations)
SHAP (feature attribution)
Global feature importance
Modular and clean system design
Fully containerized with Docker

# 🏗️ Project Structure
```
nlp-spam-classifier/
│
├── app.py # Flask API
├── model.py # Training + prediction logic
├── explain.py # Explainability (LIME + SHAP)
├── train.py # Training script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── artifacts/ # Saved model + metrics
├── data/ # Dataset
└── tests/ # Test cases
```

# ⚙️ Installation & Setup

1. Clone Repository
   
```
git clone https://github.com/<your-username>/nlp-spam-classifier.git
cd nlp-spam-classifier
```

# Build Docker Image

```
 docker build -t nlp-app .
```

# Run Container
```
docker run -p 5000:5000 nlp-app
```

# API Usage
```
http://localhost:5000
```

# Prediction Endpoint

POST/predict INPUT
```

  "text": "Win a free lottery now!!!",
  "explain": "lime",
  "num_features": 5
}
```

# OUTPUT
```

  "text": "Win a free lottery now!!!",
  "prediction": {
    "label": "spam",
    "prediction": 1,
    "confidence": 0.97
  },
  "explanation": {
    "method": "LIME",
    "top_features": [
      {"word": "free", "weight": 0.32},
      {"word": "win", "weight": 0.28}
    ]
  },
  "latency_ms": 42
}
```

# Explainability

This project implements multiple explainability techniques:

LIME Explains individual predictions Shows most important words influencing output
SHAP (LinearExplainer) Provides exact feature contribution (Shapley values) Theoretically grounded explanation method
Global Feature Importance Based on Logistic Regression coefficients Shows top spam/ham indicator words

# 🐳 Docker Commands
```
docker build -t nlp-app .
docker run -p 5000:5000 nlp-app
```

# 🧪 Testing

``` pytest tests/ ``` Includes:


API testing Model testing Explainability validation

# 📊 Model Performance

Accuracy: ~98% ROC-AUC: ~0.99 High precision and recall

# ✅ Acceptance Criteria Met

✔️ Spam vs Ham classification
✔️ Functional API
✔️ Explainability implemented (LIME + SHAP)
✔️ Docker containerized
✔️ Modular system design
✔️ Clean and readable code

# 🎯 Future Improvements

Add UI for visual explanations Deploy on cloud (AWS / Render) Add logging & monitoring Upgrade to transformer-based models

# Submitted by:

Sakshi Deshmukh
