"""CineSort web app (Flask). Run locally: python app.py  |  Production: gunicorn app:app"""
import os
import re
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).parent
bundle = joblib.load(BASE / "model" / "genre_model.joblib")
VEC, MODEL, CLASSES = bundle["vectorizer"], bundle["model"], bundle["classes"]
TEMP = bundle.get("temperature", 6)
METRICS = bundle["metrics"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # reject huge requests


def clean(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def predict(title, plot, k=5):
    x = VEC.transform([clean(f"{title} . {plot}")])
    d = MODEL.decision_function(x)[0] * TEMP
    d -= d.max()
    p = np.exp(d)
    p /= p.sum()
    return [{"genre": CLASSES[i], "prob": float(p[i])} for i in np.argsort(-p)[:k]]


@app.get("/")
def index():
    return render_template("index.html", metrics=METRICS)


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    title = str(data.get("title", ""))[:200]
    plot = str(data.get("plot", ""))[:5000]
    if len(plot.split()) < 5:
        return jsonify(error="Please enter at least a few sentences of plot description."), 400
    return jsonify(results=predict(title, plot))


@app.get("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
