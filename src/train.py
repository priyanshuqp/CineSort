"""Train the TF-IDF + logistic regression model from Prototype.ipynb."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from prepare import clean_movies

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=20000,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def train_from_frame(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    cleaned = clean_movies(df)
    if cleaned["genre"].nunique() < 2:
        raise ValueError("Need at least two genres after cleaning.")
    if len(cleaned) < 20:
        raise ValueError("Need at least 20 labeled plots after cleaning.")

    stratify = cleaned["genre"] if cleaned["genre"].value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned["text"],
        cleaned["genre"],
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )
    model = build_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "genres": sorted(cleaned["genre"].unique().tolist()),
        "counts": cleaned["genre"].value_counts().to_dict(),
        "report": report,
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics}, MODELS / "genre_model.joblib")
    return metrics


def train_from_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    return train_from_frame(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        nargs="?",
        default=str(ROOT / "data" / "seed_plots.csv"),
        help="CSV with Plot and Genre columns",
    )
    args = parser.parse_args()
    result = train_from_csv(Path(args.csv))
    print(f"Accuracy: {result['accuracy']:.3f}")
    print("Genres:", ", ".join(result["genres"]))
    print("Saved", MODELS / "genre_model.joblib")
