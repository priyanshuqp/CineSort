"""Plot-to-genre predictor — Streamlit app from Prototype.ipynb."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from prepare import clean_movies  # noqa: E402
from train import train_from_frame  # noqa: E402

MODEL_PATH = ROOT / "models" / "genre_model.joblib"
SEED_CSV = ROOT / "data" / "seed_plots.csv"


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def ensure_model():
    bundle = load_bundle()
    if bundle is not None:
        return bundle
    with st.spinner("Training the starter model…"):
        train_from_frame(pd.read_csv(SEED_CSV))
    load_bundle.clear()
    return load_bundle()


def predict_top(model, text: str, k: int = 3) -> list[tuple[str, float]]:
    probs = model.predict_proba([text])[0]
    order = probs.argsort()[::-1][:k]
    return [(model.classes_[i], float(probs[i])) for i in order]


st.set_page_config(page_title="Movie Genre Predictor", page_icon="🎬", layout="wide")
st.title("Movie genre predictor")
st.caption(
    "Paste a plot synopsis and get the top genres. "
    "Same pipeline as the notebook: TF-IDF (1–2 grams) + logistic regression."
)

bundle = ensure_model()
model = bundle["model"]
metrics = bundle.get("metrics") or {}

with st.sidebar:
    st.header("Model")
    if metrics:
        st.metric("Held-out accuracy", f"{metrics.get('accuracy', 0):.1%}")
        st.write("Genres:", ", ".join(metrics.get("genres", list(model.classes_))))
        st.caption(
            f"Trained on {metrics.get('n_train', '?')} plots, "
            f"tested on {metrics.get('n_test', '?')}."
        )
        counts = metrics.get("counts") or {}
        if counts:
            st.bar_chart(pd.Series(counts, name="plots"))

    st.divider()
    st.subheader("Retrain")
    st.caption(
        "Upload a CSV like the Wikipedia movie plots file used in the notebook "
        "(columns **Plot** and **Genre**). The app keeps the 6 most common genres "
        "after taking the first label before `|` or `/`."
    )
    uploaded = st.file_uploader("Movie plots CSV", type=["csv"])
    if st.button("Train on uploaded file", disabled=uploaded is None):
        try:
            df = pd.read_csv(uploaded)
            preview = clean_movies(df)
            with st.spinner(f"Training on {len(preview)} cleaned rows…"):
                new_metrics = train_from_frame(df)
            load_bundle.clear()
            st.success(f"New accuracy: {new_metrics['accuracy']:.1%}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if st.button("Reset to starter stories"):
        train_from_frame(pd.read_csv(SEED_CSV))
        load_bundle.clear()
        st.rerun()

left, right = st.columns((1.2, 1))

with left:
    plot = st.text_area(
        "Plot",
        height=240,
        placeholder="Two people try convincing their parents for their marriage",
        value="Two people try convincing their parents for their marriage",
    )
    go = st.button("Predict genre", type="primary")

with right:
    st.subheader("Top 3 genres")
    if go:
        text = plot.strip()
        if len(text) < 8:
            st.warning("Write a bit more of the plot so the model has something to read.")
        else:
            for genre, prob in predict_top(model, text):
                st.write(f"**{genre}** — {prob:.1%}")
                st.progress(min(max(prob, 0.0), 1.0))
    else:
        st.info("Enter a plot and click Predict.")

st.divider()
with st.expander("What this app does"):
    st.markdown(
        """
The Colab notebook uploaded a movie-plots CSV, cleaned genres to the six most
common labels, then trained:

`TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")`
plus `LogisticRegression(class_weight="balanced", max_iter=1000)`.

On the Wikipedia plots set that run scored about **61%** accuracy. The starter
model here is trained on a small built-in set so the app runs immediately.
Upload the same CSV you used in Colab to match that experiment.
"""
    )
