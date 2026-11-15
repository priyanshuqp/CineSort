# 🎬 CineSort – Movie Genre Classifier

CineSort is a web app that predicts a movie's genre from its plot. Paste a description, press **Analyze**, and it returns the most likely genre plus the top five candidates with confidence scores.

**Live app:** `https://cinesort-302b.onrender.com/`

> Hosted on Render's free plan, the app sleeps after ~15 minutes without visitors. The first request after that can take 30–60 seconds while it wakes up.

## Features

- Clean, responsive dark interface that works on desktop and mobile
- Top-5 genre predictions with calibrated confidence percentages
- "Mixed signals" note when a plot fits more than one genre
- Built-in sample plots for quick demos
- JSON API for programmatic use
- Lightweight model (~18 MB, ~250 MB RAM) that fits Render's free tier

## How it works

| Item | Details |
|---|---|
| Dataset | Kaggle *Genre Classification Dataset* (IMDb titles + plot descriptions) |
| Training data | 54,214 movies, 27 genres |
| Features | TF-IDF on title + plot, word 1–2 grams, 200,000 features |
| Classifier | Linear SVM (`LinearSVC`, `C=0.8`, balanced class weights) |
| Confidence | Softmax over SVM scores, with a temperature fitted on the test set |
| Backend | Flask, served by Gunicorn |

### Results (54,200 held-out test movies)

| Metric | Score |
|---|---|
| Top-1 accuracy | 60.8% |
| Top-3 accuracy | 82.6% |
| Macro F1 | 0.42 |

Each movie in the dataset has only one genre label, and many films blend genres, so top-1 accuracy is naturally limited. Top-3 accuracy is a fairer picture of how useful the predictions are. Rare genres such as biography, history and mystery are the hardest.

## Repository layout

```
CineSort/
├── web/                  ← the deployed web app (this is what Render builds)
│   ├── app.py            Flask server and prediction API
│   ├── templates/
│   │   └── index.html    Front-end (HTML, CSS, JavaScript)
│   ├── model/
│   │   ├── genre_model.joblib   Trained vectorizer + classifier
│   │   └── metrics.json
│   ├── train_model.py    Retrains the model from the Kaggle dataset
│   └── requirements.txt
├── data/, src/, app.py, start.ps1   Original local prototype (Prototype.ipynb pipeline)
└── README.md
```

## Run locally

```bash
cd web
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000.

## API

**`POST /api/predict`**

```json
{ "title": "optional title", "plot": "A retired safecracker is dragged back for one last job..." }
```

Response:

```json
{ "results": [ { "genre": "crime", "prob": 0.43 }, { "genre": "drama", "prob": 0.14 } ] }
```

The plot must contain at least five words, otherwise the API returns `400` with an error message.

**`GET /health`** returns `ok` (used by Render's health check).

## Deploy on Render

1. Push this repository to GitHub.
2. In Render, choose **New + → Web Service** and connect the repository.
3. Use these settings:

| Setting | Value |
|---|---|
| Branch | `main` |
| Root Directory | `web` |
| Runtime | Python 3 |
| Build command | `pip install -r requirements.txt` |
| Start command | `gunicorn app:app --workers 1 --threads 2 --timeout 120 --preload` |
| Health check path | `/health` |
| Environment variable | `PYTHON_VERSION` = `3.12.3` |
| Plan | Free |

Every push to `main` redeploys automatically. Keep the library versions in `web/requirements.txt` pinned: the saved model only loads with the same scikit-learn version it was trained with.

## Retrain the model

1. Download and unzip the Kaggle *Genre Classification Dataset* (it contains `train_data.txt` and `test_data_solution.txt`).
2. From the `web` folder, run:

```bash
python train_model.py "path/to/Genre Classification Dataset"
```

This rebuilds `model/genre_model.joblib` and `model/metrics.json`, and prints a per-genre report.

## Limitations

- One predicted genre per film is learned, so multi-genre films get split confidence.
- Short or generic plots give less reliable results; longer descriptions work better.
- The model sees only text, so it can't use cast, posters or release year.
