# 🎬 CineSort – Movie Genre Classifier

CineSort predicts a movie's genre from its plot using a TF-IDF + Linear SVM model trained on 54,214 movies (27 genres).
Test set (54,200 movies): top-1 accuracy 60.8%, top-3 accuracy 82.6%.

## Run locally
    pip install -r requirements.txt
    python app.py          # open http://localhost:5000

## Deploy on Render
1. Render dashboard -> New + -> Blueprint -> select this repo -> Apply (settings come from `render.yaml`).
2. Or New + -> Web Service: build `pip install -r requirements.txt`,
   start `gunicorn app:app --workers 1 --threads 2 --timeout 120 --preload`, health check path `/health`.

## Retrain
Download the Kaggle "Genre Classification Dataset", then:
    python train_model.py "Genre Classification Dataset"
