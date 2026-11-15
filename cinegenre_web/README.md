# CineGenre – web version (Flask + Render)

## Run locally
    pip install -r requirements.txt
    python app.py          # open http://localhost:5000

## Deploy on Render
1. Create a GitHub repo and upload everything in this folder (including `model/genre_model.joblib`, ~18 MB).
2. Render dashboard -> New + -> Blueprint (reads render.yaml) -> pick the repo -> Apply.
   (Or New + -> Web Service with: Build `pip install -r requirements.txt`,
   Start `gunicorn app:app --workers 1 --threads 2 --timeout 120 --preload`, Health check path `/health`.)
3. After the build finishes you get a public URL like https://cinegenre.onrender.com

## Retrain
    python train_model.py "Genre Classification Dataset"
