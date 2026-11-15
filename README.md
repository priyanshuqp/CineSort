# Movie Genre Predictor

A local web app from `Prototype.ipynb`: paste a movie plot, get the top 3 predicted genres.

The model is the notebook’s pipeline — TF-IDF word and bigram features with a balanced logistic regression classifier. Genres are cleaned the same way (lowercase, first label before `|` or `/`, drop `unknown`, keep the six most common).

## Run

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Then open [http://localhost:8501](http://localhost:8501).

## Train on the Wikipedia plots CSV

The notebook used a file with columns `Plot` and `Genre` (about 35k movies). Upload that CSV in the sidebar, or:

```powershell
.\.venv\Scripts\python.exe src\train.py path\to\wiki_movie_plots.csv
```

Without that file, the app trains a small starter model from `data/seed_plots.csv` so prediction still works.
