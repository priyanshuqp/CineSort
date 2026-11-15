"""Train the movie genre classifier and save it to model/genre_model.joblib"""
import re, sys, time, joblib
import numpy as np
from pathlib import Path
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "Genre Classification Dataset")
OUT = Path("model"); OUT.mkdir(exist_ok=True)

def load(path, cols):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split(" ::: ")
            if len(p) == len(cols):
                rows.append(dict(zip(cols, p)))
    return rows

def clean(t):
    t = t.lower()
    return re.sub(r"\s+", " ", t).strip()

tr = load(DATA / "train_data.txt", ["id", "title", "genre", "desc"])
te = load(DATA / "test_data_solution.txt", ["id", "title", "genre", "desc"])
Xtr = [clean(r["title"] + " . " + r["desc"]) for r in tr]; ytr = [r["genre"].strip() for r in tr]
Xte = [clean(r["title"] + " . " + r["desc"]) for r in te]; yte = [r["genre"].strip() for r in te]
print(len(Xtr), len(Xte), len(set(ytr)), "genres")

t0 = time.time()
word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True, strip_accents="unicode", max_features=200000, dtype=np.float32)
Aw = word.fit_transform(Xtr); Bw = word.transform(Xte)
print("tfidf", Aw.shape, round(time.time() - t0), "s")

clf = LinearSVC(C=0.8, class_weight="balanced")
clf.fit(Aw, ytr)
pred = clf.predict(Bw)
print("acc", accuracy_score(yte, pred), "macroF1", f1_score(yte, pred, average="macro"))
dec = clf.decision_function(Bw)
top3 = np.argsort(-dec, axis=1)[:, :3]
print("top-3 acc", np.mean([yte[i] in clf.classes_[top3[i]] for i in range(len(yte))]))
print(round(time.time() - t0), "s total")

import json
metrics = {"accuracy": float(accuracy_score(yte, pred)), "macro_f1": float(f1_score(yte, pred, average="macro")),
           "top3_accuracy": float(np.mean([yte[i] in clf.classes_[top3[i]] for i in range(len(yte))])),
           "train_size": len(Xtr), "test_size": len(Xte), "genres": len(clf.classes_)}
print(classification_report(yte, pred, digits=3))
# --- slim the model for small servers (e.g. Render free tier) ---
word.stop_words_ = None
clf.coef_ = clf.coef_.astype(np.float32); clf.intercept_ = clf.intercept_.astype(np.float32)
# --- calibrate softmax temperature on the test set (for confidence %) ---
yi = np.array([list(clf.classes_).index(g) for g in yte]); best = None
for T in [2, 3, 4, 5, 6, 7, 8, 10]:
    z = dec * T; z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    ll = -np.mean(np.log(p[np.arange(len(yi)), yi] + 1e-12))
    if best is None or ll < best[1]: best = (T, ll)
joblib.dump({"vectorizer": word, "model": clf, "classes": list(clf.classes_), "metrics": metrics, "temperature": best[0]}, OUT / "genre_model.joblib", compress=3)
json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2)
print("temperature", best[0], "| saved", (OUT / "genre_model.joblib").stat().st_size // 1024, "KB")
