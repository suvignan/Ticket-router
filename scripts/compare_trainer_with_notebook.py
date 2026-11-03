"""Proof: the pipeline's TF-IDF + LR vs the notebook's frozen Config B model, on validation.
Run from the project root: python scripts/compare_trainer_with_notebook.py"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.text import clean_text

T, L = "Consumer complaint narrative", "label"
val = pd.read_parquet(ROOT / "data/splits/val.parquet", columns=[T, L])
texts, y = val[T].map(clean_text), val[L]

preds, vocabs = {}, {}
for name, folder, tfidf, logreg in [("pipeline", "artifacts/model", "tfidf.joblib", "logreg.joblib"),
                                    ("notebook", "models", "tfidf_b.joblib", "logreg_b.joblib")]:
    vec, model = joblib.load(ROOT / folder / tfidf), joblib.load(ROOT / folder / logreg)
    vocabs[name] = vec.vocabulary_
    preds[name] = model.predict(vec.transform(texts))
    print(f"{name:9} vocab {len(vec.vocabulary_):,} | val macro-F1 {f1_score(y, preds[name], average='macro'):.4f}")

same = preds["pipeline"] == preds["notebook"]
print(f"Vocabulary identical (same words, same columns): {vocabs['pipeline'] == vocabs['notebook']}")
print(f"Prediction agreement: {same.mean():.4%} | rows that differ: {(~same).sum():,} of {len(same):,}")
if (~same).sum():
    diff = pd.DataFrame({"true": y[~same].values, "notebook": preds["notebook"][~same], "pipeline": preds["pipeline"][~same]})
    print("\nDiffering rows by TRUE class:\n" + diff["true"].value_counts().to_string())
    print("\nWho was right on the differing rows: "
          f"notebook {(diff['notebook'] == diff['true']).sum()} | pipeline {(diff['pipeline'] == diff['true']).sum()}")
