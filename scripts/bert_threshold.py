"""Choose BERT's review threshold on VALIDATION (rule written first in docs/api.md: match LR's review share at 0.6).
Needs predictions/bert_val_proba.npy from scripts/bert_val_proba.py.
    python scripts/bert_threshold.py"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.text import clean_text
from src.utils import read_yaml

val = pd.read_parquet(ROOT / "data/splits/val.parquet", columns=["Consumer complaint narrative", "label"])
labels = np.array(read_yaml(ROOT / "config/schema.yaml")["labels"])
y = val["label"].to_numpy()

lr = joblib.load(ROOT / "models/logreg_b.joblib")
assert list(lr.classes_) == list(labels)
lr_p = lr.predict_proba(joblib.load(ROOT / "models/tfidf_b.joblib").transform(val.iloc[:, 0].map(clean_text)))
bert_p = np.load(ROOT / "predictions/bert_val_proba.npy")
assert bert_p.shape == lr_p.shape


def row(name, p, t, y=y):
    conf, wrong = p.max(axis=1), labels[p.argmax(axis=1)] != y
    review = conf < t
    rev, caught = review.mean() * 100, (review & wrong).sum() / wrong.sum() * 100
    auto_acc = (~wrong[~review]).mean() * 100
    misrouted = (~review & wrong).mean() * 1000                     # wrong AND auto-routed: reaches the wrong team
    print(f"{name:<5}{t:>6.2f}{rev:>9.1f}%{caught:>10.1f}%{caught / rev:>9.2f}{auto_acc:>10.1f}%{misrouted:>11.1f}")
    return rev


print(f"{'model':<5}{'thr':>6}{'review':>10}{'caught':>11}{'per 1%':>9}{'auto acc':>11}{'bad/1000':>11}")
lr_share = row("LR", lr_p, 0.6)
for t in (0.5, 0.6, 0.7, 0.8, 0.9):
    row("BERT", bert_p, t)
chosen = round(float(np.quantile(bert_p.max(axis=1), lr_share / 100)), 2)
print(f"rule: BERT threshold matching LR's {lr_share:.1f}% review share = {chosen}")
row("BERT", bert_p, chosen)

# REPORT ONLY, the threshold is already fixed: how the validation-chosen thresholds behave on the Day 13 test set
y_test = labels[np.load(ROOT / "predictions/y_test.npy")]
print("\nTEST (report only, nothing chosen here)")
row("LR", np.load(ROOT / "predictions/proba_lr_full_test.npy"), 0.6, y_test)
row("BERT", np.load(ROOT / "predictions/proba_bert_test.npy"), chosen, y_test)
