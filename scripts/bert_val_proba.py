"""BERT validation probabilities with the SERVING engine (ONNX Runtime + src.serving.encode_for_bert), for choosing
BERT's review threshold. Offline batch job: all CPU threads, length-sorted batches (each padded only to its own
longest text), then put back in the ORIGINAL row order (Day 10 alignment lesson: asserted).
    python scripts/bert_val_proba.py"""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.serving import encode_for_bert
from src.utils import read_yaml, sha256_file

OUT = ROOT / "predictions" / "bert_val_proba.npy"
BATCH = 32


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def run(sess, encoded):
    longest = max(len(e) for e in encoded)
    ids = np.zeros((len(encoded), longest), np.int32)              # 0 = [PAD]
    for k, e in enumerate(encoded):
        ids[k, :len(e)] = e
    feeds = {"token_ids": ids, "segment_ids": np.zeros_like(ids), "padding_mask": (ids != 0).astype(np.int32)}
    return softmax(sess.run(None, feeds)[0].astype(np.float64))


val = pd.read_parquet(ROOT / "data/splits/val.parquet", columns=["Consumer complaint narrative", "label"])
labels = read_yaml(ROOT / "config/schema.yaml")["labels"]
so = ort.SessionOptions()
so.intra_op_num_threads = 16
sess = ort.InferenceSession(str(ROOT / "models/bert.onnx"), so, providers=["CPUExecutionProvider"])

t0 = time.perf_counter()
encoded = [encode_for_bert(t)[0][0] for t in val.iloc[:, 0]]
order = np.argsort([len(e) for e in encoded], kind="stable")       # shortest first
proba = np.full((len(val), len(labels)), np.nan)
for n, i in enumerate(range(0, len(order), BATCH)):
    rows = order[i:i + BATCH]
    proba[rows] = run(sess, [encoded[r] for r in rows])             # written back at the ORIGINAL row positions
    if n % 500 == 0:
        print(f"{i:,}/{len(order):,} rows | {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

# Alignment proof: 50 random rows re-scored ONE AT A TIME (no batching, no sorting) must match their row
assert not np.isnan(proba).any()
rng = np.random.default_rng(0)
for r in rng.choice(len(val), 50, replace=False):
    single = run(sess, [encoded[r]])[0]
    assert np.abs(single - proba[r]).max() < 1e-4, f"row {r} misaligned"

pred = np.array(labels)[proba.argmax(axis=1)]
OUT.parent.mkdir(exist_ok=True)
np.save(OUT, proba.astype(np.float32))
print(f"done in {(time.perf_counter() - t0) / 60:.1f} min | shape {proba.shape} | alignment: 50/50 rows match")
print(f"val macro-F1 {f1_score(val['label'], pred, average='macro'):.4f} (Day 12: 0.7205) | sha256 {sha256_file(OUT)[:12]}")
