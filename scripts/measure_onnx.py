import psutil
proc = psutil.Process()
def mb(): return proc.memory_info().rss / 1e6
print(f"1. Start:            {mb():6.1f} MB")

import sys, json, time
from pathlib import Path
import numpy as np, onnxruntime as ort
from tokenizers import BertWordPieceTokenizer
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.text import clean_text, format_for_bert
print(f"2. After imports:    {mb():6.1f} MB")

def first_350(t): return " ".join(t.split()[:350])
def softmax(z): e = np.exp(z - z.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)

so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
sess = ort.InferenceSession(str(ROOT / "models/bert.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
tok = BertWordPieceTokenizer(str(ROOT / "models/bert_vocab.txt"), lowercase=True)
tok.enable_truncation(512)
print(f"3. After session:    {mb():6.1f} MB")

raw = json.load(open(ROOT / "models/ref_raw_texts.json"))
ref_probs = np.load(ROOT / "models/ref_probs.npy")

def predict(text, pad512):
    if pad512: tok.enable_padding(length=512, pad_id=0, pad_token="[PAD]")
    else:      tok.no_padding()
    ids = np.array([tok.encode(format_for_bert(first_350(clean_text(text)))).ids], dtype=np.int32)
    feeds = {"token_ids": ids, "segment_ids": np.zeros_like(ids), "padding_mask": (ids != 0).astype(np.int32)}
    return softmax(sess.run(None, feeds)[0].astype("float32"))

def time_100(pad512):
    predict(raw[0], pad512)                      # warm-up
    times = []
    for t in raw[:100]:
        s = time.perf_counter(); predict(t, pad512); times.append((time.perf_counter() - s) * 1000)
    return np.median(times), np.percentile(times, 95)

a_med, a_p95 = time_100(pad512=True)
print(f"4. After 100 preds:  {mb():6.1f} MB")
b_med, b_p95 = time_100(pad512=False)

# Parity: B (real length) vs reference probs (padded to 512, which matched A 100%)
probs_b = np.concatenate([predict(t, pad512=False) for t in raw])
print(f"\nB vs padded-512 argmax match: {(probs_b.argmax(1) == ref_probs.argmax(1)).mean():.1%} | max |diff|: {np.abs(probs_b - ref_probs).max():.5f}")

print(f"\nA (pad 512):     median {a_med:6.1f} ms | p95 {a_p95:6.1f} ms")
print(f"B (real length): median {b_med:6.1f} ms | p95 {b_p95:6.1f} ms")
print(f"\nPeak memory: {proc.memory_info().peak_wset / 1e6:.1f} MB")
