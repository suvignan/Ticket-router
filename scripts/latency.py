"""End-to-end /predict latency (network included) on N DIFFERENT validation complaints. Day 13 rule: p95 < 200 ms.
    python scripts/latency.py https://<service>.onrender.com 100"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL, N = sys.argv[1].rstrip("/"), int(sys.argv[2]) if len(sys.argv) > 2 else 100


def post(text):
    req = urllib.request.Request(URL + "/predict", json.dumps({"text": text}).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read())
    return (time.perf_counter() - t) * 1000, body


val = pd.read_parquet(ROOT / "data/splits/val.parquet", columns=["Consumer complaint narrative"])
texts = val.iloc[np.random.default_rng(20).choice(len(val), N, replace=False), 0].tolist()   # seed 20: not the parity rows

version = json.loads(urllib.request.urlopen(URL + "/version", timeout=300).read())
print(f"model {version['model']} | git_commit {version['git_commit']}")
ms = np.array([post(t)[0] for t in texts])
print(f"{N} requests: first {ms[0]:.0f} ms | median {np.median(ms):.0f} ms | p95 {np.percentile(ms, 95):.0f} ms | max {ms.max():.0f} ms")
print(f"Day 13 rule (p95 < 200 ms): {'PASS' if np.percentile(ms, 95) < 200 else 'FAIL'}")
