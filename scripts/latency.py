"""End-to-end /predict latency (network included) on N DIFFERENT validation complaints. Day 13 rule: p95 < 200 ms.
One kept-alive connection (like a real client), plus /health (no model call) as the NETWORK baseline:
predict minus health ~= time spent in the service.        python scripts/latency.py https://<service>.onrender.com 100"""
import http.client
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL, N = urlparse(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 100
conn = (http.client.HTTPSConnection if URL.scheme == "https" else http.client.HTTPConnection)(URL.netloc, timeout=300)


def call(method, path, body=None):
    t = time.perf_counter()
    conn.request(method, path, None if body is None else json.dumps(body), {"Content-Type": "application/json"})
    data = json.loads(conn.getresponse().read())
    return (time.perf_counter() - t) * 1000, data


def stats(ms):
    return f"median {np.median(ms):.0f} ms | p95 {np.percentile(ms, 95):.0f} ms | max {ms.max():.0f} ms"


val = pd.read_parquet(ROOT / "data/splits/val.parquet", columns=["Consumer complaint narrative"])
texts = val.iloc[np.random.default_rng(20).choice(len(val), N, replace=False), 0].tolist()   # seed 20: not the parity rows

first_ms, version = call("GET", "/version")                     # also wakes a sleeping free instance
print(f"model {version['model']} | git_commit {version['git_commit'][:12]} | first call (cold if asleep) {first_ms:.0f} ms")
health = np.array([call("GET", "/health")[0] for _ in range(20)])
predict = np.array([call("POST", "/predict", {"text": t})[0] for t in texts])
print(f"network  /health  x20:  {stats(health)}")
print(f"predict  x{N}: {stats(predict)}")
print(f"service time ~ predict - network: median {np.median(predict) - np.median(health):.0f} ms")
print(f"Day 13 rule (end-to-end p95 < 200 ms): {'PASS' if np.percentile(predict, 95) < 200 else 'FAIL'}")
