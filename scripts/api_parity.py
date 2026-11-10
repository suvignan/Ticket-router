"""The API must equal the offline path. Starts uvicorn for ONE model, sends 20 validation complaints + edge cases.
    python scripts/api_parity.py lr|bert                  # starts its own uvicorn
    python scripts/api_parity.py bert http://host:port   # an ALREADY running server (Docker, Render)
Offline reference: LR = batch TF-IDF + logreg on the same rows; BERT = predictions/bert_val_proba.npy (batch ONNX job)."""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.text import clean_text

MODEL = sys.argv[1]
URL = sys.argv[2].rstrip("/") if len(sys.argv) > 2 else "http://127.0.0.1:8000"


def call(path, body=None):
    req = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


val = pd.read_parquet(ROOT / "data/splits/val.parquet", columns=["Consumer complaint narrative"])
rows = np.random.default_rng(19).choice(len(val), 20, replace=False)
texts = val.iloc[rows, 0].tolist()
labels = json.loads((ROOT / "models/label_order.json").read_text(encoding="utf-8"))
if MODEL == "lr":
    lr = joblib.load(ROOT / "models/logreg_b.joblib")
    offline = lr.predict_proba(joblib.load(ROOT / "models/tfidf_b.joblib").transform([clean_text(t) for t in texts]))
else:
    offline = np.load(ROOT / "predictions/bert_val_proba.npy")[rows]

params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
params["serving"]["model"] = MODEL
threshold = params["serving"]["thresholds"][MODEL]
tmp = Path(tempfile.mkdtemp()) / "params.yaml"
tmp.write_text(yaml.safe_dump(params), encoding="utf-8")
env = {**os.environ, "TICKET_ROUTER_PARAMS": str(tmp), "GIT_COMMIT": subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()}
server = None if len(sys.argv) > 2 else subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000", "--log-level", "warning"],
    cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
try:
    for _ in range(120):
        try:
            if call("/health")[0] == 200:
                break
        except urllib.error.URLError:
            time.sleep(1)
    else:
        raise SystemExit("server did not start")
    assert call("/version")[1]["model"] == MODEL, "server runs a different model than the one being checked"

    ok = 0
    for i, text in enumerate(texts):
        status, r = call("/predict", {"text": text})
        team, conf = labels[offline[i].argmax()], float(offline[i].max())
        decision = "review" if conf < threshold else "auto_route"
        match = status == 200 and r["team"] == team and abs(r["confidence"] - conf) < 1e-4 and r["decision"] == decision
        ok += match
        print(f"{i:>2} {'OK ' if match else 'BAD'} {r['decision']:<10} conf api {r['confidence']:.6f} offline {conf:.6f}"
              f" | {r['team'][:40]}")
    print(f"{MODEL}: {ok}/20 match (team = offline argmax, |conf diff| < 1e-4, decision at threshold {threshold})")

    for name, text in [('""', ""), ('"XXXX"', "XXXX"), ('"See attached."', "See attached."),
                       ('"help me" (too short)', "help me"), ("36,000 chars", "word " * 7200)]:
        status, r = call("/predict", {"text": text})
        r.pop("request_id")
        print(f"{name}: {status} {json.dumps(r)}")
    print("/version:", json.dumps(call("/version")[1], indent=2))
    sys.exit(0 if ok == 20 else 1)
finally:
    if server:
        server.terminate()
