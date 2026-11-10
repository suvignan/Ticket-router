"""Neon check: start the real API (LR) with DATABASE_URL from .env, send 5 requests, then query Neon directly:
5 rows, no text column, hashes = the server's log = sha256 of what we sent.      python scripts/neon_check.py"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path

import psycopg2
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8000"
DB = dotenv_values(ROOT / ".env")["DATABASE_URL"]           # never printed


def post(text):
    req = urllib.request.Request(URL + "/predict", json.dumps({"text": text}).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


TEXTS = ["I was charged twice on my credit card and the bank refuses to refund the duplicate charge.",
         "A debt collector keeps calling me about a debt that I already paid in full last year.",
         "My mortgage servicer misapplied my payment and now reports me late to the credit bureaus.",
         "XXXX",                                               # no_content
         "word " * 7200]                                       # 36,000 chars -> 413 (no hash stored)

server_log = Path(tempfile.mkdtemp()) / "server.log"
with open(server_log, "w", encoding="utf-8") as out:
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000", "--log-level", "warning"],
                              cwd=ROOT, env={**os.environ, "DATABASE_URL": DB}, stdout=out, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(URL + "/health"); break
            except urllib.error.URLError:
                time.sleep(1)
        sent = {}
        for t in TEXTS:
            status, r = post(t)
            sent[r["request_id"]] = (status, len(t), hashlib.sha256(t.encode()).hexdigest())
            print(f"sent {len(t):>6} chars -> {status} {r.get('decision') or ''} {r.get('reason') or ''}")

        with closing(psycopg2.connect(DB, connect_timeout=10)) as conn, conn.cursor() as cur:
            for _ in range(30):                                  # inserts run AFTER each response: give them a moment
                cur.execute("SELECT request_id::text, status, text_sha256, text_len, model, model_sha256, decision, team,"
                            " confidence, reason, latency_ms FROM predictions WHERE request_id::text = ANY(%s)", (list(sent),))
                rows = cur.fetchall()
                conn.commit()
                if len(rows) == len(sent):
                    break
                time.sleep(1)
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'predictions' ORDER BY ordinal_position")
            columns = [c for (c,) in cur.fetchall()]
    finally:
        server.terminate()
        server.wait()

logged = {r["request_id"]: r for r in map(json.loads, (l for l in server_log.read_text(encoding="utf-8").splitlines()
                                                       if l.startswith('{"request_id"')))}
print(f"\nNeon columns: {columns}")
print(f"raw text column present: {any('text' == c or c.endswith('_text') for c in columns)}")
ok = 0
for rid, status, sha, n, *rest in rows:
    want_status, want_len, want_sha = sent[rid]
    want_db_sha = want_sha if want_status == 200 else None
    match = (status, n, sha) == (want_status, want_len, want_db_sha) and logged[rid]["text_sha256"] == sha
    ok += match
    print(f"{'OK ' if match else 'BAD'} {rid[:8]} status {status} len {n:>6} sha {sha[:12] if sha else None} "
          f"| log sha {logged[rid]['text_sha256'][:12] if logged[rid]['text_sha256'] else None} | {rest[2]} {rest[5]} {rest[6]}")
print(f"{len(rows)}/5 rows in Neon, {ok}/5 match the request and the server log")
sys.exit(0 if ok == 5 else 1)
