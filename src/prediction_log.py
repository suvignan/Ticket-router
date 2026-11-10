"""One row per /predict response in Neon (Postgres). NO raw text column, by design: hash + length only.
Create the table once:    python -m src.prediction_log      (needs DATABASE_URL, e.g. from .env)"""
import os
from contextlib import closing

import psycopg2

COLUMNS = ("request_id", "status", "text_sha256", "text_len", "model", "model_sha256", "decision", "team",
           "confidence", "reason", "latency_ms")

DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id   uuid PRIMARY KEY,
    created_at   timestamptz NOT NULL DEFAULT now(),
    status       smallint NOT NULL,           -- 200 or 413
    text_sha256  char(64),                    -- NULL for 413: oversized text is never hashed (docs/api.md)
    text_len     integer NOT NULL,
    model        text NOT NULL,
    model_sha256 char(12) NOT NULL,
    decision     text,                        -- auto_route | review (NULL for 413)
    team         text,
    confidence   double precision,
    reason       text,
    latency_ms   real NOT NULL
    -- deliberately NO text column: the complaint is personal data and is never stored
)"""

INSERT = f"INSERT INTO predictions ({', '.join(COLUMNS)}) VALUES ({', '.join('%(' + c + ')s' for c in COLUMNS)})"


def insert(url: str, record: dict) -> None:
    # ponytail: one connection per insert (~100-300 ms to Neon, off the response path); pool it when traffic needs it
    with closing(psycopg2.connect(url, connect_timeout=5)) as conn, conn, conn.cursor() as cur:
        cur.execute(INSERT, {c: record.get(c) for c in COLUMNS})


if __name__ == "__main__":
    from dotenv import load_dotenv                 # dev only (python-dotenv is not in the serving image)

    load_dotenv()
    with closing(psycopg2.connect(os.environ["DATABASE_URL"], connect_timeout=10)) as conn, conn, conn.cursor() as cur:
        cur.execute(DDL)
    print("table predictions: ready")
