import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]       # the CODE root: schema, src/, models/ (champion)
# Read ONCE, at first import: set them before Python starts (the smoke test runs each stage as a subprocess)
DATA_ROOT = Path(os.environ.get("TICKET_ROUTER_DATA_ROOT", ROOT))   # where data/ and artifacts/ live

SCHEMA_FILE = ROOT / "config" / "schema.yaml"
PARAMS_FILE = Path(os.environ.get("TICKET_ROUTER_PARAMS", ROOT / "params.yaml"))

RAW_CSV = DATA_ROOT / "data" / "raw" / "CFPB.csv"
# Pipeline outputs get their own names, so they never overwrite the notebook files in data/interim/
INGESTED_PARQUET = DATA_ROOT / "data" / "interim" / "ingested.parquet"
VALIDATION_REPORT = DATA_ROOT / "data" / "processed" / "validation_report.json"
LABELED_PARQUET = DATA_ROOT / "data" / "interim" / "labeled.parquet"
SPLIT_DIR = DATA_ROOT / "data" / "splits"          # NOT "processed": the notebook's files live there
MODEL_DIR = DATA_ROOT / "artifacts" / "model"     # NOT models/: the frozen Day 13 files live there
ARTIFACT_DIR = DATA_ROOT / "artifacts"            # one <stage>.json per DVC stage
EVALUATION_REPORT = DATA_ROOT / "data" / "processed" / "evaluation_report.json"
