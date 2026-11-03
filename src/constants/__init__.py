from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCHEMA_FILE = ROOT / "config" / "schema.yaml"
PARAMS_FILE = ROOT / "params.yaml"

RAW_CSV = ROOT / "data" / "raw" / "CFPB.csv"
# Pipeline outputs get their own names, so they never overwrite the notebook files in data/interim/
INGESTED_PARQUET = ROOT / "data" / "interim" / "ingested.parquet"
VALIDATION_REPORT = ROOT / "data" / "processed" / "validation_report.json"
LABELED_PARQUET = ROOT / "data" / "interim" / "labeled.parquet"
SPLIT_DIR = ROOT / "data" / "splits"          # NOT "processed": the notebook's files live there
MODEL_DIR = ROOT / "artifacts" / "model"     # NOT models/: the frozen Day 13 files live there
