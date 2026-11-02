from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src import constants
from src.utils import read_yaml


@dataclass(frozen=True)
class DataIngestionConfig:
    raw_csv: Path
    columns: tuple[str, ...]          # raw columns + label-only columns (tuple: lists could be mutated)
    date_column: str
    date_format: str
    max_bad_lines: int
    cutoff_date: date                 # rows on or after this are dropped
    redaction_only_chars: str
    attachment_phrase: str
    attachment_max_len: int
    output_parquet: Path


def get_ingestion_config(schema_file=constants.SCHEMA_FILE, params_file=constants.PARAMS_FILE) -> DataIngestionConfig:
    schema, params = read_yaml(schema_file), read_yaml(params_file)
    junk = params["junk_rules"]
    return DataIngestionConfig(
        raw_csv=constants.RAW_CSV,
        columns=tuple(schema["raw_columns"]) + tuple(schema["label_only_columns"]),
        date_column=schema["date_column"],
        date_format=schema["date_format"],
        max_bad_lines=params["ingestion"]["max_bad_lines"],
        cutoff_date=date.fromisoformat(params["ingestion"]["cutoff_date"]),   # fails early on a bad date
        redaction_only_chars=junk["redaction_only_chars"],
        attachment_phrase=junk["attachment_phrase"],
        attachment_max_len=junk["attachment_max_len"],
        output_parquet=constants.INGESTED_PARQUET,
    )
