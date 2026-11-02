import dataclasses
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.constants import PARAMS_FILE, SCHEMA_FILE
from src.entity.config_entity import get_ingestion_config
from src.utils import read_yaml


def test_ingestion_config_matches_yaml():
    cfg = get_ingestion_config()
    schema, params = read_yaml(SCHEMA_FILE), read_yaml(PARAMS_FILE)
    assert cfg.cutoff_date == date.fromisoformat(params["ingestion"]["cutoff_date"])
    assert cfg.max_bad_lines == params["ingestion"]["max_bad_lines"]
    assert cfg.columns == (*schema["raw_columns"], *schema["label_only_columns"])
    assert "Sub-product" in cfg.columns            # read once at ingestion, not twice


def test_ingestion_config_is_frozen():
    cfg = get_ingestion_config()
    try:
        cfg.max_bad_lines = 99
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("config is not frozen")


if __name__ == "__main__":
    test_ingestion_config_matches_yaml()
    test_ingestion_config_is_frozen()
    print("entities: all checks passed")
