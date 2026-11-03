import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import read_yaml


def test_config():
    schema = read_yaml(ROOT / "config/schema.yaml")
    params = read_yaml(ROOT / "params.yaml")
    labels, raw = schema["labels"], schema["raw_product_values"]
    lab = params["labels"]

    assert len(labels) == 11
    assert labels == json.load(open(ROOT / "models/label_order.json"))
    assert len(raw) == 14 and len(set(raw)) == 14
    assert all(k in raw for k in lab["rename_map"])
    assert all(v in labels for v in lab["rename_map"].values())
    assert lab["card_split"]["old_product"] in raw
    assert {lab["credit_repair_move"]["from_label"], lab["credit_repair_move"]["to_label"]} <= set(labels)
    assert "Sub-product" in schema["label_only_columns"]
    split = params["split"]
    date_keys = [k for k in split if "start" in k or "end" in k]
    assert len(date_keys) == 5                                   # guard: the filter really found the dates
    assert all(isinstance(split[k], str) for k in date_keys)     # dates stay strings
    assert split["val_start"] == split["train_end_exclusive"]  # no gap, no overlap
    assert split["test_start"] == split["val_end_exclusive"]
    assert split["test_end_exclusive"] == params["ingestion"]["cutoff_date"]
    assert params["junk_rules"]["redaction_only_chars"] == r"[X/\s]"   # backslash survived


if __name__ == "__main__":
    test_config()
    print("config: all checks passed")
