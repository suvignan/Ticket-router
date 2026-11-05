import hashlib
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
    # label_order.json lives in DVC (not in a fresh clone / CI), but its sha256 is in the committed manifest.
    # The frozen file = json.dumps(labels, indent=2) written on WINDOWS (CRLF). Build those exact bytes on any OS:
    # this proves schema == deployed file, byte for byte. (The trainer now writes LF everywhere: newline="\n")
    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))["checksums"]
    as_written = json.dumps(labels, indent=2).replace("\n", "\r\n").encode("utf-8")
    assert hashlib.sha256(as_written).hexdigest() == manifest["models/label_order.json"]
    assert len(raw) == 14 and len(set(raw)) == 14
    assert all(r["from"] in raw for r in lab["rename_map"])
    assert all(r["to"] in labels for r in lab["rename_map"])
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
