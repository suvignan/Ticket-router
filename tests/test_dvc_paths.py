"""dvc.yaml must use the SAME paths as src/constants: if one is renamed and not the other,
DVC tracks a file the code never writes (or misses the one it does)."""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import constants as C
from src.entity.config_entity import get_evaluation_config


def rel(p: Path) -> str:
    """constants give absolute Windows Paths; dvc.yaml has relative forward-slash strings."""
    return Path(p).relative_to(ROOT).as_posix()


def entries(stage: dict, *keys) -> set[str]:
    """outs/metrics items are either 'path' or {'path': {cache: false}}."""
    found = set()
    for key in keys:
        for item in stage.get(key, []):
            found |= set(item) if isinstance(item, dict) else {item}
    return found


STAGES = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
A = C.ARTIFACT_DIR
CHAMPION = get_evaluation_config(Path("v"), Path("t"), Path("l")).champion

#           stage       must be in deps                                                   must be in outs (or metrics)
EXPECTED = {
    "ingest":   ([C.RAW_CSV, C.SCHEMA_FILE],                                               [C.INGESTED_PARQUET, A / "ingestion.json"]),
    "validate": ([C.INGESTED_PARQUET],                                                     [A / "validation.json", C.VALIDATION_REPORT]),
    "labels":   ([C.INGESTED_PARQUET, A / "validation.json"],                              [C.LABELED_PARQUET, A / "labels.json"]),
    "split":    ([C.LABELED_PARQUET],                                                      [C.SPLIT_DIR / f"{s}.parquet" for s in ("train", "val", "test", "test_dedup")] + [A / "split.json"]),
    "train":    ([C.SPLIT_DIR / "train.parquet", C.ROOT / "src" / "text.py"],              [C.MODEL_DIR, A / "train.json"]),
    "evaluate": ([C.SPLIT_DIR / "val.parquet", C.MODEL_DIR, A / "ingestion.json", C.ROOT / "src" / "text.py",
                  CHAMPION.tfidf_path, CHAMPION.logreg_path, CHAMPION.label_order_path],   [A / "evaluation.json", C.EVALUATION_REPORT]),
}


def test_every_stage_is_checked():
    assert set(EXPECTED) == set(STAGES)          # a new stage in dvc.yaml must get expectations here too


@pytest.mark.parametrize("name", EXPECTED)
def test_stage_paths_match_constants(name):
    deps, outs = EXPECTED[name]
    stage = STAGES[name]
    missing_deps = {rel(p) for p in deps} - entries(stage, "deps")
    missing_outs = {rel(p) for p in outs} - entries(stage, "outs", "metrics")
    assert not missing_deps, f"{name}: deps missing in dvc.yaml: {missing_deps}"
    assert not missing_outs, f"{name}: outs missing in dvc.yaml: {missing_outs}"
