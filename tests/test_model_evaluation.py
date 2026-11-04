import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.components.model_evaluation import decide_gate
from src.entity.config_entity import get_evaluation_config

CFG = get_evaluation_config(Path("val.parquet"), Path("t"), Path("l"))   # real thresholds; fake paths are never opened


def scores(macro, overrides=None):
    per_class = {label: 0.70 for label in CFG.labels}
    per_class.update(overrides or {})
    return {"macro_f1": macro, "per_class": per_class}


def test_all_fine_passes():
    passed, reasons = decide_gate(scores(0.697), scores(0.6956), CFG)
    assert passed and reasons == []


def test_floor_fails():
    passed, reasons = decide_gate(scores(0.60), scores(0.6956), CFG)
    assert not passed and any("floor" in r for r in reasons)


def test_relative_drop_fails():
    passed, reasons = decide_gate(scores(0.685), scores(0.695), CFG)      # 0.010 below
    assert not passed and any("vs champion" in r for r in reasons)


def test_small_class_drop_fails_and_names_class():
    passed, reasons = decide_gate(scores(0.697, {"Prepaid card": 0.66}), scores(0.6956), CFG)   # down 0.03
    assert not passed
    assert len(reasons) == 1                       # guard: exactly ONE failure, so the next line can't pass by luck
    assert "Prepaid card" in reasons[0]


def test_no_champion_floor_only():
    passed, reasons = decide_gate(scores(0.66), None, CFG)
    assert passed


def test_boundary_exact_tolerance_passes():
    passed, _ = decide_gate(scores(0.690), scores(0.695), CFG)   # drop is EXACTLY 0.005
    assert passed                                  # exactly 0.005 PASSES (rule is >=)
