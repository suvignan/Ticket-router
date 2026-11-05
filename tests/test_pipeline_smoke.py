"""End to end: a fresh checkout with ONLY committed files and no DagsHub runs all 6 stages on the fixture.
Each stage runs as a subprocess = the exact command DVC and CI run (env vars must exist before Python starts)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))
from src.entity.config_entity import get_trainer_config

LABELS = list(get_trainer_config(Path("x")).labels)


def snapshot(*folders) -> dict:
    """path -> (size, mtime) of every real file we must NOT touch."""
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for f in folders if f.exists() for p in f.rglob("*") if p.is_file()}


def run_stage(stage, env, tmp_path) -> dict:
    proc = subprocess.run([sys.executable, "-m", f"src.pipeline.stages.{stage}"], env=env, cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, f"{stage} failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}"   # in CI, stderr IS the debugger
    name = {"stage_01_ingest": "ingestion", "stage_02_validate": "validation", "stage_03_labels": "labels",
            "stage_04_split": "split", "stage_05_train": "train", "stage_06_evaluate": "evaluation"}[stage]
    return json.loads((tmp_path / "artifacts" / f"{name}.json").read_text(encoding="utf-8"))


def test_pipeline_smoke(tmp_path):
    raw = tmp_path / "data" / "raw" / "CFPB.csv"
    raw.parent.mkdir(parents=True)
    shutil.copy(FIXTURES / "mini_cfpb.csv", raw)
    env = os.environ | {"TICKET_ROUTER_DATA_ROOT": str(tmp_path),
                        "TICKET_ROUTER_PARAMS": str(FIXTURES / "params_fixture.yaml"),
                        "MLFLOW_DISABLED": "1", "PYTHONUTF8": "1"}
    before = snapshot(ROOT / "data", ROOT / "artifacts")

    a = run_stage("stage_01_ingest", env, tmp_path)
    assert (a["n_rows_read"], a["n_bad_lines"], a["n_junk_removed"], a["n_after_cutoff"]) == (77, 1, 3, 73), a

    v = run_stage("stage_02_validate", env, tmp_path)
    assert v["validation_status"] is True, v

    lab = run_stage("stage_03_labels", env, tmp_path)
    assert (lab["n_renamed"], lab["n_card_to_credit"], lab["n_card_to_prepaid"], lab["n_credit_repair_moved"]) == (3, 1, 1, 1), lab
    assert sorted(lab["label_counts"]) == sorted(LABELS), lab["label_counts"]

    s = run_stage("stage_04_split", env, tmp_path)
    got = {k: s[k] for k in ("n_train", "n_val", "n_test", "n_train_removed_dups", "n_ties", "n_test_seen_in_train", "n_test_dedup")}
    assert got == {"n_train": 41, "n_val": 12, "n_test": 15, "n_train_removed_dups": 5, "n_ties": 1,
                   "n_test_seen_in_train": 1, "n_test_dedup": 13}, got

    t = run_stage("stage_05_train", env, tmp_path)
    for key in ("tfidf_path", "logreg_path", "label_order_path"):
        assert Path(t[key]).exists(), key
    assert json.loads(Path(t["label_order_path"]).read_text(encoding="utf-8")) == LABELS

    e = run_stage("stage_06_evaluate", env, tmp_path)
    assert e["gate_passed"] is True and e["champion_macro_f1"] is None, e      # floor 0.0, no champion
    assert Path(e["report_path"]).exists()

    # Never touched the real project: every artifact path is under tmp_path, and real data/ + artifacts/ are unchanged
    paths = [Path(p) for d in (a, v, lab, s, t, e) for k, p in d.items() if k.endswith(("_path", "_parquet"))]
    assert len(paths) >= 10                                                   # guard: the check below really checks something
    assert all(p.resolve().is_relative_to(tmp_path.resolve()) for p in paths), paths
    assert snapshot(ROOT / "data", ROOT / "artifacts") == before
