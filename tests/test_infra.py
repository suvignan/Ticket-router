import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.exception import TicketRouterException
from src.logger import LOG_FILE, logger


def test_logger_writes_file():
    logger.info("hello")
    assert LOG_FILE.exists()
    assert "hello" in LOG_FILE.read_text(encoding="utf-8")


def test_exception_has_type_file_and_line():
    try:
        try:
            1 / 0
        except Exception as e:
            raise TicketRouterException(e) from e
    except TicketRouterException as err:
        msg = str(err)
    assert "ZeroDivisionError" in msg and "division by zero" in msg
    assert "test_infra.py" in msg
    assert re.search(r"line \d+", msg)


if __name__ == "__main__":
    test_logger_writes_file()
    test_exception_has_type_file_and_line()
    print("infra: all checks passed")


def test_artifact_json_roundtrip(tmp_path):
    from src.entity.artifact_entity import ModelEvaluationArtifact
    from src.utils import load_artifact, save_artifact
    art = ModelEvaluationArtifact(gate_passed=False, reasons=("Prepaid card dropped 0.031 (> 0.02)",),
                                  challenger_macro_f1=0.69, champion_macro_f1=0.70, macro_f1_diff=-0.01,
                                  per_class_f1_diff={"Prepaid card": -0.031}, report_path=tmp_path / "r.json")
    save_artifact(art, tmp_path / "a.json")
    assert load_artifact(ModelEvaluationArtifact, tmp_path / "a.json") == art     # Path and tuple come back typed


def test_slugify_longest_label_is_mlflow_safe():
    import re
    from src.entity.config_entity import get_trainer_config
    from src.utils import slugify
    labels = get_trainer_config(Path("x")).labels
    longest = max(labels, key=len)
    assert "," in longest                                         # guard: the test really contains the bad character
    assert slugify(longest) == "payday_loan_title_loan_personal_loan_or_advance_loan"
    assert all(re.fullmatch(r"[a-z0-9_]+", slugify(l)) for l in labels)
    assert len({slugify(l) for l in labels}) == len(labels)       # no two labels collide into one metric


def test_flatten_params():
    from src.utils import flatten
    assert flatten({"tfidf": {"min_df": 5, "ngram_range": [1, 1]}, "C": 1.0}, "mt.") == \
        {"mt.tfidf.min_df": 5, "mt.tfidf.ngram_range": "[1, 1]", "mt.C": 1.0}


def test_mlflow_off_switch_and_loud_failure(monkeypatch):
    import pytest
    from src.pipeline.stages import stage_06_evaluate as st
    monkeypatch.setattr(st, "load_dotenv", lambda *a, **k: None)   # don't read the real credentials file
    for v in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("MLFLOW_DISABLED", "1")
    assert st.log_to_mlflow(None, None) is None                    # skipped, no error
    monkeypatch.delenv("MLFLOW_DISABLED")
    with pytest.raises(RuntimeError, match="credentials missing"):
        st.log_to_mlflow(None, None)                               # no creds + no flag = fail loudly
