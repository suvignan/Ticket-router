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
