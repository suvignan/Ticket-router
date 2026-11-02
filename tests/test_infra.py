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
