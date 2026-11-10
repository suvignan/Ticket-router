"""The serving path must not pull pandas in (requirements-serve.txt has none: ~50 MB more image, or a crash in Docker)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_serving_path_imports_no_pandas():
    # Fresh process: this test session has already imported pandas. app.main loads no model at import (lifespan does).
    code = "import sys, src.serving, app.main; assert 'pandas' not in sys.modules, 'pandas imported by the serving path'"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
