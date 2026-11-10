"""API tests on a model trained from the FIXTURE (CI has no DVC data). The fixture model is the test's own champion:
its manifest is built from its own checksums, so the real startup check runs unchanged."""
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from src.utils import read_yaml, sha256_file
from tests.test_pipeline_smoke import FIXTURES, ROOT, run_stage

PARAMS = read_yaml(ROOT / "params.yaml")
FILES = {"tfidf": "artifacts/model/tfidf.joblib", "logreg": "artifacts/model/logreg.joblib",
         "label_order": "artifacts/model/label_order.json"}
CFG = {**PARAMS["serving"], "model": "lr", "files": {"lr": FILES}}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Stages 1-5 on mini_cfpb.csv (same subprocess runner as the smoke test) + a manifest of its own checksums."""
    tmp = tmp_path_factory.mktemp("api")
    (tmp / "data" / "raw").mkdir(parents=True)
    shutil.copy(FIXTURES / "mini_cfpb.csv", tmp / "data" / "raw" / "CFPB.csv")
    env = os.environ | {"TICKET_ROUTER_DATA_ROOT": str(tmp), "TICKET_ROUTER_PARAMS": str(FIXTURES / "params_fixture.yaml"),
                        "MLFLOW_DISABLED": "1", "PYTHONUTF8": "1"}
    for stage in ("stage_01_ingest", "stage_02_validate", "stage_03_labels", "stage_04_split", "stage_05_train"):
        run_stage(stage, env, tmp)
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps({"checksums": {p: sha256_file(tmp / p) for p in FILES.values()}}), encoding="utf-8")
    return tmp, manifest


@pytest.fixture(scope="module")
def client(trained):
    tmp, manifest = trained
    with TestClient(create_app(CFG, PARAMS["junk_rules"], manifest, tmp)) as c:      # `with` runs the startup (lifespan)
        yield c


def test_normal_complaint(client, trained):
    text = pd.read_parquet(trained[0] / "data" / "splits" / "val.parquet")["Consumer complaint narrative"].iloc[0]
    r = client.post("/predict", json={"text": text})
    body = r.json()
    labels = json.loads((trained[0] / FILES["label_order"]).read_text(encoding="utf-8"))
    assert r.status_code == 200, body
    assert body["team"] in labels and 0 <= body["confidence"] <= 1
    assert body["decision"] in ("auto_route", "review") and len(body["top2"]) == 2


@pytest.mark.parametrize("text", ["", "XXXX", "XX/XX/XXXX XXXX", "See attached."])
def test_no_content(client, text):
    body = client.post("/predict", json={"text": text}).json()
    assert (body["decision"], body["reason"], body["team"], body["confidence"]) == ("review", "no_content", None, None)


def test_too_short(client):
    assert client.post("/predict", json={"text": "help me"}).json()["reason"] == "too_short"


def test_over_max_chars_is_413(client):
    assert client.post("/predict", json={"text": "a" * CFG["max_chars"]}).status_code == 200          # boundary: allowed
    r = client.post("/predict", json={"text": "a" * (CFG["max_chars"] + 1)})
    assert r.status_code == 413 and "request_id" in r.json()


def test_missing_text_is_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_version_matches_own_manifest(client, trained):
    v = client.get("/version").json()
    assert v["model"] == "lr" and v["active_threshold"] == CFG["thresholds"]["lr"]
    assert v["checksums"] == json.loads(trained[1].read_text(encoding="utf-8"))["checksums"]


def test_tampered_manifest_refuses_to_start(trained, tmp_path):
    tmp, manifest = trained
    bad = json.loads(manifest.read_text(encoding="utf-8"))
    bad["checksums"][FILES["logreg"]] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to start"):
        with TestClient(create_app(CFG, PARAMS["junk_rules"], tmp_path / "manifest.json", tmp)):
            pass


def test_database_down_still_routes(trained, monkeypatch, caplog):
    """Availability decision: a failed insert is logged and the prediction is still returned."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:x@127.0.0.1:1/none")    # port 1: connection refused
    tmp, manifest = trained
    with TestClient(create_app(CFG, PARAMS["junk_rules"], manifest, tmp)) as c:
        r = c.post("/predict", json={"text": "I was charged twice on my credit card and want a refund"})
    assert r.status_code == 200 and r.json()["team"] is not None
    assert '"db_error": "OperationalError"' in caplog.text
