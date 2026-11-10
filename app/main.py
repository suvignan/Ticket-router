import hashlib
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from src import prediction_log
from src.constants import PARAMS_FILE, ROOT
from src.serving import MANIFEST, load_bundle
from src.text import clean_text, is_no_content
from src.utils import read_yaml

log = logging.getLogger("ticket_router.api")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))
log.propagate = False


class PredictRequest(BaseModel):
    text: str                                      # length checked by hand: Pydantic max_length would answer 422, contract says 413


def save_row(url: str, record: dict) -> None:
    """Runs AFTER the response is sent. A down database must never break routing: log the error, lose the row."""
    try:
        prediction_log.insert(url, record)
    except Exception as e:
        log.error(json.dumps({"request_id": record["request_id"], "db_error": type(e).__name__}))


def create_app(cfg: dict, junk_rules: dict, manifest: Path = MANIFEST, root: Path = ROOT) -> FastAPI:
    db_url = os.environ.get("DATABASE_URL")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = load_bundle(cfg, manifest, root)    # ONCE per process; raises (= no service) on a checksum mismatch
        app.state.threshold = cfg["thresholds"][cfg["model"]]
        if not db_url:
            log.warning("DATABASE_URL not set: prediction rows are NOT stored (stdout log only)")
        yield

    app = FastAPI(title="Ticket router", lifespan=lifespan)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", str(uuid.uuid4()))
        log.exception(json.dumps({"request_id": rid, "error": type(exc).__name__}))     # stack trace: server log only
        return JSONResponse(status_code=500, content={"detail": "internal error", "request_id": rid})

    @app.post("/predict")
    def predict(req: PredictRequest, request: Request):
        start, rid = time.perf_counter(), str(uuid.uuid4())
        request.state.request_id = rid
        bundle, threshold, text = request.app.state.bundle, request.app.state.threshold, req.text

        def respond(status=200, **body):
            # Privacy: the raw text is never logged or stored. Its sha256 only for texts we actually route (not 413)
            record = {"request_id": rid, "status": status, "text_len": len(text), "model": bundle.name,
                      "model_sha256": bundle.model_sha256[:12],
                      "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if status == 200 else None,
                      **{k: body.get(k) for k in ("decision", "team", "confidence", "reason")},
                      "latency_ms": round((time.perf_counter() - start) * 1000, 1)}
            log.info(json.dumps(record))
            return JSONResponse(status_code=status, content={**body, "request_id": rid},
                                background=BackgroundTask(save_row, db_url, record) if db_url else None)

        if len(text) > cfg["max_chars"]:           # FIRST: rejected with only its length touched (no hash, no cleaning)
            return respond(413, detail=f"text is {len(text):,} characters; the limit is {cfg['max_chars']:,}")

        base = dict(model=bundle.name, model_sha256=bundle.model_sha256[:12], threshold=threshold,
                    truncated_for_model=False)
        clean = clean_text(text)
        reason = "no_content" if is_no_content(text, junk_rules) else "too_short" if len(clean) < cfg["min_chars"] else None
        if reason:                                 # model NOT called: it never saw text like this
            return respond(decision="review", team=None, confidence=None, top2=None, reason=reason, **base)

        p = bundle.predict_proba(text)
        top = np.argsort(p)[::-1][:2]
        confidence = float(p[top[0]])
        low = confidence < threshold
        return respond(decision="review" if low else "auto_route", team=bundle.labels[top[0]], confidence=confidence,
                       top2=[{"team": bundle.labels[i], "probability": float(p[i])} for i in top],
                       reason="low_confidence" if low else None,
                       **{**base, "truncated_for_model": bundle.name == "bert" and len(clean.split()) > 350})

    @app.get("/health")
    def health(request: Request):
        ok = getattr(request.app.state, "bundle", None) is not None      # no model call: polled often, must cost nothing
        return JSONResponse(status_code=200 if ok else 503, content={"status": "ok" if ok else "loading"})

    @app.get("/version")
    def version(request: Request):
        b = request.app.state.bundle
        return {"model": b.name, "checksums": b.checksums, "thresholds": cfg["thresholds"],
                "active_threshold": request.app.state.threshold, "git_commit": os.environ.get("GIT_COMMIT", "unknown")}

    return app


PARAMS = read_yaml(PARAMS_FILE)
app = create_app(PARAMS["serving"], PARAMS["junk_rules"])
