# Serving image: requirements-serve.txt only (no TensorFlow, no pandas). Model files must already be real files
# (CD runs `dvc pull` + scripts/verify_manifest.py BEFORE this build; the app re-checks them at startup).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY app/ app/
COPY params.yaml .
# Only what serving.files lists (both models: switching lr <-> bert is a params change, not a new artifact)
COPY models/manifest.json models/label_order.json models/tfidf_b.joblib models/logreg_b.joblib \
     models/bert.onnx models/bert_vocab.txt models/

ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=$GIT_COMMIT

# Non-root. NO chown: the app only READS its files, so root-owned = it can't modify its own code or models
# (and `chown -R` would duplicate the 131 MB model layer)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
# Render injects PORT; 8000 locally. exec: uvicorn becomes PID 1 and receives Render's SIGTERM on redeploy
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
