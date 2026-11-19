# syntax=docker/dockerfile:1
# Build context: repository root (see docker-compose.yml).
ARG PYTHON_IMAGE=python:3.13-slim

# ---- dependencies (hash-verified) --------------------------------------------
FROM ${PYTHON_IMAGE} AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY services/model-service/requirements.txt /tmp/requirements.txt
RUN pip install --require-hashes -r /tmp/requirements.txt

# ---- train the model reproducibly (no binary artefacts in git) ----------------
FROM deps AS trainer
WORKDIR /build
COPY ml/ ml/
ARG TRAIN_ROWS=6000
ARG TRAIN_SEED=42
RUN python -m ml.train --synthetic "${TRAIN_ROWS}" --seed "${TRAIN_SEED}" --out /build/artifacts \
 && python -m ml.drift_check --baseline ml/baseline_contract.json --candidate /build/artifacts/contract.json

# ---- runtime -----------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/model.joblib \
    MANIFEST_PATH=/app/artifacts/model_manifest.json
RUN groupadd --system app && useradd --system --gid app --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=deps /opt/venv /opt/venv
COPY ml/__init__.py ml/contract.py ml/
COPY services/model-service/model_service/ model_service/
COPY --from=trainer /build/artifacts/ artifacts/
# Code and model are owned by root and read-only for the service user.
USER app
EXPOSE 8001 9100
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import sys,urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=2).status != 200)"]
CMD ["uvicorn", "--factory", "model_service.main:app_factory", "--host", "0.0.0.0", "--port", "8001", "--no-server-header", "--no-access-log"]
