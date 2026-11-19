from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from model_service.config import Settings
from model_service.main import create_app
from model_service.model_store import ModelLoadError, ModelStore
from prometheus_client import REGISTRY

TOKEN = os.environ["INTERNAL_API_TOKEN"]
ADMIN = os.environ["MODEL_ADMIN_TOKEN"]
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def artifacts(trained_artifacts: Path, tmp_path: Path) -> Path:
    for name in ("model.joblib", "model_manifest.json", "contract.json"):
        shutil.copy(trained_artifacts / name, tmp_path / name)
    return tmp_path


@pytest.fixture
def client(artifacts: Path):
    settings = Settings(
        environment="test",
        model_path=artifacts / "model.joblib",
        manifest_path=artifacts / "model_manifest.json",
        allowed_hosts=["testserver"],
        metrics_port=0,
    )
    with TestClient(create_app(settings)) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["model_version"].startswith("rf-")


def test_predict_returns_risk_and_probabilities(client, sample_features):
    resp = client.post("/v1/predict", json={"features": sample_features}, headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert list(body["probabilities"]) == ["Low", "Medium", "High"]
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_extra_features_are_ignored_not_rejected(client, sample_features):
    resp = client.post("/v1/predict", json={"features": {**sample_features, "humidity_pct": 51.0}}, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["ignored_features"] == ["humidity_pct"]


def test_contract_violations_return_422(client, sample_features):
    resp = client.post("/v1/predict", json={"features": {**sample_features, "load_pct": 500}}, headers=AUTH)
    assert resp.status_code == 422
    assert "load_pct" in resp.json()["detail"]["errors"][0]


def test_malformed_feature_names_rejected(client, sample_features):
    resp = client.post("/v1/predict", json={"features": {**sample_features, "<script>": 1}}, headers=AUTH)
    assert resp.status_code == 422


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}])
def test_predict_requires_internal_token(client, sample_features, headers):
    resp = client.post("/v1/predict", json={"features": sample_features}, headers=headers)
    assert resp.status_code == 401


def test_admin_reload_requires_admin_token(client):
    assert client.post("/v1/admin/reload", headers=AUTH).status_code == 401
    resp = client.post("/v1/admin/reload", headers={"Authorization": f"Bearer {ADMIN}"})
    assert resp.status_code == 200
    assert resp.json()["version"].startswith("rf-")


def test_request_id_is_echoed_or_generated(client):
    rid = "trace-abc-12345"
    assert client.get("/health", headers={"X-Request-ID": rid}).headers["x-request-id"] == rid
    generated = client.get("/health", headers={"X-Request-ID": "bad id\n"}).headers["x-request-id"]
    assert generated != "bad id\n"
    assert len(generated) == 32


def test_security_headers_present(client):
    headers = client.get("/health").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert "default-src 'none'" in headers["content-security-policy"]
    assert headers["cache-control"] == "no-store"


def test_docs_disabled_and_unknown_hosts_rejected(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/health", headers={"Host": "evil.example"}).status_code == 400


def test_oversized_body_rejected(client):
    resp = client.post(
        "/v1/predict",
        content=b"{" + b" " * 20_000 + b"}",
        headers={**AUTH, "Content-Type": "application/json"},
    )
    assert resp.status_code == 413


def test_metrics_are_recorded(client, sample_features):
    client.post("/v1/predict", json={"features": sample_features}, headers=AUTH)
    total = sum(
        sample.value
        for metric in REGISTRY.collect()
        if metric.name == "model_service_predictions"
        for sample in metric.samples
        if sample.name == "model_service_predictions_total"
    )
    assert total >= 1


def test_tampered_model_is_refused(artifacts: Path):
    manifest = json.loads((artifacts / "model_manifest.json").read_text())
    manifest["sha256"] = "0" * 64
    (artifacts / "model_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ModelLoadError, match="checksum"):
        ModelStore(artifacts / "model.joblib", artifacts / "model_manifest.json").load()


def test_weak_secrets_rejected_by_settings():
    with pytest.raises(ValueError):
        Settings(internal_api_token="short", model_admin_token=ADMIN)
    with pytest.raises(ValueError):
        Settings(internal_api_token="change-me-" + "x" * 40, model_admin_token=ADMIN)
