from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient
from gateway.db import Base, make_engine
from gateway.main import build_model_client, create_app
from gw_support import gateway_settings
from model_service.config import Settings as ModelSettings
from model_service.main import create_app as create_model_app


@pytest.fixture
def model_service(trained_artifacts: Path, tmp_path: Path) -> Iterator[TestClient]:
    for name in ("model.joblib", "model_manifest.json"):
        shutil.copy(trained_artifacts / name, tmp_path / name)
    settings = ModelSettings(
        environment="test",
        model_path=tmp_path / "model.joblib",
        manifest_path=tmp_path / "model_manifest.json",
        allowed_hosts=["testserver"],
        metrics_port=0,
    )
    with TestClient(create_model_app(settings)) as client:
        yield client


@pytest.fixture
def make_gateway() -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def factory(transport: httpx2.BaseTransport, **overrides) -> TestClient:
        settings = gateway_settings(**overrides)
        engine = make_engine("sqlite://")
        Base.metadata.create_all(engine)
        client_for_model = build_model_client(settings, transport=transport, sleep=lambda _: None)
        client = TestClient(create_app(settings, engine=engine, model_client=client_for_model))
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def gateway(make_gateway, model_service: TestClient) -> TestClient:
    """Gateway wired to the real model-service app (in-process, over its ASGI transport)."""
    return make_gateway(model_service._transport, model_service_url="http://testserver")
