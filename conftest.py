"""Shared pytest fixtures: a small model is trained once per test session."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Test-only secrets, generated per run. Real values come from the environment.
os.environ.setdefault("INTERNAL_API_TOKEN", "t" * 8 + os.urandom(16).hex())
os.environ.setdefault("MODEL_ADMIN_TOKEN", "a" * 8 + os.urandom(16).hex())


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from ml.data import generate
    from ml.train import save, train

    out = tmp_path_factory.mktemp("artifacts")
    result = train(generate(900, seed=7), fast=True, seed=7, n_jobs=1)
    save(result, out)
    return out


@pytest.fixture
def sample_features() -> dict:
    return {
        "machine_type": "pump",
        "temperature_c": 61.5,
        "vibration_mm_s": 3.1,
        "pressure_bar": 4.8,
        "rpm": 1790,
        "operating_hours": 6400,
        "load_pct": 72.0,
    }
