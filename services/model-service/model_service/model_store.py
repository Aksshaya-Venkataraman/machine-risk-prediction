"""Loads the trained model bundle, verifies it, and swaps it in safely on reload."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.contract import (
    CompatibilityReport,
    ContractError,
    check_compatibility,
    feature_names,
    validate_contract,
)

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    pass


class IncompatibleModelError(RuntimeError):
    def __init__(self, report: CompatibilityReport) -> None:
        super().__init__("candidate model is not compatible with the loaded model")
        self.report = report


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Any
    contract: dict[str, Any]
    version: str
    sha256: str
    trained_at: str
    metrics: dict[str, Any]
    loaded_at: str

    def predict(self, clean_features: dict[str, Any]) -> tuple[str, dict[str, float]]:
        row = pd.DataFrame(
            [[clean_features[n] for n in feature_names(self.contract)]], columns=feature_names(self.contract)
        )
        probs = self.pipeline.predict_proba(row)[0]
        by_class = {str(label): float(p) for label, p in zip(self.pipeline.classes_, probs, strict=True)}
        ordered = {label: round(by_class.get(label, 0.0), 4) for label in self.contract["classes"]}
        risk = max(ordered, key=ordered.__getitem__)
        return risk, ordered


class ModelStore:
    """Holds the current model. Reads are lock-free; reloads are serialised."""

    def __init__(self, model_path: Path, manifest_path: Path) -> None:
        self._model_path = model_path
        self._manifest_path = manifest_path
        self._current: LoadedModel | None = None
        self._reload_lock = threading.Lock()

    @property
    def current(self) -> LoadedModel | None:
        return self._current

    def _read(self) -> LoadedModel:
        try:
            manifest = json.loads(self._manifest_path.read_text())
            data = self._model_path.read_bytes()
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelLoadError(f"cannot read model artifacts: {exc}") from exc

        # joblib uses pickle, which can execute code. Only load a file whose
        # hash matches the manifest written by our own training job.
        expected = str(manifest.get("sha256", ""))
        actual = hashlib.sha256(data).hexdigest()
        if not expected or not hmac.compare_digest(expected, actual):
            raise ModelLoadError("model checksum does not match manifest; refusing to load")

        bundle = joblib.load(self._model_path)
        if not isinstance(bundle, dict) or {"pipeline", "contract"} - bundle.keys():
            raise ModelLoadError("model bundle is missing pipeline or contract")
        try:
            validate_contract(bundle["contract"])
        except ContractError as exc:
            raise ModelLoadError(f"invalid contract in model bundle: {exc}") from exc

        return LoadedModel(
            pipeline=bundle["pipeline"],
            contract=bundle["contract"],
            version=str(manifest.get("version", "unknown")),
            sha256=actual,
            trained_at=str(manifest.get("trained_at", "")),
            metrics=manifest.get("metrics", {}),
            loaded_at=datetime.now(UTC).isoformat(),
        )

    def load(self) -> LoadedModel:
        with self._reload_lock:
            self._current = self._read()
            log.info("model loaded: %s", self._current.version)
            return self._current

    def reload(self) -> tuple[LoadedModel, CompatibilityReport]:
        """Load the model from disk again, but only if it is compatible."""
        with self._reload_lock:
            candidate = self._read()
            if self._current is None:
                self._current = candidate
                return candidate, CompatibilityReport()
            report = check_compatibility(self._current.contract, candidate.contract)
            if not report.compatible:
                raise IncompatibleModelError(report)
            self._current = candidate
            return candidate, report
