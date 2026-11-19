"""Feature contract for the risk model.

A contract describes the inputs a trained model expects. It travels with the
model bundle so the model service can validate requests against the exact
model that is loaded, and so a new model can be checked for compatibility with
the one already in production before it is allowed to replace it.

Raw features are stored by the gateway as JSON, so adding a new optional input
(for example ``humidity_pct``) never needs a database migration. What *can*
break clients is a model that suddenly requires a new input, changes a type,
or narrows the accepted values. ``check_compatibility`` blocks those changes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CONTRACT_SCHEMA_VERSION = 1
NUMERIC = "numeric"
CATEGORICAL = "categorical"
_EPS = 1e-6


class ContractError(ValueError):
    """Raised when a contract document is malformed."""


@dataclass(frozen=True)
class ValidationResult:
    clean: dict[str, Any]
    errors: list[str]
    ignored: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class CompatibilityReport:
    breaking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def compatible(self) -> bool:
        return not self.breaking


def validate_contract(contract: Mapping[str, Any]) -> None:
    """Raise ContractError if ``contract`` is not a well-formed contract."""
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ContractError("unsupported contract schema_version")
    classes = contract.get("classes")
    if not isinstance(classes, list) or not classes or not all(isinstance(c, str) for c in classes):
        raise ContractError("contract.classes must be a non-empty list of strings")
    features = contract.get("features")
    if not isinstance(features, dict) or not features:
        raise ContractError("contract.features must be a non-empty object")
    for name, spec in features.items():
        ftype = spec.get("type")
        if ftype == NUMERIC:
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None and hi is not None and lo > hi:
                raise ContractError(f"{name}: min is greater than max")
            if not spec.get("required", True) and spec.get("default") is None:
                raise ContractError(f"{name}: optional numeric features need a default")
        elif ftype == CATEGORICAL:
            allowed = spec.get("allowed")
            if not isinstance(allowed, list) or not allowed:
                raise ContractError(f"{name}: categorical features need an 'allowed' list")
            if not spec.get("required", True) and spec.get("default") not in allowed:
                raise ContractError(f"{name}: optional categorical default must be allowed")
        else:
            raise ContractError(f"{name}: unknown feature type {ftype!r}")


def feature_names(contract: Mapping[str, Any]) -> list[str]:
    return list(contract["features"].keys())


def validate_features(contract: Mapping[str, Any], features: Mapping[str, Any]) -> ValidationResult:
    """Check one request's features against the contract.

    Returns cleaned values in contract order. Unknown keys are not an error:
    they are reported in ``ignored`` so callers can send extra telemetry
    (stored by the gateway) without breaking older models.
    """
    errors: list[str] = []
    clean: dict[str, Any] = {}
    specs: Mapping[str, Any] = contract["features"]

    for name, spec in specs.items():
        value = features.get(name)
        if value is None:
            if spec.get("required", True):
                errors.append(f"{name}: required feature is missing")
            else:
                clean[name] = spec["default"]
            continue

        if spec["type"] == NUMERIC:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{name}: must be a number")
                continue
            number = float(value)
            if not math.isfinite(number):
                errors.append(f"{name}: must be a finite number")
                continue
            lo, hi = spec.get("min"), spec.get("max")
            if (lo is not None and number < lo) or (hi is not None and number > hi):
                errors.append(f"{name}: must be between {lo} and {hi}")
                continue
            clean[name] = number
        else:
            if not isinstance(value, str):
                errors.append(f"{name}: must be a string")
                continue
            if value not in spec["allowed"]:
                errors.append(f"{name}: must be one of {', '.join(spec['allowed'])}")
                continue
            clean[name] = value

    ignored = sorted(key for key in features if key not in specs)
    return ValidationResult(clean=clean, errors=errors, ignored=ignored)


def check_compatibility(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> CompatibilityReport:
    """Compare a candidate model's contract with the one currently in production."""
    report = CompatibilityReport()

    if candidate.get("schema_version") != baseline.get("schema_version"):
        report.breaking.append("contract schema_version changed")
    if list(candidate.get("classes", [])) != list(baseline.get("classes", [])):
        report.breaking.append(f"output classes changed from {baseline.get('classes')} to {candidate.get('classes')}")

    base_features: Mapping[str, Any] = baseline.get("features", {})
    cand_features: Mapping[str, Any] = candidate.get("features", {})

    for name, cand in cand_features.items():
        base = base_features.get(name)
        cand_required = cand.get("required", True)
        if base is None:
            if cand_required:
                report.breaking.append(f"{name}: new feature is required (make it optional with a default)")
            else:
                report.warnings.append(f"{name}: new optional feature added")
            continue

        if cand.get("type") != base.get("type"):
            report.breaking.append(f"{name}: type changed from {base.get('type')} to {cand.get('type')}")
            continue
        if cand_required and not base.get("required", True):
            report.breaking.append(f"{name}: optional feature became required")

        if cand["type"] == CATEGORICAL:
            removed = sorted(set(base.get("allowed", [])) - set(cand.get("allowed", [])))
            if removed:
                report.breaking.append(f"{name}: categories no longer accepted: {', '.join(removed)}")
        else:
            b_lo, b_hi = base.get("min"), base.get("max")
            c_lo, c_hi = cand.get("min"), cand.get("max")
            if c_lo is not None and (b_lo is None or c_lo > b_lo):
                report.breaking.append(f"{name}: minimum raised from {b_lo} to {c_lo}")
            if c_hi is not None and (b_hi is None or c_hi < b_hi):
                report.breaking.append(f"{name}: maximum lowered from {b_hi} to {c_hi}")

    for name in base_features:
        if name not in cand_features:
            report.warnings.append(f"{name}: feature no longer used by the model (clients may still send it)")

    return report


# ---------------------------------------------------------------- data drift


def reference_stats(values: np.ndarray, *, categorical: bool, bins: int = 10) -> dict[str, Any]:
    """Summarise a training column so later data can be compared against it."""
    if categorical:
        cats, counts = np.unique(values.astype(str), return_counts=True)
        props = counts / counts.sum()
        return {"categories": cats.tolist(), "proportions": props.round(6).tolist()}
    arr = values.astype(float)
    edges = np.unique(np.quantile(arr, np.linspace(0, 1, bins + 1)))
    counts, _ = np.histogram(np.clip(arr, edges[0], edges[-1]), bins=edges)
    props = counts / counts.sum()
    return {"edges": edges.round(6).tolist(), "proportions": props.round(6).tolist()}


def population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    actual = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def column_psi(ref: Mapping[str, Any], values: np.ndarray) -> float:
    if "categories" in ref:
        cats = ref["categories"]
        vals = values.astype(str)
        actual = np.array([(vals == c).mean() for c in cats])
        return population_stability_index(np.array(ref["proportions"]), actual)
    edges = np.array(ref["edges"], dtype=float)
    counts, _ = np.histogram(np.clip(values.astype(float), edges[0], edges[-1]), bins=edges)
    actual = counts / max(counts.sum(), 1)
    return population_stability_index(np.array(ref["proportions"]), actual)
