from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ml import drift_check
from ml.contract import (
    ContractError,
    check_compatibility,
    population_stability_index,
    validate_contract,
    validate_features,
)
from ml.data import RISK_CLASSES, generate
from ml.train import feature_spec, train

BASELINE = Path(__file__).resolve().parents[1] / "baseline_contract.json"


@pytest.fixture
def contract(trained_artifacts: Path) -> dict:
    return json.loads((trained_artifacts / "contract.json").read_text())


def test_generator_is_deterministic_and_balanced():
    a, b = generate(500, seed=1), generate(500, seed=1)
    assert a.equals(b)
    shares = a["risk_level"].value_counts(normalize=True)
    assert set(shares.index) == set(RISK_CLASSES)
    assert 0.45 <= shares["Low"] <= 0.55
    assert 0.15 <= shares["High"] <= 0.25


def test_generator_rejects_tiny_datasets():
    with pytest.raises(ValueError):
        generate(10)


def test_training_reports_cv_and_per_class_metrics(trained_artifacts: Path):
    manifest = json.loads((trained_artifacts / "model_manifest.json").read_text())
    metrics = manifest["metrics"]
    assert metrics["cv_folds"] == 5
    assert set(metrics["per_class"]) == set(RISK_CLASSES)
    for values in metrics["per_class"].values():
        assert 0.0 <= values["precision"] <= 1.0
        assert 0.0 <= values["recall"] <= 1.0
    assert len(manifest["sha256"]) == 64
    assert manifest["version"].startswith("rf-")


def test_training_rejects_missing_columns():
    df = generate(200).drop(columns=["rpm"])
    with pytest.raises(ValueError, match="missing columns"):
        train(df, fast=True, n_jobs=1)


def test_training_rejects_unknown_labels():
    df = generate(200)
    df.loc[0, "risk_level"] = "Critical"
    with pytest.raises(ValueError, match="unknown risk labels"):
        train(df, fast=True, n_jobs=1)


def test_validate_features_accepts_good_input(contract, sample_features):
    result = validate_features(contract, {**sample_features, "humidity_pct": 40})
    assert result.ok
    assert result.ignored == ["humidity_pct"]
    assert list(result.clean) == list(contract["features"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"rpm": None}, "required"),
        ({"rpm": "fast"}, "must be a number"),
        ({"rpm": True}, "must be a number"),
        ({"load_pct": 140}, "between"),
        ({"temperature_c": float("nan")}, "finite"),
        ({"machine_type": "rocket"}, "must be one of"),
        ({"machine_type": 3}, "must be a string"),
    ],
)
def test_validate_features_rejects_bad_input(contract, sample_features, change, message):
    result = validate_features(contract, {**sample_features, **change})
    assert not result.ok
    assert any(message in err for err in result.errors)


def test_optional_feature_uses_default(contract, sample_features):
    contract = copy.deepcopy(contract)
    contract["features"]["humidity_pct"] = feature_spec(True)["humidity_pct"]
    result = validate_features(contract, sample_features)
    assert result.ok
    assert result.clean["humidity_pct"] == 45.0


def test_validate_contract_rejects_bad_documents(contract):
    broken = copy.deepcopy(contract)
    broken["features"]["rpm"]["type"] = "text"
    with pytest.raises(ContractError):
        validate_contract(broken)
    broken = copy.deepcopy(contract)
    broken["features"]["extra"] = {"type": "numeric", "required": False}
    with pytest.raises(ContractError, match="default"):
        validate_contract(broken)


def test_adding_optional_feature_is_compatible(contract):
    candidate = copy.deepcopy(contract)
    candidate["features"]["humidity_pct"] = feature_spec(True)["humidity_pct"]
    report = check_compatibility(contract, candidate)
    assert report.compatible
    assert any("humidity_pct" in w for w in report.warnings)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["features"].__setitem__("humidity_pct", {"type": "numeric", "required": True}),
        lambda c: c["features"]["rpm"].__setitem__("type", "categorical"),
        lambda c: c["features"]["machine_type"].__setitem__("allowed", ["pump"]),
        lambda c: c["features"]["load_pct"].__setitem__("max", 50.0),
        lambda c: c.__setitem__("classes", ["Low", "High"]),
    ],
    ids=["new-required", "type-change", "categories-removed", "range-narrowed", "classes-changed"],
)
def test_breaking_changes_are_blocked(contract, mutate):
    candidate = copy.deepcopy(contract)
    mutate(candidate)
    assert not check_compatibility(contract, candidate).compatible


def test_psi_is_zero_for_identical_and_large_for_shifted():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert population_stability_index(p, p) == pytest.approx(0.0)
    assert population_stability_index(p, np.array([0.7, 0.1, 0.1, 0.1])) > 0.25


def test_drift_check_cli_exit_codes(tmp_path, contract):
    base = tmp_path / "base.json"
    base.write_text(json.dumps(contract))
    assert drift_check.main(["--baseline", str(base), "--candidate", str(base)]) == 0

    breaking = copy.deepcopy(contract)
    breaking["features"]["rpm"]["min"] = 500.0
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps(breaking))
    assert drift_check.main(["--baseline", str(base), "--candidate", str(cand)]) == 1
    assert drift_check.main(["--baseline", str(base), "--candidate", str(tmp_path / "nope.json")]) == 2


def test_data_drift_flags_shifted_data(contract):
    reference = generate(800, seed=3)
    shifted = reference.copy()
    shifted["temperature_c"] += 25
    assert drift_check.data_drift(contract, reference)["temperature_c"]["status"] == "ok"
    assert drift_check.data_drift(contract, shifted)["temperature_c"]["status"] == "alert"


def test_committed_baseline_matches_training_schema(contract):
    baseline = json.loads(BASELINE.read_text())
    validate_contract(baseline)
    assert check_compatibility(baseline, contract).compatible
