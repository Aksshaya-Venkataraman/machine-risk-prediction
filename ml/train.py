"""Train the RandomForest risk classifier.

Hyper-parameters are tuned with GridSearchCV over 5-fold stratified
cross-validation on the training split; the untouched 20% hold-out split is
used once at the end for the reported precision/recall.

    python -m ml.train --synthetic 6000 --out artifacts
    python -m ml.train --data data/telemetry.csv --out artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.contract import (
    CATEGORICAL,
    CONTRACT_SCHEMA_VERSION,
    NUMERIC,
    reference_stats,
    validate_contract,
)
from ml.data import MACHINE_TYPES, RISK_CLASSES, generate

TARGET = "risk_level"

# Physical limits: inputs outside these ranges are sensor faults, not data.
BASE_FEATURES: dict[str, dict[str, Any]] = {
    "machine_type": {"type": CATEGORICAL, "required": True, "allowed": sorted(MACHINE_TYPES)},
    "temperature_c": {"type": NUMERIC, "required": True, "min": -40.0, "max": 200.0},
    "vibration_mm_s": {"type": NUMERIC, "required": True, "min": 0.0, "max": 100.0},
    "pressure_bar": {"type": NUMERIC, "required": True, "min": 0.0, "max": 50.0},
    "rpm": {"type": NUMERIC, "required": True, "min": 0.0, "max": 10_000.0},
    "operating_hours": {"type": NUMERIC, "required": True, "min": 0.0, "max": 100_000.0},
    "load_pct": {"type": NUMERIC, "required": True, "min": 0.0, "max": 100.0},
}
HUMIDITY_FEATURE = {"type": NUMERIC, "required": False, "default": 45.0, "min": 0.0, "max": 100.0}

FULL_GRID = {
    "clf__n_estimators": [150, 300],
    "clf__max_depth": [None, 14],
    "clf__min_samples_leaf": [1, 4],
    "clf__max_features": ["sqrt", 0.5],
}
FAST_GRID = {
    "clf__n_estimators": [40],
    "clf__max_depth": [None, 8],
    "clf__min_samples_leaf": [2],
    "clf__max_features": ["sqrt"],
}


@dataclass
class TrainResult:
    pipeline: Pipeline
    contract: dict[str, Any]
    metrics: dict[str, Any]


def feature_spec(include_humidity: bool = False) -> dict[str, dict[str, Any]]:
    spec = {name: dict(value) for name, value in BASE_FEATURES.items()}
    if include_humidity:
        spec["humidity_pct"] = dict(HUMIDITY_FEATURE)
    return spec


def build_pipeline(spec: dict[str, dict[str, Any]], seed: int) -> Pipeline:
    categorical = [n for n, s in spec.items() if s["type"] == CATEGORICAL]
    preprocess = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )
    clf = RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1)
    return Pipeline([("prep", preprocess), ("clf", clf)])


def train(
    df: pd.DataFrame,
    *,
    spec: dict[str, dict[str, Any]] | None = None,
    fast: bool = False,
    seed: int = 42,
    n_jobs: int = -1,
) -> TrainResult:
    spec = spec or feature_spec()
    missing = [c for c in [*spec, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"training data is missing columns: {missing}")
    unknown_labels = set(df[TARGET].unique()) - set(RISK_CLASSES)
    if unknown_labels:
        raise ValueError(f"unknown risk labels in training data: {sorted(unknown_labels)}")

    columns = list(spec)
    X, y = df[columns], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(
        build_pipeline(spec, seed),
        param_grid=FAST_GRID if fast else FULL_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train)
    best: Pipeline = search.best_estimator_

    y_pred = best.predict(X_test)
    report = classification_report(y_test, y_pred, labels=list(RISK_CLASSES), output_dict=True, zero_division=0)
    per_class = {
        label: {
            "precision": round(report[label]["precision"], 4),
            "recall": round(report[label]["recall"], 4),
            "f1": round(report[label]["f1-score"], 4),
            "support": int(report[label]["support"]),
        }
        for label in RISK_CLASSES
    }
    metrics = {
        "cv_folds": 5,
        "cv_best_f1_macro": round(float(search.best_score_), 4),
        "best_params": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
        "holdout_accuracy": round(report["accuracy"], 4),
        "holdout_f1_macro": round(report["macro avg"]["f1-score"], 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(RISK_CLASSES),
            "matrix": confusion_matrix(y_test, y_pred, labels=list(RISK_CLASSES)).tolist(),
        },
        "train_rows": len(X_train),
        "holdout_rows": len(X_test),
    }

    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "target": TARGET,
        "classes": list(RISK_CLASSES),
        "features": spec,
        "reference": {
            name: reference_stats(X_train[name].to_numpy(), categorical=s["type"] == CATEGORICAL)
            for name, s in spec.items()
        },
    }
    validate_contract(contract)
    return TrainResult(pipeline=best, contract=contract, metrics=metrics)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def save(result: TrainResult, out_dir: Path) -> dict[str, Any]:
    """Write model.joblib plus a manifest whose SHA-256 the service verifies on load."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(UTC)
    model_path = out_dir / "model.joblib"
    bundle = {"pipeline": result.pipeline, "contract": result.contract}
    joblib.dump(bundle, model_path, compress=3)

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = {
        "version": f"rf-{trained_at:%Y%m%d}-{digest[:8]}",
        "sha256": digest,
        "trained_at": trained_at.isoformat(),
        "sklearn_version": sklearn.__version__,
        "metrics": result.metrics,
    }
    (out_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2, default=_jsonable))
    (out_dir / "contract.json").write_text(json.dumps(result.contract, indent=2, default=_jsonable))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the machine risk classifier")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="CSV with telemetry and a risk_level column")
    source.add_argument("--synthetic", type=int, metavar="ROWS", help="generate synthetic data")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="tiny grid for CI smoke tests")
    parser.add_argument("--with-humidity", action="store_true", help="add optional humidity_pct input")
    args = parser.parse_args()

    df = pd.read_csv(args.data) if args.data else generate(args.synthetic, seed=args.seed)
    result = train(df, spec=feature_spec(args.with_humidity), fast=args.fast, seed=args.seed)
    manifest = save(result, args.out)

    print(f"model {manifest['version']} written to {args.out}")
    print(f"best params: {result.metrics['best_params']}")
    print(f"5-fold CV macro F1: {result.metrics['cv_best_f1_macro']}")
    for label, m in result.metrics["per_class"].items():
        print(f"  {label:<6} precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
