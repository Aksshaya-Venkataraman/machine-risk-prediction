"""Block incompatible model changes and report data drift.

Exit codes: 0 = compatible, 1 = breaking change found, 2 = bad input.

    python -m ml.drift_check --baseline ml/baseline_contract.json \
        --candidate artifacts/contract.json [--data new_batch.csv]

Run in CI before a new model is published, and by the model service before a
hot reload replaces the model in memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ml.contract import ContractError, check_compatibility, column_psi, validate_contract

PSI_WARN = 0.1
PSI_ALERT = 0.25


def data_drift(contract: dict[str, Any], df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Population Stability Index per feature against the training reference."""
    out: dict[str, dict[str, Any]] = {}
    for name, ref in contract.get("reference", {}).items():
        if name not in df.columns:
            out[name] = {"psi": None, "status": "missing"}
            continue
        psi = column_psi(ref, df[name].dropna().to_numpy())
        status = "alert" if psi >= PSI_ALERT else "warn" if psi >= PSI_WARN else "ok"
        out[name] = {"psi": round(psi, 4), "status": status}
    return out


def _load(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text())
    validate_contract(contract)
    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="optional CSV to check for data drift")
    args = parser.parse_args(argv)

    try:
        baseline, candidate = _load(args.baseline), _load(args.candidate)
    except (OSError, json.JSONDecodeError, ContractError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = check_compatibility(baseline, candidate)
    for warning in report.warnings:
        print(f"WARN   {warning}")
    for problem in report.breaking:
        print(f"BREAK  {problem}")

    if args.data:
        for name, result in data_drift(baseline, pd.read_csv(args.data)).items():
            print(f"DRIFT  {name:<16} psi={result['psi']} ({result['status']})")

    if not report.compatible:
        print("candidate contract is NOT compatible with the baseline", file=sys.stderr)
        return 1
    print("candidate contract is compatible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
