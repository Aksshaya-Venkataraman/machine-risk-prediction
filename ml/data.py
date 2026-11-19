"""Synthetic machine telemetry with a known risk mechanism.

Real plant data is confidential, so the repository ships a reproducible
generator. Risk is driven by wear (hours since maintenance), heat, vibration,
load and pressure deviation, plus measurement noise, so the classes overlap
the way real sensor data does. ``humidity_pct`` is recorded but is not part of
the v1 model: it demonstrates how a new input is stored and later adopted
without a schema migration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MACHINE_TYPES = ("compressor", "conveyor", "press", "pump")
RISK_CLASSES = ("Low", "Medium", "High")

_PROFILE = {
    #               temp_c  rpm    pressure_bar
    "compressor": (68.0, 3000.0, 8.0),
    "conveyor": (45.0, 900.0, 2.0),
    "press": (60.0, 1200.0, 12.0),
    "pump": (55.0, 1800.0, 5.0),
}


def generate(n: int = 6000, seed: int = 42, noise: float = 0.55) -> pd.DataFrame:
    """Return ``n`` rows of telemetry and a Low/Medium/High risk label."""
    if n < 30:
        raise ValueError("n must be at least 30")
    rng = np.random.default_rng(seed)

    machine_type = rng.choice(MACHINE_TYPES, size=n)
    base_temp = np.array([_PROFILE[m][0] for m in machine_type])
    base_rpm = np.array([_PROFILE[m][1] for m in machine_type])
    base_pressure = np.array([_PROFILE[m][2] for m in machine_type])

    operating_hours = rng.uniform(0, 12_000, n)
    wear = operating_hours / 12_000
    load_pct = np.clip(rng.normal(65, 18, n), 5, 100)

    temperature_c = base_temp + 0.22 * (load_pct - 65) + 14 * wear + rng.normal(0, 4, n)
    vibration_mm_s = np.exp(rng.normal(np.log(2.2) + 0.9 * wear + 0.006 * (load_pct - 65), 0.28))
    pressure_bar = base_pressure * (1 + rng.normal(0, 0.07, n) - 0.12 * wear * rng.random(n))
    rpm = base_rpm * (1 + rng.normal(0, 0.04, n))
    humidity_pct = np.clip(rng.normal(45, 12, n), 5, 95)

    latent = (
        2.2 * wear
        + 0.07 * (temperature_c - base_temp)
        + 1.3 * np.log(vibration_mm_s / 2.2)
        + 0.012 * (load_pct - 65)
        + 4.0 * np.abs(pressure_bar / base_pressure - 1)
        + 1.2 * wear * (load_pct > 85)
        + rng.normal(0, noise, n)
    )
    low_cut, high_cut = np.quantile(latent, [0.5, 0.8])
    risk = np.where(latent < low_cut, "Low", np.where(latent < high_cut, "Medium", "High"))

    return pd.DataFrame(
        {
            "machine_id": [f"M-{i:05d}" for i in range(n)],
            "machine_type": machine_type,
            "temperature_c": temperature_c.round(2),
            "vibration_mm_s": vibration_mm_s.round(3),
            "pressure_bar": pressure_bar.round(3),
            "rpm": rpm.round(1),
            "operating_hours": operating_hours.round(1),
            "load_pct": load_pct.round(2),
            "humidity_pct": humidity_pct.round(2),
            "risk_level": risk,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic machine telemetry")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/telemetry.csv"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    generate(args.rows, args.seed).to_csv(args.out, index=False)
    print(f"wrote {args.rows} rows to {args.out}")


if __name__ == "__main__":
    main()
