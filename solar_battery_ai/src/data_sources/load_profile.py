"""
load_profile.py — synthetic household electricity demand.

A representative AU residential profile (kW per hour):
  - 0.30 kW always-on baseline
  - morning peak ~07:30 (smaller on weekends)
  - evening peak ~19:00
  - winter heating bump (June-August in southern hemisphere)

The same load profile is used by every dispatch policy, so the comparison
between baselines and the AI agent is fair.

Future work: load forecasting from past consumption history, or user-uploaded
CSV. For now, transparent and reproducible synthesis is enough.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .. import config as cfg


def synthetic_load(
    index: pd.DatetimeIndex,
    *,
    seed: int | None = 7,
    weekend_factor: float = 0.85,
) -> pd.Series:
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    n = len(index)
    h = np.asarray(index.hour)
    is_weekday = np.asarray(index.dayofweek) < 5
    doy = np.asarray(index.dayofyear)

    base = 0.30 * np.ones(n)
    morning = 1.5 * np.exp(-((h - 7.5) ** 2) / (2 * 1.5 ** 2)) * np.where(is_weekday, 1.0, 0.5)
    evening = 2.0 * np.exp(-((h - 19) ** 2) / (2 * 1.8 ** 2))
    winter_heating = 0.4 * np.maximum(np.cos(2 * np.pi * (doy - 200) / 365), 0)

    load = base + morning + evening + winter_heating
    load *= np.where(is_weekday, 1.0, weekend_factor)

    # AR(1) noise
    noise = np.zeros(n)
    noise[0] = rng.normal(0, 0.1)
    for i in range(1, n):
        noise[i] = 0.5 * noise[i - 1] + rng.normal(0, 0.12)

    return pd.Series(np.clip(load + noise, 0.1, None), index=index, name="load_kw")


def from_csv(path: str, index: pd.DatetimeIndex) -> pd.Series:
    """Load a user-supplied CSV with columns [timestamp, load_kw], reindex to match."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "load_kw" not in df.columns:
        raise ValueError("CSV must have columns 'timestamp' and 'load_kw'.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    s = df.set_index("timestamp")["load_kw"]
    return s.reindex(index).interpolate("time").ffill().bfill()


if __name__ == "__main__":
    idx = pd.date_range("2025-01-15 00:00", periods=48, freq="h")
    s = synthetic_load(idx)
    print(s.head(12).round(2))
    print(f"Daily total: {s.iloc[:24].sum():.1f} kWh   ({s.mean():.2f} kW avg)")
