"""
multiday_weather.py — reproducible 7-day weather dataset for rolling-horizon backtest.

Generates 168 hours (7 days) of synthetic weather data with deliberately
varied solar conditions:

  Day 1 (Mon): Clear / sunny          — high radiation, low cloud
  Day 2 (Tue): Partly cloudy          — moderate solar with cloud periods
  Day 3 (Wed): Overcast / low-solar   — heavy cloud, low generation
  Day 4 (Thu): Morning cloud, clear PM— mixed day
  Day 5 (Fri): Clear / sunny          — repeat of good conditions
  Day 6 (Sat): Variable / mixed       — intermittent cloud, weekend load
  Day 7 (Sun): Mostly clear           — moderate cloud, weekend load

This variety is the entire point: a single sunny day does not stress-test the
AI agent vs the rule-based controller.  Across changing conditions the AI's
24-hour look-ahead and tariff-aware planning should produce a measurable —
if modest — advantage.

The data is deterministic (fixed seed) so every run reproduces identically.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .. import config as cfg


# ─── Day-type cloud profiles ────────────────────────────────────────────────
# Each profile returns a 24-element cloud-cover fraction [0, 1] for hours 0–23.

def _cloud_clear(rng: np.random.Generator) -> np.ndarray:
    """Mostly sunny — mean cloud ~15%."""
    base = 0.10 + 0.08 * np.sin(2 * np.pi * np.arange(24) / 24)
    return np.clip(base + rng.normal(0, 0.05, 24), 0, 0.35)


def _cloud_partly(rng: np.random.Generator) -> np.ndarray:
    """Partly cloudy — 40-60% mid-day cloud."""
    h = np.arange(24)
    base = 0.30 + 0.25 * np.exp(-((h - 12) ** 2) / (2 * 5 ** 2))
    return np.clip(base + rng.normal(0, 0.08, 24), 0.1, 0.8)


def _cloud_overcast(rng: np.random.Generator) -> np.ndarray:
    """Overcast / rainy — 70-90% cloud all day."""
    return np.clip(0.80 + rng.normal(0, 0.06, 24), 0.55, 0.95)


def _cloud_morning_then_clear(rng: np.random.Generator) -> np.ndarray:
    """Cloudy morning, clearing by noon."""
    h = np.arange(24)
    base = 0.70 * np.exp(-((h - 7) ** 2) / (2 * 3 ** 2)) + 0.10
    return np.clip(base + rng.normal(0, 0.06, 24), 0.05, 0.85)


def _cloud_variable(rng: np.random.Generator) -> np.ndarray:
    """Variable — AR(1) process, realistic intermittent cloud."""
    cc = np.zeros(24)
    cc[0] = rng.uniform(0.2, 0.6)
    for i in range(1, 24):
        cc[i] = 0.7 * cc[i - 1] + 0.3 * rng.beta(2, 3) + rng.normal(0, 0.08)
    return np.clip(cc, 0.05, 0.90)


DAY_PROFILES = [
    ("clear_sunny",        _cloud_clear),
    ("partly_cloudy",      _cloud_partly),
    ("overcast_low_solar", _cloud_overcast),
    ("morning_cloud",      _cloud_morning_then_clear),
    ("clear_sunny_2",      _cloud_clear),
    ("variable_mixed",     _cloud_variable),
    ("mostly_clear",       _cloud_partly),   # reuse partly but with different seed state
]


def generate_7day_weather(
    start: datetime | None = None,
    latitude: float = cfg.SITE_LATITUDE,
    seed: int = 2025,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Generate a deterministic 7-day (168h) weather dataset.

    Returns (weather_df, meta) in the same format as weather_client.get_weather().
    """
    rng = np.random.default_rng(seed)

    # Start on a Monday 00:00 so weekday peak tariffs apply Mon–Fri
    if start is None:
        start = datetime(2025, 3, 3, 0, 0, 0)  # a Monday
    # Ensure it starts on Monday
    days_to_monday = start.weekday()
    if days_to_monday != 0:
        start = start - timedelta(days=days_to_monday)

    hours = 168  # 7 days
    times = [start + timedelta(hours=h) for h in range(hours)]

    # ─── Build cloud cover per day ──────────────────────────────────────────
    cloud_fraction = np.zeros(hours)
    for d, (label, profile_fn) in enumerate(DAY_PROFILES):
        cloud_fraction[d * 24 : (d + 1) * 24] = profile_fn(rng)

    # ─── Solar geometry → clear-sky GHI ─────────────────────────────────────
    lat_rad = np.deg2rad(latitude)
    doy = np.array([t.timetuple().tm_yday for t in times])
    decl = np.deg2rad(23.45 * np.sin(np.deg2rad(360 / 365 * (doy - 81))))
    h_arr = np.array([t.hour + t.minute / 60.0 for t in times])
    hour_angle = np.deg2rad(15 * (h_arr - 12))
    sin_elev = (
        np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    )
    elev_deg = np.rad2deg(np.arcsin(np.clip(sin_elev, -1, 1)))
    clear_sky_ghi = np.where(elev_deg > 0, 1100 * np.sin(np.deg2rad(elev_deg)), 0.0)

    # ─── Apply cloud attenuation ────────────────────────────────────────────
    radiation = clear_sky_ghi * (1 - 0.75 * cloud_fraction)
    direct = radiation * 0.7
    diffuse = radiation * 0.3

    # ─── Temperature (seasonal + diurnal) ───────────────────────────────────
    temperature = (
        18
        + 6 * np.cos(2 * np.pi * (doy - 15) / 365)
        + 5 * np.sin(2 * np.pi * (h_arr - 9) / 24)
        + rng.normal(0, 0.5, hours)
    )

    cloud_pct = (cloud_fraction * 100).round(1)

    df = pd.DataFrame({
        "shortwave_radiation": radiation.round(1),
        "direct_radiation":    direct.round(1),
        "diffuse_radiation":   diffuse.round(1),
        "cloud_cover":         cloud_pct,
        "temperature_2m":      temperature.round(1),
        "precipitation_probability": (cloud_fraction * 80).round(0).astype(int),
    }, index=pd.DatetimeIndex([pd.Timestamp(t) for t in times], name="time"))

    meta = {
        "source": "synthetic_7day",
        "synthetic": True,
        "days": 7,
        "day_types": [label for label, _ in DAY_PROFILES],
        "seed": seed,
        "latitude": latitude,
        "longitude": cfg.SITE_LONGITUDE,
    }

    return df, meta


def describe_day_types() -> list[dict[str, str]]:
    """Return human-readable descriptions for each day type."""
    return [
        {"day": 1, "weekday": "Mon", "type": "Clear / sunny",
         "description": "High radiation, minimal cloud — best-case solar day."},
        {"day": 2, "weekday": "Tue", "type": "Partly cloudy",
         "description": "Moderate solar with intermittent cloud cover during midday."},
        {"day": 3, "weekday": "Wed", "type": "Overcast / low-solar",
         "description": "Heavy cloud, low PV generation — worst-case day for battery reliance."},
        {"day": 4, "weekday": "Thu", "type": "Morning cloud, clear afternoon",
         "description": "Cloud burns off by noon; late charge opportunity."},
        {"day": 5, "weekday": "Fri", "type": "Clear / sunny",
         "description": "Repeat sunny day — tests whether AI conserves battery for upcoming weekend."},
        {"day": 6, "weekday": "Sat", "type": "Variable / mixed",
         "description": "Intermittent cloud, weekend load profile (lower morning peak)."},
        {"day": 7, "weekday": "Sun", "type": "Mostly clear",
         "description": "Moderate cloud, weekend load profile."},
    ]


if __name__ == "__main__":
    df, meta = generate_7day_weather()
    print(f"Shape: {df.shape}")
    print(f"Day types: {meta['day_types']}")
    for d in range(7):
        day_slice = df.iloc[d * 24 : (d + 1) * 24]
        total_rad = day_slice["shortwave_radiation"].sum()
        avg_cloud = day_slice["cloud_cover"].mean()
        print(f"  Day {d+1} ({meta['day_types'][d]:20s}): "
              f"total GHI={total_rad:.0f} W·h/m²  avg cloud={avg_cloud:.0f}%")
