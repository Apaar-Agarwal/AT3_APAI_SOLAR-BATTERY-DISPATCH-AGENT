"""
weather_client.py — Open-Meteo weather/solar source client.

The agent is *source-aware*: it does not train its own forecasting model.
Instead it pulls hourly weather + solar radiation forecasts from Open-Meteo
(a free public API) and feeds them into the state builder.

Resilience:
  - try the live API
  - on any failure (no internet, rate limit, parse error) fall back to a
    cached JSON snapshot in data/cache/
  - if cache is missing, generate a deterministic synthetic snapshot
    (clear-sky model + cloud noise) so the demo still runs offline

This is the *external source estimate*, not a forecast we own.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from .. import config as cfg

log = logging.getLogger(__name__)


# ─── live fetch ────────────────────────────────────────────────────────────
def fetch_open_meteo(
    latitude: float = cfg.SITE_LATITUDE,
    longitude: float = cfg.SITE_LONGITUDE,
    timezone: str = cfg.SITE_TIMEZONE,
    timeout_s: int = 8,
) -> dict[str, Any] | None:
    """Try Open-Meteo. Return the parsed JSON, or None on any error."""
    if requests is None:
        return None
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(cfg.OPEN_METEO_VARIABLES),
        "forecast_days": 2,        # gives us at least 24h ahead from "now"
        "timezone": timezone,
    }
    try:
        r = requests.get(cfg.OPEN_METEO_URL, params=params, timeout=timeout_s)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("Open-Meteo fetch failed: %s — using cache/fallback", exc)
        return None


# ─── cache I/O ──────────────────────────────────────────────────────────────
def save_cache(payload: dict[str, Any], path: Path = cfg.CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_cache(path: Path = cfg.CACHE_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Cache read failed: %s", exc)
        return None


# ─── deterministic offline fallback ─────────────────────────────────────────
def _synthetic_payload(
    start: datetime, hours: int = 48, latitude: float = cfg.SITE_LATITUDE
) -> dict[str, Any]:
    """Physically-grounded synthetic weather snapshot for offline use.

    Snaps `start` down to the most recent midnight so the 24h horizon
    starts at 00:00 — this matches how Open-Meteo serves a full forecast
    day and means the dispatch window naturally spans a full weekday
    (including the 16-21 peak window).
    """
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    # snap to nearest Monday (so synthetic horizon always covers a weekday peak)
    days_to_monday = start.weekday()
    if days_to_monday != 0:
        start = start - timedelta(days=days_to_monday)
    rng = np.random.default_rng(42)
    times = [start + timedelta(hours=h) for h in range(hours)]

    # solar elevation → clear-sky GHI
    lat = np.deg2rad(latitude)
    doy = np.array([t.timetuple().tm_yday for t in times])
    decl = np.deg2rad(23.45 * np.sin(np.deg2rad(360 / 365 * (doy - 81))))
    h = np.array([t.hour + t.minute / 60.0 for t in times])
    hour_angle = np.deg2rad(15 * (h - 12))
    sin_elev = (
        np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    )
    elev = np.rad2deg(np.arcsin(np.clip(sin_elev, -1, 1)))
    clear_sky = np.where(elev > 0, 1100 * np.sin(np.deg2rad(elev)), 0.0)

    # AR(1) cloud cover
    cc = np.zeros(hours)
    cc[0] = rng.beta(2, 3)
    for i in range(1, hours):
        cc[i] = np.clip(0.85 * cc[i - 1] + 0.15 * 0.4 + rng.normal(0, 0.15), 0, 1)
    cloud_pct = (cc * 100).round(1)

    radiation = clear_sky * (1 - 0.75 * cc)
    direct = radiation * 0.7
    diffuse = radiation * 0.3
    temperature = 18 + 6 * np.cos(2 * np.pi * (doy[0] - 15) / 365) + 5 * np.sin(
        2 * np.pi * (h - 9) / 24
    )

    return {
        "_synthetic": True,
        "latitude": latitude,
        "longitude": cfg.SITE_LONGITUDE,
        "timezone": cfg.SITE_TIMEZONE,
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "shortwave_radiation": radiation.round(1).tolist(),
            "direct_radiation": direct.round(1).tolist(),
            "diffuse_radiation": diffuse.round(1).tolist(),
            "cloud_cover": cloud_pct.tolist(),
            "temperature_2m": temperature.round(1).tolist(),
            "precipitation_probability": (cc * 80).round(0).astype(int).tolist(),
        },
    }


# ─── primary entry point ────────────────────────────────────────────────────
def get_weather(
    *,
    prefer_cache: bool = False,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Return (hourly DataFrame, source metadata).

    Behaviour:
      - prefer_cache=True OR refresh=False+cache_exists  → use cache
      - else try live API; on fail use cache; on fail synthesise
    """
    payload: dict[str, Any] | None = None
    source = "live"

    if not refresh and (prefer_cache or cfg.CACHE_PATH.exists()):
        payload = load_cache()
        source = "cache" if payload else source

    if payload is None and not prefer_cache:
        payload = fetch_open_meteo()
        if payload is not None:
            save_cache(payload)
            source = "live"

    if payload is None:
        log.info("No live or cached data — generating synthetic snapshot.")
        payload = _synthetic_payload(datetime.now().replace(minute=0, second=0, microsecond=0))
        save_cache(payload)
        source = "synthetic_fallback"

    df = _payload_to_df(payload)
    meta = {
        "source": source,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "timezone": payload.get("timezone"),
        "synthetic": bool(payload.get("_synthetic", False)),
    }
    return df, meta


def _payload_to_df(payload: dict[str, Any]) -> pd.DataFrame:
    h = payload["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


# ─── CLI smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, meta = get_weather()
    print(f"Source: {meta['source']}  | rows: {len(df)}")
    print(df.head(6).round(1))
