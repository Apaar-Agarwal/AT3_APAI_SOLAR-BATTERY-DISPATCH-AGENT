"""
config.py — central configuration for the source-aware AI battery dispatch agent.

Holds:
  - Case-study defaults (Sydney 6.6 kW PV / 10 kWh battery)
  - Battery physics constants
  - Tariff defaults (editable by user via Streamlit)
  - Paths
  - Optimiser settings

Single source of truth so notebooks, CLI, and Streamlit all read the same values.
"""

from __future__ import annotations
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUTPUTS = DATA / "outputs"
FIGURES = ROOT / "reports" / "figures"

for _p in (CACHE, RAW, PROCESSED, OUTPUTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# ─── Site & system defaults ─────────────────────────────────────────────────
SITE_NAME = "Sydney"
SITE_LATITUDE = -33.87
SITE_LONGITUDE = 151.21
SITE_TIMEZONE = "Australia/Sydney"

PV_CAPACITY_KW = 6.6              # standard AU residential rooftop
PV_DERATE_FACTOR = 0.78           # inverter + cabling + soiling losses

BATTERY_CAPACITY_KWH = 10.0       # 10 kWh battery (Tesla Powerwall ~ 13.5; LG Resu ~ 9.8)
BATTERY_INITIAL_SOC_PCT = 50      # %
BATTERY_RESERVE_SOC_PCT = 20      # never discharge below 20%
BATTERY_MAX_CHARGE_KW = 5.0
BATTERY_MAX_DISCHARGE_KW = 5.0
BATTERY_ROUND_TRIP_EFF = 0.92     # 92% — splits to ~0.96 each direction
BATTERY_DEGRADATION_AUD_PER_KWH = 0.02  # rough lifetime-amortised cycling cost
ALLOW_GRID_CHARGING = False       # default solar-only mode

# ─── Tariff defaults (AUD/kWh) — editable in Streamlit ──────────────────────
# Indicative AU residential time-of-use schedule.
TARIFF = {
    "off_peak": 0.18,      # 22:00 - 07:00
    "shoulder": 0.30,      # 07:00 - 16:00, 21:00 - 22:00
    "peak":     0.55,      # 16:00 - 21:00 (weekdays only)
    "feed_in":  0.06,      # what the grid pays for exported solar
}

# ─── Uncertainty scenarios ──────────────────────────────────────────────────
SOLAR_SCENARIOS = {"low": 0.75, "base": 1.00, "high": 1.15}

# ─── Optimiser settings ─────────────────────────────────────────────────────
HORIZON_HOURS = 24
SOC_BINS = 41          # SoC discretisation for DP (41 = 2.5% steps)
ACTION_GRID_KW = (
    -5.0, -4.0, -3.0, -2.0, -1.0, -0.5,
    0.0,
    0.5, 1.0, 2.0, 3.0, 4.0, 5.0,
)
# Negative = charge into battery, positive = discharge from battery.

# Penalty weights in the objective (in AUD/event so they're commensurable).
RESERVE_VIOLATION_PENALTY = 5.0   # large — reserve is a soft hard constraint
PEAK_IMPORT_PENALTY = 0.0         # already priced via tariff; keep at 0 by default

# ─── Open-Meteo source ──────────────────────────────────────────────────────
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_VARIABLES = (
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
    "temperature_2m",
    "precipitation_probability",
)
CACHE_PATH = CACHE / "open_meteo_sydney_24h.json"
