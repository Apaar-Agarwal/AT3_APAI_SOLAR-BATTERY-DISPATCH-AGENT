"""
solar_client.py — PV generation estimation.

We do NOT train a forecasting model. We convert externally-sourced shortwave
radiation (W/m²) into estimated PV generation using a deliberately simple,
auditable formula:

    pv_kwh = pv_capacity_kw × (radiation_wm2 / 1000) × derate_factor

Where:
  - pv_capacity_kw    : nameplate capacity (e.g. 6.6 kW)
  - radiation_wm2     : hourly average shortwave radiation from Open-Meteo
  - derate_factor     : ~0.78 — covers inverter efficiency, soiling,
                        cabling losses, and module derate. Editable in UI.

This is intentionally transparent. The AI contribution is *downstream*:
state representation, scenario uncertainty, search-based dispatch.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .. import config as cfg


def estimate_pv_generation(
    weather_df: pd.DataFrame,
    pv_capacity_kw: float = cfg.PV_CAPACITY_KW,
    derate_factor: float = cfg.PV_DERATE_FACTOR,
    radiation_col: str = "shortwave_radiation",
) -> pd.Series:
    """
    Convert hourly shortwave radiation (W/m²) → estimated PV output (kWh per hour
    = kW for hourly data).

    Returned series has the same index as `weather_df`.
    """
    if radiation_col not in weather_df.columns:
        raise KeyError(f"Required column '{radiation_col}' missing from weather data.")

    rad = weather_df[radiation_col].to_numpy(dtype=float)
    pv_kw = pv_capacity_kw * (rad / 1000.0) * derate_factor
    pv_kw = np.clip(pv_kw, 0.0, pv_capacity_kw)  # cap at nameplate capacity
    return pd.Series(pv_kw, index=weather_df.index, name="pv_estimate_kw")


def apply_scenario(pv_series: pd.Series, scaling: float) -> pd.Series:
    """Multiply baseline PV estimate by a scenario factor (e.g. 0.75 for 'low')."""
    out = (pv_series * scaling).clip(lower=0.0)
    out.name = pv_series.name
    return out


if __name__ == "__main__":
    from .weather_client import get_weather
    weather, meta = get_weather()
    pv = estimate_pv_generation(weather)
    print(f"Source: {meta['source']}, total energy over horizon: "
          f"{pv.sum():.2f} kWh")
    print(pv.head(8).round(2))
