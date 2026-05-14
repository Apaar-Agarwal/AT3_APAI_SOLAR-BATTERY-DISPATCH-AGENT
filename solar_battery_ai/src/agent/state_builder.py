"""
state_builder.py — build the structured 24h energy state the agent reasons over.

This is the AI paradigm "intelligent agent / PEAS" in concrete form:
  PERCEPTS  : weather, solar radiation, tariff schedule, load profile
  ACTIONS   : charge / discharge / hold / import / export
  ENVIRONMENT : the household + grid + battery
  PERFORMANCE : cost, self-consumption, reserve compliance

The state object combines every input the optimiser and rules need into
a single hour-indexed DataFrame.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from .. import config as cfg
from ..data_sources.weather_client import get_weather
from ..data_sources.solar_client import estimate_pv_generation
from ..data_sources.load_profile import synthetic_load
from ..data_sources.tariff_client import TariffSchedule, default_schedule


@dataclass
class EnergyState:
    """All hour-indexed inputs the agent reasons over."""
    df: pd.DataFrame                              # cols: pv_kw, load_kw, period, import_price, export_price
    horizon: int
    pv_capacity_kw: float
    derate: float
    tariff: TariffSchedule
    source_meta: dict[str, Any] = field(default_factory=dict)

    # convenience accessors
    @property
    def pv(self) -> pd.Series:    return self.df["pv_kw"]
    @property
    def load(self) -> pd.Series:  return self.df["load_kw"]
    @property
    def import_price(self) -> pd.Series: return self.df["import_price"]
    @property
    def export_price(self) -> pd.Series: return self.df["export_price"]


def build_state(
    *,
    horizon_hours: int = cfg.HORIZON_HOURS,
    pv_capacity_kw: float = cfg.PV_CAPACITY_KW,
    derate: float = cfg.PV_DERATE_FACTOR,
    tariff: TariffSchedule | None = None,
    weather_df: pd.DataFrame | None = None,
    weather_meta: dict[str, Any] | None = None,
    load_series: pd.Series | None = None,
    prefer_cache: bool = False,
) -> EnergyState:
    """
    Construct a fully-populated EnergyState for the next `horizon_hours`.

    If weather_df is provided, it's used directly (e.g. when Streamlit user
    has already fetched). Otherwise we fetch via the weather client.
    """
    if weather_df is None:
        weather_df, weather_meta = get_weather(prefer_cache=prefer_cache)
    weather_meta = weather_meta or {}

    weather_df = weather_df.iloc[:horizon_hours].copy()
    pv = estimate_pv_generation(weather_df, pv_capacity_kw, derate)

    if load_series is None:
        load_series = synthetic_load(weather_df.index)
    else:
        load_series = load_series.reindex(weather_df.index).ffill().bfill()

    if tariff is None:
        tariff = default_schedule()
    tariff_df = tariff.schedule(weather_df.index)

    df = pd.concat(
        [
            pv.rename("pv_kw"),
            load_series.rename("load_kw"),
            weather_df[["shortwave_radiation", "cloud_cover", "temperature_2m"]],
            tariff_df,
        ],
        axis=1,
    )

    return EnergyState(
        df=df,
        horizon=len(df),
        pv_capacity_kw=pv_capacity_kw,
        derate=derate,
        tariff=tariff,
        source_meta=weather_meta,
    )


if __name__ == "__main__":
    state = build_state()
    print(f"Source: {state.source_meta.get('source')}  Horizon: {state.horizon}h")
    print(state.df.head(8).round(2))
    print(f"\nPV total : {state.pv.sum():.2f} kWh")
    print(f"Load tot : {state.load.sum():.2f} kWh")
    print(f"Period mix: {state.df['period'].value_counts().to_dict()}")
