"""
uncertainty.py — probabilistic reasoning via scenario analysis.

The system does not train a probabilistic solar forecasting model. Instead it
implements an explicit probabilistic scenario model:

  1. Take the externally sourced PV estimate as the BASE scenario.
  2. Construct LOW and HIGH PV scenarios to bracket plausible cloud variability.
  3. Assign probabilities to LOW/BASE/HIGH.
  4. Optionally update those probabilities using cloud-cover evidence.
  5. Use the probabilities downstream to compute expected cost and risk-adjusted
     policy selection.

This keeps the AI contribution honest: probability-weighted scenario reasoning,
not a hidden weather model pretending to be magic.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .. import config as cfg
from ..data_sources.solar_client import apply_scenario


DEFAULT_SCENARIO_PROBABILITIES: dict[str, float] = {
    "low": 0.25,
    "base": 0.55,
    "high": 0.20,
}


@dataclass
class ScenarioBundle:
    base: pd.Series
    low: pd.Series
    high: pd.Series

    def as_dict(self) -> dict[str, pd.Series]:
        return {"low": self.low, "base": self.base, "high": self.high}

    def total_energy(self) -> dict[str, float]:
        return {k: float(v.sum()) for k, v in self.as_dict().items()}


def normalise_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    """Return a low/base/high probability dictionary that sums to 1."""
    keys = ("low", "base", "high")
    cleaned = {k: max(0.0, float(probabilities.get(k, 0.0))) for k in keys}
    total = sum(cleaned.values())
    if total <= 0:
        return DEFAULT_SCENARIO_PROBABILITIES.copy()
    return {k: v / total for k, v in cleaned.items()}


def estimate_scenario_probabilities(
    cloud_cover_pct: pd.Series | None = None,
    prior: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Estimate posterior scenario probabilities from cloud-cover evidence.

    This is a transparent Bayesian-style update, not a trained weather model.
    The prior represents the initial chance of low/base/high solar. The likelihood
    shifts probability toward low solar when average cloud cover is high and
    toward high solar when cloud cover is low.
    """
    prior = normalise_probabilities(prior or DEFAULT_SCENARIO_PROBABILITIES)

    if cloud_cover_pct is None or len(cloud_cover_pct) == 0:
        return prior

    cloud = pd.to_numeric(cloud_cover_pct, errors="coerce").dropna()
    if cloud.empty:
        return prior

    mean_cloud = float(cloud.mean())

    if mean_cloud >= 70:
        likelihood = {"low": 0.70, "base": 0.25, "high": 0.05}
    elif mean_cloud >= 45:
        likelihood = {"low": 0.40, "base": 0.45, "high": 0.15}
    elif mean_cloud >= 25:
        likelihood = {"low": 0.20, "base": 0.55, "high": 0.25}
    else:
        likelihood = {"low": 0.10, "base": 0.40, "high": 0.50}

    posterior = {k: prior[k] * likelihood[k] for k in prior}
    return normalise_probabilities(posterior)


def make_scenarios(
    pv_base: pd.Series,
    cloud_cover_pct: pd.Series | None = None,
    scaling: dict[str, float] | None = None,
) -> ScenarioBundle:
    """
    Build (low, base, high) PV scenarios.

    If `cloud_cover_pct` is supplied, the LOW scenario is further dampened on
    hours where cloud cover > 60% (extra 0.85 multiplier). This makes the
    "what if cloudier than expected" risk concrete and observable.
    """
    scaling = scaling or cfg.SOLAR_SCENARIOS
    base = pv_base.copy()
    low = apply_scenario(pv_base, scaling["low"])
    high = apply_scenario(pv_base, scaling["high"])

    if cloud_cover_pct is not None:
        cloudy_hours = cloud_cover_pct.reindex(low.index).fillna(0) > 60
        low.loc[cloudy_hours] *= 0.85

    return ScenarioBundle(base=base, low=low, high=high)


# ─── Risk warnings ──────────────────────────────────────────────────────────
def risk_warnings(
    bundle: ScenarioBundle,
    *,
    daily_load_kwh: float,
    battery_capacity_kwh: float,
    reserve_pct: float,
) -> list[str]:
    """Return human-readable risk strings for the dashboard."""
    msgs: list[str] = []
    base_total = bundle.base.sum()
    low_total = bundle.low.sum()
    high_total = bundle.high.sum()

    ratio = low_total / max(daily_load_kwh, 1e-6)
    if ratio < 0.6:
        msgs.append(
            f"⚠️  Low-solar scenario covers only {ratio*100:.0f}% of daily load "
            f"({low_total:.1f} / {daily_load_kwh:.1f} kWh). Preserve battery reserve."
        )
    elif ratio < 1.0:
        msgs.append(
            f"⚠️  On a cloudy day, solar covers {ratio*100:.0f}% of demand — "
            f"expect grid imports."
        )
    else:
        msgs.append(
            f"✅ Even a low-solar day covers {ratio*100:.0f}% of demand — "
            f"surplus available."
        )

    spread = (high_total - low_total) / max(base_total, 1e-6)
    if spread > 0.4:
        msgs.append(
            f"⚠️  High forecast uncertainty: scenario spread is "
            f"{spread*100:.0f}% of base estimate."
        )

    # peak-window check
    peak_hours = [16, 17, 18, 19, 20]
    peak_load_proxy = daily_load_kwh * (5 / 24) * 1.5  # peaks consume more than average
    peak_solar_low = float(bundle.low.iloc[peak_hours].sum()) if len(bundle.low) >= 21 else 0.0
    if peak_solar_low < 0.3 * peak_load_proxy:
        usable_reserve = battery_capacity_kwh * (1 - reserve_pct / 100)
        msgs.append(
            f"💡 Battery has ~{usable_reserve:.1f} kWh usable above reserve — "
            f"consider conservative discharge during 4-9 pm peak under low-solar conditions."
        )
    return msgs


if __name__ == "__main__":
    from .state_builder import build_state
    state = build_state()
    bundle = make_scenarios(state.pv, state.df["cloud_cover"])
    print("Scenario daily totals (kWh):", bundle.total_energy())
    print("Scenario probabilities:", estimate_scenario_probabilities(state.df.get("cloud_cover")))
    for msg in risk_warnings(
        bundle,
        daily_load_kwh=state.load.sum(),
        battery_capacity_kwh=cfg.BATTERY_CAPACITY_KWH,
        reserve_pct=cfg.BATTERY_RESERVE_SOC_PCT,
    ):
        print(msg)
