"""
rules.py — battery model + structural knowledge representation.

This module holds the *logic / knowledge* AI paradigm.

Knowledge encoded:
  - Battery dynamics (SoC bounds, charge/discharge rates, efficiency)
  - Energy balance: solar + import + discharge = load + export + charge
  - Action feasibility constraints
  - Symbolic rules for explaining decisions:
      R1. Peak tariff & SoC > reserve → prefer DISCHARGE
      R2. Surplus solar (PV > load) & battery has headroom → CHARGE
      R3. Off-peak & SoC low & grid charging allowed → grid CHARGE
      R4. Cloudy/low-solar scenario & SoC near reserve → PRESERVE (hold)
      R5. Solar surplus but battery full → EXPORT
      R6. Load > solar & SoC ≤ reserve → IMPORT (no other choice)

The optimiser uses the dynamics; the explainer uses the rules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Iterable

from .. import config as cfg


# ════════════════════════════════════════════════════════════════════════════
# 1. Battery model
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class BatteryParams:
    capacity_kwh: float = cfg.BATTERY_CAPACITY_KWH
    initial_soc_pct: float = cfg.BATTERY_INITIAL_SOC_PCT
    reserve_soc_pct: float = cfg.BATTERY_RESERVE_SOC_PCT
    max_charge_kw: float = cfg.BATTERY_MAX_CHARGE_KW
    max_discharge_kw: float = cfg.BATTERY_MAX_DISCHARGE_KW
    round_trip_eff: float = cfg.BATTERY_ROUND_TRIP_EFF
    degradation_aud_per_kwh: float = cfg.BATTERY_DEGRADATION_AUD_PER_KWH
    allow_grid_charging: bool = cfg.ALLOW_GRID_CHARGING

    # Derived
    @property
    def one_way_eff(self) -> float:
        """sqrt(round-trip) split symmetrically across charge & discharge."""
        return math.sqrt(self.round_trip_eff)

    @property
    def reserve_kwh(self) -> float:
        return self.capacity_kwh * self.reserve_soc_pct / 100.0

    @property
    def soc_min_kwh(self) -> float:
        return self.reserve_kwh

    @property
    def soc_max_kwh(self) -> float:
        return self.capacity_kwh

    def initial_soc_kwh(self) -> float:
        return self.capacity_kwh * self.initial_soc_pct / 100.0


# ════════════════════════════════════════════════════════════════════════════
# 2. Step transition (signed-action convention)
# ════════════════════════════════════════════════════════════════════════════
# action_kw > 0 : DISCHARGE the battery (energy out, helps load)
# action_kw < 0 : CHARGE   the battery (energy in)
# action_kw = 0 : HOLD
#
# After the action is applied:
#   net_grid_kw = load - solar - action      (positive = import, negative = export)
# ════════════════════════════════════════════════════════════════════════════
def feasible_action_range(soc_kwh: float, params: BatteryParams) -> tuple[float, float]:
    """Return (min_action, max_action) feasible at the current SoC, in kW."""
    headroom_charge = params.soc_max_kwh - soc_kwh           # how much we can absorb
    headroom_discharge = soc_kwh - params.soc_min_kwh        # how much we can deliver

    max_charge = min(params.max_charge_kw, headroom_charge / params.one_way_eff)
    max_discharge = min(params.max_discharge_kw, headroom_discharge * params.one_way_eff)
    return -max_charge, max_discharge


def apply_action(
    *,
    soc_kwh: float,
    action_kw: float,
    pv_kw: float,
    load_kw: float,
    import_price: float,
    export_price: float,
    params: BatteryParams,
) -> dict:
    """
    Apply one hour of dispatch.

    Returns dict with:
      action_applied_kw : action after clamping
      soc_kwh           : updated SoC
      net_grid_kw       : positive = import, negative = export
      grid_import_kwh, grid_export_kwh, cost_aud
      degradation_cost  : amortised cycling cost
    """
    a_min, a_max = feasible_action_range(soc_kwh, params)
    a = max(a_min, min(action_kw, a_max))

    # Update SoC (account for one-way efficiency)
    if a >= 0:  # discharging
        # to deliver `a` kWh to the bus, we need a / eff kWh out of cells
        soc_new = soc_kwh - a / params.one_way_eff
    else:       # charging
        # `-a` kWh delivered to charger; only `-a * eff` makes it into cells
        soc_new = soc_kwh + (-a) * params.one_way_eff
    soc_new = max(params.soc_min_kwh, min(params.soc_max_kwh, soc_new))

    # Energy balance
    net = load_kw - pv_kw - a
    grid_import = max(net, 0.0)
    grid_export = max(-net, 0.0)
    cost = grid_import * import_price - grid_export * export_price

    deg = abs(a) * params.degradation_aud_per_kwh

    return {
        "action_applied_kw": a,
        "soc_kwh": soc_new,
        "net_grid_kw": net,
        "grid_import_kwh": grid_import,
        "grid_export_kwh": grid_export,
        "cost_aud": cost,
        "degradation_cost_aud": deg,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Symbolic rule library (used by the explainer + the rule-based baseline)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class RuleEvaluation:
    rule_id: str
    fired: bool
    rationale: str


def evaluate_rules(
    *,
    soc_kwh: float,
    pv_kw: float,
    load_kw: float,
    period: str,           # 'peak' | 'shoulder' | 'off_peak'
    scenario: str,         # 'low' | 'base' | 'high'
    params: BatteryParams,
) -> list[RuleEvaluation]:
    """Run the rule library and return which rules fired with rationales."""
    out = []
    surplus = pv_kw - load_kw
    soc_pct = 100 * soc_kwh / params.capacity_kwh
    above_reserve = soc_kwh > params.soc_min_kwh + 1e-6

    # R1 — discharge in peak when battery has energy
    out.append(RuleEvaluation(
        "R1_peak_discharge",
        period == "peak" and above_reserve,
        f"Peak tariff and SoC {soc_pct:.0f}% > reserve {params.reserve_soc_pct}% — discharge to avoid peak rate.",
    ))
    # R2 — surplus solar with headroom → charge
    out.append(RuleEvaluation(
        "R2_surplus_charge",
        surplus > 0.1 and soc_kwh < params.soc_max_kwh - 0.05,
        f"Solar surplus {surplus:+.2f} kW and battery has headroom — charge from surplus.",
    ))
    # R3 — off-peak grid charge if allowed and SoC low
    out.append(RuleEvaluation(
        "R3_offpeak_grid_charge",
        params.allow_grid_charging and period == "off_peak" and soc_pct < 30,
        f"Off-peak rate and SoC {soc_pct:.0f}% < 30% — grid charging permitted; top up cheaply.",
    ))
    # R4 — preserve battery on low-solar scenario
    out.append(RuleEvaluation(
        "R4_low_solar_preserve",
        scenario == "low" and soc_pct < params.reserve_soc_pct + 10,
        f"Low-solar scenario and SoC {soc_pct:.0f}% near reserve — hold to preserve evening reserve.",
    ))
    # R5 — battery full + surplus → export
    out.append(RuleEvaluation(
        "R5_full_export",
        surplus > 0.1 and soc_kwh >= params.soc_max_kwh - 0.05,
        f"Surplus solar {surplus:+.2f} kW with battery full — export to grid.",
    ))
    # R6 — load > solar and reserve hit → forced import
    out.append(RuleEvaluation(
        "R6_forced_import",
        load_kw > pv_kw + 0.05 and not above_reserve,
        f"Load > solar and SoC at reserve — import from grid (no battery to draw).",
    ))
    return out


def fired_rules(evaluations: Iterable[RuleEvaluation]) -> list[RuleEvaluation]:
    return [r for r in evaluations if r.fired]
