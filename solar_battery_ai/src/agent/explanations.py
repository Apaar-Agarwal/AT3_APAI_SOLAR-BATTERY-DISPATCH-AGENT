"""
explanations.py — turn a dispatch schedule into human-readable rationales.

For every hour, given (pv, load, soc, period, action), explain WHY the agent
chose that action. The explanation is grounded in the symbolic rule library
(rules.py) so we can defend it as logic-driven, not LLM-driven.
"""

from __future__ import annotations
import pandas as pd

from .rules import BatteryParams, evaluate_rules, fired_rules
from .dispatch_optimizer import label_action


def explain_schedule(
    schedule: pd.DataFrame,
    *,
    params: BatteryParams,
    scenario: str = "base",
) -> pd.DataFrame:
    """
    Add columns 'action_label' and 'explanation' to a schedule DataFrame.
    """
    out = schedule.copy()
    labels, explanations = [], []
    for ts, row in out.iterrows():
        action = float(row["action_kw"])
        labels.append(label_action(action))

        # Symbolic rules give the WHY
        evals = evaluate_rules(
            soc_kwh=float(row["soc_kwh"]),
            pv_kw=float(row["pv_kw"]),
            load_kw=float(row["load_kw"]),
            period=str(row.get("period", "")),
            scenario=scenario,
            params=params,
        )
        fired = fired_rules(evals)

        # Map action → most-relevant rationale
        if action > 0.05:
            primary = next((r for r in fired if r.rule_id == "R1_peak_discharge"), None)
            if primary is None:
                primary = next((r for r in fired if r.rule_id == "R6_forced_import"), None)
            txt = primary.rationale if primary else "Discharging to offset grid import."
        elif action < -0.05:
            primary = next((r for r in fired if r.rule_id == "R2_surplus_charge"), None)
            if primary is None:
                primary = next((r for r in fired if r.rule_id == "R3_offpeak_grid_charge"), None)
            txt = primary.rationale if primary else "Charging to capture available energy."
        else:
            primary = next((r for r in fired if r.rule_id == "R4_low_solar_preserve"), None)
            if primary is not None:
                txt = primary.rationale
            elif row["pv_kw"] < 0.05 and row.get("period") != "peak":
                txt = "No solar and not peak — hold (no economic action available)."
            else:
                txt = "Hold — better to act in a more valuable hour."
        explanations.append(txt)

    out["action_label"] = labels
    out["explanation"] = explanations
    return out


def headline(action_kw: float, period: str, soc_pct: float) -> str:
    """One-line current-hour recommendation for the dashboard."""
    if action_kw > 0.05:
        return f"Recommend DISCHARGE {action_kw:.1f} kW (period: {period}, SoC: {soc_pct:.0f}%)"
    if action_kw < -0.05:
        return f"Recommend CHARGE {abs(action_kw):.1f} kW (period: {period}, SoC: {soc_pct:.0f}%)"
    return f"Recommend HOLD (period: {period}, SoC: {soc_pct:.0f}%)"
