"""
policy_selector.py — probability-weighted policy selection.

This module turns separate policy results into a decision layer:

    policies × scenarios → expected cost → downside risk → final score

The final selected policy is the lowest safe risk-adjusted expected-cost option.
This is the bridge between probabilistic reasoning and the dashboard. Because
apparently one policy output was not enough; now the machine must run auditions.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable
import pandas as pd


@dataclass
class PolicySelectionConfig:
    risk_weight: float = 0.50
    peak_import_weight: float = 0.00
    selectable_policies: tuple[str, ...] = ("rule_based", "ai_agent", "rl_q_learning")


DISPLAY_NAMES = {
    "no_battery": "No battery baseline",
    "rule_based": "Rule-based controller",
    "ai_agent": "Dynamic Programming agent",
    "rl_q_learning": "RL Q-learning agent",
}


AI_PARADIGMS = {
    "no_battery": "Baseline",
    "rule_based": "Logic / structural knowledge representation",
    "ai_agent": "Search / dynamic programming optimisation",
    "rl_q_learning": "Reinforcement learning",
}


def _weighted_sum(values: pd.Series, probabilities: dict[str, float]) -> float:
    return float(sum(float(values.get(sc, 0.0)) * float(probabilities.get(sc, 0.0)) for sc in probabilities))


def build_policy_selection_table(
    robustness: pd.DataFrame,
    scenario_probabilities: dict[str, float],
    config: PolicySelectionConfig | None = None,
) -> pd.DataFrame:
    """
    Build a risk-adjusted policy ranking from the robustness table.

    Expected cost is calculated as:
        Σ P(scenario) × cost(policy, scenario)

    Downside risk is calculated as:
        Σ P(scenario) × max(0, cost(policy, scenario) - expected_cost)

    Final score is:
        expected_cost + risk_weight × downside_risk
        + peak_import_weight × expected_peak_import

    Lower final_score_aud is better.
    """
    config = config or PolicySelectionConfig()

    required = {"scenario", "policy", "total_cost_aud"}
    missing = required - set(robustness.columns)
    if missing:
        raise ValueError(f"Robustness table is missing columns: {sorted(missing)}")

    rows: list[dict] = []
    for policy, g in robustness.groupby("policy", sort=False):
        by_scenario = g.set_index("scenario")
        costs = by_scenario["total_cost_aud"]

        expected_cost = _weighted_sum(costs, scenario_probabilities)
        worst_case_cost = float(costs.max())
        best_case_cost = float(costs.min())
        base_cost = float(costs.get("base", expected_cost))

        downside_risk = float(
            sum(
                scenario_probabilities.get(sc, 0.0) * max(0.0, float(cost) - expected_cost)
                for sc, cost in costs.items()
            )
        )

        high_cost_threshold = base_cost * 1.10 if abs(base_cost) > 1e-9 else expected_cost * 1.10
        high_cost_probability = float(
            sum(
                scenario_probabilities.get(sc, 0.0)
                for sc, cost in costs.items()
                if float(cost) > high_cost_threshold
            )
        )

        if "peak_period_import_kwh" in by_scenario.columns:
            expected_peak_import = _weighted_sum(by_scenario["peak_period_import_kwh"], scenario_probabilities)
        else:
            expected_peak_import = 0.0

        final_score = (
            expected_cost
            + config.risk_weight * downside_risk
            + config.peak_import_weight * expected_peak_import
        )

        selectable = policy in config.selectable_policies
        rows.append({
            "policy": policy,
            "display_name": DISPLAY_NAMES.get(policy, policy),
            "ai_paradigm": AI_PARADIGMS.get(policy, "Candidate policy"),
            "selectable": selectable,
            "base_cost_aud": base_cost,
            "expected_cost_aud": expected_cost,
            "best_case_cost_aud": best_case_cost,
            "worst_case_cost_aud": worst_case_cost,
            "downside_risk_aud": downside_risk,
            "high_cost_probability_pct": high_cost_probability * 100,
            "expected_peak_import_kwh": expected_peak_import,
            "risk_adjusted_score_aud": final_score,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["selected"] = False
    selectable_mask = out["selectable"]
    if selectable_mask.any():
        selected_idx = out.loc[selectable_mask, "risk_adjusted_score_aud"].idxmin()
        out.loc[selected_idx, "selected"] = True

    out = out.sort_values(
        ["selectable", "risk_adjusted_score_aud"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return out


def selected_policy_row(policy_selection: pd.DataFrame) -> pd.Series | None:
    """Return the selected policy row, or None if nothing is selected."""
    if policy_selection is None or policy_selection.empty or "selected" not in policy_selection.columns:
        return None
    selected = policy_selection[policy_selection["selected"]]
    if selected.empty:
        return None
    return selected.iloc[0]
