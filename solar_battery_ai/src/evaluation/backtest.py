"""
backtest.py — run all policies + scenarios end-to-end.

Produces:
  - schedules per policy and scenario
  - a unified metrics table
  - an explained schedule for the AI agent (base scenario)
  - a CROSS-SCENARIO robustness table (every policy × low/base/high)
  - everything keyed for plotting + Streamlit consumption
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Any
import pandas as pd

from ..agent.rules import BatteryParams
from ..agent.state_builder import EnergyState
from ..agent.dispatch_optimizer import optimise_dispatch, DispatchResult
from ..agent.uncertainty import (
    make_scenarios, ScenarioBundle, risk_warnings, estimate_scenario_probabilities
)
from ..agent.explanations import explain_schedule
from .baselines import run_no_battery, run_rule_based, run_all_baselines
from .metrics import compute_metrics, comparison_table, relative_savings, PolicyMetrics
from .policy_selector import build_policy_selection_table
from .rl_policy import train_and_evaluate_rl


@dataclass
class BacktestReport:
    state: EnergyState
    params: BatteryParams
    scenarios: ScenarioBundle
    schedules: dict[str, pd.DataFrame]              # policy -> schedule
    scenario_results: dict[str, DispatchResult]     # 'low'|'base'|'high' -> AI dispatch
    explained: pd.DataFrame                         # AI base-scenario with rationales
    metrics: pd.DataFrame                           # comparison table (base scenario)
    metrics_with_savings: pd.DataFrame              # comparison table + savings %
    robustness: pd.DataFrame                        # policies × scenarios
    rl_schedules: dict[str, pd.DataFrame]             # low/base/high RL schedules
    risk_messages: list[str]
    scenario_probabilities: dict[str, float]       # low/base/high posterior probabilities
    policy_selection: pd.DataFrame                # risk-adjusted expected-cost ranking


def run_backtest(state: EnergyState, params: BatteryParams) -> BacktestReport:
    base_state_df = state.df.copy()

    # ─── Build scenarios ───────────────────────────────────────────────────
    bundle = make_scenarios(state.pv, state.df.get("cloud_cover"))
    scenario_probabilities = estimate_scenario_probabilities(state.df.get("cloud_cover"))

    scenario_dfs: dict[str, pd.DataFrame] = {}
    for name, pv_series in bundle.as_dict().items():
        df = base_state_df.copy()
        df["pv_kw"] = pv_series
        scenario_dfs[name] = df

    # ─── Run AI agent under each scenario ──────────────────────────────────
    scenario_results: dict[str, DispatchResult] = {}
    for name, df in scenario_dfs.items():
        scenario_results[name] = optimise_dispatch(df, params=params, scenario=name)

    # ─── Train and evaluate RL benchmark under each scenario ───────────────
    rl_schedules, rl_runtime_total = train_and_evaluate_rl(
        scenario_dfs, params=params, episodes=600
    )
    rl_runtime_each = rl_runtime_total / max(1, len(rl_schedules))

    # ─── Run baselines on the BASE scenario (the assignment headline) ──────
    base_for_baselines = base_state_df.copy()
    base_for_baselines["pv_kw"] = bundle.base
    baselines = run_all_baselines(base_for_baselines, params)

    # ─── Pull base-scenario AI schedule ────────────────────────────────────
    ai_base = scenario_results["base"]
    schedules = {
        "no_battery": baselines["no_battery"]["schedule"],
        "rule_based": baselines["rule_based"]["schedule"],
        "ai_agent_base": ai_base.schedule,
        "ai_agent_low": scenario_results["low"].schedule,
        "ai_agent_high": scenario_results["high"].schedule,
        "rl_q_learning_base": rl_schedules["base"],
        "rl_q_learning_low": rl_schedules["low"],
        "rl_q_learning_high": rl_schedules["high"],
    }
    runtimes = {
        "no_battery": baselines["no_battery"]["runtime_s"],
        "rule_based": baselines["rule_based"]["runtime_s"],
        "ai_agent_base": ai_base.runtime_s,
        "ai_agent_low": scenario_results["low"].runtime_s,
        "ai_agent_high": scenario_results["high"].runtime_s,
        "rl_q_learning_base": rl_runtime_each,
        "rl_q_learning_low": rl_runtime_each,
        "rl_q_learning_high": rl_runtime_each,
    }

    # ─── Explain the AI base schedule ──────────────────────────────────────
    explained = explain_schedule(ai_base.schedule, params=params, scenario="base")

    # ─── Metrics ──────────────────────────────────────────────────────────
    metric_objs: list[PolicyMetrics] = []
    for name, sched in schedules.items():
        metric_objs.append(
            compute_metrics(sched, policy_name=name, params=params,
                            runtime_s=runtimes.get(name, 0.0))
        )
    metrics = comparison_table(metric_objs)
    with_savings = relative_savings(metrics, baseline="no_battery")

    # ─── Cross-scenario robustness (every policy × every scenario) ─────────
    # Each policy is independently optimised/run for each scenario.
    # This is the "well-informed" comparison.
    robust_rows = []
    for sc_name, df in scenario_dfs.items():

        # No-battery
        sched_nb = run_no_battery(df)
        m_nb = compute_metrics(sched_nb, policy_name="no_battery", params=params)
        # Rule-based
        sched_rb = run_rule_based(df, params)
        m_rb = compute_metrics(sched_rb, policy_name="rule_based", params=params)
        # AI / Dynamic Programming (already computed)
        ai_sched = scenario_results[sc_name].schedule
        m_ai = compute_metrics(ai_sched, policy_name="ai_agent", params=params,
                               runtime_s=scenario_results[sc_name].runtime_s)
        # Reinforcement Learning benchmark
        rl_sched = rl_schedules[sc_name]
        m_rl = compute_metrics(rl_sched, policy_name="rl_q_learning", params=params,
                               runtime_s=rl_runtime_each)

        for m in (m_nb, m_rb, m_ai, m_rl):
            robust_rows.append({
                "scenario": sc_name,
                "policy": m.policy,
                "total_cost_aud": m.total_cost_aud,
                "total_grid_import_kwh": m.total_grid_import_kwh,
                "self_consumption_pct": m.self_consumption_pct,
                "peak_period_import_kwh": m.peak_period_import_kwh,
            })
    robustness = pd.DataFrame(robust_rows)

    # ─── Probabilistic policy selection ────────────────────────────────
    policy_selection = build_policy_selection_table(
        robustness=robustness,
        scenario_probabilities=scenario_probabilities,
    )

    # ─── Risk warnings ────────────────────────────────────────────────────
    msgs = risk_warnings(
        bundle,
        daily_load_kwh=state.load.sum(),
        battery_capacity_kwh=params.capacity_kwh,
        reserve_pct=params.reserve_soc_pct,
    )

    return BacktestReport(
        state=state,
        params=params,
        scenarios=bundle,
        schedules=schedules,
        scenario_results=scenario_results,
        explained=explained,
        metrics=metrics,
        metrics_with_savings=with_savings,
        robustness=robustness,
        rl_schedules=rl_schedules,
        risk_messages=msgs,
        scenario_probabilities=scenario_probabilities,
        policy_selection=policy_selection,
    )
