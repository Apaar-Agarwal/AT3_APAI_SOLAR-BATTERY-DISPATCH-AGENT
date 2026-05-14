"""
rl_policy.py — tabular Q-learning battery dispatch policy.

This is intentionally small and transparent. The point is not to pretend this is
a production-grade deep RL controller. The point is to show the reinforcement
learning paradigm correctly:

    state  = hour + SoC bin + PV bin + load bin + tariff period
    action = charge / hold / discharge
    reward = negative hourly cost
    episode = one simulated 24-hour day

The learned policy is evaluated beside rule-based and dynamic-programming
policies, then the policy selector decides which one should drive the dashboard.
"""

from __future__ import annotations
from collections import defaultdict
import random
import time
from typing import Iterable

import numpy as np
import pandas as pd

from ..agent.rules import BatteryParams, apply_action


RL_ACTIONS_KW = (-5.0, 0.0, 5.0)  # negative = charge, positive = discharge
PERIOD_MAP = {"off_peak": 0, "shoulder": 1, "peak": 2}


def _bin(value: float, thresholds: tuple[float, ...]) -> int:
    return int(np.digitize([value], thresholds)[0])


def _state_tuple(state_df: pd.DataFrame, t: int, soc_kwh: float, params: BatteryParams) -> tuple:
    row = state_df.iloc[t]
    hour = int(getattr(state_df.index[t], "hour", t % 24))
    soc_frac = soc_kwh / max(params.capacity_kwh, 1e-9)
    pv = float(row["pv_kw"])
    load = float(row["load_kw"])
    period = str(row.get("period", ""))
    return (
        hour,
        _bin(soc_frac, (0.2, 0.4, 0.6, 0.8)),
        _bin(pv, (0.5, 1.5, 3.0)),
        _bin(load, (0.5, 1.5, 2.5)),
        PERIOD_MAP.get(period, 1),
    )


def _safe_action(action_kw: float, pv_kw: float, load_kw: float, params: BatteryParams) -> float:
    """Prevent the RL policy from grid-charging when grid charging is disabled."""
    if action_kw < 0 and not params.allow_grid_charging:
        surplus = max(float(pv_kw) - float(load_kw), 0.0)
        return -min(abs(action_kw), surplus, params.max_charge_kw)
    return float(action_kw)


def _step(
    state_df: pd.DataFrame,
    t: int,
    soc_kwh: float,
    action_idx: int,
    params: BatteryParams,
) -> tuple[float, float, dict]:
    row = state_df.iloc[t]
    action_kw = _safe_action(float(RL_ACTIONS_KW[action_idx]), row["pv_kw"], row["load_kw"], params)
    step = apply_action(
        soc_kwh=soc_kwh,
        action_kw=action_kw,
        pv_kw=float(row["pv_kw"]),
        load_kw=float(row["load_kw"]),
        import_price=float(row["import_price"]),
        export_price=float(row["export_price"]),
        params=params,
    )
    total_cost = float(step["cost_aud"] + step["degradation_cost_aud"])
    reward = -total_cost
    return float(step["soc_kwh"]), reward, step


def train_q_learning_policy(
    training_days: Iterable[pd.DataFrame],
    params: BatteryParams,
    *,
    episodes: int = 600,
    alpha: float = 0.12,
    gamma: float = 0.95,
    epsilon: float = 0.25,
    epsilon_decay: float = 0.992,
    min_epsilon: float = 0.03,
    seed: int = 42,
) -> dict[tuple, np.ndarray]:
    """Train a tabular Q-learning policy over simulated daily episodes."""
    rng = random.Random(seed)
    days = [d.copy() for d in training_days if d is not None and len(d) > 0]
    if not days:
        raise ValueError("At least one training day is required for Q-learning.")

    q = defaultdict(lambda: np.zeros(len(RL_ACTIONS_KW), dtype=float))
    eps = float(epsilon)

    for _ in range(episodes):
        df = rng.choice(days)
        soc = params.initial_soc_kwh()
        for t in range(len(df)):
            s = _state_tuple(df, t, soc, params)
            if rng.random() < eps:
                a = rng.randrange(len(RL_ACTIONS_KW))
            else:
                a = int(np.argmax(q[s]))

            next_soc, reward, _ = _step(df, t, soc, a, params)
            done = t == len(df) - 1
            if done:
                target = reward
            else:
                ns = _state_tuple(df, t + 1, next_soc, params)
                target = reward + gamma * float(np.max(q[ns]))
            q[s][a] += alpha * (target - q[s][a])
            soc = next_soc
        eps = max(min_epsilon, eps * epsilon_decay)

    return dict(q)


def run_q_learning_policy(
    state_df: pd.DataFrame,
    params: BatteryParams,
    q_table: dict[tuple, np.ndarray],
    *,
    scenario: str = "base",
) -> pd.DataFrame:
    """Roll out the learned greedy Q policy and return a schedule dataframe."""
    rows = []
    soc = params.initial_soc_kwh()

    for t, (ts, row) in enumerate(state_df.iterrows()):
        s = _state_tuple(state_df, t, soc, params)
        values = q_table.get(s)
        action_idx = int(np.argmax(values)) if values is not None else 1  # hold fallback
        next_soc, _, step = _step(state_df, t, soc, action_idx, params)
        rows.append({
            "timestamp": ts,
            "pv_kw": float(row["pv_kw"]),
            "load_kw": float(row["load_kw"]),
            "period": row.get("period", ""),
            "import_price": float(row["import_price"]),
            "export_price": float(row["export_price"]),
            "action_kw": float(step["action_applied_kw"]),
            "soc_kwh": float(step["soc_kwh"]),
            "soc_pct": 100 * float(step["soc_kwh"]) / params.capacity_kwh,
            "net_grid_kw": float(step["net_grid_kw"]),
            "grid_import_kwh": float(step["grid_import_kwh"]),
            "grid_export_kwh": float(step["grid_export_kwh"]),
            "cost_aud": float(step["cost_aud"]),
            "degradation_cost_aud": float(step["degradation_cost_aud"]),
            "scenario": scenario,
        })
        soc = next_soc

    return pd.DataFrame(rows).set_index("timestamp")


def train_and_evaluate_rl(
    scenario_dfs: dict[str, pd.DataFrame],
    params: BatteryParams,
    *,
    episodes: int = 600,
) -> tuple[dict[str, pd.DataFrame], float]:
    """Train once across scenario days, then evaluate the learned policy in each scenario."""
    t0 = time.perf_counter()
    q_table = train_q_learning_policy(scenario_dfs.values(), params, episodes=episodes)
    training_runtime = time.perf_counter() - t0

    schedules: dict[str, pd.DataFrame] = {}
    eval_start = time.perf_counter()
    for name, df in scenario_dfs.items():
        schedules[name] = run_q_learning_policy(df, params, q_table, scenario=name)
    total_runtime = training_runtime + (time.perf_counter() - eval_start)
    return schedules, total_runtime
