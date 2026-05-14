"""
rolling_backtest.py — 7-day rolling-horizon backtest.

Compares four policies over 168 hours (7 days) of varied weather:

  1. no_battery       — solar first, surplus exported, deficit imported
  2. rule_based       — simple charge-from-surplus / discharge-when-deficit
  3. dp_rolling       — receding-horizon Dynamic Programming: at each hour,
                        re-optimise over the next 24h, apply the first action,
                        advance, repeat
  4. rl_q_learning    — tabular Q-learning controller trained on simulated
                        daily episodes, then rolled out over the same 7-day data

The backtest now matches the updated app logic: the system compares logic-based,
planning-based and reinforcement-learning policies instead of only showing the
old no-battery / rule-based / AI-rolling trio.

All four policies see the same solar/load/tariff data and start with the same
battery state, so the comparison is fair.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .. import config as cfg
from ..agent.rules import BatteryParams, apply_action
from ..agent.dispatch_optimizer import optimise_dispatch
from ..agent.state_builder import EnergyState
from ..data_sources.solar_client import estimate_pv_generation
from ..data_sources.load_profile import synthetic_load
from ..data_sources.tariff_client import TariffSchedule, default_schedule
from ..data_sources.multiday_weather import generate_7day_weather, describe_day_types
from ..evaluation.baselines import run_no_battery, run_rule_based
from ..evaluation.metrics import compute_metrics, PolicyMetrics
from ..evaluation.rl_policy import train_q_learning_policy, run_q_learning_policy


# ════════════════════════════════════════════════════════════════════════════
# Rolling-horizon AI dispatch
# ════════════════════════════════════════════════════════════════════════════
def run_rolling_horizon(
    state_df: pd.DataFrame,
    params: BatteryParams,
    *,
    lookahead: int = 24,
    terminal_soc_value: float = 0.30,
) -> pd.DataFrame:
    """
    At each hour h:
      1. Build a lookahead window [h, h + lookahead) (capped at horizon end)
      2. Run DP over that window with current SOC, using an adaptive action grid
         that includes the exact surplus/deficit for each hour
      3. Apply only the FIRST action
      4. Advance to h+1

    terminal_soc_value: AUD/kWh value assigned to energy remaining in the battery
    at the end of each planning window. This prevents the rolling horizon from
    emptying the battery near the end of its lookahead, since the SOC carries
    over to the next window. Set to ~average import price for best results.

    Returns a full schedule DataFrame in the same format as baselines.
    """
    H = len(state_df)
    soc = params.initial_soc_kwh()
    rows = []

    # Build an enriched action grid that includes surplus/deficit match values.
    # The standard grid has coarse steps (0, ±0.5, ±1, ..., ±5).
    # Adding the exact surplus per hour lets the DP capture the same
    # self-consumption that the rule-based controller achieves.
    pv_arr = state_df["pv_kw"].to_numpy(dtype=float)
    load_arr = state_df["load_kw"].to_numpy(dtype=float)
    surplus_arr = pv_arr - load_arr  # positive = surplus, negative = deficit

    for h in range(H):
        # Determine lookahead window
        end = min(h + lookahead, H)
        window_df = state_df.iloc[h:end].copy()

        if len(window_df) == 0:
            break

        # Build per-window adaptive action grid
        base_actions = set(cfg.ACTION_GRID_KW)
        for wh in range(h, end):
            s = surplus_arr[wh]
            if s > 0.05:
                # Rule-based would charge full surplus: action = -surplus
                base_actions.add(-round(min(s, params.max_charge_kw), 3))
            elif s < -0.05:
                # Rule-based would discharge to cover deficit: action = -surplus (positive)
                base_actions.add(round(min(-s, params.max_discharge_kw), 3))
        adaptive_grid = tuple(sorted(base_actions))

        # Use terminal value when there are hours remaining after this window
        use_terminal = terminal_soc_value if end < H else 0.0

        # Run DP over the window
        result = optimise_dispatch(
            window_df,
            params=params,
            initial_soc_kwh=soc,
            scenario="base",
            terminal_soc_value=use_terminal,
            action_grid=adaptive_grid,
        )

        # Extract only the first action from the optimised schedule
        first_row = result.schedule.iloc[0]
        action = float(first_row["action_kw"])

        # Apply the action to get the actual step result
        ts = state_df.index[h]
        r = state_df.iloc[h]
        step = apply_action(
            soc_kwh=soc,
            action_kw=action,
            pv_kw=float(r["pv_kw"]),
            load_kw=float(r["load_kw"]),
            import_price=float(r["import_price"]),
            export_price=float(r["export_price"]),
            params=params,
        )

        rows.append({
            "timestamp": ts,
            "pv_kw": float(r["pv_kw"]),
            "load_kw": float(r["load_kw"]),
            "period": r.get("period", ""),
            "import_price": float(r["import_price"]),
            "export_price": float(r["export_price"]),
            "action_kw": float(step["action_applied_kw"]),
            "soc_kwh": float(step["soc_kwh"]),
            "soc_pct": 100 * float(step["soc_kwh"]) / params.capacity_kwh,
            "net_grid_kw": float(step["net_grid_kw"]),
            "grid_import_kwh": float(step["grid_import_kwh"]),
            "grid_export_kwh": float(step["grid_export_kwh"]),
            "cost_aud": float(step["cost_aud"]),
            "degradation_cost_aud": float(step["degradation_cost_aud"]),
        })
        soc = step["soc_kwh"]

    return pd.DataFrame(rows).set_index("timestamp")


# ════════════════════════════════════════════════════════════════════════════
# Full 7-day rolling backtest
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class RollingBacktestReport:
    """All results from the 7-day rolling-horizon backtest."""
    state_df: pd.DataFrame                        # full 168h energy state
    weather_meta: dict[str, Any]
    params: BatteryParams
    schedules: dict[str, pd.DataFrame]            # policy → 168h schedule
    metrics: pd.DataFrame                         # comparison table
    runtimes: dict[str, float]
    day_types: list[dict[str, str]]
    daily_costs: pd.DataFrame                     # day × policy cost breakdown


def build_7day_state(
    tariff: TariffSchedule | None = None,
    pv_capacity_kw: float = cfg.PV_CAPACITY_KW,
    derate: float = cfg.PV_DERATE_FACTOR,
    seed: int = 2025,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the full 168h energy state for the 7-day backtest."""
    weather_df, meta = generate_7day_weather(seed=seed)
    pv = estimate_pv_generation(weather_df, pv_capacity_kw, derate)
    load = synthetic_load(weather_df.index, seed=42)

    if tariff is None:
        tariff = default_schedule()
    tariff_df = tariff.schedule(weather_df.index)

    state_df = pd.concat([
        pv.rename("pv_kw"),
        load.rename("load_kw"),
        weather_df[["shortwave_radiation", "cloud_cover", "temperature_2m"]],
        tariff_df,
    ], axis=1)

    return state_df, meta


def run_rolling_backtest(
    params: BatteryParams | None = None,
    tariff: TariffSchedule | None = None,
    pv_capacity_kw: float = cfg.PV_CAPACITY_KW,
    lookahead: int = 24,
    seed: int = 2025,
) -> RollingBacktestReport:
    """
    Run the full 7-day backtest.

    Compares: no_battery, rule_based, dp_rolling, rl_q_learning.
    """
    if params is None:
        params = BatteryParams()

    state_df, weather_meta = build_7day_state(
        tariff=tariff, pv_capacity_kw=pv_capacity_kw, seed=seed
    )

    schedules = {}
    runtimes = {}

    # 1. No-battery baseline
    t0 = time.perf_counter()
    schedules["no_battery"] = run_no_battery(state_df)
    runtimes["no_battery"] = time.perf_counter() - t0

    # 2. Rule-based baseline
    t0 = time.perf_counter()
    schedules["rule_based"] = run_rule_based(state_df, params)
    runtimes["rule_based"] = time.perf_counter() - t0

    # 3. Dynamic Programming rolling-horizon policy
    t0 = time.perf_counter()
    schedules["dp_rolling"] = run_rolling_horizon(
        state_df, params, lookahead=lookahead
    )
    runtimes["dp_rolling"] = time.perf_counter() - t0

    # 4. Reinforcement Learning policy
    # Train on the seven individual simulated days, then roll out once over
    # the full 168-hour sequence. This is a benchmark policy, not the main
    # controller. If it underperforms, that is still a valid empirical result.
    t0 = time.perf_counter()
    training_days = [
        state_df.iloc[d * 24:(d + 1) * 24].copy()
        for d in range(7)
    ]
    q_table = train_q_learning_policy(training_days, params, episodes=800)
    schedules["rl_q_learning"] = run_q_learning_policy(
        state_df, params, q_table, scenario="7day"
    )
    runtimes["rl_q_learning"] = time.perf_counter() - t0

    # ─── Compute metrics ────────────────────────────────────────────────────
    metric_objs = []
    for name, sched in schedules.items():
        metric_objs.append(
            compute_metrics(sched, policy_name=name, params=params,
                            runtime_s=runtimes.get(name, 0.0))
        )

    from ..evaluation.metrics import comparison_table, relative_savings
    metrics = comparison_table(metric_objs)
    metrics = relative_savings(metrics, baseline="no_battery")

    # ─── Per-day cost breakdown ─────────────────────────────────────────────
    daily_rows = []
    for d in range(7):
        start_h = d * 24
        end_h = (d + 1) * 24
        for name, sched in schedules.items():
            day_slice = sched.iloc[start_h:end_h]
            cost = float(day_slice["cost_aud"].sum()
                         + day_slice.get("degradation_cost_aud", pd.Series(0)).sum())
            grid_import = float(day_slice["grid_import_kwh"].sum())
            peak_mask = day_slice["period"] == "peak" if "period" in day_slice.columns else pd.Series(False, index=day_slice.index)
            peak_import = float(day_slice.loc[peak_mask, "grid_import_kwh"].sum())
            daily_rows.append({
                "day": d + 1,
                "policy": name,
                "cost_aud": cost,
                "grid_import_kwh": grid_import,
                "peak_import_kwh": peak_import,
            })
    daily_costs = pd.DataFrame(daily_rows)

    return RollingBacktestReport(
        state_df=state_df,
        weather_meta=weather_meta,
        params=params,
        schedules=schedules,
        metrics=metrics,
        runtimes=runtimes,
        day_types=describe_day_types(),
        daily_costs=daily_costs,
    )


def format_rolling_metrics(report: RollingBacktestReport) -> str:
    """Format the rolling backtest metrics as a printable string."""
    m = report.metrics
    lines = [
        "=" * 72,
        "7-DAY ROLLING-HORIZON BACKTEST RESULTS",
        "=" * 72,
    ]

    # Summary metrics
    cols = ["total_cost_aud", "total_grid_import_kwh", "total_grid_export_kwh",
            "self_consumption_pct", "peak_period_import_kwh",
            "battery_throughput_kwh", "battery_cycles_equivalent",
            "runtime_s"]
    cols = [c for c in cols if c in m.columns]
    lines.append(m[cols].round(2).to_string())

    # Savings
    if "cost_saving_pct" in m.columns:
        lines.append("\nCost saving vs no-battery:")
        for policy in m.index:
            if policy != "no_battery":
                saving = m.loc[policy, "cost_saving_pct"]
                lines.append(f"  {policy:20s}: {saving:.1f}%")

    # Best controllable policy summary
    controllable = [p for p in ["rule_based", "dp_rolling", "rl_q_learning"] if p in m.index]
    if controllable:
        best = m.loc[controllable, "total_cost_aud"].idxmin()
        best_cost = float(m.loc[best, "total_cost_aud"])
        lines.append(f"\nBest controllable 7-day policy: {best} (${best_cost:.2f})")
        if "rule_based" in m.index and best != "rule_based":
            rb_cost = float(m.loc["rule_based", "total_cost_aud"])
            diff = rb_cost - best_cost
            pct = 100 * diff / abs(rb_cost) if abs(rb_cost) > 1e-6 else 0.0
            lines.append(f"  Best policy vs rule-based: ${diff:.2f} ({pct:.1f}%)")

    # Daily breakdown
    lines.append("\nDaily cost breakdown (AUD):")
    pivot = report.daily_costs.pivot(index="day", columns="policy", values="cost_aud")
    cols_order = [c for c in ["no_battery", "rule_based", "dp_rolling", "rl_q_learning"] if c in pivot.columns]
    lines.append(pivot[cols_order].round(2).to_string())

    # Runtimes
    lines.append("\nRuntimes:")
    for name, rt in report.runtimes.items():
        lines.append(f"  {name:20s}: {rt:.2f}s")

    return "\n".join(lines)


if __name__ == "__main__":
    report = run_rolling_backtest()
    print(format_rolling_metrics(report))
