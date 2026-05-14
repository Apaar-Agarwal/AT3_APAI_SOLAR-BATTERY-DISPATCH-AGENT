"""
baselines.py — non-AI dispatch policies for comparison.

Two baselines:
  1. NO BATTERY     : solar first; surplus exported, deficit imported.
  2. SIMPLE RULE    : charge from solar surplus, discharge when load > solar.
                       No tariff awareness, no scenario reasoning.

The AI agent (dispatch_optimizer) is then compared against these.
"""

from __future__ import annotations
import time
import pandas as pd

from ..agent.rules import BatteryParams, apply_action


# ────────────────────────────────────────────────────────────────────────────
# Baseline 1 — no battery
# ────────────────────────────────────────────────────────────────────────────
def run_no_battery(state_df: pd.DataFrame) -> pd.DataFrame:
    """Solar meets load; surplus exported, deficit imported."""
    rows = []
    for ts, r in state_df.iterrows():
        net = float(r["load_kw"]) - float(r["pv_kw"])  # +import / -export
        grid_import = max(net, 0.0)
        grid_export = max(-net, 0.0)
        cost = grid_import * float(r["import_price"]) - grid_export * float(r["export_price"])
        rows.append({
            "timestamp": ts,
            "pv_kw": float(r["pv_kw"]),
            "load_kw": float(r["load_kw"]),
            "period": r.get("period", ""),
            "import_price": float(r["import_price"]),
            "export_price": float(r["export_price"]),
            "action_kw": 0.0,
            "soc_kwh": 0.0,
            "soc_pct": 0.0,
            "net_grid_kw": net,
            "grid_import_kwh": grid_import,
            "grid_export_kwh": grid_export,
            "cost_aud": cost,
            "degradation_cost_aud": 0.0,
        })
    return pd.DataFrame(rows).set_index("timestamp")


# ────────────────────────────────────────────────────────────────────────────
# Baseline 2 — simple rule-based (no tariff awareness)
# ────────────────────────────────────────────────────────────────────────────
def run_rule_based(state_df: pd.DataFrame, params: BatteryParams) -> pd.DataFrame:
    """
    For each hour:
        if solar > load: charge with surplus (capped by max_charge)
        elif solar < load: discharge to cover deficit (capped by max_discharge)
        else: hold
    """
    rows = []
    soc = params.initial_soc_kwh()
    for ts, r in state_df.iterrows():
        pv = float(r["pv_kw"]); load = float(r["load_kw"])
        surplus = pv - load
        if surplus > 0.05:
            action = -min(surplus, params.max_charge_kw)         # charge (signed −)
        elif surplus < -0.05:
            action = min(-surplus, params.max_discharge_kw)      # discharge (signed +)
        else:
            action = 0.0
        step = apply_action(
            soc_kwh=soc, action_kw=action,
            pv_kw=pv, load_kw=load,
            import_price=float(r["import_price"]),
            export_price=float(r["export_price"]),
            params=params,
        )
        rows.append({
            "timestamp": ts,
            "pv_kw": pv, "load_kw": load,
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


# ────────────────────────────────────────────────────────────────────────────
# Convenience runner
# ────────────────────────────────────────────────────────────────────────────
def run_all_baselines(state_df: pd.DataFrame, params: BatteryParams) -> dict[str, dict]:
    out = {}
    t0 = time.perf_counter()
    nb = run_no_battery(state_df)
    out["no_battery"] = {"schedule": nb, "runtime_s": time.perf_counter() - t0}

    t0 = time.perf_counter()
    rb = run_rule_based(state_df, params)
    out["rule_based"] = {"schedule": rb, "runtime_s": time.perf_counter() - t0}
    return out
