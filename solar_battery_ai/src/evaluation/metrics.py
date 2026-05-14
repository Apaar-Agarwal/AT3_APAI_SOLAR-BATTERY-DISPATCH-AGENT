"""
metrics.py — assignment-required performance metrics.

For each policy schedule we compute:
    total_cost_aud
    total_grid_import_kwh
    total_grid_export_kwh
    self_consumption_pct
    peak_period_import_kwh
    battery_throughput_kwh
    battery_cycles_equivalent
    runtime_s

Plus comparison helpers (vs. no_battery and vs. rule_based).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import pandas as pd

from ..agent.rules import BatteryParams


@dataclass
class PolicyMetrics:
    policy: str
    total_cost_aud: float
    total_grid_import_kwh: float
    total_grid_export_kwh: float
    total_pv_kwh: float
    total_load_kwh: float
    self_consumption_pct: float
    peak_period_import_kwh: float
    battery_throughput_kwh: float
    battery_cycles_equivalent: float
    runtime_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    schedule: pd.DataFrame,
    *,
    policy_name: str,
    params: BatteryParams,
    runtime_s: float = 0.0,
) -> PolicyMetrics:
    pv_total = float(schedule["pv_kw"].sum())
    load_total = float(schedule["load_kw"].sum())
    import_total = float(schedule["grid_import_kwh"].sum())
    export_total = float(schedule["grid_export_kwh"].sum())

    # Self-consumption: how much of generated solar was used on-site
    # (= total PV − exported), as a fraction of PV. Bounded [0, 1].
    if pv_total > 1e-6:
        self_cons = max(0.0, min(1.0, (pv_total - export_total) / pv_total))
    else:
        self_cons = 0.0

    peak_import = float(
        schedule.loc[schedule.get("period", "") == "peak", "grid_import_kwh"].sum()
        if "period" in schedule.columns else 0.0
    )

    throughput = float(schedule["action_kw"].abs().sum())  # kWh of battery activity
    cycles = throughput / (2 * params.capacity_kwh) if params.capacity_kwh > 0 else 0.0

    return PolicyMetrics(
        policy=policy_name,
        total_cost_aud=float(schedule["cost_aud"].sum() + schedule.get("degradation_cost_aud", pd.Series(0)).sum()),
        total_grid_import_kwh=import_total,
        total_grid_export_kwh=export_total,
        total_pv_kwh=pv_total,
        total_load_kwh=load_total,
        self_consumption_pct=100 * self_cons,
        peak_period_import_kwh=peak_import,
        battery_throughput_kwh=throughput,
        battery_cycles_equivalent=cycles,
        runtime_s=runtime_s,
    )


def comparison_table(metrics: list[PolicyMetrics]) -> pd.DataFrame:
    """Tidy comparison DataFrame across policies."""
    df = pd.DataFrame([m.to_dict() for m in metrics]).set_index("policy")
    return df.round(3)


def relative_savings(df: pd.DataFrame, baseline: str = "no_battery") -> pd.DataFrame:
    """% saving vs. a chosen baseline for cost and grid import."""
    if baseline not in df.index:
        return df
    base_cost = df.loc[baseline, "total_cost_aud"]
    base_import = df.loc[baseline, "total_grid_import_kwh"]
    out = df.copy()
    out["cost_saving_pct"] = (
        (base_cost - df["total_cost_aud"]) / base_cost * 100
        if abs(base_cost) > 1e-9 else 0.0
    )
    out["grid_import_reduction_pct"] = (
        (base_import - df["total_grid_import_kwh"]) / base_import * 100
        if abs(base_import) > 1e-9 else 0.0
    )
    return out.round(2)
