"""
dispatch_optimizer.py — search-based dispatch optimiser (the AI core).

Solves the 24-hour battery dispatch problem via DYNAMIC PROGRAMMING over a
discretised state space:

    state = (hour, soc_bin)
    action ∈ ACTION_GRID_KW   (default 9 discrete levels from -5 to +5 kW)

For each (hour, soc_bin) the optimiser computes the minimum total cost-to-go
to the end of the horizon, by exhaustive search over all feasible actions
followed by a backward Bellman recursion.

Why DP and not LP:
  - Spec asks for dynamic programming / discrete search.
  - DP gives an EXPLAINABLE optimal action at every (hour, SoC) — useful for
    the dashboard.
  - Action grid is small enough that DP is fast (~5 ms for 24h × 21 bins × 9
    actions).
  - Easier to extend with non-linear objectives (degradation cost, peak-import
    penalty, reserve violation penalty) without refactoring.

Objective per hour:
    cost = grid_import * import_price - grid_export * export_price
         + degradation_aud_per_kwh * |action_kw|
         + reserve_violation_penalty if action would force soc < reserve

The optimiser never produces an infeasible action (we mask them in the
forward pass), so reserve_violation_penalty mainly guards against numerical
edge cases.
"""

from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd

from .. import config as cfg
from .rules import BatteryParams, apply_action, evaluate_rules


# ════════════════════════════════════════════════════════════════════════════
# DP solver
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class DispatchResult:
    schedule: pd.DataFrame      # hour-indexed, columns: action_kw, soc_kwh, soc_pct, ...
    objective_aud: float        # total cost over horizon
    runtime_s: float
    actions_grid: tuple
    soc_grid_kwh: np.ndarray


def optimise_dispatch(
    state_df: pd.DataFrame,
    *,
    params: BatteryParams,
    initial_soc_kwh: float | None = None,
    action_grid: tuple = cfg.ACTION_GRID_KW,
    soc_bins: int = cfg.SOC_BINS,
    scenario: str = "base",
    terminal_soc_value: float = 0.0,
) -> DispatchResult:
    """
    Solve the 24-hour dispatch problem via DP.

    `state_df` must have columns: pv_kw, load_kw, import_price, export_price.

    `terminal_soc_value` (AUD/kWh): if > 0, the DP values ending with high SoC.
    Used in rolling-horizon mode where the battery state carries over to the next
    planning window.  A value ~= average import price (e.g. 0.30) is reasonable.
    """
    t0 = time.perf_counter()
    H = len(state_df)
    pv = state_df["pv_kw"].to_numpy(dtype=float)
    load = state_df["load_kw"].to_numpy(dtype=float)
    import_p = state_df["import_price"].to_numpy(dtype=float)
    export_p = state_df["export_price"].to_numpy(dtype=float)

    soc_grid = np.linspace(params.soc_min_kwh, params.soc_max_kwh, soc_bins)
    if initial_soc_kwh is None:
        initial_soc_kwh = params.initial_soc_kwh()
    initial_bin = int(np.argmin(np.abs(soc_grid - initial_soc_kwh)))

    A = np.array(action_grid, dtype=float)
    nA = len(A)

    # Value function: V[h, b] = min cost-to-go from (hour h, soc_bin b) to end
    V = np.full((H + 1, soc_bins), np.inf)
    # Terminal value: reward high SOC at end of horizon (for rolling-horizon continuity)
    if terminal_soc_value > 0:
        # Higher SOC at the end → lower cost (negative reward = benefit)
        V[H, :] = -terminal_soc_value * soc_grid
    else:
        V[H, :] = 0.0
    # Best action and resulting soc bin per (h, b) — for forward reconstruction
    best_action = np.full((H, soc_bins), -1, dtype=int)
    best_next_bin = np.full((H, soc_bins), -1, dtype=int)

    # ─── Backward induction ────────────────────────────────────────────────
    for h in range(H - 1, -1, -1):
        for b in range(soc_bins):
            soc = soc_grid[b]
            best_cost = np.inf
            best_a_idx = -1
            best_nb = -1
            for a_idx in range(nA):
                a = A[a_idx]
                # Feasibility
                if a < 0 and not params.allow_grid_charging:
                    # charging: only allowed from solar surplus when grid charging is off
                    surplus = pv[h] - load[h]
                    if -a > surplus + 1e-6:
                        continue
                step = apply_action(
                    soc_kwh=soc, action_kw=a,
                    pv_kw=pv[h], load_kw=load[h],
                    import_price=import_p[h], export_price=export_p[h],
                    params=params,
                )
                # Bin the resulting SoC
                next_soc = step["soc_kwh"]
                nb = int(np.argmin(np.abs(soc_grid - next_soc)))
                step_cost = step["cost_aud"] + step["degradation_cost_aud"]
                # Reserve violation guard (extremely rare after clamping)
                if step["soc_kwh"] < params.soc_min_kwh - 1e-3:
                    step_cost += cfg.RESERVE_VIOLATION_PENALTY
                cost = step_cost + V[h + 1, nb]
                if cost < best_cost:
                    best_cost = cost
                    best_a_idx = a_idx
                    best_nb = nb
            V[h, b] = best_cost
            best_action[h, b] = best_a_idx
            best_next_bin[h, b] = best_nb

    # ─── Forward roll-out from initial_bin to recover optimal schedule ─────
    rows = []
    soc_kwh = soc_grid[initial_bin]
    b = initial_bin
    total_cost = 0.0

    for h in range(H):
        a_idx = best_action[h, b]
        action = float(A[a_idx]) if a_idx >= 0 else 0.0
        step = apply_action(
            soc_kwh=soc_kwh, action_kw=action,
            pv_kw=pv[h], load_kw=load[h],
            import_price=import_p[h], export_price=export_p[h],
            params=params,
        )
        rows.append({
            "timestamp": state_df.index[h],
            "pv_kw": float(pv[h]),
            "load_kw": float(load[h]),
            "period": state_df["period"].iloc[h] if "period" in state_df.columns else "",
            "import_price": float(import_p[h]),
            "export_price": float(export_p[h]),
            "action_kw": float(step["action_applied_kw"]),
            "soc_kwh": float(step["soc_kwh"]),
            "soc_pct": 100 * float(step["soc_kwh"]) / params.capacity_kwh,
            "net_grid_kw": float(step["net_grid_kw"]),
            "grid_import_kwh": float(step["grid_import_kwh"]),
            "grid_export_kwh": float(step["grid_export_kwh"]),
            "cost_aud": float(step["cost_aud"]),
            "degradation_cost_aud": float(step["degradation_cost_aud"]),
        })
        total_cost += step["cost_aud"] + step["degradation_cost_aud"]
        soc_kwh = step["soc_kwh"]
        b = best_next_bin[h, b] if best_next_bin[h, b] >= 0 else int(np.argmin(np.abs(soc_grid - soc_kwh)))

    schedule = pd.DataFrame(rows).set_index("timestamp")
    schedule["scenario"] = scenario

    return DispatchResult(
        schedule=schedule,
        objective_aud=float(total_cost),
        runtime_s=time.perf_counter() - t0,
        actions_grid=action_grid,
        soc_grid_kwh=soc_grid,
    )


# ════════════════════════════════════════════════════════════════════════════
# Helper: derive a human-readable action label
# ════════════════════════════════════════════════════════════════════════════
def label_action(action_kw: float) -> str:
    if action_kw > 0.05:
        return f"DISCHARGE {action_kw:.1f} kW"
    if action_kw < -0.05:
        return f"CHARGE {abs(action_kw):.1f} kW"
    return "HOLD"


# ════════════════════════════════════════════════════════════════════════════
# Smoke test
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from .state_builder import build_state
    state = build_state()
    params = BatteryParams()
    result = optimise_dispatch(state.df, params=params, scenario="base")
    print(f"Objective: ${result.objective_aud:.2f}   "
          f"Runtime: {result.runtime_s*1000:.1f} ms")
    print(result.schedule[["pv_kw", "load_kw", "period", "action_kw",
                           "soc_pct", "net_grid_kw", "cost_aud"]].head(10).round(2))
