"""
main.py — CLI pipeline for the source-aware AI battery dispatch agent.

Steps:
  1. Load weather (live → cache → synthetic fallback)
  2. Build the energy state (PV estimate × load × tariff)
  3. Construct uncertainty scenarios
  4. Run baselines (no_battery, rule_based)
  5. Run AI dispatch under each scenario
  6. Compute metrics + savings
  7. Save outputs (CSVs + JSON + figures)
  8. (Optional) Run 7-day rolling-horizon backtest

Run:
    python main.py
    python main.py --refresh           (force a live API call)
    python main.py --offline           (use cache only — never hit the API)
    python main.py --grid-charging     (allow grid charging in agent)
    python main.py --rolling-backtest  (run 7-day rolling-horizon backtest)
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src import config as cfg
from src.agent.rules import BatteryParams
from src.agent.state_builder import build_state
from src.evaluation.backtest import run_backtest
from src.visualisation.plots import save_all_figures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Source-aware AI battery dispatch agent")
    p.add_argument("--refresh", action="store_true", help="Force a live Open-Meteo fetch.")
    p.add_argument("--offline", action="store_true", help="Use cache/synthetic only.")
    p.add_argument("--grid-charging", action="store_true",
                   help="Allow agent to charge from grid in off-peak.")
    p.add_argument("--horizon", type=int, default=cfg.HORIZON_HOURS,
                   help=f"Planning horizon (hours). Default {cfg.HORIZON_HOURS}.")
    p.add_argument("--rolling-backtest", action="store_true",
                   help="Run 7-day rolling-horizon backtest after the single-day run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_total = time.perf_counter()

    print("=" * 72)
    print("Source-Aware AI Battery Dispatch Agent")
    print("=" * 72)

    # --- 1. Build state ---
    print("\n[1/4] Fetching source data and building energy state ...")
    state = build_state(
        horizon_hours=args.horizon,
        prefer_cache=args.offline and not args.refresh,
    )
    print(f"      source     : {state.source_meta.get('source')}")
    print(f"      synthetic? : {state.source_meta.get('synthetic')}")
    print(f"      horizon    : {state.horizon}h")
    print(f"      PV total   : {state.pv.sum():.2f} kWh")
    print(f"      Load total : {state.load.sum():.2f} kWh")

    # --- 2. Battery parameters ---
    params = BatteryParams(allow_grid_charging=args.grid_charging)

    # --- 3. Run full backtest (baselines + AI under 3 scenarios) ---
    print("\n[2/4] Running baselines + AI dispatch under low/base/high scenarios ...")
    report = run_backtest(state, params)

    # --- 4. Persist outputs ---
    print("\n[3/4] Persisting outputs ...")
    out = cfg.OUTPUTS

    for name, sched in report.schedules.items():
        sched.to_csv(out / f"schedule_{name}.csv")
    report.explained.to_csv(out / "schedule_ai_agent_base_explained.csv")
    report.metrics.to_csv(out / "metrics_comparison.csv")
    report.metrics_with_savings.to_csv(out / "metrics_with_savings.csv")
    report.robustness.to_csv(out / "metrics_robustness.csv", index=False)
    (out / "risk_messages.json").write_text(json.dumps(report.risk_messages, indent=2))
    report.state.df.to_csv(out / "energy_state_24h.csv")

    print(f"      -> {out}/")
    for p in sorted(out.glob("*")):
        print(f"        {p.name}")

    # --- 5. Figures ---
    print("\n[4/4] Generating figures ...")
    figs = save_all_figures(report)
    for k, v in figs.items():
        print(f"      {k:<22} -> {v.relative_to(cfg.ROOT)}")

    # --- 6. Headline summary ---
    df = report.metrics_with_savings
    print("\n" + "=" * 72)
    print("RESULTS  (24h horizon - base scenario)")
    print("=" * 72)
    cols = ["total_cost_aud", "total_grid_import_kwh", "total_grid_export_kwh",
            "self_consumption_pct", "cost_saving_pct", "grid_import_reduction_pct"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string())

    print("\nRisk warnings:")
    for msg in report.risk_messages:
        print(f"  {msg}")

    print("\nROBUSTNESS  (cost AUD per policy x scenario):")
    pivot = report.robustness.pivot(index="policy", columns="scenario",
                                    values="total_cost_aud")
    pivot = pivot[[c for c in ("low", "base", "high") if c in pivot.columns]]
    print(pivot.round(2).to_string())

    print(f"\nTotal pipeline runtime: {time.perf_counter() - t_total:.2f}s")

    # --- 7. Rolling-horizon 7-day backtest (optional) ---
    if args.rolling_backtest:
        _run_rolling_backtest(params)


def _run_rolling_backtest(params: BatteryParams) -> None:
    """Run and save the 7-day rolling-horizon backtest."""
    from src.evaluation.rolling_backtest import (
        run_rolling_backtest, format_rolling_metrics,
    )
    from src.visualisation.plots import save_rolling_figures

    print("\n" + "=" * 72)
    print("7-DAY ROLLING-HORIZON BACKTEST")
    print("=" * 72)
    print("Running 7-day policy comparison: no-battery, rule-based, DP rolling, and RL Q-learning...")

    t0 = time.perf_counter()
    rolling = run_rolling_backtest(params=params)
    runtime = time.perf_counter() - t0

    print(format_rolling_metrics(rolling))

    # Save outputs
    out = cfg.OUTPUTS
    for name, sched in rolling.schedules.items():
        sched.to_csv(out / f"rolling_7day_schedule_{name}.csv")
    rolling.metrics.to_csv(out / "rolling_7day_metrics.csv")
    rolling.daily_costs.to_csv(out / "rolling_7day_daily_costs.csv", index=False)
    rolling.state_df.to_csv(out / "rolling_7day_energy_state.csv")

    print(f"\n      Schedules + metrics saved to {out}/")

    # Figures
    figs = save_rolling_figures(rolling)
    for k, v in figs.items():
        print(f"      {k:<30} -> {v.relative_to(cfg.ROOT)}")

    # Honest interpretation
    m = rolling.metrics
    controllable = [p for p in ["rule_based", "dp_rolling", "rl_q_learning"] if p in m.index]
    if controllable:
        best = m.loc[controllable, "total_cost_aud"].idxmin()
        best_cost = float(m.loc[best, "total_cost_aud"])
        print("\n" + "-" * 72)
        print("INTERPRETATION")
        print("-" * 72)
        print(f"  Best controllable 7-day policy: {best} (${best_cost:.2f}).")
        if "rule_based" in m.index and best != "rule_based":
            rb_cost = float(m.loc["rule_based", "total_cost_aud"])
            diff = rb_cost - best_cost
            print(f"  It saves ${diff:.2f} vs rule-based over 7 days.")
        elif best == "rule_based" and "dp_rolling" in m.index:
            dp_cost = float(m.loc["dp_rolling", "total_cost_aud"])
            print(f"  Rule-based is ${dp_cost - best_cost:.2f} cheaper than DP rolling in this run.")
        if "rl_q_learning" in m.index:
            rl_cost = float(m.loc["rl_q_learning", "total_cost_aud"])
            print(f"  RL Q-learning is included as a learning-based benchmark (${rl_cost:.2f}).")
        print("  This is reported as empirical comparison, not forced AI superiority.")
        print("  DP provides transparent planning; RL demonstrates learned control; rule-based")
        print("  remains a strong simple baseline under residential TOU conditions.")

    print(f"\nRolling backtest total runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
