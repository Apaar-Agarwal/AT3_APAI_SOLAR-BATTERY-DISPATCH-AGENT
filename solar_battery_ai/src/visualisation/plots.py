"""
plots.py — figures for the report and Streamlit dashboard.

Two style sets:
  - matplotlib  : saves PNGs into reports/figures/  (used by main.py)
  - plotly      : returns figures for Streamlit interactive display
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .. import config as cfg
from ..agent.uncertainty import ScenarioBundle


PALETTE = {
    "no_battery": "#94a3b8",
    "rule_based": "#fb923c",
    "ai_agent_base": "#dc2626",
    "ai_agent_low": "#0891b2",
    "ai_agent_high": "#16a34a",
    "pv": "#d97706",
    "load": "#0891b2",
    "soc": "#7c3aed",
}


# ════════════════════════════════════════════════════════════════════════════
# matplotlib (PNG output for the report)
# ════════════════════════════════════════════════════════════════════════════
def plot_inputs(state_df: pd.DataFrame, out: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.plot(state_df.index, state_df["pv_kw"], color=PALETTE["pv"], lw=2, label="PV estimate (kW)")
    ax1.plot(state_df.index, state_df["load_kw"], color=PALETTE["load"], lw=2, label="Load (kW)")
    ax1.set_ylabel("kW")
    ax1.grid(alpha=0.3)
    # Tariff period shading
    if "period" in state_df.columns:
        for ts, p in state_df["period"].items():
            color = {"peak": "#fee2e2", "shoulder": "#fef3c7", "off_peak": "#ecfeff"}.get(p, "white")
            ax1.axvspan(ts, ts + pd.Timedelta(hours=1), color=color, alpha=0.5, lw=0)
    ax1.legend(loc="upper left")
    ax1.set_title("Energy state — next 24 hours (shaded by tariff period)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def plot_cost_comparison(metrics_df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    df = metrics_df.copy()
    bars = ax.bar(
        df.index, df["total_cost_aud"],
        color=[PALETTE.get(p, "#7c3aed") for p in df.index],
    )
    ax.set_ylabel("Total cost over 24h (AUD)")
    ax.set_title("Electricity cost by policy  (lower = better)")
    ax.grid(alpha=0.3, axis="y")
    for bar, v in zip(bars, df["total_cost_aud"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"${v:.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontweight="bold")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def plot_grid_io(metrics_df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(metrics_df))
    w = 0.35
    ax.bar([i - w / 2 for i in x], metrics_df["total_grid_import_kwh"], width=w,
           label="Grid import (kWh)", color="#dc2626")
    ax.bar([i + w / 2 for i in x], metrics_df["total_grid_export_kwh"], width=w,
           label="Grid export (kWh)", color="#16a34a")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics_df.index, rotation=20)
    ax.set_ylabel("kWh")
    ax.set_title("Grid import vs export by policy")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def plot_soc_curve(schedules: dict[str, pd.DataFrame], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, df in schedules.items():
        if name == "no_battery": continue
        ax.plot(df.index, df["soc_pct"], lw=2, label=name, color=PALETTE.get(name, "#7c3aed"))
    ax.set_ylabel("Battery SoC (%)")
    ax.set_ylim(0, 100)
    ax.axhline(cfg.BATTERY_RESERVE_SOC_PCT, ls="--", color="red", lw=1, alpha=0.6,
               label=f"reserve {cfg.BATTERY_RESERVE_SOC_PCT}%")
    ax.set_title("Battery state of charge over 24h")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def plot_scenarios(bundle: ScenarioBundle, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.fill_between(bundle.base.index, bundle.low, bundle.high,
                    color="#0891b2", alpha=0.2, label="low ↔ high range")
    ax.plot(bundle.base.index, bundle.base, color="#dc2626", lw=2, label="base estimate")
    ax.plot(bundle.base.index, bundle.low, color="#94a3b8", lw=1, ls="--", label="low (cloudy)")
    ax.plot(bundle.base.index, bundle.high, color="#16a34a", lw=1, ls="--", label="high (sunny)")
    ax.set_ylabel("PV estimate (kW)")
    ax.set_title("Solar uncertainty — low / base / high scenarios")
    ax.legend(); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def plot_peak_imports(metrics_df: pd.DataFrame, out: Path) -> Path:
    """Bar of peak-period grid imports — where AI's tariff awareness is visible."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df = metrics_df.copy()
    bars = ax.bar(
        df.index, df["peak_period_import_kwh"],
        color=[PALETTE.get(p, "#7c3aed") for p in df.index],
    )
    ax.set_ylabel("Peak-period grid import (kWh)")
    ax.set_title("Imports during 4-9pm peak  (lower = better tariff awareness)")
    ax.grid(alpha=0.3, axis="y")
    for bar, v in zip(bars, df["peak_period_import_kwh"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontweight="bold")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def plot_robustness(robustness_df: pd.DataFrame, out: Path) -> Path:
    """Cost across (policy × scenario) — shows which policy degrades gracefully."""
    pivot = robustness_df.pivot(index="policy", columns="scenario",
                                values="total_cost_aud")
    # consistent column ordering
    cols = [c for c in ("low", "base", "high") if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(pivot.index))
    w = 0.27
    colours = {"low": "#0891b2", "base": "#dc2626", "high": "#16a34a"}
    for i, sc in enumerate(cols):
        offset = (i - 1) * w
        ax.bar([xi + offset for xi in x], pivot[sc], width=w,
               label=f"{sc} solar", color=colours.get(sc, "#7c3aed"))
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=20)
    ax.set_ylabel("Total cost over 24h (AUD)")
    ax.set_title("Cost robustness: each policy under low / base / high solar")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def save_all_figures(report, out_dir: Path = cfg.FIGURES) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "01_inputs": plot_inputs(report.state.df, out_dir / "01_inputs.png"),
        "02_cost_comparison": plot_cost_comparison(
            report.metrics.loc[["no_battery", "rule_based", "ai_agent_base"]],
            out_dir / "02_cost_comparison.png"),
        "03_grid_io": plot_grid_io(
            report.metrics.loc[["no_battery", "rule_based", "ai_agent_base"]],
            out_dir / "03_grid_io.png"),
        "04_soc_curve": plot_soc_curve(
            {k: v for k, v in report.schedules.items() if k in
             ("rule_based", "ai_agent_base", "ai_agent_low", "ai_agent_high")},
            out_dir / "04_soc_curve.png"),
        "05_scenarios": plot_scenarios(report.scenarios, out_dir / "05_scenarios.png"),
        "06_peak_imports": plot_peak_imports(
            report.metrics.loc[["no_battery", "rule_based", "ai_agent_base"]],
            out_dir / "06_peak_imports.png"),
        "07_robustness": plot_robustness(report.robustness, out_dir / "07_robustness.png"),
    }
    return paths


# ════════════════════════════════════════════════════════════════════════════
# Rolling backtest (7-day) matplotlib figures
# ════════════════════════════════════════════════════════════════════════════
ROLLING_PALETTE = {
    "no_battery": "#94a3b8",
    "rule_based": "#fb923c",
    "dp_rolling": "#dc2626",
    "rl_q_learning": "#7c3aed",
}


def plot_rolling_cumulative_cost(
    schedules: dict[str, pd.DataFrame], out: Path
) -> Path:
    """Cumulative cost over 7 days for each policy."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, sched in schedules.items():
        total_cost = sched["cost_aud"] + sched.get("degradation_cost_aud", 0)
        cum = total_cost.cumsum()
        ax.plot(sched.index, cum, lw=2, label=name,
                color=ROLLING_PALETTE.get(name, "#7c3aed"))
    ax.set_ylabel("Cumulative cost (AUD)")
    ax.set_title("7-day rolling backtest — cumulative electricity cost")
    ax.legend()
    ax.grid(alpha=0.3)
    # Day separators
    for d in range(1, 7):
        ax.axvline(sched.index[d * 24], color="#e2e8f0", ls=":", lw=1)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def plot_rolling_soc(
    schedules: dict[str, pd.DataFrame], out: Path
) -> Path:
    """Battery SOC over 7 days for controllable battery policies."""
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, sched in schedules.items():
        if name == "no_battery":
            continue
        ax.plot(sched.index, sched["soc_pct"], lw=1.8, label=name,
                color=ROLLING_PALETTE.get(name, "#7c3aed"))
    ax.axhline(cfg.BATTERY_RESERVE_SOC_PCT, ls="--", color="red", lw=1, alpha=0.6,
               label=f"reserve {cfg.BATTERY_RESERVE_SOC_PCT}%")
    ax.set_ylabel("Battery SoC (%)")
    ax.set_ylim(0, 105)
    ax.set_title("7-day battery state of charge comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    for d in range(1, 7):
        ax.axvline(sched.index[d * 24], color="#e2e8f0", ls=":", lw=1)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def plot_rolling_grid_io(
    schedules: dict[str, pd.DataFrame], out: Path
) -> Path:
    """Daily grid import/export as grouped bars."""
    rows = []
    for name, sched in schedules.items():
        for d in range(7):
            day_slice = sched.iloc[d * 24 : (d + 1) * 24]
            rows.append({
                "day": d + 1,
                "policy": name,
                "grid_import_kwh": float(day_slice["grid_import_kwh"].sum()),
                "grid_export_kwh": float(day_slice["grid_export_kwh"].sum()),
            })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(
        axes,
        ["grid_import_kwh", "grid_export_kwh"],
        ["Grid import (kWh/day)", "Grid export (kWh/day)"],
    ):
        policies = ["no_battery", "rule_based", "dp_rolling", "rl_q_learning"]
        policies = [p for p in policies if p in df["policy"].unique()]
        x = range(7)
        w = 0.25
        for i, p in enumerate(policies):
            vals = df[df["policy"] == p][metric].values
            offset = (i - 1) * w
            ax.bar([xi + offset for xi in x], vals, width=w, label=p,
                   color=ROLLING_PALETTE.get(p, "#7c3aed"))
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"D{d+1}" for d in range(7)])
        ax.set_ylabel("kWh")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def plot_rolling_daily_cost(daily_costs: pd.DataFrame, out: Path) -> Path:
    """Per-day cost comparison as grouped bars."""
    pivot = daily_costs.pivot(index="day", columns="policy", values="cost_aud")
    cols_order = [c for c in ["no_battery", "rule_based", "dp_rolling", "rl_q_learning"]
                  if c in pivot.columns]
    pivot = pivot[cols_order]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(pivot))
    w = 0.25
    for i, col in enumerate(cols_order):
        offset = (i - 1) * w
        bars = ax.bar([xi + offset for xi in x], pivot[col], width=w,
                      label=col, color=ROLLING_PALETTE.get(col, "#7c3aed"))
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Day {d}" for d in pivot.index])
    ax.set_ylabel("Daily cost (AUD)")
    ax.set_title("7-day rolling backtest — daily cost by policy")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.axhline(0, color="black", lw=0.5)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def plot_rolling_solar_load_overview(state_df: pd.DataFrame, out: Path) -> Path:
    """7-day solar/load/tariff overview."""
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.fill_between(state_df.index, state_df["pv_kw"], alpha=0.3,
                     color=PALETTE["pv"], label="PV estimate")
    ax1.plot(state_df.index, state_df["load_kw"], color=PALETTE["load"],
             lw=1.5, label="Load")
    ax1.set_ylabel("kW")
    ax1.set_title("7-day solar generation and household load")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    for d in range(1, 7):
        ax1.axvline(state_df.index[d * 24], color="#e2e8f0", ls=":", lw=1)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def save_rolling_figures(rolling_report, out_dir: Path = cfg.FIGURES) -> dict[str, Path]:
    """Save all 7-day rolling backtest figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "08_rolling_overview": plot_rolling_solar_load_overview(
            rolling_report.state_df, out_dir / "08_rolling_overview.png"),
        "09_rolling_cumulative_cost": plot_rolling_cumulative_cost(
            rolling_report.schedules, out_dir / "09_rolling_cumulative_cost.png"),
        "10_rolling_soc": plot_rolling_soc(
            rolling_report.schedules, out_dir / "10_rolling_soc.png"),
        "11_rolling_grid_io": plot_rolling_grid_io(
            rolling_report.schedules, out_dir / "11_rolling_grid_io.png"),
        "12_rolling_daily_cost": plot_rolling_daily_cost(
            rolling_report.daily_costs, out_dir / "12_rolling_daily_cost.png"),
    }
    return paths


# ════════════════════════════════════════════════════════════════════════════
# Plotly versions (used by Streamlit)
# ════════════════════════════════════════════════════════════════════════════
def plotly_inputs(state_df: pd.DataFrame) -> go.Figure:
    state_df = normalise_schedule_columns(state_df)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if state_df.empty:
        fig.update_layout(
            title="Energy state — no data",
            height=300,
            annotations=[dict(
                text="No energy state available.", xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False, font=dict(color="#64748b"),
            )],
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig
    fig.add_trace(go.Scatter(x=state_df.index, y=state_df["pv_kw"],
                             name="PV estimate", line=dict(color=PALETTE["pv"], width=2.5)))
    fig.add_trace(go.Scatter(x=state_df.index, y=state_df["load_kw"],
                             name="Load", line=dict(color=PALETTE["load"], width=2.5)))
    if "import_price" in state_df.columns:
        fig.add_trace(
            go.Scatter(x=state_df.index, y=state_df["import_price"],
                       name="Import $/kWh",
                       line=dict(color="#7c3aed", width=1.5, dash="dot")),
            secondary_y=True,
        )
    fig.update_layout(
        title="Energy state — next 24h",
        xaxis_title="Time",
        yaxis_title="kW",
        hovermode="x unified",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_yaxes(title_text="$/kWh", secondary_y=True)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Schedule schema normalisation
# ════════════════════════════════════════════════════════════════════════════
#
# Different parts of the project may produce schedule dataframes with slightly
# different column names (e.g. single-day backtest vs 7-day rolling backtest,
# or a user-provided dataframe loaded from CSV). The plotting layer must NOT
# rely on a specific upstream call site — it normalises whatever it gets so
# the UI never crashes on a column-name mismatch.
#
# Canonical schedule schema:
#   timestamp / time : datetime index
#   pv_kw            : solar generation estimate (kW)
#   load_kw          : household demand (kW)
#   action_kw        : battery action, +discharge / -charge
#   soc_pct          : battery state of charge (%)
#   period           : 'peak' | 'shoulder' | 'off_peak'
#   cost_aud         : AUD spent (or earned, if negative) for that hour
#   action_label     : human label e.g. "DISCHARGE 2.0 kW"
#   explanation      : symbolic-rule rationale
# ════════════════════════════════════════════════════════════════════════════
def normalise_schedule_columns(schedule: pd.DataFrame) -> pd.DataFrame:
    """Coerce a schedule dataframe to the canonical column schema.

    - Renames known aliases (e.g. ``pv_kwh`` -> ``pv_kw``, ``solar_kw`` ->
      ``pv_kw``, ``reason`` -> ``explanation``).
    - Fills any missing required columns with safe defaults so plotting
      functions can rely on the schema.
    - If a ``timestamp`` or ``time`` column exists, parses it and uses it as
      the index (without dropping it from the frame).
    - Returns an empty DataFrame unchanged so callers can short-circuit.
    """
    if schedule is None:
        return pd.DataFrame()

    schedule = schedule.copy()

    rename_map = {
        # PV / solar
        "pv_kwh": "pv_kw",
        "pv_generation_kw": "pv_kw",
        "solar_kw": "pv_kw",
        "solar_generation_kw": "pv_kw",
        "pv_estimate_kw": "pv_kw",
        # Load / demand
        "load_kwh": "load_kw",
        "household_load_kw": "load_kw",
        "demand_kw": "load_kw",
        # SoC
        "battery_soc_pct": "soc_pct",
        "soc_percent": "soc_pct",
        # Action
        "action": "action_kw",
        "battery_action_kw": "action_kw",
        "net_action_kw": "action_kw",
        # Cost
        "cost": "cost_aud",
        "electricity_cost_aud": "cost_aud",
        # Explanation
        "reason": "explanation",
    }
    for old, new in rename_map.items():
        if old in schedule.columns and new not in schedule.columns:
            schedule = schedule.rename(columns={old: new})

    # If schedule is empty AFTER renames, just return it. Callers should
    # check `.empty` before plotting.
    if schedule.empty:
        return schedule

    required_defaults = {
        "pv_kw": 0.0,
        "load_kw": 0.0,
        "action_kw": 0.0,
        "soc_pct": 0.0,
        "period": "unknown",
        "cost_aud": 0.0,
        "action_label": "HOLD",
        "explanation": "",
    }
    for col, default in required_defaults.items():
        if col not in schedule.columns:
            schedule[col] = default

    # If timestamp/time columns exist and the index isn't already datetime,
    # promote them to the index. We keep the original column with drop=False
    # so downstream code that reads `schedule["timestamp"]` still works.
    if not pd.api.types.is_datetime64_any_dtype(schedule.index):
        if "timestamp" in schedule.columns:
            schedule["timestamp"] = pd.to_datetime(schedule["timestamp"], errors="coerce")
            schedule = schedule.set_index("timestamp", drop=False)
        elif "time" in schedule.columns:
            schedule["time"] = pd.to_datetime(schedule["time"], errors="coerce")
            schedule = schedule.set_index("time", drop=False)

    return schedule


def plotly_dispatch(schedule: pd.DataFrame, title: str = "AI dispatch") -> go.Figure:
    """3-row stack: PV/load on top, action bars in the middle, SoC at the bottom.

    Robust to column-name variation and empty inputs. If the schedule is
    empty (e.g. simulator has consumed the full horizon), returns a placeholder
    figure rather than raising.
    """
    schedule = normalise_schedule_columns(schedule)

    if schedule.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{title} — no schedule available",
            height=300,
            annotations=[dict(
                text="No dispatch plan available — run optimisation or reset the simulation.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color="#64748b"),
            )],
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.42, 0.28, 0.30])
    fig.add_trace(go.Scatter(x=schedule.index, y=schedule["pv_kw"], name="PV",
                             line=dict(color=PALETTE["pv"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=schedule.index, y=schedule["load_kw"], name="Load",
                             line=dict(color=PALETTE["load"], width=2)), row=1, col=1)

    colors = ["#16a34a" if a > 0 else ("#dc2626" if a < 0 else "#94a3b8")
              for a in schedule["action_kw"]]
    fig.add_trace(go.Bar(x=schedule.index, y=schedule["action_kw"],
                         name="Action (kW)", marker_color=colors,
                         hovertemplate="%{y:.2f} kW<br>(+ discharge / − charge)"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=schedule.index, y=schedule["soc_pct"],
                             name="SoC", line=dict(color=PALETTE["soc"], width=2)),
                  row=3, col=1)

    fig.add_hline(y=cfg.BATTERY_RESERVE_SOC_PCT, line_dash="dash",
                  line_color="red", opacity=0.5, row=3, col=1,
                  annotation_text=f"reserve {cfg.BATTERY_RESERVE_SOC_PCT}%")

    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="kW (signed)", row=2, col=1)
    fig.update_yaxes(title_text="SoC %", range=[0, 100], row=3, col=1)
    fig.update_layout(title=title, height=620, hovermode="x unified",
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plotly_cost_bar(metrics_df: pd.DataFrame) -> go.Figure:
    df = metrics_df.copy()
    fig = go.Figure(go.Bar(
        x=df.index, y=df["total_cost_aud"],
        marker_color=[PALETTE.get(p, "#7c3aed") for p in df.index],
        text=[f"${v:.2f}" for v in df["total_cost_aud"]], textposition="outside",
    ))
    fig.update_layout(title="Cost by policy (lower = better)",
                      yaxis_title="AUD over 24h", height=380,
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plotly_grid_io(metrics_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Import", x=metrics_df.index,
                         y=metrics_df["total_grid_import_kwh"], marker_color="#dc2626"))
    fig.add_trace(go.Bar(name="Export", x=metrics_df.index,
                         y=metrics_df["total_grid_export_kwh"], marker_color="#16a34a"))
    fig.update_layout(title="Grid import vs export", barmode="group",
                      yaxis_title="kWh", height=380,
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plotly_soc(schedules: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for name, df in schedules.items():
        if name == "no_battery":
            continue
        df = normalise_schedule_columns(df)
        if df.empty:
            continue
        fig.add_trace(go.Scatter(x=df.index, y=df["soc_pct"], name=name,
                                 line=dict(color=PALETTE.get(name, "#7c3aed"), width=2)))
    fig.add_hline(y=cfg.BATTERY_RESERVE_SOC_PCT, line_dash="dash",
                  line_color="red", opacity=0.5,
                  annotation_text=f"reserve {cfg.BATTERY_RESERVE_SOC_PCT}%")
    fig.update_layout(title="Battery state of charge over 24h",
                      yaxis_title="SoC %", yaxis_range=[0, 100],
                      height=380, hovermode="x unified",
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plotly_scenarios(bundle: ScenarioBundle) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bundle.base.index, y=bundle.high,
                             line=dict(width=0), showlegend=False, name="high"))
    fig.add_trace(go.Scatter(x=bundle.base.index, y=bundle.low,
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(8,145,178,0.18)",
                             name="low ↔ high"))
    fig.add_trace(go.Scatter(x=bundle.base.index, y=bundle.base,
                             line=dict(color="#dc2626", width=2.5), name="base"))
    fig.add_trace(go.Scatter(x=bundle.base.index, y=bundle.low,
                             line=dict(color="#94a3b8", width=1, dash="dash"),
                             name="low"))
    fig.add_trace(go.Scatter(x=bundle.base.index, y=bundle.high,
                             line=dict(color="#16a34a", width=1, dash="dash"),
                             name="high"))
    fig.update_layout(title="Solar uncertainty — low / base / high scenarios",
                      yaxis_title="PV estimate (kW)", height=380,
                      hovermode="x unified",
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plotly_peak_imports(metrics_df: pd.DataFrame) -> go.Figure:
    df = metrics_df.copy()
    fig = go.Figure(go.Bar(
        x=df.index, y=df["peak_period_import_kwh"],
        marker_color=[PALETTE.get(p, "#7c3aed") for p in df.index],
        text=[f"{v:.2f} kWh" for v in df["peak_period_import_kwh"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Peak-period grid import (4-9 pm) — lower = better tariff awareness",
        yaxis_title="kWh", height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plotly_robustness(robustness_df: pd.DataFrame) -> go.Figure:
    pivot = robustness_df.pivot(index="policy", columns="scenario",
                                values="total_cost_aud")
    cols = [c for c in ("low", "base", "high") if c in pivot.columns]
    pivot = pivot[cols]
    colours = {"low": "#0891b2", "base": "#dc2626", "high": "#16a34a"}
    fig = go.Figure()
    for sc in cols:
        fig.add_trace(go.Bar(
            name=f"{sc} solar", x=pivot.index, y=pivot[sc],
            marker_color=colours.get(sc, "#7c3aed"),
            text=[f"${v:.2f}" for v in pivot[sc]], textposition="outside",
        ))
    fig.update_layout(
        title="Cost robustness — every policy under low / base / high solar",
        yaxis_title="AUD over 24h", barmode="group",
        height=420, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Plotly versions for rolling backtest (Streamlit)
# ════════════════════════════════════════════════════════════════════════════
def plotly_rolling_cumulative_cost(schedules: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for name, sched in schedules.items():
        sched = normalise_schedule_columns(sched)
        if sched.empty or "cost_aud" not in sched.columns:
            continue
        total_cost = sched["cost_aud"] + sched.get("degradation_cost_aud", 0)
        cum = total_cost.cumsum()
        fig.add_trace(go.Scatter(
            x=sched.index, y=cum, name=name,
            line=dict(color=ROLLING_PALETTE.get(name, "#7c3aed"), width=2.5),
        ))
    fig.update_layout(
        title="7-day cumulative electricity cost",
        yaxis_title="Cumulative cost (AUD)", hovermode="x unified",
        height=420, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plotly_rolling_soc(schedules: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for name, sched in schedules.items():
        if name == "no_battery":
            continue
        sched = normalise_schedule_columns(sched)
        if sched.empty or "soc_pct" not in sched.columns:
            continue
        fig.add_trace(go.Scatter(
            x=sched.index, y=sched["soc_pct"], name=name,
            line=dict(color=ROLLING_PALETTE.get(name, "#7c3aed"), width=2),
        ))
    fig.add_hline(y=cfg.BATTERY_RESERVE_SOC_PCT, line_dash="dash",
                  line_color="red", opacity=0.5,
                  annotation_text=f"reserve {cfg.BATTERY_RESERVE_SOC_PCT}%")
    fig.update_layout(
        title="7-day battery state of charge",
        yaxis_title="SoC %", yaxis_range=[0, 105],
        height=420, hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plotly_rolling_daily_cost(daily_costs: pd.DataFrame) -> go.Figure:
    pivot = daily_costs.pivot(index="day", columns="policy", values="cost_aud")
    cols_order = [c for c in ["no_battery", "rule_based", "dp_rolling", "rl_q_learning"]
                  if c in pivot.columns]
    fig = go.Figure()
    for col in cols_order:
        fig.add_trace(go.Bar(
            name=col, x=[f"Day {d}" for d in pivot.index], y=pivot[col],
            marker_color=ROLLING_PALETTE.get(col, "#7c3aed"),
            text=[f"${v:.2f}" for v in pivot[col]], textposition="outside",
        ))
    fig.update_layout(
        title="Daily cost by policy (7-day backtest)",
        yaxis_title="AUD / day", barmode="group",
        height=420, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
