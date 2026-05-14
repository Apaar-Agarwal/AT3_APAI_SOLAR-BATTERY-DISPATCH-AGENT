"""
streamlit_app.py — Source-Aware AI Battery Dispatch Agent.

Decision-support prototype for UTS 36121 AT3.

The agent observes external (or cached) solar radiation, weather, tariff,
and load data; builds a structured energy state; reasons about uncertainty
through low/base/high PV scenarios; plans 24 hours of battery dispatch via
dynamic programming; explains every action through a symbolic rule library;
and applies actions only in simulation. It does not control a physical
battery.

Tabs (in order):
  1. Dashboard
  2. Agent Loop
  3. Dispatch Plan
  4. Evaluation & Policy Selection
  5. Uncertainty & Risk
  6. 7-Day Backtest
"""

from __future__ import annotations
import sys
from pathlib import Path

# Make `src` importable when running `streamlit run app/streamlit_app.py` from project root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import config as cfg
from src.agent.rules import BatteryParams, apply_action, evaluate_rules, fired_rules
from src.agent.state_builder import build_state
from src.agent.dispatch_optimizer import optimise_dispatch
from src.agent.uncertainty import make_scenarios
from src.agent.explanations import explain_schedule
from src.data_sources.weather_client import get_weather
from src.data_sources.tariff_client import TariffSchedule
from src.evaluation.backtest import run_backtest
from src.evaluation.policy_selector import selected_policy_row
from src.visualisation.plots import (
    plotly_inputs, plotly_dispatch, plotly_cost_bar, plotly_grid_io,
    plotly_soc, plotly_scenarios, plotly_peak_imports, plotly_robustness,
    plotly_rolling_cumulative_cost, plotly_rolling_soc, plotly_rolling_daily_cost,
    normalise_schedule_columns,
)


# ════════════════════════════════════════════════════════════════════════════
# Page setup
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Source-Aware AI Battery Dispatch Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# Stitch design system — CSS
# ════════════════════════════════════════════════════════════════════════════
STITCH_CSS = """
<style>
  :root {
    --kin-primary:    #1A202C;
    --kin-accent-blue:#3182CE;
    --kin-accent-green:#38A169;
    --kin-accent-orange:#DD6B20;
    --kin-grey:       #718096;
    --kin-grey-soft:  #A0AEC0;
    --kin-surface:    #FFFFFF;
    --kin-canvas:     #F7FAFC;
    --kin-border:     #E2E8F0;
    --kin-text:       #1A202C;
    --kin-text-soft:  #4A5568;
    --kin-shadow:     0 4px 6px -1px rgba(0,0,0,0.05),
                      0 2px 4px -1px rgba(0,0,0,0.03);
  }

  /* Body / canvas */
  html, body, [class*="css"] {
    font-family: 'Public Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Roboto, Helvetica, Arial, sans-serif;
    color: var(--kin-text);
  }
  .stApp { background-color: var(--kin-canvas); }

  /* Cards */
  .kin-card {
    background: var(--kin-surface);
    border: 1px solid var(--kin-border);
    border-radius: 14px;
    padding: 24px 28px;
    box-shadow: var(--kin-shadow);
    margin-bottom: 16px;
  }
  .kin-card-tight {
    background: var(--kin-surface);
    border: 1px solid var(--kin-border);
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: var(--kin-shadow);
    margin-bottom: 12px;
  }

  /* KPI cards */
  .kin-kpi-label {
    font-size: 0.72rem;
    color: var(--kin-grey);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 6px;
  }
  .kin-kpi-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--kin-text);
    line-height: 1.15;
    margin-bottom: 4px;
  }
  .kin-kpi-sub {
    font-size: 0.85rem;
    color: var(--kin-grey);
    margin-top: 2px;
  }

  /* Section headers */
  .kin-section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--kin-text);
    margin-top: 12px;
    margin-bottom: 4px;
  }
  .kin-section-sub {
    font-size: 0.9rem;
    color: var(--kin-text-soft);
    margin-bottom: 12px;
  }

  /* Chips */
  .kin-chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.02em;
    line-height: 1.4;
  }
  .kin-chip-hold       { background: rgba(49,130,206,0.10);  color: #2C5282; }
  .kin-chip-charge     { background: rgba(56,161,105,0.10);  color: #276749; }
  .kin-chip-discharge  { background: rgba(221,107,32,0.12);  color: #9C4221; }
  .kin-chip-peak       { background: rgba(221,107,32,0.10);  color: #9C4221; }
  .kin-chip-shoulder   { background: rgba(214,158,46,0.12);  color: #975A16; }
  .kin-chip-off_peak   { background: rgba(56,161,105,0.10);  color: #276749; }
  .kin-chip-status-ok  { background: rgba(56,161,105,0.10);  color: #276749; }
  .kin-chip-status-warn{ background: rgba(214,158,46,0.14);  color: #744210; }
  .kin-chip-status-info{ background: rgba(49,130,206,0.10);  color: #2C5282; }

  /* Status row inside agent status card */
  .kin-status-row {
    display: flex; flex-wrap: wrap; gap: 18px 32px;
    padding: 4px 0;
  }
  .kin-status-item { font-size: 0.92rem; color: var(--kin-text-soft); }
  .kin-status-item strong { color: var(--kin-text); }

  /* Reasoning card */
  .kin-reason {
    background: var(--kin-surface);
    border: 1px solid var(--kin-border);
    border-left: 4px solid var(--kin-accent-blue);
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: var(--kin-shadow);
    margin-bottom: 14px;
  }
  .kin-reason p { margin: 8px 0 0 0; color: var(--kin-text-soft); }

  /* Step indicator */
  .kin-steps { display:flex; gap:0; align-items:center; flex-wrap: wrap; margin: 6px 0 10px 0; }
  .kin-step {
    display:flex; align-items:center; gap:8px;
    padding: 6px 14px;
    border-radius: 999px;
    background: var(--kin-canvas);
    border: 1px solid var(--kin-border);
    font-size: 0.85rem; font-weight: 600;
    color: var(--kin-text-soft);
  }
  .kin-step-active {
    background: rgba(49,130,206,0.10);
    border-color: rgba(49,130,206,0.30);
    color: #2C5282;
  }
  .kin-step-done {
    background: rgba(56,161,105,0.10);
    border-color: rgba(56,161,105,0.30);
    color: #276749;
  }
  .kin-step-arrow {
    color: var(--kin-grey-soft); font-weight: 700; padding: 0 8px;
  }

  /* Subtle info / interpretation card */
  .kin-info {
    background: var(--kin-canvas);
    border: 1px solid var(--kin-border);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
    color: var(--kin-text-soft);
    font-size: 0.93rem;
  }
  .kin-info strong { color: var(--kin-text); }

  /* Buttons – primary alignment with primary-color */
  .stButton>button[kind="primary"] {
    background-color: var(--kin-primary) !important;
    border-color: var(--kin-primary) !important;
    border-radius: 8px;
    font-weight: 600;
  }
  .stButton>button {
    border-radius: 8px;
    font-weight: 500;
  }

  /* Tabs styling: compact, no clipping under Streamlit header */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 1px solid var(--kin-border);
    min-height: 44px;
    overflow-x: auto;
    overflow-y: hidden;
    flex-wrap: nowrap;
    padding-top: 2px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    color: var(--kin-text-soft);
    font-weight: 500;
    padding: 10px 12px;
    border-radius: 8px 8px 0 0;
    min-height: 40px;
    white-space: nowrap;
    font-size: 0.86rem;
  }
  .stTabs [data-baseweb="tab"] p {
    white-space: nowrap;
    margin: 0;
  }
  .stTabs [aria-selected="true"] {
    background: var(--kin-surface);
    color: var(--kin-text);
    font-weight: 600;
    border-bottom: 2px solid var(--kin-primary);
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: var(--kin-surface);
    border-right: 1px solid var(--kin-border);
  }
  .kin-brand {
    font-size: 1.05rem; font-weight: 700; color: var(--kin-text);
    margin-bottom: 0;
  }
  .kin-brand-sub {
    font-size: 0.82rem; color: var(--kin-grey);
    letter-spacing: 0.01em;
    margin-top: 0; margin-bottom: 14px;
  }

  /* Tables */
  div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* Keep content below Streamlit's fixed header so tab labels are not cut */
  header[data-testid="stHeader"] {
    background: rgba(247, 250, 252, 0.96);
    height: 2.8rem;
  }
  .block-container {
    padding-top: 4.4rem !important;
    padding-bottom: 3rem;
    max-width: 1500px;
  }
</style>
"""
st.markdown(STITCH_CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Render helpers
# ════════════════════════════════════════════════════════════════════════════
def render_kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kin-kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kin-card-tight">'
        f'<div class="kin-kpi-label">{label}</div>'
        f'<div class="kin-kpi-value">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )


def render_kpi_card_chip(label: str, chip_html: str, sub: str = "") -> str:
    sub_html = f'<div class="kin-kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kin-card-tight">'
        f'<div class="kin-kpi-label">{label}</div>'
        f'<div style="margin-top:4px;margin-bottom:4px;">{chip_html}</div>'
        f'{sub_html}'
        f'</div>'
    )


def render_section_header(title: str, sub: str = "") -> None:
    sub_html = f'<div class="kin-section-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kin-section-title">{title}</div>{sub_html}',
        unsafe_allow_html=True,
    )


def action_chip(action_kw: float) -> str:
    if action_kw > 0.05:
        return f'<span class="kin-chip kin-chip-discharge">DISCHARGE {action_kw:.1f} kW</span>'
    if action_kw < -0.05:
        return f'<span class="kin-chip kin-chip-charge">CHARGE {abs(action_kw):.1f} kW</span>'
    return '<span class="kin-chip kin-chip-hold">HOLD</span>'


def period_chip(period: str) -> str:
    label = {"peak": "PEAK", "shoulder": "SHOULDER", "off_peak": "OFF-PEAK"}.get(
        period, period.upper() if period else "—"
    )
    cls_key = period if period in ("peak", "shoulder", "off_peak") else "shoulder"
    return f'<span class="kin-chip kin-chip-{cls_key}">{label}</span>'


def status_chip(text: str, kind: str = "ok") -> str:
    cls = {"ok": "kin-chip-status-ok", "warn": "kin-chip-status-warn",
           "info": "kin-chip-status-info"}.get(kind, "kin-chip-status-ok")
    return f'<span class="kin-chip {cls}">{text}</span>'


POLICY_SHORT_NAMES = {
    "Dynamic Programming agent": "DP Agent",
    "Rule-based controller": "Rule-Based",
    "RL Q-learning agent": "RL Agent",
    "No battery baseline": "No Battery",
    "no_battery": "No Battery",
    "rule_based": "Rule-Based",
    "ai_agent": "DP Agent",
    "rl_q_learning": "RL Agent",
}


def short_policy_name(name: str) -> str:
    """Compact label for KPI cards where long names wrap badly."""
    return POLICY_SHORT_NAMES.get(str(name), str(name))


def render_step_indicator(steps: list[str], current_idx: int) -> str:
    """Horizontal step pill row. current_idx: which step is 'active'.
    Steps before current are 'done', after are pending."""
    parts = []
    for i, s in enumerate(steps):
        if i < current_idx:
            cls = "kin-step kin-step-done"
        elif i == current_idx:
            cls = "kin-step kin-step-active"
        else:
            cls = "kin-step"
        parts.append(f'<div class="{cls}">{s}</div>')
        if i < len(steps) - 1:
            parts.append('<span class="kin-step-arrow">›</span>')
    return f'<div class="kin-steps">{"".join(parts)}</div>'


# ════════════════════════════════════════════════════════════════════════════
# Presets (preserved exactly)
# ════════════════════════════════════════════════════════════════════════════
PRESETS = {
    "Sydney household (default)": {
        "pv_capacity_kw": 6.6,
        "battery_capacity_kwh": 10.0,
        "initial_soc_pct": 50,
        "reserve_soc_pct": 20,
        "tariff": dict(cfg.TARIFF),
        "allow_grid_charging": False,
        "scenario_override": None,
        "description": "Standard 6.6 kW PV / 10 kWh battery, 50% start of charge, default ToU tariff.",
    },
    "Cloudy day stress test": {
        "pv_capacity_kw": 6.6,
        "battery_capacity_kwh": 10.0,
        "initial_soc_pct": 50,
        "reserve_soc_pct": 20,
        "tariff": dict(cfg.TARIFF),
        "allow_grid_charging": False,
        "scenario_override": "low",
        "description": "Same hardware, but PV truth is the low-solar (cloudy) scenario. The agent must protect reserve for the evening peak.",
    },
    "Low battery case": {
        "pv_capacity_kw": 6.6,
        "battery_capacity_kwh": 10.0,
        "initial_soc_pct": 25,
        "reserve_soc_pct": 20,
        "tariff": dict(cfg.TARIFF),
        "allow_grid_charging": True,
        "scenario_override": None,
        "description": "Battery starts near reserve. Off-peak grid charging is enabled — the agent decides whether to use it.",
    },
    "High tariff evening case": {
        "pv_capacity_kw": 6.6,
        "battery_capacity_kwh": 10.0,
        "initial_soc_pct": 50,
        "reserve_soc_pct": 20,
        "tariff": {"off_peak": 0.20, "shoulder": 0.35, "peak": 0.85, "feed_in": 0.06},
        "allow_grid_charging": False,
        "scenario_override": None,
        "description": "Aggressive peak rate (~85 c/kWh). The agent should preserve battery for the 4-9 pm window.",
    },
    "Custom": {
        "description": "Edit values manually in the Advanced settings expander below.",
    },
}


# ════════════════════════════════════════════════════════════════════════════
# Session-state defaults
# ════════════════════════════════════════════════════════════════════════════
def _init_session() -> None:
    defaults = {
        "weather_df": None,
        "weather_meta": None,
        "preset_name": "Sydney household (default)",
        "pv_capacity_kw": cfg.PV_CAPACITY_KW,
        "battery_capacity_kwh": cfg.BATTERY_CAPACITY_KWH,
        "initial_soc_pct": cfg.BATTERY_INITIAL_SOC_PCT,
        "reserve_soc_pct": cfg.BATTERY_RESERVE_SOC_PCT,
        "tariff_rates": dict(cfg.TARIFF),
        "allow_grid_charging": cfg.ALLOW_GRID_CHARGING,
        "custom_load_series": None,
        "report": None,
        "current_time_index": 0,
        "current_soc_kwh": None,
        "simulation_history": [],
        "latest_recommended_action": None,
        "scenario_override": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


def _reset_simulation():
    st.session_state.simulation_history = []
    st.session_state.current_time_index = 0
    st.session_state.current_soc_kwh = None
    st.session_state.latest_recommended_action = None


def apply_preset(name: str) -> None:
    p = PRESETS.get(name)
    if p is None or name == "Custom":
        return
    for key in ("pv_capacity_kw", "battery_capacity_kwh", "initial_soc_pct",
                "reserve_soc_pct", "allow_grid_charging", "scenario_override"):
        if key in p:
            st.session_state[key] = p[key]
    if "tariff" in p:
        st.session_state.tariff_rates = dict(p["tariff"])
    st.session_state.report = None
    _reset_simulation()


# ════════════════════════════════════════════════════════════════════════════
# Sidebar — minimal, preset-driven, advanced settings collapsed
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="kin-brand">Source-Aware AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="kin-brand-sub">Battery Dispatch Agent</div>', unsafe_allow_html=True)

    new_preset = st.selectbox(
        "Scenario preset",
        list(PRESETS.keys()),
        index=list(PRESETS.keys()).index(st.session_state.preset_name),
        help="Each preset configures the household, tariff, and scenario truth.",
    )
    if new_preset != st.session_state.preset_name:
        st.session_state.preset_name = new_preset
        apply_preset(new_preset)
        st.rerun()
    st.caption(PRESETS[st.session_state.preset_name].get("description", ""))

    if st.button("Re-plan with current settings", use_container_width=True, type="primary"):
        st.session_state.report = None
        _reset_simulation()
        st.rerun()

    with st.expander("Advanced household settings", expanded=False):
        st.session_state.pv_capacity_kw = st.number_input(
            "PV size (kW)", 1.0, 20.0,
            float(st.session_state.pv_capacity_kw), 0.1)
        st.session_state.battery_capacity_kwh = st.number_input(
            "Battery capacity (kWh)", 1.0, 30.0,
            float(st.session_state.battery_capacity_kwh), 0.5)
        st.session_state.initial_soc_pct = st.slider(
            "Starting SoC (%)", 0, 100, int(st.session_state.initial_soc_pct))
        st.session_state.reserve_soc_pct = st.slider(
            "Reserve SoC (%)", 0, 50, int(st.session_state.reserve_soc_pct))
        st.session_state.allow_grid_charging = st.toggle(
            "Allow grid charging (off-peak)",
            value=st.session_state.allow_grid_charging,
        )
        st.markdown("**Tariff (AUD/kWh)**")
        rates = st.session_state.tariff_rates
        rates["off_peak"] = st.number_input("off-peak", 0.0, 2.0, float(rates["off_peak"]), 0.01)
        rates["shoulder"] = st.number_input("shoulder", 0.0, 2.0, float(rates["shoulder"]), 0.01)
        rates["peak"]     = st.number_input("peak",     0.0, 2.0, float(rates["peak"]),     0.01)
        rates["feed_in"]  = st.number_input("feed-in (export)", 0.0, 2.0, float(rates["feed_in"]),  0.01)


# ════════════════════════════════════════════════════════════════════════════
# Backend helpers (signatures preserved)
# ════════════════════════════════════════════════════════════════════════════
def _build_params() -> BatteryParams:
    return BatteryParams(
        capacity_kwh=st.session_state.battery_capacity_kwh,
        initial_soc_pct=st.session_state.initial_soc_pct,
        reserve_soc_pct=st.session_state.reserve_soc_pct,
        allow_grid_charging=st.session_state.allow_grid_charging,
    )


def _ensure_state(prefer_cache: bool = True, refresh: bool = False):
    """The agent observes its environment: load weather, build energy state."""
    if st.session_state.weather_df is None or refresh:
        df, meta = get_weather(prefer_cache=prefer_cache, refresh=refresh)
        st.session_state.weather_df = df
        st.session_state.weather_meta = meta

    state = build_state(
        horizon_hours=cfg.HORIZON_HOURS,
        pv_capacity_kw=st.session_state.pv_capacity_kw,
        derate=cfg.PV_DERATE_FACTOR,
        tariff=TariffSchedule(rates=dict(st.session_state.tariff_rates)),
        weather_df=st.session_state.weather_df,
        weather_meta=st.session_state.weather_meta,
        load_series=st.session_state.get("custom_load_series"),
        prefer_cache=prefer_cache,
    )

    override = st.session_state.get("scenario_override")
    if override in ("low", "base", "high"):
        bundle = make_scenarios(state.pv, state.df.get("cloud_cover"))
        state.df = state.df.copy()
        state.df["pv_kw"] = bundle.as_dict()[override]
    return state


def _ensure_report():
    """Ensure the backtest report is computed for the current settings."""
    if st.session_state.report is not None:
        return st.session_state.report

    state = _ensure_state(prefer_cache=True)
    params = _build_params()
    with st.spinner("Agent observing environment, building state, planning 24 h dispatch…"):
        report = run_backtest(state, params)
    st.session_state.report = report
    if st.session_state.current_soc_kwh is None:
        st.session_state.current_soc_kwh = params.initial_soc_kwh()
    return report


def _selected_policy_info(report):
    """Return selected policy metadata from the probability-weighted selector."""
    row = selected_policy_row(getattr(report, "policy_selection", pd.DataFrame()))
    if row is None:
        return {
            "policy": "ai_agent",
            "display_name": "Dynamic Programming agent",
            "expected_cost_aud": float(report.metrics.loc["ai_agent_base", "total_cost_aud"]),
            "risk_adjusted_score_aud": float(report.metrics.loc["ai_agent_base", "total_cost_aud"]),
            "base_cost_aud": float(report.metrics.loc["ai_agent_base", "total_cost_aud"]),
        }
    return row.to_dict()


def _selected_schedule_for_report(report) -> pd.DataFrame:
    """Return the base-scenario schedule for the selected policy."""
    info = _selected_policy_info(report)
    policy = info.get("policy", "ai_agent")
    if policy == "rule_based":
        return report.schedules.get("rule_based", report.explained)
    if policy == "no_battery":
        return report.schedules.get("no_battery", report.explained)
    if policy == "rl_q_learning":
        return report.schedules.get("rl_q_learning_base", report.explained)
    return report.explained


# ════════════════════════════════════════════════════════════════════════════
# Simulator (preserved): apply next action to advance the simulated state
# ════════════════════════════════════════════════════════════════════════════
def apply_next_action_to_simulation(
    schedule: pd.DataFrame,
    current_soc: float,
    battery_config: BatteryParams,
) -> dict:
    """Apply the FIRST row of `schedule` as one hour of dispatch.

    Returns a log row containing the applied action and resulting SoC.
    This is a SIMULATION step, not a real control signal.
    """
    if schedule is None or len(schedule) == 0:
        raise ValueError("Empty schedule; nothing to apply.")
    row = schedule.iloc[0]
    step = apply_action(
        soc_kwh=current_soc,
        action_kw=float(row["action_kw"]),
        pv_kw=float(row["pv_kw"]),
        load_kw=float(row["load_kw"]),
        import_price=float(row["import_price"]),
        export_price=float(row["export_price"]),
        params=battery_config,
    )
    return {
        "timestamp": schedule.index[0],
        "period": str(row.get("period", "")),
        "pv_kw": float(row["pv_kw"]),
        "load_kw": float(row["load_kw"]),
        "action_kw_planned": float(row["action_kw"]),
        "action_kw_applied": float(step["action_applied_kw"]),
        "soc_before_kwh": current_soc,
        "soc_after_kwh": float(step["soc_kwh"]),
        "net_grid_kw": float(step["net_grid_kw"]),
        "cost_aud": float(step["cost_aud"]),
    }


def _replan_from_simulation_state() -> pd.DataFrame:
    """Rebuild the state from current_time_index forward, re-run DP from
    the simulated current_soc_kwh. Returns explained schedule."""
    state = _ensure_state(prefer_cache=True)
    params = _build_params()

    idx = st.session_state.current_time_index
    if idx >= len(state.df):
        return pd.DataFrame()

    sub_df = state.df.iloc[idx:].copy()
    soc = float(st.session_state.current_soc_kwh)
    result = optimise_dispatch(
        sub_df, params=params, initial_soc_kwh=soc, scenario="base",
    )
    return explain_schedule(result.schedule, params=params, scenario="base")


def _live_schedule_for_dashboard(report) -> pd.DataFrame:
    """If simulation has progressed, re-plan from current state; else use
    the original explained schedule."""
    if st.session_state.current_time_index > 0 and st.session_state.simulation_history:
        live = _replan_from_simulation_state()
        if live is None or len(live) == 0:
            return _selected_schedule_for_report(report)
        return live
    return _selected_schedule_for_report(report)


# ════════════════════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════════════════════
TAB_LABELS = [
    "Dashboard",
    "Agent Loop",
    "Dispatch Plan",
    "Evaluation & Policy Selection",
    "Uncertainty & Risk",
    "7-Day Backtest",
]
tabs = st.tabs(TAB_LABELS)


# ───────────────────────────────────────────────────────────────────────────
# 1. Dashboard
# ───────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("## Agent Dashboard")
    st.caption(
        "The agent observes external (or cached) source data, builds a structured "
        "energy state, reasons about uncertainty, plans 24 hours of dispatch via "
        "dynamic programming, and explains every action."
    )

    report = _ensure_report()
    live_schedule = _live_schedule_for_dashboard(report)
    live_schedule = normalise_schedule_columns(live_schedule)

    if live_schedule is not None and not live_schedule.empty:
        current_row = live_schedule.iloc[0]
        action_kw = float(current_row["action_kw"])
        period = str(current_row.get("period", ""))
        soc_pct = float(current_row.get("soc_pct", 0.0))
    else:
        current_row = None
        action_kw, period, soc_pct = 0.0, "", 0.0

    # Note about "current"
    st.markdown(
        '<div class="kin-info">'
        '<strong>Note.</strong> "Current action" refers to the first hour of the '
        'agent\'s current 24-hour plan. Stepping the simulation forward (Agent Loop tab) '
        'advances the simulated state and triggers re-planning.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── KPI row ────────────────────────────────────────────────────────────
    selected_info = _selected_policy_info(report)
    selected_cost = float(selected_info.get("expected_cost_aud", 0.0))
    selected_score = float(selected_info.get("risk_adjusted_score_aud", selected_cost))
    selected_name = str(selected_info.get("display_name", selected_info.get("policy", "Selected policy")))
    selected_short_name = short_policy_name(selected_name)
    nb_rows = report.policy_selection[report.policy_selection["policy"] == "no_battery"] if hasattr(report, "policy_selection") else pd.DataFrame()
    nb_cost = float(nb_rows.iloc[0]["expected_cost_aud"]) if not nb_rows.empty else float(report.metrics.loc["no_battery", "total_cost_aud"])
    saving_aud = nb_cost - selected_cost
    saving_pct = (saving_aud / nb_cost * 100) if abs(nb_cost) > 1e-9 else 0.0
    n_warnings = sum(1 for m in report.risk_messages if m.startswith("⚠"))
    if n_warnings == 0:
        risk_chip = status_chip("All clear", "ok")
        risk_sub = "no scenarios trigger warnings"
    else:
        risk_chip = status_chip(f"{n_warnings} warning{'s' if n_warnings != 1 else ''}", "warn")
        risk_sub = "see Uncertainty & Risk tab"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            render_kpi_card_chip(
                "Recommended action now",
                action_chip(action_kw),
                f"period {period or '—'} · SoC {soc_pct:.0f}%",
            ),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            render_kpi_card(
                "Expected daily cost",
                f"${selected_cost:.2f}",
                f"{selected_short_name} · score ${selected_score:.2f}",
            ),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            render_kpi_card(
                "Saving vs no-battery",
                f"{saving_pct:.0f}%",
                f"${saving_aud:.2f} per day",
            ),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            render_kpi_card_chip("Risk status", risk_chip, risk_sub),
            unsafe_allow_html=True,
        )

    # ── AI Agent Status card ───────────────────────────────────────────────
    meta = st.session_state.weather_meta or {}
    src = meta.get("source", "—")
    src_label = {"live": "live", "cache": "cached", "synthetic": "synthetic fallback"}.get(src, src)
    pv_total = float(report.state.pv.sum())
    load_src = "uploaded CSV" if st.session_state.get("custom_load_series") is not None else "default synthetic"
    peak_rate = float(st.session_state.tariff_rates["peak"])

    status_html = (
        f'<div class="kin-card">'
        f'<div class="kin-section-title">AI Agent Status</div>'
        f'<div class="kin-section-sub">Live snapshot of the agent\'s observations and configuration.</div>'
        f'<div class="kin-status-row">'
        f'  <div class="kin-status-item"><strong>Weather source</strong><br/>{status_chip(src_label, "ok")}</div>'
        f'  <div class="kin-status-item"><strong>PV estimate</strong><br/>{status_chip("ready", "ok")} '
        f'<span style="color:var(--kin-grey);">{pv_total:.1f} kWh / {report.state.horizon} h</span></div>'
        f'  <div class="kin-status-item"><strong>Tariff profile</strong><br/>{status_chip("loaded", "ok")} '
        f'<span style="color:var(--kin-grey);">peak ${peak_rate:.2f}/kWh</span></div>'
        f'  <div class="kin-status-item"><strong>Load profile</strong><br/>{status_chip(load_src, "ok")}</div>'
        f'  <div class="kin-status-item"><strong>Selected policy</strong><br/>{status_chip(selected_short_name, "info")}</div>'
        f'  <div class="kin-status-item"><strong>Horizon</strong><br/>'
        f'<span style="color:var(--kin-text);">Next {report.state.horizon} hours</span></div>'
        f'</div></div>'
    )
    st.markdown(status_html, unsafe_allow_html=True)

    # ── Why this action? ───────────────────────────────────────────────────
    render_section_header("Why this action?",
                          "The agent's reasoning behind the current recommendation.")

    if current_row is not None:
        params = _build_params()
        evals = evaluate_rules(
            soc_kwh=float(current_row["soc_kwh"]),
            pv_kw=float(current_row["pv_kw"]),
            load_kw=float(current_row["load_kw"]),
            period=period, scenario="base", params=params,
        )
        fired = fired_rules(evals)
        primary_explanation = current_row.get("explanation", "—")

        st.markdown(
            f'<div class="kin-reason">'
            f'{action_chip(action_kw)}'
            f'<p>{primary_explanation}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Always-shown reasoning checklist.
        # Symbolic rules explain local conditions; the selected policy/planner chooses the final action.
        rule_lookup = {e.rule_id: e for e in evals} if evals else {}
        canonical_rules = [
            ("reserve_floor", "Reserve rule",
             "protect SoC above the reserve floor for evening peak"),
            ("tariff_peak_discharge", "Tariff rule",
             "prefer discharging during peak; charge during off-peak when allowed"),
            ("solar_surplus_charge", "Solar rule",
             "absorb surplus PV into the battery rather than exporting at low feed-in"),
            ("battery_capacity", "Battery constraint",
             "respect rated charge / discharge power and SoC bounds"),
        ]
        st.markdown("**Symbolic reasoning layer**")
        st.caption(
            "These rules represent domain knowledge used to explain the current recommendation. "
            "The final dispatch policy is selected using risk-adjusted expected cost."
        )
        for rid, name, desc in canonical_rules:
            ev = rule_lookup.get(rid)
            if ev is not None and getattr(ev, "fired", False):
                badge = status_chip("active", "ok")
                detail = getattr(ev, "rationale", desc)
            else:
                badge = status_chip("not driving this hour", "info")
                detail = desc
            st.markdown(
                f"- **{name}** — {detail} &nbsp; {badge}",
                unsafe_allow_html=True,
            )
        st.markdown(
            "- **Policy selection layer** — compares rule-based, Dynamic Programming and RL policies "
            "under probability-weighted solar scenarios &nbsp; "
            f"{status_chip('active selector', 'ok')}",
            unsafe_allow_html=True,
        )

        if fired:
            with st.expander("All symbolic rules currently firing"):
                for r in fired:
                    st.markdown(f"- `{r.rule_id}` — {r.rationale}")
    else:
        st.info("No active recommendation. Reset the simulation to plan from hour 0.")

    # ── 24-hour energy outlook ─────────────────────────────────────────────
    render_section_header(
        "24-hour energy outlook",
        "Solar generation estimate, household load, and tariff periods over the planning horizon.",
    )
    st.plotly_chart(plotly_inputs(report.state.df), use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
# 2. Agent Loop
# ───────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("## Agent Loop")
    st.caption(
        "The cycle the agent runs each step: observe data, build state, plan via "
        "dynamic programming, act in simulation, then re-plan from the new state."
    )

    report = _ensure_report()
    sim_schedule = _live_schedule_for_dashboard(report)
    sim_schedule = normalise_schedule_columns(sim_schedule)

    # Horizontal step indicator
    steps = ["Observe", "Build state", "Plan", "Act in simulation", "Re-plan"]
    if st.session_state.current_time_index == 0:
        active_idx = 2  # ready to act
    elif sim_schedule is None or sim_schedule.empty:
        active_idx = 4
    else:
        active_idx = 4  # post-act, ready to re-plan / act again
    st.markdown(render_step_indicator(steps, active_idx), unsafe_allow_html=True)

    st.markdown(
        '<div class="kin-info">'
        '<strong>Simulation only.</strong> "Apply next action" advances the simulated '
        'battery state and re-runs the planner. The agent does not control any real '
        'inverter or battery system.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Two-column layout: control + current conditions
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown('<div class="kin-card">', unsafe_allow_html=True)
        st.markdown('<div class="kin-section-title">Simulation control</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Apply the agent's next action to the simulated battery, advance "
            "time by one hour, then re-plan from the new state."
        )
        b1, b2 = st.columns(2)
        with b1:
            apply_disabled = sim_schedule is None or sim_schedule.empty
            if st.button(
                "Apply next action",
                type="primary",
                use_container_width=True,
                disabled=apply_disabled,
                key="agent_loop_apply",
            ):
                params = _build_params()
                log_row = apply_next_action_to_simulation(
                    schedule=sim_schedule,
                    current_soc=float(st.session_state.current_soc_kwh),
                    battery_config=params,
                )
                st.session_state.simulation_history.append(log_row)
                st.session_state.current_soc_kwh = log_row["soc_after_kwh"]
                st.session_state.current_time_index += 1
                st.session_state.latest_recommended_action = log_row
                st.rerun()
        with b2:
            if st.button("Reset simulation", use_container_width=True,
                         key="agent_loop_reset"):
                _reset_simulation()
                st.session_state.current_soc_kwh = _build_params().initial_soc_kwh()
                st.rerun()

        m1, m2 = st.columns(2)
        cap = max(1e-9, _build_params().capacity_kwh)
        soc_pct_live = (st.session_state.current_soc_kwh or 0) / cap * 100
        m1.metric("Sim hours elapsed", st.session_state.current_time_index)
        m2.metric("Current simulated SoC", f"{soc_pct_live:.0f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="kin-card">', unsafe_allow_html=True)
        st.markdown('<div class="kin-section-title">Current conditions</div>',
                    unsafe_allow_html=True)
        if sim_schedule is not None and not sim_schedule.empty:
            row = sim_schedule.iloc[0]
            params = _build_params()
            period_now = str(row.get("period", ""))
            tariff_rate = float(row.get("import_price", 0.0))
            pv_now = float(row["pv_kw"])
            load_now = float(row["load_kw"])
            soc_kwh_now = float(row.get("soc_kwh", 0.0))
            soc_pct_now = float(row.get("soc_pct", 0.0))
            st.markdown(
                f'<div class="kin-status-row">'
                f'<div class="kin-status-item"><strong>Period</strong><br/>{period_chip(period_now)}</div>'
                f'<div class="kin-status-item"><strong>Import tariff</strong><br/>${tariff_rate:.2f}/kWh</div>'
                f'<div class="kin-status-item"><strong>PV (kW)</strong><br/>{pv_now:.2f}</div>'
                f'<div class="kin-status-item"><strong>Load (kW)</strong><br/>{load_now:.2f}</div>'
                f'<div class="kin-status-item"><strong>SoC</strong><br/>{soc_pct_now:.0f}% ({soc_kwh_now:.2f} kWh)</div>'
                f'<div class="kin-status-item"><strong>Battery capacity</strong><br/>{params.capacity_kwh:.1f} kWh</div>'
                f'<div class="kin-status-item"><strong>Reserve floor</strong><br/>{params.reserve_soc_pct}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Planning horizon consumed. Reset the simulation to start over.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Reasoning snapshot
    if sim_schedule is not None and not sim_schedule.empty:
        row = sim_schedule.iloc[0]
        action_kw_now = float(row["action_kw"])
        explanation_now = row.get("explanation", "—")
        st.markdown(
            f'<div class="kin-reason">'
            f'<div class="kin-section-title" style="margin-top:0;">Reasoning snapshot</div>'
            f'{action_chip(action_kw_now)}'
            f'<p>{explanation_now}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Simulation timeline + log
    if st.session_state.simulation_history:
        render_section_header("Simulation timeline",
                              "Hourly record of applied actions and resulting state.")
        sim_df = pd.DataFrame(st.session_state.simulation_history).copy()
        sim_df["soc_after_pct"] = (
            sim_df["soc_after_kwh"] / max(1e-9, _build_params().capacity_kwh) * 100
        ).round(1)
        display_df = sim_df[
            ["timestamp", "period", "pv_kw", "load_kw",
             "action_kw_applied", "soc_after_pct", "cost_aud"]
        ].copy()
        display_df.columns = [
            "time", "period", "PV kW", "load kW",
            "action kW", "SoC %", "cost AUD",
        ]
        st.dataframe(display_df.round(2), use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────────────────────
# 3. Dispatch Plan
# ───────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("## Dispatch Plan")
    st.caption(
        "The full 24-hour plan, with action chips and symbolic rule explanations "
        "for every hour. Action is signed kW: positive = discharge, negative = charge."
    )

    report = _ensure_report()

    if st.session_state.current_time_index > 0 and st.session_state.simulation_history:
        plan = _replan_from_simulation_state()
        st.markdown(
            f'<div class="kin-info">'
            f'Showing live re-plan from simulation hour '
            f'{st.session_state.current_time_index}, '
            f'SoC {float(st.session_state.current_soc_kwh):.2f} kWh.'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        plan = report.explained.copy()

    plan = normalise_schedule_columns(plan)

    if plan is None or plan.empty:
        st.warning(
            "No dispatch plan available — the simulation has consumed the full "
            "planning horizon. Click **Reset simulation** in the Agent Loop tab "
            "to plan again from hour 0."
        )
    else:
        # KPI cards
        first = plan.iloc[0]
        action_now_kw = float(first["action_kw"])
        peak_mask = plan["period"] == "peak"
        peak_action_kw = plan.loc[peak_mask, "action_kw"].sum() if peak_mask.any() else 0.0
        if peak_mask.any():
            peak_strategy = ("Discharging across peak"
                             if peak_action_kw > 0.5 else
                             "Holding through peak"
                             if abs(peak_action_kw) <= 0.5 else
                             "Charging through peak")
        else:
            peak_strategy = "No peak hours in horizon"
        plan_grid_import = float(plan.get("grid_import_kwh", pd.Series([0])).sum())
        plan_final_soc = float(plan.iloc[-1]["soc_pct"])

        kp1, kp2, kp3, kp4 = st.columns(4)
        with kp1:
            st.markdown(
                render_kpi_card_chip("Current action", action_chip(action_now_kw),
                                     f"hour 1 of plan · period {first.get('period','—')}"),
                unsafe_allow_html=True,
            )
        with kp2:
            st.markdown(
                render_kpi_card("Peak-period strategy", peak_strategy,
                                f"net action across peak: {peak_action_kw:+.1f} kWh"),
                unsafe_allow_html=True,
            )
        with kp3:
            st.markdown(
                render_kpi_card("Expected grid import",
                                f"{plan_grid_import:.1f} kWh",
                                "across the 24-hour plan"),
                unsafe_allow_html=True,
            )
        with kp4:
            st.markdown(
                render_kpi_card("Final SoC", f"{plan_final_soc:.0f}%",
                                "battery state at end of horizon"),
                unsafe_allow_html=True,
            )

        # Action summary timeline (5 phases of the day)
        def _phase_action(start_h: int, end_h: int) -> str:
            mask = (plan.index.hour >= start_h) & (plan.index.hour < end_h)
            sub = plan.loc[mask]
            if sub.empty:
                return "—"
            net = sub["action_kw"].sum()
            if net > 0.5:
                return f"Discharge ({net:+.1f} kWh net)"
            if net < -0.5:
                return f"Charge ({net:+.1f} kWh net)"
            return "Hold"

        render_section_header("Action summary by phase",
                              "Net action across each part of the day.")
        phases = [
            ("Overnight (00-06)", _phase_action(0, 6)),
            ("Morning (06-10)", _phase_action(6, 10)),
            ("Midday (10-16)", _phase_action(10, 16)),
            ("Evening peak (16-21)", _phase_action(16, 21)),
            ("Late (21-24)", _phase_action(21, 24)),
        ]
        phase_df = pd.DataFrame(phases, columns=["Phase", "Net action"])
        st.dataframe(phase_df, use_container_width=True, hide_index=True)

        # Dispatch chart
        render_section_header("Dispatch chart",
                              "PV/load (top), action bars (middle), SoC (bottom).")
        st.plotly_chart(
            plotly_dispatch(plan, "AI dispatch — base scenario"),
            use_container_width=True,
        )

        # Full hourly table
        with st.expander("View full hourly plan"):
            rows = []
            for ts, row in plan.iterrows():
                rows.append({
                    "Time": ts.strftime("%a %H:%M") if hasattr(ts, "strftime") else str(ts),
                    "Action": action_chip(float(row["action_kw"])),
                    "kW": f"{float(row['action_kw']):+.1f}",
                    "SoC %": f"{float(row['soc_pct']):.0f}%",
                    "Period": period_chip(str(row.get("period", ""))),
                    "PV / load (kW)": f"{float(row['pv_kw']):.1f} / {float(row['load_kw']):.1f}",
                    "Cost (AUD)": f"{float(row['cost_aud']):+.2f}",
                    "Reason": row.get("explanation", "—"),
                })
            df_html = pd.DataFrame(rows).to_html(escape=False, index=False)
            st.markdown(
                f'<div style="overflow-x:auto;max-height:600px;overflow-y:auto;">'
                f'{df_html}</div>',
                unsafe_allow_html=True,
            )

        st.download_button(
            "Download dispatch CSV",
            data=plan.to_csv().encode(),
            file_name="ai_dispatch_schedule.csv",
            mime="text/csv",
        )


# ───────────────────────────────────────────────────────────────────────────
# 4. Evaluation & Policy Selection
# ───────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## Evaluation & Policy Selection")
    st.caption(
        "The system compares candidate dispatch policies under low/base/high solar "
        "scenarios and selects the lowest risk-adjusted expected-cost policy."
    )

    report = _ensure_report()
    m = report.metrics_with_savings.copy()
    selection = getattr(report, "policy_selection", pd.DataFrame()).copy()

    st.markdown(
        '<div class="kin-info">'
        '<strong>Selection logic.</strong> Each candidate policy is evaluated under '
        'low, base and high solar scenarios. Scenario probabilities are used to compute '
        'expected cost. A downside-risk penalty is then added, and the selected policy is '
        'the lowest risk-adjusted expected-cost policy among controllable battery policies. '
        'The no-battery case remains a benchmark, not the preferred controller.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not selection.empty:
        selected = selected_policy_row(selection)
        k1, k2, k3, k4 = st.columns(4)
        if selected is not None:
            with k1:
                st.markdown(
                    render_kpi_card("Selected policy", short_policy_name(str(selected["display_name"])),
                                    "lowest risk-adjusted expected cost"),
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    render_kpi_card("Expected cost", f"${float(selected['expected_cost_aud']):.2f}",
                                    "probability-weighted scenarios"),
                    unsafe_allow_html=True,
                )
            with k3:
                st.markdown(
                    render_kpi_card("Worst-case cost", f"${float(selected['worst_case_cost_aud']):.2f}",
                                    "highest cost among low/base/high"),
                    unsafe_allow_html=True,
                )
            with k4:
                st.markdown(
                    render_kpi_card("Risk-adjusted score", f"${float(selected['risk_adjusted_score_aud']):.2f}",
                                    "expected cost + downside risk"),
                    unsafe_allow_html=True,
                )

        render_section_header(
            "Risk-adjusted expected cost comparison",
            "This is the actual policy selector table. Lower score is better."
        )
        display = selection.copy()
        display["selected"] = display["selected"].map(lambda x: "Yes" if bool(x) else "No")
        display["role"] = display["selectable"].map(lambda x: "Candidate" if bool(x) else "Benchmark")

        compact_cols = [
            "display_name", "ai_paradigm", "role", "base_cost_aud",
            "expected_cost_aud", "worst_case_cost_aud",
            "risk_adjusted_score_aud", "selected",
        ]
        compact_cols = [c for c in compact_cols if c in display.columns]
        compact = display[compact_cols].copy()
        compact = compact.rename(columns={
            "display_name": "Policy",
            "ai_paradigm": "AI paradigm",
            "role": "Role",
            "base_cost_aud": "Base cost ($)",
            "expected_cost_aud": "Expected cost ($)",
            "worst_case_cost_aud": "Worst-case ($)",
            "risk_adjusted_score_aud": "Risk-adjusted score ($)",
            "selected": "Selected",
        })
        st.dataframe(compact.round(3), use_container_width=True, hide_index=True)

        with st.expander("Detailed risk diagnostics"):
            detail_cols = [
                "display_name", "best_case_cost_aud", "downside_risk_aud",
                "high_cost_probability_pct", "expected_peak_import_kwh",
                "constraint_violations", "selection_reason",
            ]
            detail_cols = [c for c in detail_cols if c in display.columns]
            detail = display[detail_cols].copy().rename(columns={
                "display_name": "Policy",
                "best_case_cost_aud": "Best-case ($)",
                "downside_risk_aud": "Downside risk ($)",
                "high_cost_probability_pct": "High-cost probability (%)",
                "expected_peak_import_kwh": "Expected peak import (kWh)",
                "constraint_violations": "Constraint violations",
                "selection_reason": "Selection reason",
            })
            st.dataframe(detail.round(3), use_container_width=True, hide_index=True)

    show_cols = [
        "total_cost_aud", "total_grid_import_kwh", "total_grid_export_kwh",
        "peak_period_import_kwh", "self_consumption_pct",
        "battery_throughput_kwh", "battery_cycles_equivalent", "runtime_s",
    ]
    show_cols = [c for c in show_cols if c in m.columns]
    headline_rows = ["no_battery", "rule_based", "ai_agent_base", "rl_q_learning_base"]
    headline_rows = [r for r in headline_rows if r in m.index]

    if all(r in m.index for r in ["no_battery", "rule_based", "ai_agent_base"]):
        ai_cost = float(m.loc["ai_agent_base", "total_cost_aud"])
        rb_cost = float(m.loc["rule_based", "total_cost_aud"])
        nb_cost = float(m.loc["no_battery", "total_cost_aud"])
        ai_peak = float(m.loc["ai_agent_base", "peak_period_import_kwh"])
        rb_peak = float(m.loc["rule_based", "peak_period_import_kwh"])

        interp_lines = []
        interp_lines.append(
            f"<strong>Dynamic Programming vs no-battery:</strong> ${nb_cost - ai_cost:.2f} cheaper "
            f"({(nb_cost - ai_cost) / max(abs(nb_cost), 1e-9) * 100:.0f}% saving over 24 h). "
            f"This remains the clearest base-case cost reduction."
        )
        if abs(ai_cost - rb_cost) < 0.10:
            nuance = (
                f"<strong>Dynamic Programming vs rule-based:</strong> essentially tied "
                f"(${ai_cost:.2f} vs ${rb_cost:.2f}). On a simple sunny day, a rule can capture "
                f"much of the benefit. The selector therefore uses expected cost and downside risk, "
                f"not only the base-case cost."
            )
        elif ai_cost < rb_cost:
            nuance = (
                f"<strong>Dynamic Programming vs rule-based:</strong> DP saves an additional "
                f"${rb_cost - ai_cost:.2f} ({(rb_cost - ai_cost) / max(abs(rb_cost), 1e-9) * 100:.1f}%) "
                f"in the base scenario. Longer horizons are reported separately rather than assumed."
            )
        else:
            nuance = (
                f"<strong>Dynamic Programming vs rule-based:</strong> the simple rule slightly wins on this "
                f"single day (${rb_cost:.2f} vs ${ai_cost:.2f}). This is why policy selection is based "
                f"on scenario-weighted expected cost and risk, not marketing optimism."
            )
        interp_lines.append(nuance)
        interp_lines.append(
            "<strong>Why this matters:</strong> the project now demonstrates evaluation-driven AI: "
            "candidate policies are compared empirically before the final recommendation is shown."
        )
        st.markdown(
            '<div class="kin-info">' + "<br/><br/>".join(interp_lines) + "</div>",
            unsafe_allow_html=True,
        )

    render_section_header("Base-case policy metrics (24-hour horizon)",
                          "Base solar case only: no-battery, rule-based, DP and RL policies.")
    st.dataframe(m.loc[headline_rows, show_cols].round(2), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plotly_cost_bar(m.loc[headline_rows]), use_container_width=True)
    with c2:
        st.plotly_chart(plotly_grid_io(m.loc[headline_rows]), use_container_width=True)

    render_section_header("Battery state of charge across all policies", "")
    st.plotly_chart(plotly_soc(report.schedules), use_container_width=True)

    render_section_header("Where tariff awareness shows: peak-period imports", "")
    st.plotly_chart(plotly_peak_imports(m.loc[headline_rows]),
                    use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
# 5. Uncertainty & Risk
# ───────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("## Uncertainty & Risk")
    st.caption(
        "The agent reasons about solar uncertainty by assigning probabilities to "
        "low/base/high solar scenarios, then calculating expected cost and downside risk."
    )

    report = _ensure_report()
    totals = report.scenarios.total_energy()
    probs = getattr(report, "scenario_probabilities", {"low": 0.25, "base": 0.55, "high": 0.20})

    prob_df = pd.DataFrame([
        {"Scenario": "low", "Probability": probs.get("low", 0.0), "PV total kWh": totals.get("low", 0.0)},
        {"Scenario": "base", "Probability": probs.get("base", 0.0), "PV total kWh": totals.get("base", 0.0)},
        {"Scenario": "high", "Probability": probs.get("high", 0.0), "PV total kWh": totals.get("high", 0.0)},
    ])
    prob_df["Probability %"] = prob_df["Probability"] * 100

    render_section_header("Scenario probabilities",
                          "Probabilities are updated from cloud-cover evidence using a transparent Bayesian-style rule.")
    st.dataframe(prob_df[["Scenario", "Probability %", "PV total kWh"]].round(2),
                 use_container_width=True, hide_index=True)

    sc_cols = st.columns(3)
    with sc_cols[0]:
        cost_low = float(report.metrics.loc["ai_agent_low", "total_cost_aud"])
        st.markdown(
            f'<div class="kin-card-tight">'
            f'<div class="kin-kpi-label">Low solar scenario</div>'
            f'<div class="kin-kpi-value">{totals["low"]:.1f} kWh</div>'
            f'<div class="kin-kpi-sub">PV total · AI agent cost ${cost_low:.2f}</div>'
            f'<div style="margin-top:8px;">{status_chip("higher risk", "warn")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with sc_cols[1]:
        cost_base = float(report.metrics.loc["ai_agent_base", "total_cost_aud"])
        st.markdown(
            f'<div class="kin-card-tight">'
            f'<div class="kin-kpi-label">Base estimate</div>'
            f'<div class="kin-kpi-value">{totals["base"]:.1f} kWh</div>'
            f'<div class="kin-kpi-sub">PV total · AI agent cost ${cost_base:.2f}</div>'
            f'<div style="margin-top:8px;">{status_chip("nominal", "info")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with sc_cols[2]:
        cost_high = float(report.metrics.loc["ai_agent_high", "total_cost_aud"])
        st.markdown(
            f'<div class="kin-card-tight">'
            f'<div class="kin-kpi-label">High solar scenario</div>'
            f'<div class="kin-kpi-value">{totals["high"]:.1f} kWh</div>'
            f'<div class="kin-kpi-sub">PV total · AI agent cost ${cost_high:.2f}</div>'
            f'<div style="margin-top:8px;">{status_chip("lower risk", "ok")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    render_section_header("Scenario comparison",
                          "PV trajectories used by the planner under each scenario.")
    st.plotly_chart(plotly_scenarios(report.scenarios), use_container_width=True)

    render_section_header("Robustness — every policy under every scenario",
                          "Low-solar scenarios show why planning matters: battery capacity "
                          "is scarce and how it is allocated has the largest effect on cost.")
    st.plotly_chart(plotly_robustness(report.robustness), use_container_width=True)
    with st.expander("Robustness numbers"):
        st.dataframe(
            report.robustness.round(3),
            use_container_width=True, hide_index=True,
        )

    render_section_header("Risk warnings", "")
    if not report.risk_messages:
        st.success("No risk warnings active.")
    else:
        for msg in report.risk_messages:
            clean = msg.lstrip("⚠✅ ").strip()
            if msg.startswith("⚠"):
                st.warning(clean)
            elif msg.startswith("✅"):
                st.success(clean)
            else:
                st.info(clean)


# ───────────────────────────────────────────────────────────────────────────
# 6. 7-Day Backtest
# ───────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("## 7-Day Backtest")
    st.caption(
        "Rolling-horizon evaluation across 168 hours. The tab now compares the "
        "updated policy set: no-battery baseline, rule-based controller, DP rolling "
        "planner, and RL Q-learning controller."
    )

    out_dir = ROOT / "data" / "outputs"
    sched_files = {
        "no_battery":    out_dir / "rolling_7day_schedule_no_battery.csv",
        "rule_based":    out_dir / "rolling_7day_schedule_rule_based.csv",
        "dp_rolling":    out_dir / "rolling_7day_schedule_dp_rolling.csv",
        "rl_q_learning": out_dir / "rolling_7day_schedule_rl_q_learning.csv",
    }
    metrics_file = out_dir / "rolling_7day_metrics.csv"
    daily_file   = out_dir / "rolling_7day_daily_costs.csv"

    if not metrics_file.exists() or not all(p.exists() for p in sched_files.values()):
        st.markdown(
            '<div class="kin-info">'
            'No 7-day backtest outputs were found in <code>data/outputs/</code>. '
            'Generate them by running the CLI from the project root:'
            '<br/><br/>'
            '<code>python main.py --offline --rolling-backtest</code>'
            '<br/><br/>'
            'This produces <code>rolling_7day_metrics.csv</code>, '
            '<code>rolling_7day_daily_costs.csv</code>, and per-policy schedule CSVs '
            'for <code>no_battery</code>, <code>rule_based</code>, '
            '<code>dp_rolling</code>, and <code>rl_q_learning</code>.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # Load
        schedules: dict[str, pd.DataFrame] = {}
        for name, path in sched_files.items():
            try:
                df = pd.read_csv(path, parse_dates=["timestamp"])
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp", drop=False)
                schedules[name] = normalise_schedule_columns(df)
            except Exception as e:
                st.warning(f"Could not read {path.name}: {e}")
        try:
            metrics_df = pd.read_csv(metrics_file, index_col=0)
        except Exception as e:
            st.error(f"Could not read 7-day metrics: {e}")
            metrics_df = pd.DataFrame()
        try:
            daily_df = pd.read_csv(daily_file)
        except Exception:
            daily_df = pd.DataFrame()

        st.markdown(
            '<div class="kin-info">'
            'Across 7 consecutive days, the app compares the updated candidate policies on the same '
            'solar, load and tariff sequence. This checks whether the 24-hour policy-selection result '
            'generalises over time, rather than assuming one good day proves robustness.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Headline KPIs
        if not metrics_df.empty:
            cost_col = "total_cost_aud" if "total_cost_aud" in metrics_df.columns else metrics_df.columns[0]
            nb_total = float(metrics_df.loc["no_battery", cost_col]) if "no_battery" in metrics_df.index else float("nan")
            rb_total = float(metrics_df.loc["rule_based", cost_col]) if "rule_based" in metrics_df.index else float("nan")

            controllable_order = ["rule_based", "dp_rolling", "rl_q_learning"]
            controllable = [p for p in controllable_order if p in metrics_df.index]
            best_policy = None
            best_total = float("nan")
            if controllable:
                best_policy = metrics_df.loc[controllable, cost_col].astype(float).idxmin()
                best_total = float(metrics_df.loc[best_policy, cost_col])

            short_names = {
                "no_battery": "No Battery",
                "rule_based": "Rule-Based",
                "dp_rolling": "DP Rolling",
                "rl_q_learning": "RL Q-learning",
            }

            kk1, kk2, kk3 = st.columns(3)
            with kk1:
                if best_policy is not None and pd.notna(best_total):
                    st.markdown(
                        render_kpi_card(
                            "Best 7-day policy",
                            f"{short_names.get(best_policy, best_policy)} — ${best_total:.2f}",
                            "lowest total cost over 168-hour backtest",
                        ),
                        unsafe_allow_html=True,
                    )
            with kk2:
                if pd.notna(nb_total) and pd.notna(best_total) and abs(nb_total) > 1e-9:
                    save_pct = (nb_total - best_total) / nb_total * 100
                    st.markdown(
                        render_kpi_card(
                            "Saving vs no-battery (7 d)",
                            f"{save_pct:.0f}%",
                            f"${nb_total - best_total:.2f} over the week",
                        ),
                        unsafe_allow_html=True,
                    )
            with kk3:
                if best_policy is not None and pd.notna(rb_total) and pd.notna(best_total) and abs(rb_total) > 1e-9:
                    diff = rb_total - best_total
                    edge_pct = diff / rb_total * 100
                    if best_policy == "rule_based":
                        value = "Rule selected"
                        note = "rule-based has lowest 7-day cost"
                    else:
                        value = f"{abs(edge_pct):.1f}% better" if diff >= 0 else f"{abs(edge_pct):.1f}% higher cost"
                        note = f"{short_names.get(best_policy, best_policy)} ${abs(diff):.2f} cheaper" if diff >= 0 else f"{short_names.get(best_policy, best_policy)} ${abs(diff):.2f} more expensive"
                    st.markdown(
                        render_kpi_card("Best vs rule-based (7 d)", value, note),
                        unsafe_allow_html=True,
                    )

            render_section_header("7-day metrics by policy", "")
            display_metrics = metrics_df.copy()
            rename_index = {
                "no_battery": "No Battery",
                "rule_based": "Rule-Based",
                "dp_rolling": "DP Rolling",
                "rl_q_learning": "RL Q-learning",
            }
            display_metrics.index = [rename_index.get(i, i) for i in display_metrics.index]
            st.dataframe(display_metrics.round(2), use_container_width=True)

        if schedules:
            render_section_header("Cumulative cost over 7 days", "")
            st.plotly_chart(plotly_rolling_cumulative_cost(schedules),
                            use_container_width=True)

            render_section_header("Battery state of charge over 7 days", "")
            st.plotly_chart(plotly_rolling_soc(schedules),
                            use_container_width=True)

        if not daily_df.empty and {"day", "policy", "cost_aud"}.issubset(daily_df.columns):
            render_section_header("Daily cost by policy", "")
            st.plotly_chart(plotly_rolling_daily_cost(daily_df),
                            use_container_width=True)
