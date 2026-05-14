# Source-Aware AI Agent for Solar Battery Dispatch Optimisation

**36121 — AI Principles and Applications · Assessment 3**

## 1. Project objective

This project develops a *source-aware AI battery dispatch agent* for an
Australian household with rooftop solar and a home battery. The agent
retrieves external weather and solar radiation forecasts from a public source,
converts these inputs into a structured 24-hour energy state, reasons about
solar uncertainty through low / base / high scenarios, and uses a search-based
dynamic programming optimiser to recommend an hourly battery dispatch
schedule. The system is evaluated against two baselines — *no battery* and a
*simple rule-based controller* — using electricity cost, grid import / export,
self-consumption, battery cycling, and runtime under uncertain weather.

The core academic claim is **decision quality, not forecast accuracy**.

## 2. Why "source-aware"

The agent does **not** train a machine-learning solar forecasting model.
Solar generation is estimated transparently from externally-sourced shortwave
radiation:

```
pv_kw = pv_capacity × (radiation_wm² / 1000) × derate_factor
```

This is auditable and parameterised by site characteristics, not learned from
historical data. The AI contribution is downstream: state representation,
scenario uncertainty reasoning, and search-based dispatch optimisation.

## 3. AI paradigms demonstrated

| Paradigm | Concrete role |
|---|---|
| **Intelligent agent / PEAS** | Perceives weather + tariff + battery state; acts via charge/discharge/hold |
| **Logic / structural knowledge** | Battery dynamics, action feasibility, symbolic rule library (`R1`–`R6`) |
| **Probabilistic reasoning** | Low (×0.75) / base / high (×1.15) PV scenarios, cloud-aware low-scenario adjustment, risk warnings |
| **Search-based optimisation** | Dynamic programming over (hour, SoC bin), 13 discrete actions, exhaustive backward induction |

LLMs are explicitly *not* used as the AI core; the agent's reasoning is
grounded in optimisation and symbolic logic, not statistical language models.

## 4. Data sources

| Source | Used for | Cached |
|---|---|---|
| Open-Meteo public forecast API | shortwave_radiation, cloud_cover, temperature | `data/cache/open_meteo_sydney_24h.json` |
| Editable static AU ToU tariff | import / export prices | in-memory (sidebar editable) |
| Synthetic residential load profile | household demand | regenerated each run |

The app falls back to cached → synthetic data on any API failure, so it always
runs offline.

## 5. Setup

```bash
pip install -r requirements.txt
```

## 6. Running the app

```bash
streamlit run app/streamlit_app.py
```

The dashboard exposes 8 tabs: Overview, Inputs & Sources, Energy State,
Dispatch, Comparison, Uncertainty, Multi-day Backtest, Limitations.

## 7. Running the CLI pipeline

```bash
python main.py                  # use cache, fall back to synthetic
python main.py --refresh        # force a live Open-Meteo call
python main.py --offline        # never hit the API
python main.py --grid-charging  # allow off-peak grid charging
python main.py --rolling-backtest   # also run 7-day rolling-horizon backtest
python main.py --offline --rolling-backtest   # full offline run with backtest
```

The pipeline runs in roughly 2 seconds for single-day; the rolling backtest
adds ~20 seconds. All outputs are written to `data/outputs/` and
`reports/figures/`.

## 8. Outputs produced

| Path | Description |
|---|---|
| `data/outputs/schedule_*.csv` | 24-hour schedule for each policy and scenario |
| `data/outputs/schedule_ai_agent_base_explained.csv` | AI schedule with rule-based explanations per hour |
| `data/outputs/metrics_comparison.csv` | All policies × all metrics |
| `data/outputs/metrics_with_savings.csv` | Same, plus % savings vs no-battery baseline |
| `data/outputs/metrics_robustness.csv` | Cross-scenario robustness table |
| `data/outputs/risk_messages.json` | Generated risk warnings |
| `data/outputs/energy_state_24h.csv` | The state the agent reasoned over |
| `data/outputs/rolling_7day_schedule_*.csv` | 7-day schedule for each policy |
| `data/outputs/rolling_7day_metrics.csv` | 7-day metrics comparison |
| `data/outputs/rolling_7day_daily_costs.csv` | Per-day cost breakdown |
| `data/outputs/rolling_7day_energy_state.csv` | 7-day energy state |
| `reports/figures/01_inputs.png` | PV / load / tariff overlay |
| `reports/figures/02_cost_comparison.png` | Cost by policy bar chart |
| `reports/figures/03_grid_io.png` | Grid import vs export by policy |
| `reports/figures/04_soc_curve.png` | Battery SoC trajectories |
| `reports/figures/05_scenarios.png` | Low / base / high uncertainty fan chart |
| `reports/figures/06_peak_imports.png` | Peak-period import comparison |
| `reports/figures/07_robustness.png` | Cross-scenario robustness |
| `reports/figures/08_rolling_overview.png` | 7-day solar/load overview |
| `reports/figures/09_rolling_cumulative_cost.png` | 7-day cumulative cost |
| `reports/figures/10_rolling_soc.png` | 7-day SOC comparison |
| `reports/figures/11_rolling_grid_io.png` | 7-day daily grid import/export |
| `reports/figures/12_rolling_daily_cost.png` | 7-day daily cost by policy |

## 9. Multi-day Rolling-Horizon Backtest

### What it tests

The rolling-horizon experiment evaluates whether the AI agent's planning
advantage becomes more visible across changing solar conditions. A 7-day
synthetic dataset includes: clear/sunny, partly cloudy, overcast/low-solar,
morning cloud, variable/mixed, and weekend load patterns. All conditions
are deterministic (fixed seed) so every run reproduces identically.

### Why it is more realistic than a single-day comparison

A single sunny day does not stress-test the AI agent vs the rule-based
controller. Across changing conditions — particularly cloudy/low-solar days
where the battery cannot fully recharge from solar — the AI's 24-hour
look-ahead and tariff-aware planning should produce measurable, if modest,
benefits.

### What policies are compared

| Policy | Description |
|---|---|
| `no_battery` | Solar first; surplus exported, deficit imported |
| `rule_based` | Greedy: charge from any solar surplus, discharge for any deficit |
| `dp_rolling` | Receding-horizon DP: re-optimise 24h ahead each hour, apply first action |
| `rl_q_learning` | Tabular Q-learning controller trained on simulated daily episodes |

All three policies see the same solar, load, and tariff data and start with
the same battery state (50% SoC).

### Metrics

For each policy: total cost (AUD), grid import (kWh), grid export (kWh),
peak-period import (kWh), self-consumption rate, battery cycles, runtime.

### Honest interpretation of results

A typical 7-day run produces:

| Policy | Total cost (AUD) | Grid import (kWh) | Self-consumption |
|---|---:|---:|---:|
| No battery | $19.95 | 85.8 | 27% |
| Rule-based | $3.14 | 29.2 | 57% |
| **AI rolling** | **$3.34** | 30.7 | 56% |

Key observations, stated honestly:

1. **Both battery policies save ~83% vs no-battery.** This is the primary
   assignment claim and it is genuine.

2. **The AI agent and rule-based are competitive** (gap: ~$0.20 over 7 days).
   The rule-based controller is remarkably effective under simple residential
   TOU conditions because maximising self-consumption is near-optimal when
   import prices always exceed export + degradation costs.

3. **The AI wins on the challenging days** — typically the overcast/low-solar
   days where tariff-aware planning matters most (e.g., Day 3 overcast:
   AI saves ~$0.10 vs rule-based). On sunny days, the rule-based's greedy
   strategy is hard to beat.

4. **The remaining AI–rule gap is partly due to DP discretisation.** The
   dynamic programming optimiser uses discrete SoC bins and action steps,
   while the rule-based controller uses continuous values that exactly match
   the solar surplus/deficit. This is an inherent trade-off of the DP approach.

5. **The AI agent's value is structural, not purely cost-based:**
   - **Explicit planning** — the AI plans across 24 hours, accounting for
     future tariff periods and expected solar availability.
   - **Scenario handling** — can evaluate performance under low/base/high
     solar conditions.
   - **Explainability** — every action is grounded in the symbolic rule
     library and shown alongside the schedule.
   - **Extensibility** — the same DP framework trivially extends to dynamic
     pricing, demand response, multi-battery systems, and vehicle-to-grid.

Stating these limits up front is part of the assignment's "limitations and
trade-offs" requirement.

## 10. Repository layout

```
solar_battery_ai/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── cache/                ← API cache + synthetic fallback
│   ├── raw/                  ← raw fetched data (optional)
│   ├── processed/            ← processed snapshots
│   └── outputs/              ← schedules, metrics, JSON
├── reports/
│   └── figures/              ← PNG figures for the report
├── src/
│   ├── config.py
│   ├── data_sources/
│   │   ├── weather_client.py     Open-Meteo client + cache fallback
│   │   ├── solar_client.py       Transparent PV estimator
│   │   ├── tariff_client.py      AU ToU tariff schedule
│   │   ├── load_profile.py       Synthetic household demand
│   │   └── multiday_weather.py   7-day synthetic weather dataset
│   ├── agent/
│   │   ├── state_builder.py      Composes the 24h energy state
│   │   ├── rules.py              Battery model + symbolic rule library
│   │   ├── uncertainty.py        Low / base / high scenarios + risk warnings
│   │   ├── dispatch_optimizer.py Dynamic programming optimiser (the core)
│   │   └── explanations.py       Rule-grounded action explanations
│   ├── evaluation/
│   │   ├── baselines.py          No-battery + simple rule-based
│   │   ├── metrics.py            All required performance metrics
│   │   ├── backtest.py           Single-day orchestrator across policies + scenarios
│   │   └── rolling_backtest.py   7-day rolling-horizon backtest
│   └── visualisation/
│       └── plots.py              matplotlib for files, plotly for the app
├── main.py                       CLI runner
├── requirements.txt
└── README.md
```

## 11. Limitations and known caveats

- **Tariff is indicative.** Real retailers' time-of-use windows and rates
  differ; users can edit them in the Streamlit sidebar.
- **Synthetic load profile.** A representative residential pattern; a real
  household's demand will diverge.
- **PV estimation is a transparent physical model**, not a forecast — error
  grows as conditions diverge from clear-sky / standard derate assumptions.
- **External API reliability.** Open-Meteo is free and reliable but not
  guaranteed. Cache + synthetic fallback ensure offline operation.
- **Battery degradation is simplified** (flat AUD/kWh cycled cost). Real
  degradation depends on depth-of-discharge, temperature, and calendar age.
- **DP discretisation introduces approximation error.** With discrete SoC
  bins (41 bins × 0.2 kWh resolution) and action steps, the optimiser cannot
  exactly match the continuous surplus/deficit values that the rule-based
  controller uses. This is a known trade-off of the DP approach.
- **Not a commercial energy management system.** This is decision-support
  only; no control signals are sent to any physical device.

## 12. Assignment framing — the claim

> *This project develops a solar–battery optimisation pipeline that combines
> external source-aware data ingestion, scenario-based uncertainty reasoning,
> and search-based dispatch optimisation to support smarter household energy
> management. The system is evaluated using forecast-conditional cost, grid
> import, export behaviour, self-consumption, and battery cycling, showing
> whether AI-based control provides measurable benefits over simpler baseline
> strategies.*

The dispatch optimiser is the core AI artefact; everything upstream
(transparent PV estimate, ToU tariff, synthetic load) is engineered to be
auditable so that the optimiser's contribution is clearly attributable.

## 13. Honest findings

A typical 24h horizon on a sunny weekday produces:

| Policy | Cost (AUD) | Grid import (kWh) | Peak import (kWh) | Self-consumption |
|---|---:|---:|---:|---:|
| No battery | $4.78 | 14.4 | **6.6** | 38% |
| Simple rule-based | $1.20 | 3.9 | 0.0 | 86% |
| **AI dispatch agent** | **$1.17** | 5.1 | 0.07 | 79% |

Three things worth noting honestly:

1. **The headline saving — 75.6% vs no-battery — is real and large**, and it is
   the assignment's primary claim.
2. **The AI agent vs simple rule-based gap is small** ($0.03 / day on a sunny
   day). On a single-day horizon with moderate solar, a well-designed simple
   rule captures most of the achievable benefit. The AI's edge appears in:
   - **Tariff-aware peak-period planning** — the rule charges on every solar
     surplus and only "happens to" survive the peak on sunny days.
   - **Cross-scenario consistency** — across low/base/high solar the AI
     outperforms rule-based by a small but consistent margin.
   - **Explainability** — every action is grounded in the symbolic rule
     library and shown alongside the schedule.
3. **The multi-day rolling backtest reveals the structural story.** Over 7 days
   of mixed weather, the AI and rule-based remain competitive (~$0.20 gap over
   168 hours), but the AI wins on challenging cloudy days where planning
   matters. The rule-based's greedy self-consumption is near-optimal under
   simple TOU tariffs — this is a genuine finding, not a limitation to hide.

## 14. Reproducibility

All random seeds are fixed in `src/config.py` and the data-source modules.
With cached data, the entire pipeline reproduces bit-for-bit on every run.
