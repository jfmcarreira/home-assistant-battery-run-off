# Home Battery Runtime

Home Battery Runtime is a local Home Assistant custom integration that estimates
when a home battery will reach its configured reserve state of charge (SoC). It
combines current battery conditions with a learned time-of-week household load
profile. No cloud service, external API, telemetry, or machine-learning library
is required.

> [!IMPORTANT]
> This integration requires Home Assistant `2026.8.0` or newer.

The forecast is conditional: it answers **when the battery would reach reserve
if active discharge continues**. It does not predict future solar production,
charging, or a scheduled change in battery operating mode.

## Installation

### HACS

Install the integration as a HACS custom repository:

1. Open HACS in Home Assistant.
2. Open the three-dot menu and select **Custom repositories**.
3. Enter
   `https://github.com/jfmcarreira/home-assistant-battery-run-off` and select
   **Integration**.
4. Select **Download**, then restart Home Assistant.

### Manual

1. Place `custom_components/battery_runtime` from this repository in the
   `custom_components` directory under the Home Assistant configuration
   directory.
2. Restart Home Assistant.
3. Open **Settings > Devices & services > Add integration**.
4. Search for **Home Battery Runtime**.

## Setup

Setup and reconfiguration are handled in the Home Assistant UI. Add one config
entry for each independently measured battery.

| Field | Requirement |
| --- | --- |
| Name | A non-empty name for this battery |
| Battery SoC entity | Numeric percentage sensor |
| Home consumption power entity | Non-negative whole-home demand in W or kW |
| Battery power entity | Numeric battery power in W or kW |
| Battery capacity | Positive usable, deliverable capacity in kWh |
| Reserve SoC | Percentage from 0 through 99 |
| Battery power sign | Whether positive or negative values mean discharge |
| Discharge threshold | Non-negative noise threshold in watts; default 50 W |

The home consumption entity must represent whole-home demand, not grid import or
grid export. The battery normally needs to supply that full demand while it is
discharging for the learned relationship to be meaningful.

Capacity must represent the energy between 0% and 100% SoC at the same
measurement boundary as the battery power sensor. The integration does not add
a hidden efficiency factor. If the battery reports positive power while
discharging, keep the default positive-discharge sign. Select
negative-discharge when discharge values are negative.

Changing a source entity, the battery power sign, or Home Assistant's timezone
invalidates incompatible learned history. Changing the name, capacity, reserve,
or discharge threshold preserves it.

## Sensors

Home Assistant derives entity IDs from the configured name. The examples below
assume the name `Home Battery`; check **Developer tools > States** for the actual
entity IDs.

| Sensor | Semantics |
| --- | --- |
| Estimated battery depletion time | Timezone-aware timestamp at which SoC is forecast to reach reserve |
| Estimated battery runtime | Decimal hours remaining until reserve, exposed as a duration sensor |

Both primary sensors are unknown while the battery is idle or charging, source
data is invalid or unavailable, or no meaningful forecast exists. They never
retain an old result and present it as a current forecast. A forecast that does
not reach reserve within 48 hours is also unknown; the status sensor explains
that result. While actively discharging at or below reserve, runtime is zero and
depletion time is the current time.

The following diagnostic sensors are disabled by default. Enable them from the
integration device's entity list when needed.

| Diagnostic sensor | Value and meaning |
| --- | --- |
| Prediction status | `discharging`, `idle`, `charging`, `learning`, `source_unavailable`, or `beyond_horizon` |
| Prediction confidence | `low`, `medium`, or `high`; model-quality classification, not a probability |
| History coverage | Percentage of forecast-relevant time slots backed by learned observations |
| Forecast average discharge | Average power demand in watts used by the active forecast |

Status values have these exact meanings:

| Status | Meaning |
| --- | --- |
| `discharging` | A profile-based forecast is active |
| `idle` | Normalized battery discharge is below the configured threshold |
| `charging` | Normalized battery power indicates charging |
| `learning` | History is insufficient and the constant-current fallback is active |
| `source_unavailable` | At least one configured source has no valid numeric state or supported unit |
| `beyond_horizon` | Predicted energy use does not reach reserve within 48 hours |

## Calculation

Available energy above reserve is:

```text
remaining_kWh = capacity_kWh * max(current_soc - reserve_soc, 0) / 100
```

Battery power is converted to watts and normalized so discharge is always
positive internally:

```text
normalized_discharge_w = battery_power_w * configured_sign
configured_sign = +1 for positive-discharge, -1 for negative-discharge
```

After sign normalization, power less than the negative threshold is charging,
power from the negative threshold through the positive threshold is idle, and
power greater than the positive threshold is active discharge. During a
profile-based forecast, each future baseline is:

```text
baseline_discharge_w = historical_home_load_w * learned_battery_share
```

The model chooses weekday or weekend history for each future local date. It
blends current battery power into the first 30 minutes, then decays the effect of
current home-load conditions toward the learned profile over about two hours.
Predicted watts are integrated in five-minute steps and converted to kWh. The
final step is interpolated to produce a more precise depletion timestamp, and
forecasting stops after 48 hours.

## Learning And Fallback

On first setup, the integration asks Home Assistant Recorder for up to 28 days
of available five-minute mean statistics. Recorder can retain less high-
resolution history; all compatible data that is available is used.

Observations are converted to Home Assistant's local timezone and grouped by:

- Weekday or weekend.
- One of 96 fifteen-minute slots in a day.

Each slot learns an exponentially weighted mean home load, variance, effective
sample weight, valid-observation count, and last-update time. During active
discharge, the integration also learns the ratio of battery discharge to home
load. This battery-share ratio accounts for modest inverter losses or different
measurement points. It defaults to `1.0` until three valid ratio samples exist,
ignores ratio learning below 50 W home load, and clamps each learned ratio to
the range `0.5` through `1.5`.

Live learning samples every five minutes. Home consumption is learned while
charging, idle, and discharging; the battery-share ratio is learned only during
active discharge. Unknown, unavailable, non-numeric, non-finite, and negative
home-load samples are rejected. Ratio outliers are bounded, and only aggregate
profile data, not raw power history, is persisted.

Each weekday or weekend category reached during profile integration must have at
least four populated cells. A missing exact future slot then uses the
effective-weighted mean of populated cells from the same category; it never
borrows from the other category. Exact-slot history coverage excludes these
substituted values.

If the forecast reaches a weekday or weekend category with fewer than four
populated cells, runtime uses smoothed current battery discharge as a
constant-power fallback:

```text
runtime_h = remaining_kWh / (smoothed_discharge_w / 1000)
depletion_time = current_time + runtime_h
```

Fallback results have status `learning` and low confidence. Learning continues
automatically, is saved in batches, and survives Home Assistant restarts.

## Limitations

- Predictions are produced only during active battery discharge.
- Future solar generation and charging are not modeled.
- Weather, tariffs, battery schedules, and per-appliance demand are not modeled.
- Multiple config entries are independent; batteries are not combined into one
  virtual battery.
- Capacity is not estimated and battery degradation is not inferred.
- Confidence describes input coverage and consistency, not an error interval or
  statistical probability.
- Large temporary loads can move the near-term estimate, although smoothing and
  decay are intended to prevent a short spike from dominating the full runtime.
- Forecast quality depends on accurate SoC, whole-home demand, battery power,
  units, sign convention, usable capacity, reserve, Recorder history, and local
  timezone.
- No prediction extends beyond the 48-hour forecast horizon.

Do not use this forecast as the sole control for safety-critical equipment or
loads that cannot tolerate power loss.

## Diagnostics And Troubleshooting

When no forecast is shown, first enable **Prediction status** and check:

- All three source entities have numeric states and the power entities use `W`
  or `kW`.
- Battery SoC is above reserve.
- The selected sign makes discharge positive.
- Normalized discharge is above the threshold.
- `beyond_horizon` is not active.

Use **Settings > Devices & services > Home Battery Runtime > Download
diagnostics** when reporting a problem. Diagnostics are designed to exclude raw
historical power data. Review the generated file before sharing it, as with any
Home Assistant diagnostic download.

Temporary debug logging can be enabled in `configuration.yaml`:

```yaml
logger:
  logs:
    custom_components.battery_runtime: debug
```

Restart Home Assistant after changing logger configuration, reproduce the
problem, then remove debug logging to limit log volume. Do not manually edit the
integration's aggregate data in Home Assistant's `.storage` directory.

## Dashboard Example

Replace these illustrative entity IDs with the IDs created for the configured
battery:

```yaml
type: vertical-stack
cards:
  - type: entities
    title: Home battery forecast
    show_header_toggle: false
    entities:
      - entity: sensor.home_battery_estimated_battery_depletion_time
        name: Expected reserve time
      - entity: sensor.home_battery_estimated_battery_runtime
        name: Runtime remaining
      - entity: sensor.home_battery_prediction_status
      - entity: sensor.home_battery_prediction_confidence
      - entity: sensor.home_battery_forecast_average_discharge
  - type: gauge
    entity: sensor.home_battery_history_coverage
    name: Learned history coverage
    min: 0
    max: 100
```

## Automation Examples

The diagnostic status sensor must be enabled for these examples. Replace the
illustrative entity IDs before use.

Create a persistent notification after runtime remains below two hours for five
minutes:

```yaml
alias: Home battery runtime is low
triggers:
  - trigger: numeric_state
    entity_id: sensor.home_battery_estimated_battery_runtime
    below: 2
    for: "00:05:00"
conditions:
  - condition: state
    entity_id: sensor.home_battery_prediction_status
    state: discharging
actions:
  - action: persistent_notification.create
    data:
      title: Home battery runtime is low
      message: The home battery is forecast to reach reserve in under two hours.
mode: single
```

Report a source that remains unavailable for ten minutes:

```yaml
alias: Home battery forecast source unavailable
triggers:
  - trigger: state
    entity_id: sensor.home_battery_prediction_status
    to: source_unavailable
    for: "00:10:00"
actions:
  - action: persistent_notification.create
    data:
      title: Home battery forecast unavailable
      message: Check the configured SoC, home demand, and battery power sensors.
mode: single
```

## Development

Development uses Python 3.14.2 or newer, [uv](https://docs.astral.sh/uv/), pytest with
`pytest-homeassistant-custom-component`, and Ruff:

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

HACS and hassfest validation run in separate GitHub Actions workflows. The test
dependency is deliberately pinned so CI targets a known Home Assistant test
harness; update that pin intentionally when advancing the tested Home Assistant
version.

## License

Home Battery Runtime is licensed under the
[GNU General Public License v3.0](LICENSE).
