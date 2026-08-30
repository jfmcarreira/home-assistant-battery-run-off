"""Tests for the pure battery runtime prediction model."""

import math
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from custom_components.battery_runtime.model import (
    BATTERY_SHARE_MAX_RATIO,
    BatteryPowerSign,
    BatteryRuntimeModel,
    BatteryShare,
    ConsumptionProfile,
    EWMAStatistics,
    ModelConfig,
    PredictionConfidence,
    PredictionInput,
    PredictionStatus,
    ProfileCell,
    available_energy_kwh,
    constant_power_runtime_hours,
    normalize_discharge_power,
    power_to_watts,
)

UTC_NOW = datetime(2026, 1, 5, 12, tzinfo=UTC)


def _config(
    *,
    capacity_kwh: float = 10.0,
    reserve_soc_percent: float = 0.0,
    timezone_name: str = "UTC",
    battery_power_sign: BatteryPowerSign = BatteryPowerSign.POSITIVE_DISCHARGE,
    discharge_threshold_w: float = 50.0,
) -> ModelConfig:
    return ModelConfig(
        capacity_kwh=capacity_kwh,
        reserve_soc_percent=reserve_soc_percent,
        timezone_name=timezone_name,
        battery_power_sign=battery_power_sign,
        discharge_threshold_w=discharge_threshold_w,
    )


def _fill_cells(cells: list[ProfileCell], value: float, *, samples: int = 12) -> None:
    for cell in cells:
        for sample in range(samples):
            cell.update(value, UTC_NOW + timedelta(days=sample))


def _profile(
    *,
    timezone_name: str = "UTC",
    home_load_w: float = 1000.0,
    samples: int = 12,
) -> ConsumptionProfile:
    profile = ConsumptionProfile(timezone_name)
    _fill_cells(profile.cells, home_load_w, samples=samples)
    return profile


def _predict(
    model: BatteryRuntimeModel,
    *,
    timestamp: datetime = UTC_NOW,
    soc_percent: float | None = 100.0,
    home_load_w: float | None = 1000.0,
    battery_power_w: float | None = 1000.0,
):
    return model.predict(
        PredictionInput(
            timestamp=timestamp,
            soc_percent=soc_percent,
            home_load_w=home_load_w,
            battery_power_w=battery_power_w,
        )
    )


def test_energy_power_and_sign_formulas() -> None:
    """Core formulas validate units, reserve, and both sign conventions."""

    assert available_energy_kwh(13.5, 80, 20) == pytest.approx(8.1)
    assert available_energy_kwh(10, 10, 20) == 0
    assert constant_power_runtime_hours(2.5, 1000) == 2.5
    assert constant_power_runtime_hours(2.5, 0) is None
    assert normalize_discharge_power(850, 1) == 850
    assert normalize_discharge_power(-850, -1) == 850
    assert power_to_watts(1000, "W") == power_to_watts(1, "kW")


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: available_energy_kwh(0, 50, 10), "greater than zero"),
        (lambda: available_energy_kwh(10, math.nan, 10), "finite"),
        (lambda: normalize_discharge_power(100, 0), "1 or -1"),
        (lambda: power_to_watts(1, "MW"), "W or kW"),
        (lambda: ModelConfig(10, 100, "UTC"), "at most 99"),
        (lambda: ModelConfig(10, 10, "Not/AZone"), "unknown timezone"),
    ],
)
def test_numeric_configuration_validation(call, match: str) -> None:
    """Invalid and non-finite numeric configuration fails explicitly."""

    with pytest.raises(ValueError, match=match):
        call()


def test_ewma_tracks_mean_variance_weight_count_and_timestamp() -> None:
    """Profile cell updates use the documented conventional EWMA formulas."""

    cell = ProfileCell()
    cell.update(100, UTC_NOW)
    cell.update(200, UTC_NOW + timedelta(minutes=5))

    assert cell.mean == pytest.approx(120)
    assert cell.variance == pytest.approx(1600)
    assert cell.effective_weight == pytest.approx(1.8)
    assert cell.sample_count == 2
    assert cell.last_update == UTC_NOW + timedelta(minutes=5)


@pytest.mark.parametrize(
    "serialized",
    [
        {
            "mean": 1.0,
            "variance": 0.0,
            "effective_weight": 0.0,
            "sample_count": 0,
            "last_update": None,
        },
        {
            "mean": 0.0,
            "variance": 1.0,
            "effective_weight": 0.0,
            "sample_count": 0,
            "last_update": None,
        },
        {
            "mean": 0.0,
            "variance": 0.0,
            "effective_weight": 1.0,
            "sample_count": 0,
            "last_update": None,
        },
        {
            "mean": 100.0,
            "variance": 0.0,
            "effective_weight": 0.5,
            "sample_count": 1,
            "last_update": UTC_NOW.isoformat(),
        },
        {
            "mean": 100.0,
            "variance": 0.0,
            "effective_weight": 3.0,
            "sample_count": 2,
            "last_update": UTC_NOW.isoformat(),
        },
        {
            "mean": 100.0,
            "variance": 0.0,
            "effective_weight": 1.7,
            "sample_count": 2,
            "last_update": UTC_NOW.isoformat(),
        },
        {
            "mean": 100.0,
            "variance": 1.0,
            "effective_weight": 1.0,
            "sample_count": 1,
            "last_update": UTC_NOW.isoformat(),
        },
    ],
)
def test_ewma_deserialization_rejects_inconsistent_cross_fields(
    serialized: dict[str, object],
) -> None:
    """Stored EWMA fields must describe a state reachable by updates."""

    with pytest.raises(ValueError, match="inconsistent state"):
        EWMAStatistics.from_dict(serialized)


def test_profile_shares_local_slots_across_all_days() -> None:
    """The same local quarter-hour shares history regardless of the day."""

    profile = ConsumptionProfile("Europe/Lisbon")
    monday = datetime(2026, 1, 5, 9, 14, tzinfo=ZoneInfo("Europe/Lisbon"))
    saturday = datetime(2026, 1, 10, 9, 14, tzinfo=ZoneInfo("Europe/Lisbon"))
    profile.update(monday, 500)

    assert profile.slot_for(monday) == 36
    assert profile.slot_for(saturday) == 36
    assert profile.estimate_home_load(monday).home_load_w == 500
    assert profile.estimate_home_load(saturday).home_load_w == 500

    profile.update(saturday, 1500)
    assert profile.estimate_home_load(monday).home_load_w == pytest.approx(700)


def test_profile_converts_utc_before_selecting_slot() -> None:
    """A UTC instant maps to the correct local date across midnight."""

    profile = ConsumptionProfile("Pacific/Auckland")
    instant = datetime(2026, 1, 9, 11, 5, tzinfo=UTC)

    assert profile.slot_for(instant) == 0


def test_battery_share_defaults_then_learns_and_clamps_outliers() -> None:
    """Battery-share evidence is gated and implausible ratios are bounded."""

    share = BatteryShare()
    assert share.value == 1.0
    assert not share.update(100, 10, UTC_NOW)
    for minute in (0, 5, 10):
        assert share.update(10_000, 1000, UTC_NOW + timedelta(minutes=minute))

    assert share.value == BATTERY_SHARE_MAX_RATIO
    assert share.statistics.sample_count == 3


def test_observe_learns_home_load_in_all_battery_modes() -> None:
    """Charging still trains demand, but it does not train battery share."""

    model = BatteryRuntimeModel(_config())
    assert model.observe(UTC_NOW, 800, -500)
    assert model.profile.cell_at(UTC_NOW).mean == 800
    assert model.battery_share.statistics.sample_count == 0
    assert not model.observe(UTC_NOW, -1, 500)


def test_constant_current_fallback_is_exact_and_marked_learning() -> None:
    """An empty profile produces a useful low-confidence exact fallback."""

    model = BatteryRuntimeModel(_config(capacity_kwh=10))
    result = _predict(model, soc_percent=50, home_load_w=1000, battery_power_w=1000)

    assert result.status is PredictionStatus.LEARNING
    assert result.confidence is PredictionConfidence.LOW
    assert result.runtime_hours == pytest.approx(5)
    assert result.depletion_time == UTC_NOW + timedelta(hours=5)
    assert result.average_discharge_w == pytest.approx(1000)
    assert result.coverage_percent == 0
    assert result.used_fallback


def test_idle_history_does_not_dilute_new_discharge_session() -> None:
    """A discharge starting after idle uses its full current power immediately."""

    model = BatteryRuntimeModel(_config(capacity_kwh=1))
    for sample in range(6):
        model.update_current_conditions(UTC_NOW + timedelta(minutes=sample), 1000, 0)

    result = _predict(
        model,
        timestamp=UTC_NOW + timedelta(minutes=6),
        battery_power_w=1000,
    )

    assert result.status is PredictionStatus.LEARNING
    assert result.runtime_hours == pytest.approx(1)
    assert result.average_discharge_w == pytest.approx(1000)


def test_charging_history_does_not_dilute_new_discharge_session() -> None:
    """Charging current readings reset rather than contribute zero discharge."""

    model = BatteryRuntimeModel(_config(capacity_kwh=1))
    for sample in range(3):
        model.update_current_conditions(
            UTC_NOW + timedelta(minutes=sample * 5), 1000, -1000
        )

    result = _predict(
        model,
        timestamp=UTC_NOW + timedelta(minutes=15),
        battery_power_w=1000,
    )

    assert result.status is PredictionStatus.LEARNING
    assert result.runtime_hours == pytest.approx(1)
    assert result.average_discharge_w == pytest.approx(1000)


@pytest.mark.parametrize(
    ("outage_home_w", "outage_battery_w"),
    [(None, None), (None, 1000), (1000, None)],
)
def test_outage_resets_stale_current_smoothing_for_one_call_recovery(
    outage_home_w: float | None,
    outage_battery_w: float | None,
) -> None:
    """One current update after any source outage starts from recovered values."""

    model = BatteryRuntimeModel(_config(capacity_kwh=1))
    model.update_current_conditions(UTC_NOW, 5000, 5000)
    assert not model.update_current_conditions(
        UTC_NOW + timedelta(minutes=1), outage_home_w, outage_battery_w
    )
    assert model.update_current_conditions(UTC_NOW + timedelta(minutes=2), 1000, 1000)

    result = _predict(
        model,
        timestamp=UTC_NOW + timedelta(minutes=2),
        home_load_w=1000,
        battery_power_w=1000,
    )

    assert result.runtime_hours == pytest.approx(1)
    assert result.average_discharge_w == pytest.approx(1000)


def test_sustained_active_discharge_still_smooths_power_noise() -> None:
    """Readings within an active session continue through the discharge EWMA."""

    model = BatteryRuntimeModel(_config(capacity_kwh=1))
    for sample in range(3):
        model.update_current_conditions(UTC_NOW + timedelta(minutes=sample), 1000, 1000)
    model.update_current_conditions(UTC_NOW + timedelta(minutes=3), 2000, 2000)

    result = _predict(
        model,
        timestamp=UTC_NOW + timedelta(minutes=3),
        home_load_w=2000,
        battery_power_w=2000,
    )

    assert result.average_discharge_w is not None
    assert 1000 < result.average_discharge_w < 2000


def test_profile_constant_load_runtime_and_reserve_are_exact() -> None:
    """Five-minute integration preserves the constant-load energy formula."""

    model = BatteryRuntimeModel(
        _config(capacity_kwh=10, reserve_soc_percent=20),
        profile=_profile(),
    )
    result = _predict(model, soc_percent=60)

    assert result.status is PredictionStatus.DISCHARGING
    assert result.runtime_hours == pytest.approx(4)
    assert result.depletion_time == UTC_NOW + timedelta(hours=4)
    assert result.average_discharge_w == pytest.approx(1000)
    assert result.coverage_percent == 100
    assert result.confidence is PredictionConfidence.HIGH
    assert not result.used_fallback


def test_soc_at_or_below_reserve_has_zero_runtime() -> None:
    """No positive runtime is exposed once reserve has been reached."""

    model = BatteryRuntimeModel(_config(capacity_kwh=10, reserve_soc_percent=20))
    result = _predict(model, soc_percent=20)

    assert result.status is PredictionStatus.DISCHARGING
    assert result.runtime_hours == 0
    assert result.depletion_time == UTC_NOW


def test_negative_discharge_sign_produces_same_prediction() -> None:
    """Configured battery sign is normalized before mode and energy logic."""

    model = BatteryRuntimeModel(
        _config(
            capacity_kwh=1,
            battery_power_sign=BatteryPowerSign.NEGATIVE_DISCHARGE,
        )
    )
    result = _predict(model, battery_power_w=-1000)

    assert result.status is PredictionStatus.LEARNING
    assert result.runtime_hours == pytest.approx(1)


def test_forecasts_use_the_same_profile_on_every_day() -> None:
    """A populated profile remains ready across weekdays and weekends."""

    profile = _profile(home_load_w=1000)
    model = BatteryRuntimeModel(_config(capacity_kwh=2), profile=profile)
    monday = datetime(2026, 1, 5, 12, tzinfo=UTC)
    saturday = datetime(2026, 1, 10, 12, tzinfo=UTC)

    monday_result = _predict(
        model, timestamp=monday, home_load_w=1000, battery_power_w=1000
    )
    saturday_result = _predict(
        model, timestamp=saturday, home_load_w=1000, battery_power_w=1000
    )

    assert monday_result.runtime_hours == pytest.approx(2)
    assert saturday_result.runtime_hours == pytest.approx(2)


def test_forecast_uses_elapsed_time_across_spring_dst_transition() -> None:
    """Runtime and depletion timestamp advance on the real UTC timeline."""

    timezone_name = "America/New_York"
    model = BatteryRuntimeModel(
        _config(capacity_kwh=2, timezone_name=timezone_name),
        profile=_profile(timezone_name=timezone_name),
    )
    before_jump = datetime(2025, 3, 9, 1, 30, tzinfo=ZoneInfo(timezone_name))
    result = _predict(model, timestamp=before_jump)

    assert result.runtime_hours == pytest.approx(2)
    assert result.depletion_time == datetime(
        2025, 3, 9, 4, 30, tzinfo=ZoneInfo(timezone_name)
    )
    assert result.depletion_time is not None
    assert result.depletion_time.utcoffset() == timedelta(hours=-4)


def test_forecast_uses_second_fold_across_fall_dst_transition() -> None:
    """A real two-hour runtime lands in the repeated hour's second fold."""

    timezone_name = "America/New_York"
    model = BatteryRuntimeModel(
        _config(capacity_kwh=2, timezone_name=timezone_name),
        profile=_profile(timezone_name=timezone_name),
    )
    before_fall_back = datetime(2025, 11, 2, 0, 30, tzinfo=ZoneInfo(timezone_name))
    result = _predict(model, timestamp=before_fall_back)

    assert result.runtime_hours == pytest.approx(2)
    assert result.depletion_time is not None
    assert result.depletion_time.fold == 1
    assert result.depletion_time.utcoffset() == timedelta(hours=-5)
    assert result.depletion_time.astimezone(UTC) == datetime(
        2025, 11, 2, 6, 30, tzinfo=UTC
    )


def test_transient_smoothing_is_independent_of_update_frequency() -> None:
    """Time-based smoothing gives the same result for chatty source sensors."""

    sparse = BatteryRuntimeModel(_config(capacity_kwh=1))
    chatty = BatteryRuntimeModel(_config(capacity_kwh=1))

    sparse.update_current_conditions(UTC_NOW, 1000, 1000)
    chatty.update_current_conditions(UTC_NOW, 1000, 1000)

    sparse.update_current_conditions(UTC_NOW + timedelta(minutes=1), 2000, 2000)
    sparse.update_current_conditions(UTC_NOW + timedelta(minutes=11), 2000, 2000)
    for minute in range(1, 12):
        chatty.update_current_conditions(
            UTC_NOW + timedelta(minutes=minute), 2000, 2000
        )

    sparse_result = _predict(
        sparse,
        timestamp=UTC_NOW + timedelta(minutes=11),
        home_load_w=2000,
        battery_power_w=2000,
    )
    chatty_result = _predict(
        chatty,
        timestamp=UTC_NOW + timedelta(minutes=11),
        home_load_w=2000,
        battery_power_w=2000,
    )

    assert sparse_result.average_discharge_w == pytest.approx(
        chatty_result.average_discharge_w
    )
    assert sparse_result.runtime_hours == pytest.approx(chatty_result.runtime_hours)


def test_short_current_spike_has_limited_profile_forecast_effect() -> None:
    """A two-minute load spike only nudges the learned-profile forecast."""

    profile = _profile()
    normal_model = BatteryRuntimeModel(_config(capacity_kwh=2), profile=profile)
    spike_model = BatteryRuntimeModel(
        _config(capacity_kwh=2), profile=ConsumptionProfile.from_dict(profile.to_dict())
    )

    for minute in range(2):
        timestamp = UTC_NOW + timedelta(minutes=minute)
        normal_model.update_current_conditions(timestamp, 1000, 1000)
        spike_model.update_current_conditions(timestamp, 1000, 1000)

    for minute in range(2, 4):
        timestamp = UTC_NOW + timedelta(minutes=minute)
        normal_model.update_current_conditions(timestamp, 1000, 1000)
        spike_model.update_current_conditions(timestamp, 5000, 5000)

    normal = _predict(
        normal_model,
        timestamp=UTC_NOW + timedelta(minutes=3),
    )
    spike = _predict(
        spike_model,
        timestamp=UTC_NOW + timedelta(minutes=3),
        home_load_w=5000,
        battery_power_w=5000,
    )

    assert normal.runtime_hours == pytest.approx(2)
    assert spike.runtime_hours is not None
    assert normal.runtime_hours is not None
    assert normal.runtime_hours * 0.9 < spike.runtime_hours < normal.runtime_hours


def test_sustained_load_change_still_adjusts_profile_forecast() -> None:
    """A lasting load increase gradually becomes part of the forecast."""

    model = BatteryRuntimeModel(_config(capacity_kwh=2), profile=_profile())
    model.update_current_conditions(UTC_NOW, 1000, 1000)
    model.update_current_conditions(UTC_NOW + timedelta(minutes=1), 1000, 1000)

    for minute in range(2, 32):
        model.update_current_conditions(UTC_NOW + timedelta(minutes=minute), 2000, 2000)

    result = _predict(
        model,
        timestamp=UTC_NOW + timedelta(minutes=31),
        home_load_w=2000,
        battery_power_w=2000,
    )

    assert result.runtime_hours is not None
    assert 1.3 < result.runtime_hours < 1.7


def test_missing_cells_use_shared_average_and_report_low_coverage() -> None:
    """Sparse exact cells use the shared average without introducing gaps."""

    profile = ConsumptionProfile("UTC")
    for slot in (0, 20, 40, 60):
        for sample in range(8):
            profile.cells[slot].update(1000, UTC_NOW + timedelta(days=sample))
    model = BatteryRuntimeModel(_config(capacity_kwh=1), profile=profile)
    monday_midnight = datetime(2026, 1, 5, tzinfo=UTC)
    result = _predict(model, timestamp=monday_midnight)

    assert result.status is PredictionStatus.DISCHARGING
    assert result.runtime_hours == pytest.approx(1)
    assert 0 < result.coverage_percent < 50
    assert result.confidence is PredictionConfidence.LOW


def test_profile_history_is_available_on_every_day() -> None:
    """History learned on one day makes the same slots ready on every day."""

    profile = ConsumptionProfile("UTC")
    _fill_cells(profile.cells, 1000)
    model = BatteryRuntimeModel(_config(capacity_kwh=1), profile=profile)
    saturday = datetime(2026, 1, 10, 12, tzinfo=UTC)

    result = _predict(model, timestamp=saturday)

    assert result.status is PredictionStatus.DISCHARGING
    assert result.runtime_hours == pytest.approx(1)
    assert result.coverage_percent == pytest.approx(100)
    assert not result.used_fallback


def test_final_five_minute_step_is_interpolated() -> None:
    """Depletion can occur midway through the final integration step."""

    model = BatteryRuntimeModel(
        _config(capacity_kwh=1),
        profile=_profile(),
    )
    result = _predict(model, soc_percent=12.5)

    assert result.runtime_hours == pytest.approx(0.125)
    assert result.depletion_time == UTC_NOW + timedelta(minutes=7, seconds=30)


def test_profile_forecast_enforces_48_hour_horizon() -> None:
    """A profile forecast too long to resolve returns beyond_horizon."""

    model = BatteryRuntimeModel(
        _config(capacity_kwh=10),
        profile=_profile(home_load_w=100),
    )
    result = _predict(model, home_load_w=100, battery_power_w=100)

    assert result.status is PredictionStatus.BEYOND_HORIZON
    assert result.runtime_hours is None
    assert result.depletion_time is None
    assert result.average_discharge_w == pytest.approx(100)


def test_fallback_forecast_enforces_48_hour_horizon() -> None:
    """The constant-current path applies the same maximum horizon."""

    model = BatteryRuntimeModel(_config(capacity_kwh=10))
    result = _predict(model, home_load_w=100, battery_power_w=100)

    assert result.status is PredictionStatus.BEYOND_HORIZON
    assert result.used_fallback


def test_confidence_decreases_for_high_variability() -> None:
    """Deterministic thresholds distinguish stable and variable profiles."""

    stable = BatteryRuntimeModel(_config(capacity_kwh=1), profile=_profile())
    variable_profile = ConsumptionProfile("UTC")
    for cell in variable_profile.cells:
        for sample in range(20):
            cell.update(
                0 if sample % 2 == 0 else 2000,
                UTC_NOW + timedelta(days=sample),
            )
    variable = BatteryRuntimeModel(_config(capacity_kwh=1), profile=variable_profile)
    variable_current = variable_profile.cells[48].mean

    stable_result = _predict(stable)
    variable_result = _predict(
        variable,
        home_load_w=variable_current,
        battery_power_w=variable_current,
    )

    assert stable_result.confidence is PredictionConfidence.HIGH
    assert variable_result.confidence is not PredictionConfidence.HIGH


def test_confidence_classification_exact_threshold_boundaries() -> None:
    """Inclusive confidence thresholds classify their exact boundary values."""

    classify = BatteryRuntimeModel._classify_confidence

    assert classify(0.9, 5.0, 0.7, 1000, 1000) is PredictionConfidence.HIGH
    assert (
        classify(math.nextafter(0.9, 0.0), 5.0, 0.7, 1000, 1000)
        is PredictionConfidence.MEDIUM
    )
    assert classify(0.5, 0.0, 0.0, 1000, 1000) is PredictionConfidence.MEDIUM
    assert (
        classify(math.nextafter(0.5, 0.0), 0.0, 0.0, 1000, 1000)
        is PredictionConfidence.LOW
    )


@pytest.mark.parametrize(
    ("soc", "home", "battery"),
    [
        (None, 1000, 1000),
        (50, -1, 1000),
        (50, 1000, math.inf),
    ],
)
def test_unavailable_sources_do_not_produce_forecast_values(soc, home, battery) -> None:
    """Invalid source data never retains prediction values."""

    result = _predict(
        BatteryRuntimeModel(_config()),
        soc_percent=soc,
        home_load_w=home,
        battery_power_w=battery,
    )

    assert result.status is PredictionStatus.SOURCE_UNAVAILABLE
    assert result.runtime_hours is None
    assert result.depletion_time is None
    assert result.average_discharge_w is None


@pytest.mark.parametrize(
    ("soc", "battery", "expected_status", "expected_runtime"),
    [
        (50, 50, PredictionStatus.IDLE, 5),
        (100, 0, PredictionStatus.IDLE, 10),
        (50, -100, PredictionStatus.CHARGING, 5),
    ],
)
def test_idle_and_charging_forecast_discharge_starting_now(
    soc, battery, expected_status, expected_runtime
) -> None:
    """Non-discharging batteries use current demand for a hypothetical forecast."""

    result = _predict(
        BatteryRuntimeModel(_config()),
        soc_percent=soc,
        home_load_w=1000,
        battery_power_w=battery,
    )

    assert result.status is expected_status
    assert result.runtime_hours == pytest.approx(expected_runtime)
    assert result.depletion_time == UTC_NOW + timedelta(hours=expected_runtime)
    assert result.average_discharge_w == pytest.approx(1000)
    assert result.used_fallback


def test_charging_profile_forecast_ignores_current_charge_power() -> None:
    """A hypothetical forecast uses expected demand, not incoming charge power."""

    result = _predict(
        BatteryRuntimeModel(_config(capacity_kwh=1), profile=_profile()),
        battery_power_w=-5000,
    )

    assert result.status is PredictionStatus.CHARGING
    assert result.runtime_hours == pytest.approx(1)
    assert result.average_discharge_w == pytest.approx(1000)
    assert not result.used_fallback


def test_profile_and_model_round_trip_store_serialization() -> None:
    """All aggregate fields survive a JSON-compatible Store round trip."""

    config = _config(timezone_name="Europe/Lisbon")
    model = BatteryRuntimeModel(config)
    local = datetime(2026, 1, 5, 18, tzinfo=ZoneInfo("Europe/Lisbon"))
    for minute in (0, 5, 10):
        model.observe(local + timedelta(minutes=minute), 1000, 1200)

    serialized = model.to_dict()
    restored = BatteryRuntimeModel.from_dict(config, serialized)

    assert restored.to_dict() == serialized
    assert restored.profile.timezone_name == "Europe/Lisbon"
    assert restored.profile.cell_at(local).sample_count == 3
    assert restored.battery_share.value == pytest.approx(1.2)
    assert restored.last_sample == local + timedelta(minutes=10)


def test_legacy_profile_storage_merges_day_categories() -> None:
    """Legacy weekday and weekend cells restore into one shared profile."""

    config = _config()
    weekday = ConsumptionProfile("UTC")
    weekend = ConsumptionProfile("UTC")
    weekday.cells[48].update(800, UTC_NOW)
    weekend.cells[48].update(1000, UTC_NOW + timedelta(days=5))
    serialized = BatteryRuntimeModel(config).to_dict()
    serialized["version"] = 1
    serialized["profile"] = {
        "timezone": "UTC",
        "weekday": weekday.to_dict()["cells"],
        "weekend": weekend.to_dict()["cells"],
    }

    restored = BatteryRuntimeModel.from_dict(config, serialized)
    merged = restored.profile.cells[48]

    assert merged.sample_count == 2
    assert merged.mean == pytest.approx(900)
    assert merged.variance == pytest.approx(10_000)
    assert restored.to_dict()["version"] == 2
    restored_profile = restored.to_dict()["profile"]
    assert isinstance(restored_profile, dict)
    assert "cells" in restored_profile


def test_bootstrap_observations_match_restored_aggregate_model_prediction() -> None:
    """Historical learning has no transient state absent from serialization."""

    config = _config(capacity_kwh=1)
    bootstrapped = BatteryRuntimeModel(config)
    for week in range(3):
        bootstrapped.observe(UTC_NOW - timedelta(days=7 * (week + 1)), 5000, 5000)

    restored = BatteryRuntimeModel.from_dict(config, bootstrapped.to_dict())
    current = PredictionInput(
        timestamp=UTC_NOW,
        soc_percent=100,
        home_load_w=1000,
        battery_power_w=1000,
    )

    bootstrapped_result = bootstrapped.predict(current)
    restored_result = restored.predict(current)

    assert restored.to_dict() == bootstrapped.to_dict()
    assert bootstrapped_result == restored_result
    assert bootstrapped_result.average_discharge_w == pytest.approx(1000)


def test_serialization_rejects_malformed_or_incompatible_data() -> None:
    """Corrupt profile values and timezone changes fail safely for rebuilding."""

    config = _config()
    serialized = BatteryRuntimeModel(config).to_dict()
    profile = serialized["profile"]
    assert isinstance(profile, dict)
    cells = profile["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    cell["mean"] = math.nan

    with pytest.raises(ValueError, match="finite"):
        BatteryRuntimeModel.from_dict(config, serialized)

    valid = BatteryRuntimeModel(config).to_dict()
    with pytest.raises(ValueError, match="timezone"):
        BatteryRuntimeModel.from_dict(_config(timezone_name="Europe/Lisbon"), valid)


def test_naive_timestamps_are_rejected() -> None:
    """Slot selection and forecasting require an unambiguous aware timestamp."""

    statistics = EWMAStatistics()
    with pytest.raises(ValueError, match="timezone-aware"):
        statistics.update(1, datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="timezone-aware"):
        _predict(
            BatteryRuntimeModel(_config()),
            timestamp=datetime(2026, 1, 1),
        )
