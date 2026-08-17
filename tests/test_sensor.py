"""Tests for Home Battery Runtime sensor entities."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE, EntityCategory, UnitOfPower, UnitOfTime
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.battery_runtime.const import DOMAIN
from custom_components.battery_runtime.model import (
    PredictionConfidence,
    PredictionResult,
    PredictionStatus,
)
from custom_components.battery_runtime.sensor import (
    SENSOR_DESCRIPTIONS,
    BatteryRuntimeSensor,
    async_setup_entry,
)

pytestmark = pytest.mark.usefixtures("enable_custom_integrations")

NOW = datetime(2026, 8, 18, 12, tzinfo=UTC)


def _result(
    status: PredictionStatus = PredictionStatus.DISCHARGING,
    *,
    confidence: PredictionConfidence = PredictionConfidence.MEDIUM,
    depletion_time: datetime | None = NOW,
    runtime_hours: float | None = 4.5,
    coverage_percent: float = 75.0,
    average_discharge_w: float | None = 900.0,
) -> PredictionResult:
    """Build one coordinator sensor snapshot."""
    return PredictionResult(
        status=status,
        confidence=confidence,
        depletion_time=depletion_time,
        runtime_hours=runtime_hours,
        coverage_percent=coverage_percent,
        average_discharge_w=average_discharge_w,
        used_fallback=status is PredictionStatus.LEARNING,
    )


def _entities(
    entry: MockConfigEntry, result: PredictionResult
) -> tuple[Mock, dict[str, BatteryRuntimeSensor]]:
    """Create all entities against one lightweight coordinator snapshot."""
    coordinator = Mock()
    coordinator.data = result
    coordinator.last_update_success = True
    entry.runtime_data = coordinator
    entities = {
        description.key: BatteryRuntimeSensor(entry, description)
        for description in SENSOR_DESCRIPTIONS
    }
    return coordinator, entities


async def test_platform_adds_all_entities_with_virtual_device(
    hass: HomeAssistant, battery_entry: MockConfigEntry
) -> None:
    """All entities share stable IDs and one virtual battery device."""
    coordinator = Mock()
    coordinator.data = _result()
    coordinator.last_update_success = True
    battery_entry.runtime_data = coordinator
    add_entities = Mock()

    await async_setup_entry(hass, battery_entry, add_entities)
    entities = list(add_entities.call_args.args[0])

    assert len(entities) == 6
    assert {entity.unique_id for entity in entities} == {
        f"{battery_entry.entry_id}_{description.key}"
        for description in SENSOR_DESCRIPTIONS
    }
    assert all(
        entity.device_info["identifiers"] == {(DOMAIN, battery_entry.entry_id)}
        for entity in entities
    )
    assert all(entity.device_info["name"] == battery_entry.title for entity in entities)


def test_sensor_descriptions_match_home_assistant_semantics(
    battery_entry: MockConfigEntry,
) -> None:
    """Primary forecasts and disabled diagnostic sensors use native metadata."""
    _, entities = _entities(battery_entry, _result())

    depletion = entities["estimated_depletion_time"].entity_description
    runtime = entities["estimated_runtime"].entity_description
    status = entities["prediction_status"].entity_description
    confidence = entities["prediction_confidence"].entity_description
    coverage = entities["history_coverage"].entity_description
    average = entities["forecast_average_discharge"].entity_description

    assert depletion.device_class is SensorDeviceClass.TIMESTAMP
    assert depletion.state_class is None
    assert runtime.device_class is SensorDeviceClass.DURATION
    assert runtime.native_unit_of_measurement == UnitOfTime.HOURS
    assert runtime.state_class is None

    for description in (status, confidence, coverage, average):
        assert description.entity_category is EntityCategory.DIAGNOSTIC
        assert not description.entity_registry_enabled_default
    assert status.device_class is SensorDeviceClass.ENUM
    assert status.options == [value.value for value in PredictionStatus]
    assert confidence.device_class is SensorDeviceClass.ENUM
    assert confidence.options == [value.value for value in PredictionConfidence]
    assert coverage.native_unit_of_measurement == PERCENTAGE
    assert coverage.state_class is SensorStateClass.MEASUREMENT
    assert average.device_class is SensorDeviceClass.POWER
    assert average.native_unit_of_measurement == UnitOfPower.WATT
    assert average.state_class is SensorStateClass.MEASUREMENT


@pytest.mark.parametrize(
    "status",
    [
        PredictionStatus.IDLE,
        PredictionStatus.CHARGING,
        PredictionStatus.SOURCE_UNAVAILABLE,
    ],
)
def test_non_active_states_never_expose_stale_forecasts(
    battery_entry: MockConfigEntry, status: PredictionStatus
) -> None:
    """Only the reason sensor remains known when no forecast is meaningful."""
    coordinator, entities = _entities(battery_entry, _result())
    coordinator.data = _result(status)

    assert entities["estimated_depletion_time"].native_value is None
    assert entities["estimated_runtime"].native_value is None
    assert entities["prediction_status"].native_value == status.value
    assert entities["prediction_confidence"].native_value is None
    assert entities["history_coverage"].native_value is None
    assert entities["forecast_average_discharge"].native_value is None


@pytest.mark.parametrize(
    "status", [PredictionStatus.DISCHARGING, PredictionStatus.LEARNING]
)
def test_actionable_forecast_values_are_exposed(
    battery_entry: MockConfigEntry, status: PredictionStatus
) -> None:
    """Profile and fallback forecasts expose both primary native values."""
    _, entities = _entities(battery_entry, _result(status))

    assert entities["estimated_depletion_time"].native_value == NOW
    assert entities["estimated_runtime"].native_value == 4.5
    assert entities["prediction_confidence"].native_value == "medium"
    assert entities["history_coverage"].native_value == 75.0
    assert entities["forecast_average_discharge"].native_value == 900.0


def test_beyond_horizon_keeps_only_meaningful_diagnostics(
    battery_entry: MockConfigEntry,
) -> None:
    """A beyond-horizon result has no timestamp but still describes the model."""
    result = _result(
        PredictionStatus.BEYOND_HORIZON,
        depletion_time=None,
        runtime_hours=None,
    )
    _, entities = _entities(battery_entry, result)

    assert entities["estimated_depletion_time"].native_value is None
    assert entities["estimated_runtime"].native_value is None
    assert entities["prediction_status"].native_value == "beyond_horizon"
    assert entities["prediction_confidence"].native_value == "medium"
    assert entities["history_coverage"].native_value == 75.0
    assert entities["forecast_average_discharge"].native_value == 900.0
