"""Tests for Home Battery Runtime diagnostics."""

import json
from datetime import UTC, datetime

import pytest
from homeassistant.components.diagnostics import REDACTED
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.battery_runtime.const import (
    CONF_BATTERY_POWER_ENTITY,
    CONF_BATTERY_SOC_ENTITY,
    CONF_HOME_CONSUMPTION_ENTITY,
)
from custom_components.battery_runtime.coordinator import BatteryRuntimeCoordinator
from custom_components.battery_runtime.diagnostics import (
    async_get_config_entry_diagnostics,
)
from custom_components.battery_runtime.model import (
    PredictionConfidence,
    PredictionResult,
    PredictionStatus,
)

pytestmark = pytest.mark.usefixtures("enable_custom_integrations")

WEEKDAY = datetime(2026, 8, 17, 12, tzinfo=UTC)
WEEKEND = datetime(2026, 8, 22, 12, tzinfo=UTC)


def _coordinator_with_aggregates(
    hass: HomeAssistant, entry: MockConfigEntry
) -> BatteryRuntimeCoordinator:
    """Build a loaded coordinator containing only aggregate learning data."""
    coordinator = BatteryRuntimeCoordinator(hass, entry)
    coordinator.model.observe(WEEKDAY, 800.0, 900.0)
    coordinator.model.observe(WEEKEND, 1000.0, 1100.0)
    coordinator.last_save = WEEKEND
    coordinator.async_set_updated_data(
        PredictionResult(
            status=PredictionStatus.DISCHARGING,
            confidence=PredictionConfidence.MEDIUM,
            depletion_time=WEEKEND.replace(hour=16),
            runtime_hours=4.0,
            coverage_percent=50.0,
            average_discharge_w=1000.0,
            used_fallback=False,
        )
    )
    entry.runtime_data = coordinator
    return coordinator


async def test_diagnostics_redact_sources_and_summarize_profile(
    hass: HomeAssistant,
    battery_entry: MockConfigEntry,
    battery_config_data: dict,
) -> None:
    """Diagnostics contain useful aggregates but no source identity or raw history."""
    _coordinator_with_aggregates(hass, battery_entry)

    diagnostics = await async_get_config_entry_diagnostics(hass, battery_entry)

    config = diagnostics["config_entry"]["data"]
    assert config[CONF_NAME] == REDACTED
    for key in (
        CONF_BATTERY_SOC_ENTITY,
        CONF_HOME_CONSUMPTION_ENTITY,
        CONF_BATTERY_POWER_ENTITY,
    ):
        assert config[key] == REDACTED

    serialized = json.dumps(diagnostics)
    for key in (
        CONF_BATTERY_SOC_ENTITY,
        CONF_HOME_CONSUMPTION_ENTITY,
        CONF_BATTERY_POWER_ENTITY,
    ):
        assert battery_config_data[key] not in serialized

    profile = diagnostics["profile"]
    assert profile["timezone"] == hass.config.time_zone
    assert profile["weekday"]["total_slots"] == 96
    assert profile["weekday"]["observed_slots"] == 1
    assert profile["weekday"]["observation_count"] == 1
    assert profile["weekday"]["average_home_load_w"] == pytest.approx(800)
    assert profile["weekend"]["observed_slots"] == 1
    assert profile["weekend"]["average_home_load_w"] == pytest.approx(1000)
    assert profile["battery_share"]["observation_count"] == 2
    assert "weekday_cells" not in serialized
    assert "weekend_cells" not in serialized
    assert "raw" not in serialized.lower()


async def test_diagnostics_include_only_calculated_runtime_output(
    hass: HomeAssistant, battery_entry: MockConfigEntry
) -> None:
    """Runtime diagnostics expose the coordinator result rather than source states."""
    _coordinator_with_aggregates(hass, battery_entry)

    diagnostics = await async_get_config_entry_diagnostics(hass, battery_entry)

    runtime = diagnostics["runtime"]
    assert runtime["last_update_success"]
    assert runtime["last_sample"] == WEEKEND.isoformat()
    assert runtime["last_save"] == WEEKEND.isoformat()
    assert runtime["prediction"] == {
        "status": "discharging",
        "confidence": "medium",
        "depletion_time": WEEKEND.replace(hour=16).isoformat(),
        "runtime_hours": 4.0,
        "coverage_percent": 50.0,
        "average_discharge_w": 1000.0,
        "used_fallback": False,
    }


async def test_diagnostics_handle_missing_prediction_snapshot(
    hass: HomeAssistant, battery_entry: MockConfigEntry
) -> None:
    """Diagnostics remain downloadable before a coordinator result is present."""
    coordinator = BatteryRuntimeCoordinator(hass, battery_entry)
    battery_entry.runtime_data = coordinator

    diagnostics = await async_get_config_entry_diagnostics(hass, battery_entry)

    assert diagnostics["runtime"]["prediction"] is None
    assert diagnostics["profile"]["weekday"]["observed_slots"] == 0
    assert diagnostics["profile"]["weekend"]["observed_slots"] == 0
