"""Shared fixtures for Home Battery Runtime tests."""

from __future__ import annotations

from typing import Any

import pytest
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.battery_runtime.const import (
    CONF_BATTERY_POWER_ENTITY,
    CONF_BATTERY_POWER_SIGN,
    CONF_BATTERY_SOC_ENTITY,
    CONF_CAPACITY_KWH,
    CONF_DISCHARGE_THRESHOLD_W,
    CONF_HOME_CONSUMPTION_ENTITY,
    CONF_RESERVE_SOC,
    DOMAIN,
    BatteryPowerSign,
)


@pytest.fixture
def battery_config_data() -> dict[str, Any]:
    """Return valid configuration shared by integration-facing tests."""
    return {
        CONF_NAME: "House battery",
        CONF_BATTERY_SOC_ENTITY: "sensor.home_battery_soc",
        CONF_HOME_CONSUMPTION_ENTITY: "sensor.home_consumption",
        CONF_BATTERY_POWER_ENTITY: "sensor.home_battery_power",
        CONF_CAPACITY_KWH: 10.0,
        CONF_RESERVE_SOC: 10.0,
        CONF_BATTERY_POWER_SIGN: BatteryPowerSign.POSITIVE.value,
        CONF_DISCHARGE_THRESHOLD_W: 50.0,
    }


@pytest.fixture
def battery_entry(
    hass: HomeAssistant, battery_config_data: dict[str, Any]
) -> MockConfigEntry:
    """Create and register a Home Battery Runtime config entry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="House battery",
        data=battery_config_data,
    )
    entry.add_to_hass(hass)
    return entry
