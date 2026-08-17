"""Tests for the Home Battery Runtime config flow."""

from unittest.mock import patch

import pytest
from homeassistant import config_entries
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import (
    ATTR_DEVICE_CLASS,
    ATTR_UNIT_OF_MEASUREMENT,
    CONF_NAME,
    PERCENTAGE,
    UnitOfPower,
)
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType, InvalidData
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.battery_runtime.config_flow import (
    POWER_ENTITY_SELECTOR,
    _normalize_input,
)
from custom_components.battery_runtime.const import (
    CONF_BATTERY_POWER_ENTITY,
    CONF_BATTERY_POWER_SIGN,
    CONF_BATTERY_SOC_ENTITY,
    CONF_CAPACITY_KWH,
    CONF_DISCHARGE_THRESHOLD_W,
    CONF_HOME_CONSUMPTION_ENTITY,
    CONF_RESERVE_SOC,
    DEFAULT_DISCHARGE_THRESHOLD_W,
    DOMAIN,
    BatteryPowerSign,
)

pytestmark = pytest.mark.usefixtures("enable_custom_integrations")

SOC_ENTITY = "sensor.home_battery_soc"
HOME_POWER_ENTITY = "sensor.home_consumption"
BATTERY_POWER_ENTITY = "sensor.home_battery_power"


def _set_valid_states(hass: HomeAssistant) -> None:
    """Set valid source states."""
    hass.states.async_set(
        SOC_ENTITY,
        "72",
        {
            ATTR_DEVICE_CLASS: SensorDeviceClass.BATTERY,
            ATTR_UNIT_OF_MEASUREMENT: PERCENTAGE,
        },
    )
    hass.states.async_set(
        HOME_POWER_ENTITY,
        "850",
        {
            ATTR_UNIT_OF_MEASUREMENT: UnitOfPower.WATT,
        },
    )
    hass.states.async_set(
        BATTERY_POWER_ENTITY,
        "0.9",
        {
            ATTR_UNIT_OF_MEASUREMENT: UnitOfPower.KILO_WATT,
        },
    )


def _valid_input(**updates: object) -> dict[str, object]:
    """Return valid user input with optional overrides."""
    return {
        CONF_NAME: "House battery",
        CONF_BATTERY_SOC_ENTITY: SOC_ENTITY,
        CONF_HOME_CONSUMPTION_ENTITY: HOME_POWER_ENTITY,
        CONF_BATTERY_POWER_ENTITY: BATTERY_POWER_ENTITY,
        CONF_CAPACITY_KWH: 13.5,
        CONF_RESERVE_SOC: 10,
        CONF_BATTERY_POWER_SIGN: BatteryPowerSign.POSITIVE.value,
        CONF_DISCHARGE_THRESHOLD_W: 50,
        **updates,
    }


async def _start_user_flow(hass: HomeAssistant) -> dict:
    """Start a user flow."""
    return await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )


def test_power_selector_filters_by_domain_and_unit_only() -> None:
    """Test power sensors do not need to declare a power device class."""
    assert POWER_ENTITY_SELECTOR.config["domain"] == ["sensor"]
    assert POWER_ENTITY_SELECTOR.config["filter"] == [
        {
            "domain": ["sensor"],
            "unit_of_measurement": [UnitOfPower.WATT, UnitOfPower.KILO_WATT],
        }
    ]


async def test_user_flow(hass: HomeAssistant) -> None:
    """Test a complete user flow stores normalized data."""
    _set_valid_states(hass)
    result = await _start_user_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    with patch(
        "custom_components.battery_runtime.async_setup_entry",
        return_value=True,
        create=True,
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            _valid_input(**{CONF_NAME: "  House battery  "}),
        )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "House battery"
    assert result["data"] == _valid_input()


async def test_default_threshold(hass: HomeAssistant) -> None:
    """Test the optional threshold receives its documented default."""
    _set_valid_states(hass)
    user_input = _valid_input()
    user_input.pop(CONF_DISCHARGE_THRESHOLD_W)
    result = await _start_user_flow(hass)

    with patch(
        "custom_components.battery_runtime.async_setup_entry",
        return_value=True,
        create=True,
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input
        )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_DISCHARGE_THRESHOLD_W] == DEFAULT_DISCHARGE_THRESHOLD_W


@pytest.mark.parametrize(
    ("updates", "field", "error"),
    [
        ({CONF_NAME: "   "}, CONF_NAME, "empty_name"),
        ({CONF_CAPACITY_KWH: 0}, CONF_CAPACITY_KWH, "invalid_capacity"),
        ({CONF_CAPACITY_KWH: float("inf")}, CONF_CAPACITY_KWH, "invalid_capacity"),
    ],
)
async def test_invalid_configuration_values(
    hass: HomeAssistant, updates: dict[str, object], field: str, error: str
) -> None:
    """Test non-entity configuration validation."""
    _set_valid_states(hass)
    result = await _start_user_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _valid_input(**updates)
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"][field] == error


@pytest.mark.parametrize(
    "updates",
    [
        {CONF_RESERVE_SOC: -1},
        {CONF_RESERVE_SOC: 100},
        {CONF_DISCHARGE_THRESHOLD_W: -1},
        {CONF_BATTERY_POWER_SIGN: "zero"},
    ],
)
async def test_selector_constraints(
    hass: HomeAssistant, updates: dict[str, object]
) -> None:
    """Test Home Assistant rejects values outside selector constraints."""
    _set_valid_states(hass)
    result = await _start_user_flow(hass)

    with pytest.raises(InvalidData):
        await hass.config_entries.flow.async_configure(
            result["flow_id"], _valid_input(**updates)
        )


def test_normalization_defense_in_depth() -> None:
    """Test validation still rejects malformed data if selectors are bypassed."""
    user_input = _valid_input(
        **{
            CONF_BATTERY_SOC_ENTITY: "light.not_a_sensor",
            CONF_RESERVE_SOC: 100,
            CONF_DISCHARGE_THRESHOLD_W: -1,
            CONF_BATTERY_POWER_SIGN: "zero",
        }
    )
    user_input.pop(CONF_CAPACITY_KWH)

    _, errors = _normalize_input(user_input)

    assert errors == {
        CONF_BATTERY_SOC_ENTITY: "invalid_entity",
        CONF_CAPACITY_KWH: "invalid_capacity",
        CONF_RESERVE_SOC: "invalid_reserve",
        CONF_DISCHARGE_THRESHOLD_W: "invalid_threshold",
        CONF_BATTERY_POWER_SIGN: "invalid_power_sign",
    }


def test_power_sign_multipliers() -> None:
    """Test sign values expose their discharge normalization multiplier."""
    assert BatteryPowerSign.POSITIVE.multiplier == 1
    assert BatteryPowerSign.NEGATIVE.multiplier == -1


@pytest.mark.parametrize(
    ("entity_id", "state", "attributes", "field", "error"),
    [
        (
            SOC_ENTITY,
            "unknown",
            {ATTR_UNIT_OF_MEASUREMENT: PERCENTAGE},
            CONF_BATTERY_SOC_ENTITY,
            "entity_not_numeric",
        ),
        (
            SOC_ENTITY,
            "72",
            {ATTR_UNIT_OF_MEASUREMENT: UnitOfPower.WATT},
            CONF_BATTERY_SOC_ENTITY,
            "invalid_soc_unit",
        ),
        (
            SOC_ENTITY,
            "101",
            {ATTR_UNIT_OF_MEASUREMENT: PERCENTAGE},
            CONF_BATTERY_SOC_ENTITY,
            "invalid_soc_value",
        ),
        (
            HOME_POWER_ENTITY,
            "850",
            {ATTR_UNIT_OF_MEASUREMENT: "MW"},
            CONF_HOME_CONSUMPTION_ENTITY,
            "invalid_power_unit",
        ),
        (
            HOME_POWER_ENTITY,
            "-1",
            {ATTR_UNIT_OF_MEASUREMENT: UnitOfPower.WATT},
            CONF_HOME_CONSUMPTION_ENTITY,
            "negative_home_consumption",
        ),
        (
            BATTERY_POWER_ENTITY,
            "nan",
            {ATTR_UNIT_OF_MEASUREMENT: UnitOfPower.KILO_WATT},
            CONF_BATTERY_POWER_ENTITY,
            "entity_not_numeric",
        ),
    ],
)
async def test_invalid_entity_state(
    hass: HomeAssistant,
    entity_id: str,
    state: str,
    attributes: dict[str, str],
    field: str,
    error: str,
) -> None:
    """Test source state and unit validation."""
    _set_valid_states(hass)
    hass.states.async_set(entity_id, state, attributes)
    result = await _start_user_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _valid_input()
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"][field] == error


async def test_missing_entity(hass: HomeAssistant) -> None:
    """Test that an entity must currently exist."""
    _set_valid_states(hass)
    hass.states.async_remove(SOC_ENTITY)
    result = await _start_user_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _valid_input()
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"][CONF_BATTERY_SOC_ENTITY] == "entity_not_found"


async def test_duplicate_sources_abort(hass: HomeAssistant) -> None:
    """Test exact duplicate source entities cannot be configured twice."""
    _set_valid_states(hass)
    MockConfigEntry(domain=DOMAIN, title="Existing", data=_valid_input()).add_to_hass(
        hass
    )
    result = await _start_user_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _valid_input(**{CONF_NAME: "Duplicate"})
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_duplicate_battery_with_different_home_source_aborts(
    hass: HomeAssistant,
) -> None:
    """Test a battery cannot be duplicated by selecting another home sensor."""
    _set_valid_states(hass)
    alternate_home = "sensor.alternate_home_consumption"
    hass.states.async_set(
        alternate_home, "900", {ATTR_UNIT_OF_MEASUREMENT: UnitOfPower.WATT}
    )
    MockConfigEntry(domain=DOMAIN, title="Existing", data=_valid_input()).add_to_hass(
        hass
    )
    result = await _start_user_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        _valid_input(**{CONF_HOME_CONSUMPTION_ENTITY: alternate_home}),
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_reconfigure(hass: HomeAssistant) -> None:
    """Test reconfiguration replaces data, title, and schedules a reload."""
    _set_valid_states(hass)
    entry = MockConfigEntry(domain=DOMAIN, title="Old name", data=_valid_input())
    entry.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "entry_id": entry.entry_id,
        },
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"

    updated = _valid_input(
        **{
            CONF_NAME: "Updated battery",
            CONF_RESERVE_SOC: 20,
            CONF_BATTERY_POWER_SIGN: BatteryPowerSign.NEGATIVE.value,
        }
    )
    with patch.object(hass.config_entries, "async_schedule_reload") as schedule_reload:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], updated
        )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.title == "Updated battery"
    assert entry.data == updated
    schedule_reload.assert_called_once_with(entry.entry_id)


async def test_reconfigure_rejects_duplicate(hass: HomeAssistant) -> None:
    """Test reconfiguration cannot take another entry's source entities."""
    _set_valid_states(hass)
    other_soc = "sensor.other_battery_soc"
    other_home = "sensor.other_home_consumption"
    other_power = "sensor.other_battery_power"
    hass.states.async_set(other_soc, "60", {ATTR_UNIT_OF_MEASUREMENT: PERCENTAGE})
    hass.states.async_set(
        other_home, "700", {ATTR_UNIT_OF_MEASUREMENT: UnitOfPower.WATT}
    )
    hass.states.async_set(
        other_power, "600", {ATTR_UNIT_OF_MEASUREMENT: UnitOfPower.WATT}
    )

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Other battery",
        data=_valid_input(
            **{
                CONF_BATTERY_SOC_ENTITY: other_soc,
                CONF_HOME_CONSUMPTION_ENTITY: other_home,
                CONF_BATTERY_POWER_ENTITY: other_power,
            }
        ),
    )
    entry.add_to_hass(hass)
    MockConfigEntry(
        domain=DOMAIN, title="House battery", data=_valid_input()
    ).add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={
            "source": config_entries.SOURCE_RECONFIGURE,
            "entry_id": entry.entry_id,
        },
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], _valid_input(**{CONF_NAME: "Conflict"})
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "already_configured"}
    assert entry.title == "Other battery"
