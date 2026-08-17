"""Config flow for Home Battery Runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    CONF_NAME,
    PERCENTAGE,
    UnitOfEnergy,
    UnitOfPower,
)
from homeassistant.core import HomeAssistant, valid_entity_id
from homeassistant.helpers import selector

from .const import (
    BATTERY_IDENTITY_KEYS,
    CONF_BATTERY_POWER_ENTITY,
    CONF_BATTERY_POWER_SIGN,
    CONF_BATTERY_SOC_ENTITY,
    CONF_CAPACITY_KWH,
    CONF_DISCHARGE_THRESHOLD_W,
    CONF_HOME_CONSUMPTION_ENTITY,
    CONF_RESERVE_SOC,
    DEFAULT_BATTERY_POWER_SIGN,
    DEFAULT_DISCHARGE_THRESHOLD_W,
    DOMAIN,
    MAX_RESERVE_SOC_PERCENT,
    MAX_SOC_PERCENT,
    MIN_RESERVE_SOC_PERCENT,
    MIN_SOC_PERCENT,
    SOURCE_ENTITY_KEYS,
    SUPPORTED_POWER_UNITS,
    BatteryPowerSign,
)

SOC_ENTITY_SELECTOR = selector.EntitySelector(
    selector.EntitySelectorConfig(
        domain=SENSOR_DOMAIN,
        filter=[
            selector.EntityFilterSelectorConfig(
                domain=SENSOR_DOMAIN,
                device_class=SensorDeviceClass.BATTERY,
            ),
            selector.EntityFilterSelectorConfig(
                domain=SENSOR_DOMAIN,
                unit_of_measurement=PERCENTAGE,
            ),
        ],
    )
)

POWER_ENTITY_SELECTOR = selector.EntitySelector(
    selector.EntitySelectorConfig(
        domain=SENSOR_DOMAIN,
        filter=selector.EntityFilterSelectorConfig(
            domain=SENSOR_DOMAIN,
            unit_of_measurement=[UnitOfPower.WATT, UnitOfPower.KILO_WATT],
        ),
    )
)

CONFIG_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_NAME): selector.TextSelector(),
        vol.Required(CONF_BATTERY_SOC_ENTITY): SOC_ENTITY_SELECTOR,
        vol.Required(CONF_HOME_CONSUMPTION_ENTITY): POWER_ENTITY_SELECTOR,
        vol.Required(CONF_BATTERY_POWER_ENTITY): POWER_ENTITY_SELECTOR,
        vol.Required(CONF_CAPACITY_KWH): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0,
                step="any",
                mode=selector.NumberSelectorMode.BOX,
                unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
            )
        ),
        vol.Required(CONF_RESERVE_SOC): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=MIN_RESERVE_SOC_PERCENT,
                max=MAX_RESERVE_SOC_PERCENT,
                step="any",
                mode=selector.NumberSelectorMode.BOX,
                unit_of_measurement=PERCENTAGE,
            )
        ),
        vol.Required(
            CONF_BATTERY_POWER_SIGN,
            default=DEFAULT_BATTERY_POWER_SIGN.value,
        ): selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[sign.value for sign in BatteryPowerSign],
                mode=selector.SelectSelectorMode.DROPDOWN,
                translation_key="battery_power_sign",
            )
        ),
        vol.Optional(
            CONF_DISCHARGE_THRESHOLD_W,
            default=DEFAULT_DISCHARGE_THRESHOLD_W,
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0,
                step="any",
                mode=selector.NumberSelectorMode.BOX,
                unit_of_measurement=UnitOfPower.WATT,
            )
        ),
    }
)


def _normalize_input(
    user_input: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Normalize submitted data and validate non-entity fields."""
    data = dict(user_input)
    errors: dict[str, str] = {}

    name = data.get(CONF_NAME)
    if not isinstance(name, str) or not (name := name.strip()):
        errors[CONF_NAME] = "empty_name"
    else:
        data[CONF_NAME] = name

    for key in SOURCE_ENTITY_KEYS:
        entity_id = data.get(key)
        if (
            not isinstance(entity_id, str)
            or not valid_entity_id(entity_id)
            or not entity_id.startswith(f"{SENSOR_DOMAIN}.")
        ):
            errors[key] = "invalid_entity"

    numeric_fields = (
        (CONF_CAPACITY_KWH, "invalid_capacity", lambda value: value > 0),
        (
            CONF_RESERVE_SOC,
            "invalid_reserve",
            lambda value: MIN_RESERVE_SOC_PERCENT <= value <= MAX_RESERVE_SOC_PERCENT,
        ),
        (
            CONF_DISCHARGE_THRESHOLD_W,
            "invalid_threshold",
            lambda value: value >= 0,
        ),
    )
    data.setdefault(CONF_DISCHARGE_THRESHOLD_W, DEFAULT_DISCHARGE_THRESHOLD_W)
    for key, error, is_valid in numeric_fields:
        try:
            value = float(data[key])
        except KeyError, TypeError, ValueError:
            errors[key] = error
            continue
        if not math.isfinite(value) or not is_valid(value):
            errors[key] = error
            continue
        data[key] = value

    try:
        sign = BatteryPowerSign(data.get(CONF_BATTERY_POWER_SIGN))
    except TypeError, ValueError:
        errors[CONF_BATTERY_POWER_SIGN] = "invalid_power_sign"
    else:
        data[CONF_BATTERY_POWER_SIGN] = sign.value

    return data, errors


def _validate_entity_state(
    hass: HomeAssistant,
    entity_id: str,
    *,
    expected_units: frozenset[str],
    value_range: tuple[float, float] | None = None,
    non_negative: bool = False,
) -> str | None:
    """Validate that a selected entity has a usable numeric state and unit."""
    state = hass.states.get(entity_id)
    if state is None:
        return "entity_not_found"

    try:
        value = float(state.state)
    except TypeError, ValueError:
        return "entity_not_numeric"
    if not math.isfinite(value):
        return "entity_not_numeric"

    unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
    if unit not in expected_units:
        return (
            "invalid_soc_unit" if PERCENTAGE in expected_units else "invalid_power_unit"
        )

    if value_range is not None and not value_range[0] <= value <= value_range[1]:
        return "invalid_soc_value"
    if non_negative and value < 0:
        return "negative_home_consumption"
    return None


def _validate_source_entities(
    hass: HomeAssistant, data: Mapping[str, Any]
) -> dict[str, str]:
    """Validate all selected source entities against their current states."""
    validations = (
        (
            CONF_BATTERY_SOC_ENTITY,
            {
                "expected_units": frozenset((PERCENTAGE,)),
                "value_range": (MIN_SOC_PERCENT, MAX_SOC_PERCENT),
            },
        ),
        (
            CONF_HOME_CONSUMPTION_ENTITY,
            {"expected_units": SUPPORTED_POWER_UNITS, "non_negative": True},
        ),
        (
            CONF_BATTERY_POWER_ENTITY,
            {"expected_units": SUPPORTED_POWER_UNITS},
        ),
    )
    errors: dict[str, str] = {}
    for key, kwargs in validations:
        if error := _validate_entity_state(hass, data[key], **kwargs):
            errors[key] = error
    return errors


class BatteryRuntimeConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle configuration for Home Battery Runtime."""

    VERSION = 1

    def _has_duplicate_sources(
        self, data: Mapping[str, Any], *, exclude_entry_id: str | None = None
    ) -> bool:
        """Return whether another entry represents the same battery."""
        return any(
            entry.entry_id != exclude_entry_id
            and all(
                entry.data.get(key) == data.get(key) for key in BATTERY_IDENTITY_KEYS
            )
            for entry in self._async_current_entries()
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle setup initiated by a user."""
        if user_input is not None:
            data, errors = _normalize_input(user_input)
            if not errors and self._has_duplicate_sources(data):
                return self.async_abort(reason="already_configured")
            if not errors:
                errors = _validate_source_entities(self.hass, data)
            if not errors:
                return self.async_create_entry(title=data[CONF_NAME], data=data)

            return self.async_show_form(
                step_id="user",
                data_schema=self.add_suggested_values_to_schema(
                    CONFIG_SCHEMA, user_input
                ),
                errors=errors,
            )

        return self.async_show_form(step_id="user", data_schema=CONFIG_SCHEMA)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Update required configuration and reload the config entry."""
        entry = self._get_reconfigure_entry()
        suggested_values: Mapping[str, Any] = entry.data

        if user_input is not None:
            suggested_values = user_input
            data, errors = _normalize_input(user_input)
            if not errors and self._has_duplicate_sources(
                data, exclude_entry_id=entry.entry_id
            ):
                errors["base"] = "already_configured"
            if not errors:
                errors = _validate_source_entities(self.hass, data)
            if not errors:
                return self.async_update_reload_and_abort(
                    entry,
                    title=data[CONF_NAME],
                    data=data,
                )

        else:
            errors = {}

        defaults = dict(suggested_values)
        defaults.setdefault(CONF_BATTERY_POWER_SIGN, DEFAULT_BATTERY_POWER_SIGN.value)
        defaults.setdefault(CONF_DISCHARGE_THRESHOLD_W, DEFAULT_DISCHARGE_THRESHOLD_W)
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(CONFIG_SCHEMA, defaults),
            errors=errors or None,
        )
