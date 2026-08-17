"""Sensor platform for Home Battery Runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import (
    PERCENTAGE,
    EntityCategory,
    UnitOfPower,
    UnitOfTime,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from . import BatteryRuntimeConfigEntry
from .const import DOMAIN, INTEGRATION_NAME
from .coordinator import BatteryRuntimeCoordinator
from .model import (
    PredictionConfidence,
    PredictionResult,
    PredictionStatus,
)

type NativeSensorValue = str | int | float | datetime | None

_FORECAST_STATUSES = frozenset(
    (PredictionStatus.DISCHARGING, PredictionStatus.LEARNING)
)
_DIAGNOSTIC_STATUSES = _FORECAST_STATUSES | {PredictionStatus.BEYOND_HORIZON}


def _forecast_value(
    result: PredictionResult, value: NativeSensorValue
) -> NativeSensorValue:
    """Return a forecast value only while a prediction is actionable."""
    return value if result.status in _FORECAST_STATUSES else None


def _diagnostic_value(
    result: PredictionResult, value: NativeSensorValue
) -> NativeSensorValue:
    """Return forecast diagnostics only when they describe an active forecast."""
    return value if result.status in _DIAGNOSTIC_STATUSES else None


@dataclass(frozen=True, kw_only=True)
class BatteryRuntimeSensorEntityDescription(SensorEntityDescription):
    """Describe a Home Battery Runtime sensor."""

    value_fn: Callable[[PredictionResult], NativeSensorValue]


SENSOR_DESCRIPTIONS: tuple[BatteryRuntimeSensorEntityDescription, ...] = (
    BatteryRuntimeSensorEntityDescription(
        key="estimated_depletion_time",
        translation_key="estimated_depletion_time",
        device_class=SensorDeviceClass.TIMESTAMP,
        value_fn=lambda result: _forecast_value(result, result.depletion_time),
    ),
    BatteryRuntimeSensorEntityDescription(
        key="estimated_runtime",
        translation_key="estimated_runtime",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.HOURS,
        suggested_display_precision=2,
        value_fn=lambda result: _forecast_value(result, result.runtime_hours),
    ),
    BatteryRuntimeSensorEntityDescription(
        key="prediction_status",
        translation_key="prediction_status",
        device_class=SensorDeviceClass.ENUM,
        options=[status.value for status in PredictionStatus],
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
        value_fn=lambda result: result.status.value,
    ),
    BatteryRuntimeSensorEntityDescription(
        key="prediction_confidence",
        translation_key="prediction_confidence",
        device_class=SensorDeviceClass.ENUM,
        options=[confidence.value for confidence in PredictionConfidence],
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
        value_fn=lambda result: _diagnostic_value(result, result.confidence.value),
    ),
    BatteryRuntimeSensorEntityDescription(
        key="history_coverage",
        translation_key="history_coverage",
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=1,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
        value_fn=lambda result: _diagnostic_value(result, result.coverage_percent),
    ),
    BatteryRuntimeSensorEntityDescription(
        key="forecast_average_discharge",
        translation_key="forecast_average_discharge",
        device_class=SensorDeviceClass.POWER,
        native_unit_of_measurement=UnitOfPower.WATT,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=0,
        entity_category=EntityCategory.DIAGNOSTIC,
        entity_registry_enabled_default=False,
        value_fn=lambda result: _diagnostic_value(result, result.average_discharge_w),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: BatteryRuntimeConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Home Battery Runtime sensors from a config entry."""
    async_add_entities(
        BatteryRuntimeSensor(entry, description) for description in SENSOR_DESCRIPTIONS
    )


class BatteryRuntimeSensor(CoordinatorEntity[BatteryRuntimeCoordinator], SensorEntity):
    """Expose one value from the shared prediction coordinator."""

    entity_description: BatteryRuntimeSensorEntityDescription
    _attr_has_entity_name = True

    def __init__(
        self,
        entry: BatteryRuntimeConfigEntry,
        description: BatteryRuntimeSensorEntityDescription,
    ) -> None:
        """Initialize a Home Battery Runtime sensor."""
        super().__init__(entry.runtime_data)
        self.entity_description = description
        self._attr_unique_id = f"{entry.entry_id}_{description.key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer=INTEGRATION_NAME,
            model="Battery runtime estimator",
        )

    @property
    def native_value(self) -> NativeSensorValue:
        """Return the latest coordinator value without performing I/O."""
        return self.entity_description.value_fn(self.coordinator.data)


__all__ = [
    "SENSOR_DESCRIPTIONS",
    "BatteryRuntimeSensor",
    "BatteryRuntimeSensorEntityDescription",
    "async_setup_entry",
]
