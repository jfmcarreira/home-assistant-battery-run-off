"""Diagnostics for Home Battery Runtime."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant

from . import BatteryRuntimeConfigEntry
from .const import SOURCE_ENTITY_KEYS, STORAGE_VERSION
from .model import DayType, PredictionResult, ProfileCell


def _datetime_string(value: datetime | None) -> str | None:
    """Return an ISO timestamp suitable for JSON diagnostics."""
    return value.isoformat() if value is not None else None


def _profile_summary(cells: Sequence[ProfileCell]) -> dict[str, Any]:
    """Summarize profile aggregates without exposing individual time slots."""
    observed = [cell for cell in cells if cell.sample_count > 0]
    effective_weight = sum(cell.effective_weight for cell in observed)
    average_home_load_w = (
        sum(cell.mean * cell.effective_weight for cell in observed) / effective_weight
        if effective_weight > 0
        else None
    )
    last_updates = [
        cell.last_update for cell in observed if cell.last_update is not None
    ]
    return {
        "total_slots": len(cells),
        "observed_slots": len(observed),
        "observation_count": sum(cell.sample_count for cell in observed),
        "effective_sample_weight": effective_weight,
        "average_home_load_w": average_home_load_w,
        "last_update": _datetime_string(max(last_updates, default=None)),
    }


def _prediction_summary(result: PredictionResult | None) -> dict[str, Any] | None:
    """Return only calculated forecast output."""
    if result is None:
        return None
    return {
        "status": result.status.value,
        "confidence": result.confidence.value,
        "depletion_time": _datetime_string(result.depletion_time),
        "runtime_hours": result.runtime_hours,
        "coverage_percent": result.coverage_percent,
        "average_discharge_w": result.average_discharge_w,
        "used_fallback": result.used_fallback,
    }


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: BatteryRuntimeConfigEntry
) -> dict[str, Any]:
    """Return redacted configuration and aggregate-only runtime diagnostics."""
    coordinator = entry.runtime_data
    model = coordinator.model
    profile = model.profile
    battery_share = model.battery_share.statistics

    config = dict(entry.data)
    entity_keys = {
        key
        for key, value in config.items()
        if key in SOURCE_ENTITY_KEYS
        or (key.endswith("_entity") and isinstance(value, str))
    }
    redacted_config = async_redact_data(config, entity_keys | {CONF_NAME})

    return {
        "config_entry": {
            "version": entry.version,
            "minor_version": entry.minor_version,
            "data": redacted_config,
        },
        "runtime": {
            "last_update_success": coordinator.last_update_success,
            "last_sample": _datetime_string(model.last_sample),
            "last_save": _datetime_string(coordinator.last_save),
            "prediction": _prediction_summary(coordinator.data),
        },
        "profile": {
            "storage_schema_version": STORAGE_VERSION,
            "timezone": profile.timezone_name,
            DayType.WEEKDAY.value: _profile_summary(profile.weekday_cells),
            DayType.WEEKEND.value: _profile_summary(profile.weekend_cells),
            "battery_share": {
                "value": model.battery_share.value,
                "observation_count": battery_share.sample_count,
                "effective_sample_weight": battery_share.effective_weight,
                "last_update": _datetime_string(battery_share.last_update),
            },
        },
    }


__all__ = ["async_get_config_entry_diagnostics"]
