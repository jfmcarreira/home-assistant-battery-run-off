"""Home Battery Runtime integration lifecycle."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import PLATFORMS, STORAGE_KEY_PREFIX, STORAGE_VERSION
from .coordinator import BatteryRuntimeCoordinator

type BatteryRuntimeConfigEntry = ConfigEntry[BatteryRuntimeCoordinator]


async def async_setup_entry(
    hass: HomeAssistant, entry: BatteryRuntimeConfigEntry
) -> bool:
    """Set up Home Battery Runtime from a config entry."""
    coordinator = BatteryRuntimeCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()

    entry.runtime_data = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: BatteryRuntimeConfigEntry
) -> bool:
    """Unload a Home Battery Runtime config entry."""
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False

    await entry.runtime_data.async_shutdown()
    return True


async def async_remove_entry(
    hass: HomeAssistant, entry: BatteryRuntimeConfigEntry
) -> None:
    """Remove persisted learning data for a deleted config entry."""
    await Store(
        hass,
        STORAGE_VERSION,
        f"{STORAGE_KEY_PREFIX}.{entry.entry_id}",
    ).async_remove()


__all__ = [
    "BatteryRuntimeConfigEntry",
    "async_remove_entry",
    "async_setup_entry",
    "async_unload_entry",
]
