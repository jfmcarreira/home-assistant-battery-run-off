"""Tests for the Home Battery Runtime config-entry lifecycle."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.battery_runtime import (
    async_remove_entry,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.battery_runtime.const import (
    PLATFORMS,
    STORAGE_KEY_PREFIX,
    STORAGE_VERSION,
)

pytestmark = pytest.mark.usefixtures("enable_custom_integrations")


async def test_setup_entry_uses_one_shared_coordinator(
    hass: HomeAssistant, battery_entry: MockConfigEntry
) -> None:
    """Setup refreshes before exposing and forwarding the coordinator."""
    coordinator = Mock()
    coordinator.async_config_entry_first_refresh = AsyncMock()

    with (
        patch(
            "custom_components.battery_runtime.BatteryRuntimeCoordinator",
            return_value=coordinator,
        ) as coordinator_class,
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            AsyncMock(),
        ) as forward_setups,
    ):
        assert await async_setup_entry(hass, battery_entry)

    coordinator_class.assert_called_once_with(hass, battery_entry)
    coordinator.async_config_entry_first_refresh.assert_awaited_once_with()
    assert battery_entry.runtime_data is coordinator
    forward_setups.assert_awaited_once_with(battery_entry, PLATFORMS)
    assert battery_entry.update_listeners == []


async def test_failed_first_refresh_does_not_forward_platforms(
    hass: HomeAssistant, battery_entry: MockConfigEntry
) -> None:
    """A failed initial refresh leaves the platform and runtime data untouched."""
    coordinator = Mock()
    coordinator.async_config_entry_first_refresh = AsyncMock(
        side_effect=RuntimeError("refresh failed")
    )

    with (
        patch(
            "custom_components.battery_runtime.BatteryRuntimeCoordinator",
            return_value=coordinator,
        ),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            AsyncMock(),
        ) as forward_setups,
        pytest.raises(RuntimeError, match="refresh failed"),
    ):
        await async_setup_entry(hass, battery_entry)

    assert not hasattr(battery_entry, "runtime_data")
    forward_setups.assert_not_awaited()


@pytest.mark.parametrize("platform_unloaded", [True, False])
async def test_unload_entry_shuts_down_only_after_platform_unload(
    hass: HomeAssistant,
    battery_entry: MockConfigEntry,
    platform_unloaded: bool,
) -> None:
    """Coordinator listeners and storage stop only for a successful unload."""
    coordinator = Mock()
    coordinator.async_shutdown = AsyncMock()
    battery_entry.runtime_data = coordinator

    with patch.object(
        hass.config_entries,
        "async_unload_platforms",
        AsyncMock(return_value=platform_unloaded),
    ) as unload_platforms:
        result = await async_unload_entry(hass, battery_entry)

    assert result is platform_unloaded
    unload_platforms.assert_awaited_once_with(battery_entry, PLATFORMS)
    if platform_unloaded:
        coordinator.async_shutdown.assert_awaited_once_with()
    else:
        coordinator.async_shutdown.assert_not_awaited()


async def test_remove_entry_deletes_entry_scoped_storage(
    hass: HomeAssistant, battery_entry: MockConfigEntry
) -> None:
    """Deleting an entry also deletes its persisted aggregate profile."""
    store = Mock()
    store.async_remove = AsyncMock()

    with patch(
        "custom_components.battery_runtime.Store", return_value=store
    ) as store_class:
        await async_remove_entry(hass, battery_entry)

    store_class.assert_called_once_with(
        hass,
        STORAGE_VERSION,
        f"{STORAGE_KEY_PREFIX}.{battery_entry.entry_id}",
    )
    store.async_remove.assert_awaited_once_with()
