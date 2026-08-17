"""Tests for versioned Home Battery Runtime aggregate storage."""

import asyncio
from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

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
    STORAGE_KEY_PREFIX,
    STORAGE_VERSION,
    BatteryPowerSign,
)
from custom_components.battery_runtime.coordinator import BatteryRuntimeCoordinator

pytestmark = pytest.mark.usefixtures("enable_custom_integrations")

NOW = datetime(2026, 8, 17, 12, tzinfo=UTC)


class FakeStore:
    """Small Store test double retaining delayed and immediate records."""

    def __init__(self, loaded: object = None) -> None:
        self.loaded = loaded
        self.delayed: list[tuple[object, float]] = []
        self.saved: list[dict] = []

    async def async_load(self):
        """Return configured stored data."""
        return self.loaded

    def async_delay_save(self, data_func, delay: float) -> None:
        """Record one delayed write without running its timer."""
        self.delayed.append((data_func, delay))

    async def async_save(self, data: dict) -> None:
        """Record immediate flush data."""
        self.saved.append(data)


def _data(**updates: object) -> dict[str, object]:
    """Return config data suitable for storage identity tests."""
    return {
        CONF_NAME: "House battery",
        CONF_BATTERY_SOC_ENTITY: "sensor.battery_soc",
        CONF_HOME_CONSUMPTION_ENTITY: "sensor.home_power",
        CONF_BATTERY_POWER_ENTITY: "sensor.battery_power",
        CONF_CAPACITY_KWH: 10.0,
        CONF_RESERVE_SOC: 10.0,
        CONF_BATTERY_POWER_SIGN: BatteryPowerSign.POSITIVE.value,
        CONF_DISCHARGE_THRESHOLD_W: 50.0,
        **updates,
    }


def _coordinator(
    hass: HomeAssistant,
    store: FakeStore,
    **updates: object,
) -> BatteryRuntimeCoordinator:
    """Build a coordinator backed by a fake Store."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="House battery",
        data=_data(**updates),
    )
    entry.add_to_hass(hass)
    with patch(
        "custom_components.battery_runtime.coordinator.Store",
        return_value=store,
    ):
        return BatteryRuntimeCoordinator(hass, entry)


def test_store_uses_version_and_one_key_per_entry(hass: HomeAssistant) -> None:
    """Coordinator constructs a versioned Store scoped to its config entry."""
    entry = MockConfigEntry(domain=DOMAIN, title="House battery", data=_data())

    with patch("custom_components.battery_runtime.coordinator.Store") as store_cls:
        BatteryRuntimeCoordinator(hass, entry)

    assert store_cls.call_args.args[1] == STORAGE_VERSION
    assert store_cls.call_args.args[2] == f"{STORAGE_KEY_PREFIX}.{entry.entry_id}"


async def test_aggregate_model_survives_round_trip_and_capacity_change(
    hass: HomeAssistant,
) -> None:
    """Profile aggregates restore while non-identity forecast settings may change."""
    first_store = FakeStore()
    first = _coordinator(hass, first_store)
    assert first.model.observe(NOW, 900, 1000)
    first._schedule_save()

    await first.async_flush_storage()

    assert len(first_store.saved) == 1
    stored = first_store.saved[0]
    assert stored["schema_version"] == STORAGE_VERSION
    assert "profile" not in stored
    assert "model" in stored
    assert stored["model"]["last_sample"] == NOW.isoformat()

    second = _coordinator(
        hass,
        FakeStore(stored),
        **{
            CONF_CAPACITY_KWH: 20.0,
            CONF_RESERVE_SOC: 20.0,
            CONF_DISCHARGE_THRESHOLD_W: 100.0,
        },
    )
    assert await second._async_restore_model()
    assert second.model.config.capacity_kwh == 20
    assert second.model.config.reserve_soc_percent == 20
    assert second.model.profile.cell_at(NOW).sample_count == 1
    assert second.model.profile.cell_at(NOW).mean == pytest.approx(900)


@pytest.mark.parametrize(
    "mutate_identity",
    [
        lambda identity: identity.update(timezone="Europe/Lisbon"),
        lambda identity: identity.update(battery_power_sign="negative"),
        lambda identity: identity["source_entities"].update(
            home_consumption_entity="sensor.new_home_power"
        ),
    ],
)
async def test_timezone_source_or_sign_change_invalidates_storage(
    hass: HomeAssistant, mutate_identity
) -> None:
    """Every profile compatibility field invalidates stale learned data."""
    original = _coordinator(hass, FakeStore())
    original.model.observe(NOW, 900, 1000)
    stored = original._storage_data()
    incompatible = deepcopy(stored)
    mutate_identity(incompatible["identity"])
    incompatible_store = FakeStore(incompatible)
    restored = _coordinator(hass, incompatible_store)

    assert not await restored._async_restore_model()
    assert restored.model.last_sample is None
    assert restored.model.profile.observed_cell_count() == 0
    restored._schedule_save()
    assert len(incompatible_store.delayed) == 1


@pytest.mark.parametrize(
    "stored",
    [
        "not-a-record",
        {"schema_version": STORAGE_VERSION + 1},
        {
            "schema_version": STORAGE_VERSION,
            "identity": {},
            "model": {},
            "last_save": None,
        },
    ],
)
async def test_malformed_or_unsupported_storage_is_safely_rebuilt(
    hass: HomeAssistant, stored: object
) -> None:
    """Malformed and unsupported records never prevent coordinator setup."""
    store = FakeStore(stored)
    coordinator = _coordinator(hass, store)

    assert not await coordinator._async_restore_model()
    assert coordinator.model.last_sample is None
    coordinator._schedule_save()
    assert len(store.delayed) == 1


async def test_storage_load_exception_disables_session_persistence(
    hass: HomeAssistant,
) -> None:
    """A failed Store read cannot be overwritten by delayed or unload writes."""
    store = FakeStore()

    async def _failed_load():
        raise RuntimeError("temporary storage failure")

    store.async_load = _failed_load
    coordinator = _coordinator(hass, store)
    with patch.object(
        coordinator,
        "_async_bootstrap_recorder",
        AsyncMock(return_value=0),
    ) as bootstrap:
        await coordinator._async_setup()

    bootstrap.assert_awaited_once_with()
    assert coordinator.model.observe(NOW, 500, 600)
    assert coordinator.model.profile.cell_at(NOW).mean == 500
    coordinator._schedule_save()
    await coordinator.async_shutdown()

    assert not coordinator._persistence_enabled
    assert not coordinator._storage_load_initialized
    assert store.delayed == []
    assert store.saved == []


async def test_batched_write_and_idempotent_unload_flush(hass: HomeAssistant) -> None:
    """Repeated changes share one delayed write and unload flushes only once."""
    store = FakeStore()
    coordinator = _coordinator(hass, store)
    coordinator.model.observe(NOW, 500, 600)

    coordinator._schedule_save()
    coordinator._schedule_save()
    assert len(store.delayed) == 1

    await coordinator.async_shutdown()
    await coordinator.async_shutdown()

    assert len(store.saved) == 1
    assert store.saved[0]["model"]["last_sample"] == NOW.isoformat()


async def test_flush_uses_identity_captured_before_entry_update(
    hass: HomeAssistant,
) -> None:
    """An old coordinator cannot persist aggregates under reconfigured identity."""
    store = FakeStore()
    coordinator = _coordinator(hass, store)
    original_identity = coordinator._storage_identity()
    coordinator.model.observe(NOW, 500, 600)
    coordinator._schedule_save()

    hass.config_entries.async_update_entry(
        coordinator.entry,
        data=_data(
            **{
                CONF_HOME_CONSUMPTION_ENTITY: "sensor.reconfigured_home_power",
                CONF_BATTERY_POWER_SIGN: BatteryPowerSign.NEGATIVE.value,
            }
        ),
    )
    await coordinator.async_shutdown()

    assert store.saved[-1]["identity"] == original_identity


async def test_shutdown_saves_after_delayed_callback_consumed_dirty(
    hass: HomeAssistant,
) -> None:
    """Final awaited save closes the delayed Store callback/write race."""
    store = FakeStore()
    coordinator = _coordinator(hass, store)
    assert not await coordinator._async_restore_model()
    coordinator.model.observe(NOW, 500, 600)
    coordinator._schedule_save()
    delayed_data = store.delayed[0][0]()
    assert not coordinator._dirty
    assert delayed_data["model"]["last_sample"] == NOW.isoformat()

    await coordinator.async_shutdown()
    await coordinator.async_shutdown()

    assert len(store.saved) == 1
    assert store.saved[0]["model"] == delayed_data["model"]


async def test_cancelled_initial_storage_load_does_not_overwrite_history(
    hass: HomeAssistant,
) -> None:
    """Unload during a pending initial Store read must not save an empty model."""
    load_started = asyncio.Event()
    never_complete = asyncio.Event()
    store = FakeStore()

    async def _blocked_load():
        load_started.set()
        await never_complete.wait()

    store.async_load = _blocked_load
    coordinator = _coordinator(hass, store)
    setup_task = asyncio.create_task(coordinator._async_setup())
    await load_started.wait()

    setup_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await setup_task
    await coordinator.async_shutdown()

    assert not coordinator._storage_load_initialized
    assert store.saved == []


def test_delayed_write_serializes_latest_aggregate_batch(hass: HomeAssistant) -> None:
    """Delayed Store callback snapshots all samples accumulated in its batch."""
    store = FakeStore()
    coordinator = _coordinator(hass, store)
    coordinator.model.observe(NOW, 500, 600)
    coordinator._schedule_save()
    coordinator.model.observe(NOW.replace(minute=17), 1000, 1100)

    data_func = store.delayed[0][0]
    saved = data_func()

    assert saved["model"]["last_sample"] == NOW.replace(minute=17).isoformat()
    assert not coordinator._dirty
