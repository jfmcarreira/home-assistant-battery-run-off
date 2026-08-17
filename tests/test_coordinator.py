"""Tests for Home Battery Runtime orchestration."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    CONF_NAME,
    EVENT_CORE_CONFIG_UPDATE,
    PERCENTAGE,
    STATE_UNAVAILABLE,
    UnitOfPower,
)
from homeassistant.core import Event, HomeAssistant
from homeassistant.util.unit_conversion import PowerConverter
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
    SAMPLE_INTERVAL,
    BatteryPowerSign,
)
from custom_components.battery_runtime.coordinator import (
    ACTIVE_REFRESH_INTERVAL,
    STORAGE_SAVE_DELAY_SECONDS,
    BatteryRuntimeCoordinator,
)
from custom_components.battery_runtime.model import PredictionStatus

pytestmark = pytest.mark.usefixtures("enable_custom_integrations")

SOC_ENTITY = "sensor.home_battery_soc"
HOME_POWER_ENTITY = "sensor.home_consumption"
BATTERY_POWER_ENTITY = "sensor.home_battery_power"
NOW = datetime(2026, 8, 17, 12, tzinfo=UTC)


def _entry_data(**updates: object) -> dict[str, object]:
    """Return valid config-entry data."""
    return {
        CONF_NAME: "House battery",
        CONF_BATTERY_SOC_ENTITY: SOC_ENTITY,
        CONF_HOME_CONSUMPTION_ENTITY: HOME_POWER_ENTITY,
        CONF_BATTERY_POWER_ENTITY: BATTERY_POWER_ENTITY,
        CONF_CAPACITY_KWH: 10.0,
        CONF_RESERVE_SOC: 10.0,
        CONF_BATTERY_POWER_SIGN: BatteryPowerSign.POSITIVE.value,
        CONF_DISCHARGE_THRESHOLD_W: 50.0,
        **updates,
    }


def _entry(hass: HomeAssistant, **updates: object) -> MockConfigEntry:
    """Create and register a test config entry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="House battery",
        data=_entry_data(**updates),
    )
    entry.add_to_hass(hass)
    return entry


def _set_states(
    hass: HomeAssistant,
    *,
    soc: str = "60",
    home: str = "0.8",
    battery: str = "1",
    home_unit: str = UnitOfPower.KILO_WATT,
    battery_unit: str = UnitOfPower.KILO_WATT,
) -> None:
    """Set source states with independently selectable power units."""
    hass.states.async_set(
        SOC_ENTITY,
        soc,
        {ATTR_UNIT_OF_MEASUREMENT: PERCENTAGE},
    )
    hass.states.async_set(
        HOME_POWER_ENTITY,
        home,
        {ATTR_UNIT_OF_MEASUREMENT: home_unit},
    )
    hass.states.async_set(
        BATTERY_POWER_ENTITY,
        battery,
        {ATTR_UNIT_OF_MEASUREMENT: battery_unit},
    )


@pytest.mark.parametrize(
    ("sign", "battery_state"),
    [
        (BatteryPowerSign.POSITIVE, "1"),
        (BatteryPowerSign.NEGATIVE, "-1"),
    ],
)
async def test_power_conversion_sign_mapping_and_active_refresh(
    hass: HomeAssistant,
    sign: BatteryPowerSign,
    battery_state: str,
) -> None:
    """W/kW conversion and config-to-model sign mapping produce equal forecasts."""
    _set_states(hass, battery=battery_state)
    coordinator = BatteryRuntimeCoordinator(
        hass,
        _entry(hass, **{CONF_BATTERY_POWER_SIGN: sign.value}),
    )

    result = await coordinator._async_update_data()

    assert result.status is PredictionStatus.LEARNING
    assert result.average_discharge_w == pytest.approx(1000)
    assert result.runtime_hours == pytest.approx(5)
    assert coordinator.update_interval == ACTIVE_REFRESH_INTERVAL


async def test_unavailable_source_clears_prediction_and_recovers(
    hass: HomeAssistant,
) -> None:
    """An unavailable upstream value is unknown rather than a stale forecast."""
    _set_states(hass)
    coordinator = BatteryRuntimeCoordinator(hass, _entry(hass))
    active = await coordinator._async_update_data()
    assert active.runtime_hours is not None

    hass.states.async_set(
        HOME_POWER_ENTITY,
        STATE_UNAVAILABLE,
        {ATTR_UNIT_OF_MEASUREMENT: UnitOfPower.KILO_WATT},
    )
    unavailable = await coordinator._async_update_data()
    assert unavailable.status is PredictionStatus.SOURCE_UNAVAILABLE
    assert unavailable.runtime_hours is None
    assert unavailable.depletion_time is None
    assert coordinator.update_interval is None

    hass.states.async_set(
        HOME_POWER_ENTITY,
        "800",
        {ATTR_UNIT_OF_MEASUREMENT: UnitOfPower.WATT},
    )
    recovered = await coordinator._async_update_data()
    assert recovered.status is PredictionStatus.LEARNING
    assert recovered.runtime_hours is not None


async def test_idle_does_not_schedule_minute_refresh(hass: HomeAssistant) -> None:
    """Periodic minute recalculation is enabled only during active discharge."""
    _set_states(hass, battery="0")
    coordinator = BatteryRuntimeCoordinator(hass, _entry(hass))

    result = await coordinator._async_update_data()

    assert result.status is PredictionStatus.IDLE
    assert coordinator.update_interval is None


async def test_setup_registers_listeners_and_shutdown_is_idempotent(
    hass: HomeAssistant,
) -> None:
    """Every source listener and learning timer is removed exactly once."""
    coordinator = BatteryRuntimeCoordinator(hass, _entry(hass))
    remove_states = Mock()
    remove_sample_timer = Mock()
    remove_config_listener = Mock()

    with (
        patch.object(coordinator, "_async_restore_model", AsyncMock(return_value=True)),
        patch(
            "custom_components.battery_runtime.coordinator."
            "async_track_state_change_event",
            return_value=remove_states,
        ) as track_states,
        patch(
            "custom_components.battery_runtime.coordinator.async_track_time_interval",
            return_value=remove_sample_timer,
        ) as track_interval,
        patch(
            "homeassistant.core.EventBus.async_listen",
            return_value=remove_config_listener,
        ) as listen,
    ):
        await coordinator._async_setup()
        await coordinator.async_shutdown()
        await coordinator.async_shutdown()

    assert track_states.call_args.args[1] == (
        SOC_ENTITY,
        HOME_POWER_ENTITY,
        BATTERY_POWER_ENTITY,
    )
    assert track_interval.call_args.args[2] == SAMPLE_INTERVAL
    assert listen.call_args.args == (
        EVENT_CORE_CONFIG_UPDATE,
        coordinator._async_core_config_updated,
    )
    remove_states.assert_called_once_with()
    remove_sample_timer.assert_called_once_with()
    remove_config_listener.assert_called_once_with()


async def test_timezone_change_schedules_one_reload(hass: HomeAssistant) -> None:
    """A changed slot timezone reloads once; unrelated core updates do nothing."""
    coordinator = BatteryRuntimeCoordinator(hass, _entry(hass))
    event = Event(EVENT_CORE_CONFIG_UPDATE, {})

    with patch.object(hass.config_entries, "async_schedule_reload") as schedule_reload:
        coordinator._async_core_config_updated(event)
        schedule_reload.assert_not_called()

        await hass.config.async_set_time_zone("Europe/Lisbon")
        coordinator._async_core_config_updated(event)
        coordinator._async_core_config_updated(event)
        await coordinator.async_shutdown()
        coordinator._async_core_config_updated(event)

    schedule_reload.assert_called_once_with(coordinator.entry.entry_id)


async def test_timezone_changed_during_restore_schedules_one_reload(
    hass: HomeAssistant,
) -> None:
    """The post-registration check catches a core update missed during restore."""
    coordinator = BatteryRuntimeCoordinator(hass, _entry(hass))

    async def _restore_model() -> bool:
        await hass.config.async_set_time_zone("Europe/Lisbon")
        return True

    with (
        patch.object(
            coordinator,
            "_async_restore_model",
            AsyncMock(side_effect=_restore_model),
        ),
        patch.object(hass.config_entries, "async_schedule_reload") as schedule_reload,
    ):
        await coordinator._async_setup()
        coordinator._async_core_config_updated(Event(EVENT_CORE_CONFIG_UPDATE, {}))
        await coordinator.async_shutdown()

    schedule_reload.assert_called_once_with(coordinator.entry.entry_id)


async def test_removed_source_logs_and_requests_debounced_refresh(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    """A removed configured entity is visible in logs and triggers recalculation."""
    coordinator = BatteryRuntimeCoordinator(hass, _entry(hass))
    event = Event(
        "state_changed",
        {
            "entity_id": HOME_POWER_ENTITY,
            "old_state": None,
            "new_state": None,
        },
    )

    with patch.object(
        coordinator, "async_request_refresh", AsyncMock()
    ) as request_refresh:
        await coordinator._async_source_state_changed(event)

    request_refresh.assert_awaited_once_with()
    assert HOME_POWER_ENTITY in caplog.text
    assert "was removed" in caplog.text


async def test_live_sampling_learns_and_batches_storage(
    hass: HomeAssistant,
) -> None:
    """Five-minute observations train aggregates without one write per sample."""
    _set_states(hass, home="750", battery="800", home_unit="W", battery_unit="W")
    coordinator = BatteryRuntimeCoordinator(hass, _entry(hass))

    with (
        patch.object(coordinator._store, "async_delay_save") as delay_save,
        patch.object(coordinator, "async_request_refresh", AsyncMock()),
    ):
        await coordinator._async_live_sample(NOW)
        await coordinator._async_live_sample(NOW + SAMPLE_INTERVAL)

    cell = coordinator.model.profile.cell_at(NOW)
    assert cell.sample_count == 2
    assert cell.mean == pytest.approx(750)
    delay_save.assert_called_once()
    assert delay_save.call_args.args[1] == STORAGE_SAVE_DELAY_SECONDS


async def test_live_sample_refresh_updates_current_smoothing_once(
    hass: HomeAssistant,
) -> None:
    """Learning remains aggregate-only; its requested refresh smooths current data."""
    _set_states(hass, home="750", battery="800", home_unit="W", battery_unit="W")
    coordinator = BatteryRuntimeCoordinator(hass, _entry(hass))
    update_current = Mock(wraps=coordinator.model.update_current_conditions)

    async def _refresh() -> None:
        await coordinator._async_update_data()

    with (
        patch.object(
            coordinator.model,
            "update_current_conditions",
            update_current,
        ),
        patch.object(
            coordinator,
            "async_request_refresh",
            AsyncMock(side_effect=_refresh),
        ),
        patch.object(coordinator._store, "async_delay_save"),
    ):
        await coordinator._async_live_sample(NOW)

    assert update_current.call_count == 1
    assert coordinator.model._recent_home_load.sample_count == 1
    assert coordinator.model._recent_discharge.sample_count == 1


async def test_recorder_bootstrap_uses_executor_and_requested_watts(
    hass: HomeAssistant,
) -> None:
    """Recorder bootstrap uses its executor API and learns converted means."""
    coordinator = BatteryRuntimeCoordinator(hass, _entry(hass))
    start = NOW.timestamp()
    statistics = {
        HOME_POWER_ENTITY: [
            {"start": start, "mean": 1200.0},
            {"start": start + 300, "mean": -1.0},
            {"start": start + 600, "mean": float("nan")},
        ],
        BATTERY_POWER_ENTITY: [
            {"start": start, "mean": 1000.0},
        ],
    }
    recorder = Mock()
    recorder.async_add_executor_job = AsyncMock(return_value=statistics)

    with patch(
        "custom_components.battery_runtime.coordinator.get_instance",
        return_value=recorder,
    ):
        observations = await coordinator._async_bootstrap_recorder()

    assert observations == 1
    call = recorder.async_add_executor_job.await_args
    assert call.args[0].__name__ == "statistics_during_period"
    assert call.args[4] == {HOME_POWER_ENTITY, BATTERY_POWER_ENTITY}
    assert call.args[5] == "5minute"
    assert call.args[6] == {PowerConverter.UNIT_CLASS: UnitOfPower.WATT}
    assert call.args[7] == {"mean"}
    assert call.args[2] < call.args[3]
    assert call.args[3] - call.args[2] == timedelta(days=28)
    assert coordinator.model.profile.cell_at(NOW).mean == pytest.approx(1200)
    assert coordinator.model.battery_share.statistics.sample_count == 1
    assert coordinator.model._recent_home_load.sample_count == 0
    assert coordinator.model._recent_discharge.sample_count == 0


async def test_missing_recorder_history_starts_with_empty_live_profile(
    hass: HomeAssistant,
) -> None:
    """No Recorder rows is a valid bootstrap outcome, not a setup failure."""
    coordinator = BatteryRuntimeCoordinator(hass, _entry(hass))
    recorder = Mock()
    recorder.async_add_executor_job = AsyncMock(return_value={})

    with patch(
        "custom_components.battery_runtime.coordinator.get_instance",
        return_value=recorder,
    ):
        observations = await coordinator._async_bootstrap_recorder()

    assert observations == 0
    assert coordinator.model.last_sample is None
