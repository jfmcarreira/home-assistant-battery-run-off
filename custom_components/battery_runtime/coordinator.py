"""Runtime orchestration for Home Battery Runtime."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    EVENT_CORE_CONFIG_UPDATE,
    PERCENTAGE,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfPower,
)
from homeassistant.core import Event, HomeAssistant, State, callback
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import PowerConverter

from .const import (
    CONF_BATTERY_POWER_ENTITY,
    CONF_BATTERY_POWER_SIGN,
    CONF_BATTERY_SOC_ENTITY,
    CONF_CAPACITY_KWH,
    CONF_DISCHARGE_THRESHOLD_W,
    CONF_HOME_CONSUMPTION_ENTITY,
    CONF_RESERVE_SOC,
    DEFAULT_BATTERY_POWER_SIGN,
    DEFAULT_DISCHARGE_THRESHOLD_W,
    RECORDER_HISTORY_DAYS,
    SAMPLE_INTERVAL,
    SOURCE_ENTITY_KEYS,
    STORAGE_KEY_PREFIX,
    STORAGE_VERSION,
)
from .const import (
    BatteryPowerSign as ConfiguredBatteryPowerSign,
)
from .model import (
    BatteryPowerSign as ModelBatteryPowerSign,
)
from .model import (
    BatteryRuntimeModel,
    ModelConfig,
    PredictionInput,
    PredictionResult,
    PredictionStatus,
    power_to_watts,
)

_LOGGER = logging.getLogger(__name__)

ACTIVE_PREDICTION_STATUSES = frozenset(
    (
        PredictionStatus.DISCHARGING,
        PredictionStatus.LEARNING,
        PredictionStatus.BEYOND_HORIZON,
    )
)
ACTIVE_REFRESH_INTERVAL = timedelta(minutes=1)
RECALCULATION_DEBOUNCE_SECONDS = 1.0
STORAGE_SAVE_DELAY_SECONDS = 15 * 60


class BatteryRuntimeCoordinator(DataUpdateCoordinator[PredictionResult]):
    """Coordinate source states, profile learning, and prediction updates."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize one config-entry coordinator."""
        self.entry = entry
        self.soc_entity_id = entry.data[CONF_BATTERY_SOC_ENTITY]
        self.home_power_entity_id = entry.data[CONF_HOME_CONSUMPTION_ENTITY]
        self.battery_power_entity_id = entry.data[CONF_BATTERY_POWER_ENTITY]
        self.source_entity_ids = (
            self.soc_entity_id,
            self.home_power_entity_id,
            self.battery_power_entity_id,
        )
        self._source_identity = dict(
            zip(SOURCE_ENTITY_KEYS, self.source_entity_ids, strict=True)
        )
        self.timezone_name = hass.config.time_zone
        self.configured_power_sign = ConfiguredBatteryPowerSign(
            entry.data.get(
                CONF_BATTERY_POWER_SIGN,
                DEFAULT_BATTERY_POWER_SIGN.value,
            )
        )
        self.model = BatteryRuntimeModel(self._model_config())
        self._store: Store[dict[str, Any]] = Store(
            hass,
            STORAGE_VERSION,
            f"{STORAGE_KEY_PREFIX}.{entry.entry_id}",
        )
        self._unsubscribers: list[Callable[[], None]] = []
        self._setup_complete = False
        self._stopped = False
        self._reload_requested = False
        self._storage_load_initialized = False
        self._persistence_enabled = True
        self._dirty = False
        self.last_save: datetime | None = None

        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=f"{entry.title} battery runtime",
            update_interval=None,
            request_refresh_debouncer=Debouncer(
                hass,
                _LOGGER,
                cooldown=RECALCULATION_DEBOUNCE_SECONDS,
                immediate=False,
            ),
        )

    def _model_config(self) -> ModelConfig:
        """Translate config-entry values to the pure model boundary."""
        return ModelConfig(
            capacity_kwh=float(self.entry.data[CONF_CAPACITY_KWH]),
            reserve_soc_percent=float(self.entry.data[CONF_RESERVE_SOC]),
            timezone_name=self.timezone_name,
            battery_power_sign=ModelBatteryPowerSign(
                self.configured_power_sign.multiplier
            ),
            discharge_threshold_w=float(
                self.entry.data.get(
                    CONF_DISCHARGE_THRESHOLD_W,
                    DEFAULT_DISCHARGE_THRESHOLD_W,
                )
            ),
        )

    async def _async_setup(self) -> None:
        """Restore or bootstrap learning and attach runtime listeners."""
        if self._setup_complete or self._stopped:
            return

        restored = await self._async_restore_model()
        if not restored:
            await self._async_bootstrap_recorder()
            self._schedule_save()

        if self._stopped:
            return
        self._unsubscribers.extend(
            (
                async_track_state_change_event(
                    self.hass,
                    self.source_entity_ids,
                    self._async_source_state_changed,
                ),
                async_track_time_interval(
                    self.hass,
                    self._async_live_sample,
                    SAMPLE_INTERVAL,
                    name=f"{self.entry.title} battery runtime learning",
                    cancel_on_shutdown=True,
                ),
                self.hass.bus.async_listen(
                    EVENT_CORE_CONFIG_UPDATE,
                    self._async_core_config_updated,
                ),
            )
        )
        self._setup_complete = True
        self._async_core_config_updated(None)

    async def _async_update_data(self) -> PredictionResult:
        """Read current source states and calculate a prediction."""
        now = dt_util.utcnow()
        soc_percent, home_load_w, battery_power_w = self._read_sources()
        self.model.update_current_conditions(now, home_load_w, battery_power_w)
        result = self.model.predict(
            PredictionInput(
                timestamp=now,
                soc_percent=soc_percent,
                home_load_w=home_load_w,
                battery_power_w=battery_power_w,
            )
        )
        self.update_interval = (
            ACTIVE_REFRESH_INTERVAL
            if result.status in ACTIVE_PREDICTION_STATUSES
            else None
        )
        return result

    def _read_sources(self) -> tuple[float | None, float | None, float | None]:
        """Read and normalize all source entities at one point in time."""
        return (
            self._read_soc(self.hass.states.get(self.soc_entity_id)),
            self._read_power(
                self.hass.states.get(self.home_power_entity_id), non_negative=True
            ),
            self._read_power(self.hass.states.get(self.battery_power_entity_id)),
        )

    @staticmethod
    def _read_soc(state: State | None) -> float | None:
        """Return a valid percentage state, otherwise ``None``."""
        if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            return None
        if state.attributes.get(ATTR_UNIT_OF_MEASUREMENT) != PERCENTAGE:
            return None
        try:
            value = float(state.state)
        except TypeError, ValueError:
            return None
        return value if math.isfinite(value) and 0 <= value <= 100 else None

    @staticmethod
    def _read_power(state: State | None, *, non_negative: bool = False) -> float | None:
        """Return a finite W value from a W or kW source state."""
        if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            return None
        unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
        if unit not in (UnitOfPower.WATT, UnitOfPower.KILO_WATT):
            return None
        try:
            value = power_to_watts(float(state.state), unit)
        except TypeError, ValueError:
            return None
        if non_negative and value < 0:
            return None
        return value

    async def _async_source_state_changed(self, event: Event) -> None:
        """Debounce a prediction refresh after any source state change."""
        if self._stopped:
            return
        if event.data.get("new_state") is None:
            _LOGGER.warning(
                "Configured source entity %s was removed; predictions will remain "
                "unavailable until it is restored or the entry is reconfigured",
                event.data.get("entity_id"),
            )
        await self.async_request_refresh()

    @callback
    def _async_core_config_updated(self, _event: Event | None) -> None:
        """Reload once when profile slot timezone compatibility changes."""
        if (
            self._stopped
            or self._reload_requested
            or self.hass.config.time_zone == self.timezone_name
        ):
            return
        self._reload_requested = True
        _LOGGER.info(
            "Home Assistant timezone changed from %s to %s; reloading %s",
            self.timezone_name,
            self.hass.config.time_zone,
            self.entry.title,
        )
        self.hass.config_entries.async_schedule_reload(self.entry.entry_id)

    async def _async_live_sample(self, now: datetime) -> None:
        """Learn one aggregate profile observation every five minutes."""
        if self._stopped:
            return
        _, home_load_w, battery_power_w = self._read_sources()
        if self.model.observe(now, home_load_w, battery_power_w):
            self._schedule_save()
        await self.async_request_refresh()

    async def _async_bootstrap_recorder(self) -> int:
        """Bootstrap profile aggregates from supported Recorder statistics APIs."""
        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=RECORDER_HISTORY_DAYS)
        statistic_ids = {
            self.home_power_entity_id,
            self.battery_power_entity_id,
        }
        requested_units = {PowerConverter.UNIT_CLASS: UnitOfPower.WATT}
        try:
            statistics = await get_instance(self.hass).async_add_executor_job(
                statistics_during_period,
                self.hass,
                start_time,
                end_time,
                statistic_ids,
                "5minute",
                requested_units,
                {"mean"},
            )
        except Exception:  # Recorder absence or damaged statistics is non-fatal.
            _LOGGER.warning(
                "Unable to bootstrap %s from Recorder; starting with live learning",
                self.entry.title,
                exc_info=True,
            )
            return 0

        if not isinstance(statistics, Mapping):
            _LOGGER.warning(
                "Recorder returned malformed statistics for %s; starting with live "
                "learning",
                self.entry.title,
            )
            return 0

        battery_by_start: dict[float, float] = {}
        battery_rows = statistics.get(self.battery_power_entity_id, ())
        if isinstance(battery_rows, list | tuple):
            for row in battery_rows:
                parsed = self._parse_statistics_row(row, non_negative=False)
                if parsed is not None:
                    timestamp, value = parsed
                    battery_by_start[timestamp.timestamp()] = value

        observations = 0
        home_rows = statistics.get(self.home_power_entity_id, ())
        if not isinstance(home_rows, list | tuple):
            return 0
        parsed_home_rows = [
            parsed
            for row in home_rows
            if (parsed := self._parse_statistics_row(row, non_negative=True))
            is not None
        ]
        for timestamp, home_load_w in sorted(parsed_home_rows):
            battery_power_w = battery_by_start.get(timestamp.timestamp())
            if self.model.observe(timestamp, home_load_w, battery_power_w):
                observations += 1

        if observations:
            _LOGGER.debug(
                "Bootstrapped %s with %s Recorder observations",
                self.entry.title,
                observations,
            )
        return observations

    @staticmethod
    def _parse_statistics_row(
        row: object, *, non_negative: bool
    ) -> tuple[datetime, float] | None:
        """Validate one Recorder row already requested in watts."""
        if not isinstance(row, Mapping):
            return None
        start = row.get("start")
        mean = row.get("mean")
        try:
            value = float(mean)
        except TypeError, ValueError:
            return None
        if not math.isfinite(value) or (non_negative and value < 0):
            return None

        if isinstance(start, datetime):
            if start.tzinfo is None or start.utcoffset() is None:
                return None
            timestamp = start.astimezone(UTC)
        elif isinstance(start, bool) or not isinstance(start, int | float):
            return None
        else:
            try:
                timestamp = datetime.fromtimestamp(float(start), UTC)
            except OverflowError, OSError, ValueError:
                return None
        return timestamp, value

    async def _async_restore_model(self) -> bool:
        """Restore compatible aggregate model data from Store."""
        try:
            stored = await self._store.async_load()
        except Exception:
            self._persistence_enabled = False
            _LOGGER.warning(
                "Unable to load stored learning data for %s; continuing with "
                "in-memory learning and persistence disabled for this session",
                self.entry.title,
                exc_info=True,
            )
            return False

        self._storage_load_initialized = True
        if stored is None:
            return False
        try:
            if not isinstance(stored, Mapping):
                raise ValueError("stored record is not a mapping")
            if stored.get("schema_version") != STORAGE_VERSION:
                raise ValueError("unsupported stored schema version")
            if stored.get("identity") != self._storage_identity():
                _LOGGER.info(
                    "Stored learning data for %s does not match its current sources, "
                    "power sign, or timezone; rebuilding it",
                    self.entry.title,
                )
                return False
            model = BatteryRuntimeModel.from_dict(
                self._model_config(), stored.get("model")
            )
            last_save = self._parse_stored_datetime(stored.get("last_save"))
        except TypeError, ValueError:
            _LOGGER.warning(
                "Stored learning data for %s is malformed or unsupported; rebuilding it",
                self.entry.title,
                exc_info=True,
            )
            return False

        self.model = model
        self.last_save = last_save
        return True

    def _storage_identity(self) -> dict[str, object]:
        """Return fields whose changes invalidate learned aggregate data."""
        return {
            "timezone": self.timezone_name,
            "source_entities": dict(self._source_identity),
            "battery_power_sign": self.configured_power_sign.value,
        }

    @staticmethod
    def _parse_stored_datetime(value: object) -> datetime | None:
        """Parse an optional aware ISO datetime from storage."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("last_save must be an ISO datetime string")
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("last_save must be timezone-aware")
        return parsed

    @callback
    def _schedule_save(self) -> None:
        """Schedule one batched Store write for all pending observations."""
        if not self._persistence_enabled or self._stopped or self._dirty:
            return
        self._dirty = True
        self._store.async_delay_save(
            self._storage_data_for_delayed_save,
            STORAGE_SAVE_DELAY_SECONDS,
        )

    @callback
    def _storage_data_for_delayed_save(self) -> dict[str, Any]:
        """Build delayed Store data and mark the current batch consumed."""
        self._dirty = False
        return self._storage_data()

    def _storage_data(self) -> dict[str, Any]:
        """Serialize only aggregate model state and compatibility metadata."""
        self.last_save = dt_util.utcnow()
        return {
            "schema_version": STORAGE_VERSION,
            "identity": self._storage_identity(),
            "model": self.model.to_dict(),
            "last_save": self.last_save.isoformat(),
        }

    async def async_flush_storage(self, *, force: bool = False) -> None:
        """Immediately persist any pending aggregate changes."""
        if not self._persistence_enabled:
            self._dirty = False
            return
        should_force_save = force and self._storage_load_initialized
        if not self._dirty and not should_force_save:
            return
        self._dirty = False
        await self._store.async_save(self._storage_data())

    async def async_shutdown(self) -> None:
        """Idempotently cancel runtime work and flush pending learning."""
        if self._stopped:
            return
        self._stopped = True
        while self._unsubscribers:
            self._unsubscribers.pop()()
        await super().async_shutdown()
        await self.async_flush_storage(force=True)


__all__ = ["BatteryRuntimeCoordinator"]
