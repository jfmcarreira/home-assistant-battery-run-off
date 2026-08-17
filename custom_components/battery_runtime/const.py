"""Constants for the Home Battery Runtime integration."""

from datetime import timedelta
from enum import StrEnum
from typing import Final

from homeassistant.const import Platform, UnitOfPower

DOMAIN: Final = "battery_runtime"
INTEGRATION_NAME: Final = "Home Battery Runtime"

PLATFORMS: Final = (Platform.SENSOR,)

CONF_BATTERY_SOC_ENTITY: Final = "battery_soc_entity"
CONF_HOME_CONSUMPTION_ENTITY: Final = "home_consumption_entity"
CONF_BATTERY_POWER_ENTITY: Final = "battery_power_entity"
CONF_CAPACITY_KWH: Final = "capacity_kwh"
CONF_RESERVE_SOC: Final = "reserve_soc"
CONF_BATTERY_POWER_SIGN: Final = "battery_power_sign"
CONF_DISCHARGE_THRESHOLD_W: Final = "discharge_threshold_w"

SOURCE_ENTITY_KEYS: Final = (
    CONF_BATTERY_SOC_ENTITY,
    CONF_HOME_CONSUMPTION_ENTITY,
    CONF_BATTERY_POWER_ENTITY,
)
BATTERY_IDENTITY_KEYS: Final = (
    CONF_BATTERY_SOC_ENTITY,
    CONF_BATTERY_POWER_ENTITY,
)


class BatteryPowerSign(StrEnum):
    """Sign conventions supported by battery power sensors."""

    POSITIVE = "positive"
    NEGATIVE = "negative"

    @property
    def multiplier(self) -> int:
        """Return the multiplier that normalizes discharge to positive watts."""
        return 1 if self is BatteryPowerSign.POSITIVE else -1


class PredictionStatus(StrEnum):
    """Reasons a prediction is or is not available."""

    DISCHARGING = "discharging"
    IDLE = "idle"
    CHARGING = "charging"
    LEARNING = "learning"
    SOURCE_UNAVAILABLE = "source_unavailable"
    BEYOND_HORIZON = "beyond_horizon"


class PredictionConfidence(StrEnum):
    """Prediction confidence levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


DEFAULT_BATTERY_POWER_SIGN: Final = BatteryPowerSign.POSITIVE
DEFAULT_DISCHARGE_THRESHOLD_W: Final = 50.0

MIN_SOC_PERCENT: Final = 0.0
MAX_SOC_PERCENT: Final = 100.0
MIN_RESERVE_SOC_PERCENT: Final = 0.0
MAX_RESERVE_SOC_PERCENT: Final = 99.0
SUPPORTED_POWER_UNITS: Final[frozenset[str]] = frozenset(
    (UnitOfPower.WATT, UnitOfPower.KILO_WATT)
)

RECORDER_HISTORY_DAYS: Final = 28
PROFILE_SLOT_MINUTES: Final = 15
PROFILE_SLOTS_PER_DAY: Final = 96
SAMPLE_INTERVAL: Final = timedelta(minutes=5)
FORECAST_STEP: Final = timedelta(minutes=5)
FORECAST_HORIZON: Final = timedelta(hours=48)
CURRENT_POWER_BLEND_DURATION: Final = timedelta(minutes=30)
CURRENT_LOAD_ADJUSTMENT_DECAY: Final = timedelta(hours=2)

STORAGE_VERSION: Final = 1
STORAGE_KEY_PREFIX: Final = DOMAIN
