"""Pure learning and prediction model for Home Battery Runtime.

All power values at this boundary are watts and all energy values are kWh.  The
module deliberately has no Home Assistant imports so it can be tested and used
by recorder/bootstrap code without framework state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import IntEnum, StrEnum
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SLOTS_PER_DAY = 96
SLOT_MINUTES = 15
FORECAST_STEP = timedelta(minutes=5)
FORECAST_STEP_HOURS = FORECAST_STEP.total_seconds() / 3600
MAX_FORECAST_HOURS = 48.0

PROFILE_EWMA_ALPHA = 0.2
CURRENT_EWMA_ALPHA = 0.3
CURRENT_READING_WEIGHT = 0.35
CURRENT_BLEND_HOURS = 0.5
CURRENT_ADJUSTMENT_DECAY_HOURS = 2.0
MIN_PROFILE_CELLS = 4

BATTERY_SHARE_MIN_SAMPLES = 3
BATTERY_SHARE_MIN_HOME_LOAD_W = 50.0
BATTERY_SHARE_MIN_RATIO = 0.5
BATTERY_SHARE_MAX_RATIO = 1.5
CURRENT_ADJUSTMENT_MIN = 0.5
CURRENT_ADJUSTMENT_MAX = 2.0

CONFIDENCE_MEDIUM_SCORE = 0.55
CONFIDENCE_HIGH_SCORE = 0.82
CONFIDENCE_MEDIUM_COVERAGE = 0.5
CONFIDENCE_HIGH_COVERAGE = 0.9
CONFIDENCE_FULL_EFFECTIVE_WEIGHT = 5.0
CONFIDENCE_MAX_COEFFICIENT_OF_VARIATION = 1.0

MODEL_SCHEMA_VERSION = 1


class DayType(StrEnum):
    """Profile day categories."""

    WEEKDAY = "weekday"
    WEEKEND = "weekend"


class BatteryPowerSign(IntEnum):
    """Multiplier which normalizes battery power to positive discharge."""

    POSITIVE_DISCHARGE = 1
    NEGATIVE_DISCHARGE = -1


class PredictionStatus(StrEnum):
    """Reason a prediction is or is not available."""

    DISCHARGING = "discharging"
    IDLE = "idle"
    CHARGING = "charging"
    LEARNING = "learning"
    SOURCE_UNAVAILABLE = "source_unavailable"
    BEYOND_HORIZON = "beyond_horizon"


class PredictionConfidence(StrEnum):
    """Deterministic forecast quality classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _require_finite(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return number


def _source_number(
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    try:
        return _require_finite(value, "source value", minimum=minimum, maximum=maximum)
    except ValueError:
        return None


def _require_aware(value: datetime, name: str = "timestamp") -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _parse_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO datetime string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as err:
        raise ValueError(f"{name} must be an ISO datetime string") from err
    return _require_aware(parsed, name)


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _timeline_add(value: datetime, elapsed: timedelta) -> datetime:
    """Add real elapsed time, rather than wall time, across a DST transition."""

    return (value.astimezone(UTC) + elapsed).astimezone(value.tzinfo)


def power_to_watts(value: float, unit: str) -> float:
    """Convert a finite W or kW value to watts."""

    number = _require_finite(value, "power")
    if not isinstance(unit, str):
        raise ValueError("power unit must be W or kW")
    normalized_unit = unit.strip().lower()
    if normalized_unit in {"w", "watt", "watts"}:
        return number
    if normalized_unit in {"kw", "kilowatt", "kilowatts"}:
        return number * 1000.0
    raise ValueError("power unit must be W or kW")


def normalize_discharge_power(
    battery_power_w: float, sign: BatteryPowerSign | int
) -> float:
    """Return battery power with positive values representing discharge."""

    power = _require_finite(battery_power_w, "battery_power_w")
    if isinstance(sign, bool):
        raise ValueError("sign must be 1 or -1")
    try:
        normalized_sign = BatteryPowerSign(sign)
    except (TypeError, ValueError) as err:
        raise ValueError("sign must be 1 or -1") from err
    return power * int(normalized_sign)


def available_energy_kwh(
    capacity_kwh: float, soc_percent: float, reserve_soc_percent: float
) -> float:
    """Calculate usable energy remaining above the configured reserve."""

    capacity = _require_finite(capacity_kwh, "capacity_kwh", minimum=0.0)
    if capacity == 0:
        raise ValueError("capacity_kwh must be greater than zero")
    soc = _require_finite(soc_percent, "soc_percent", minimum=0.0, maximum=100.0)
    reserve = _require_finite(
        reserve_soc_percent,
        "reserve_soc_percent",
        minimum=0.0,
        maximum=100.0,
    )
    return capacity * max(soc - reserve, 0.0) / 100.0


def constant_power_runtime_hours(
    remaining_energy_kwh: float, discharge_power_w: float
) -> float | None:
    """Return constant-power runtime, or ``None`` for non-positive power."""

    energy = _require_finite(remaining_energy_kwh, "remaining_energy_kwh", minimum=0.0)
    power = _require_finite(discharge_power_w, "discharge_power_w")
    if power <= 0:
        return None
    return energy * 1000.0 / power


@dataclass(slots=True)
class EWMAStatistics:
    """Exponentially weighted online mean and variance."""

    mean: float = 0.0
    variance: float = 0.0
    effective_weight: float = 0.0
    sample_count: int = 0
    last_update: datetime | None = None

    def update(
        self,
        value: float,
        updated_at: datetime,
        *,
        alpha: float = PROFILE_EWMA_ALPHA,
    ) -> None:
        """Add a non-negative observation using a conventional EWMA update."""

        observation = _require_finite(value, "observation", minimum=0.0)
        timestamp = _require_aware(updated_at)
        smoothing = _require_finite(alpha, "alpha", minimum=0.0, maximum=1.0)
        if smoothing == 0:
            raise ValueError("alpha must be greater than zero")

        if self.sample_count == 0:
            self.mean = observation
            self.variance = 0.0
            self.effective_weight = 1.0
        else:
            delta = observation - self.mean
            self.mean += smoothing * delta
            self.variance = max(
                (1.0 - smoothing) * (self.variance + smoothing * delta * delta),
                0.0,
            )
            self.effective_weight = (1.0 - smoothing) * self.effective_weight + 1.0
        self.sample_count += 1
        if self.last_update is None or timestamp > self.last_update:
            self.last_update = timestamp

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible aggregate state."""

        return {
            "mean": self.mean,
            "variance": self.variance,
            "effective_weight": self.effective_weight,
            "sample_count": self.sample_count,
            "last_update": (
                self.last_update.isoformat() if self.last_update is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: object) -> EWMAStatistics:
        """Load and strictly validate aggregate state."""

        record = _require_mapping(data, "EWMA statistics")
        mean = _require_finite(record.get("mean"), "mean", minimum=0.0)
        variance = _require_finite(record.get("variance"), "variance", minimum=0.0)
        effective_weight = _require_finite(
            record.get("effective_weight"), "effective_weight", minimum=0.0
        )
        sample_count_value = record.get("sample_count")
        if (
            isinstance(sample_count_value, bool)
            or not isinstance(sample_count_value, int)
            or sample_count_value < 0
        ):
            raise ValueError("sample_count must be a non-negative integer")

        last_update_value = record.get("last_update")
        last_update = (
            None
            if last_update_value is None
            else _parse_datetime(last_update_value, "last_update")
        )
        if sample_count_value == 0:
            if (
                mean != 0
                or variance != 0
                or effective_weight != 0
                or last_update is not None
            ):
                raise ValueError("empty statistics have inconsistent state")
        else:
            expected_weight = (
                1.0 - (1.0 - PROFILE_EWMA_ALPHA) ** sample_count_value
            ) / PROFILE_EWMA_ALPHA
            if (
                effective_weight < 1
                or effective_weight > sample_count_value
                or not math.isclose(
                    effective_weight,
                    expected_weight,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                or last_update is None
                or (sample_count_value == 1 and variance != 0)
            ):
                raise ValueError("populated statistics have inconsistent state")

        return cls(
            mean=mean,
            variance=variance,
            effective_weight=effective_weight,
            sample_count=sample_count_value,
            last_update=last_update,
        )


@dataclass(slots=True)
class ProfileCell(EWMAStatistics):
    """EWMA statistics for one local quarter-hour home-load slot."""


@dataclass(frozen=True, slots=True)
class ProfileEstimate:
    """A profile lookup and whether it came from the exact quarter-hour cell."""

    home_load_w: float | None
    exact: bool
    cell: ProfileCell | None = None


class ConsumptionProfile:
    """Weekday/weekend home-load profile with 96 cells per day type."""

    def __init__(
        self,
        timezone_name: str,
        *,
        weekday_cells: Sequence[ProfileCell] | None = None,
        weekend_cells: Sequence[ProfileCell] | None = None,
    ) -> None:
        if not isinstance(timezone_name, str) or not timezone_name:
            raise ValueError("timezone_name must be non-empty")
        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as err:
            raise ValueError(f"unknown timezone: {timezone_name}") from err
        self.timezone_name = timezone_name
        self.timezone = timezone
        self.weekday_cells = self._prepare_cells(weekday_cells, DayType.WEEKDAY)
        self.weekend_cells = self._prepare_cells(weekend_cells, DayType.WEEKEND)

    @staticmethod
    def _prepare_cells(
        cells: Sequence[ProfileCell] | None, day_type: DayType
    ) -> list[ProfileCell]:
        if cells is None:
            return [ProfileCell() for _ in range(SLOTS_PER_DAY)]
        if isinstance(cells, (str, bytes)) or len(cells) != SLOTS_PER_DAY:
            raise ValueError(f"{day_type.value} profile must contain 96 cells")
        prepared = list(cells)
        if not all(isinstance(cell, ProfileCell) for cell in prepared):
            raise ValueError(f"{day_type.value} profile contains an invalid cell")
        return prepared

    def day_type_and_slot(self, timestamp: datetime) -> tuple[DayType, int]:
        """Map an aware timestamp to its configured local day type and slot."""

        local = _require_aware(timestamp).astimezone(self.timezone)
        day_type = DayType.WEEKEND if local.weekday() >= 5 else DayType.WEEKDAY
        slot = (local.hour * 60 + local.minute) // SLOT_MINUTES
        return day_type, slot

    def cells_for(self, day_type: DayType) -> list[ProfileCell]:
        """Return the cells for a day category."""

        return self.weekday_cells if day_type is DayType.WEEKDAY else self.weekend_cells

    def cell_at(self, timestamp: datetime) -> ProfileCell:
        """Return the exact profile cell for a timestamp."""

        day_type, slot = self.day_type_and_slot(timestamp)
        return self.cells_for(day_type)[slot]

    def update(self, timestamp: datetime, home_load_w: float) -> ProfileCell:
        """Learn a valid non-negative home-load observation."""

        cell = self.cell_at(timestamp)
        cell.update(home_load_w, timestamp)
        return cell

    def estimate_home_load(self, timestamp: datetime) -> ProfileEstimate:
        """Estimate home load, falling back only within the matching day type.

        A missing exact slot uses the effective-weighted mean of populated cells
        for the same weekday/weekend category.  It never borrows the other day
        type; callers may then use a current-condition fallback when that entire
        category is unlearned.
        """

        day_type, slot = self.day_type_and_slot(timestamp)
        cells = self.cells_for(day_type)
        exact_cell = cells[slot]
        if exact_cell.sample_count > 0:
            return ProfileEstimate(exact_cell.mean, True, exact_cell)

        populated = [cell for cell in cells if cell.sample_count > 0]
        total_weight = sum(cell.effective_weight for cell in populated)
        if total_weight <= 0:
            return ProfileEstimate(None, False)
        mean = (
            sum(cell.mean * cell.effective_weight for cell in populated) / total_weight
        )
        return ProfileEstimate(mean, False)

    def observed_cell_count(self, day_type: DayType | None = None) -> int:
        """Return the number of exact cells containing observations."""

        if day_type is None:
            cells = self.weekday_cells + self.weekend_cells
        else:
            cells = self.cells_for(day_type)
        return sum(cell.sample_count > 0 for cell in cells)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible profile state suitable for HA Store."""

        return {
            "timezone": self.timezone_name,
            DayType.WEEKDAY.value: [cell.to_dict() for cell in self.weekday_cells],
            DayType.WEEKEND.value: [cell.to_dict() for cell in self.weekend_cells],
        }

    @classmethod
    def from_dict(cls, data: object) -> ConsumptionProfile:
        """Load and strictly validate a serialized profile."""

        record = _require_mapping(data, "profile")
        timezone_name = record.get("timezone")
        if not isinstance(timezone_name, str):
            raise ValueError("profile timezone must be a string")

        loaded: dict[DayType, list[ProfileCell]] = {}
        for day_type in DayType:
            raw_cells = record.get(day_type.value)
            if (
                not isinstance(raw_cells, Sequence)
                or isinstance(raw_cells, (str, bytes))
                or len(raw_cells) != SLOTS_PER_DAY
            ):
                raise ValueError(f"{day_type.value} profile must contain 96 cells")
            loaded[day_type] = [ProfileCell.from_dict(cell) for cell in raw_cells]

        return cls(
            timezone_name,
            weekday_cells=loaded[DayType.WEEKDAY],
            weekend_cells=loaded[DayType.WEEKEND],
        )


@dataclass(slots=True)
class BatteryShare:
    """Learn the ratio between battery discharge and measured home load."""

    statistics: EWMAStatistics = field(default_factory=EWMAStatistics)

    @property
    def value(self) -> float:
        """Return a learned ratio, or the neutral default until it is credible."""

        if self.statistics.sample_count < BATTERY_SHARE_MIN_SAMPLES:
            return 1.0
        return min(
            max(self.statistics.mean, BATTERY_SHARE_MIN_RATIO),
            BATTERY_SHARE_MAX_RATIO,
        )

    def update(
        self,
        discharge_power_w: float,
        home_load_w: float,
        updated_at: datetime,
    ) -> bool:
        """Learn a clamped ratio from an active-discharge observation."""

        discharge = _require_finite(discharge_power_w, "discharge_power_w", minimum=0.0)
        home_load = _require_finite(home_load_w, "home_load_w", minimum=0.0)
        _require_aware(updated_at)
        if discharge <= 0 or home_load < BATTERY_SHARE_MIN_HOME_LOAD_W:
            return False
        ratio = min(
            max(discharge / home_load, BATTERY_SHARE_MIN_RATIO),
            BATTERY_SHARE_MAX_RATIO,
        )
        self.statistics.update(ratio, updated_at)
        return True

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible aggregate state."""

        return self.statistics.to_dict()

    @classmethod
    def from_dict(cls, data: object) -> BatteryShare:
        """Load battery-share aggregate state."""

        statistics = EWMAStatistics.from_dict(data)
        if statistics.sample_count > 0 and not (
            BATTERY_SHARE_MIN_RATIO <= statistics.mean <= BATTERY_SHARE_MAX_RATIO
        ):
            raise ValueError("battery-share mean is outside the allowed range")
        return cls(statistics)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Validated model settings for one battery."""

    capacity_kwh: float
    reserve_soc_percent: float
    timezone_name: str
    battery_power_sign: BatteryPowerSign | int = BatteryPowerSign.POSITIVE_DISCHARGE
    discharge_threshold_w: float = 50.0

    def __post_init__(self) -> None:
        capacity = _require_finite(self.capacity_kwh, "capacity_kwh", minimum=0.0)
        if capacity == 0:
            raise ValueError("capacity_kwh must be greater than zero")
        reserve = _require_finite(
            self.reserve_soc_percent,
            "reserve_soc_percent",
            minimum=0.0,
            maximum=99.0,
        )
        threshold = _require_finite(
            self.discharge_threshold_w, "discharge_threshold_w", minimum=0.0
        )
        if isinstance(self.battery_power_sign, bool):
            raise ValueError("battery_power_sign must be 1 or -1")
        try:
            sign = BatteryPowerSign(self.battery_power_sign)
        except (TypeError, ValueError) as err:
            raise ValueError("battery_power_sign must be 1 or -1") from err
        if not isinstance(self.timezone_name, str) or not self.timezone_name:
            raise ValueError("timezone_name must be non-empty")
        try:
            ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError as err:
            raise ValueError(f"unknown timezone: {self.timezone_name}") from err

        object.__setattr__(self, "capacity_kwh", capacity)
        object.__setattr__(self, "reserve_soc_percent", reserve)
        object.__setattr__(self, "discharge_threshold_w", threshold)
        object.__setattr__(self, "battery_power_sign", sign)


@dataclass(frozen=True, slots=True)
class PredictionInput:
    """Current source readings used for one forecast."""

    timestamp: datetime
    soc_percent: float | None
    home_load_w: float | None
    battery_power_w: float | None


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Complete prediction and diagnostic output for coordinator consumers."""

    status: PredictionStatus
    confidence: PredictionConfidence
    depletion_time: datetime | None
    runtime_hours: float | None
    coverage_percent: float
    average_discharge_w: float | None
    used_fallback: bool


class BatteryRuntimeModel:
    """Own learned aggregates, recent smoothing, and runtime forecasts."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        profile: ConsumptionProfile | None = None,
        battery_share: BatteryShare | None = None,
    ) -> None:
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be ModelConfig")
        self.config = config
        self.profile = profile or ConsumptionProfile(config.timezone_name)
        if self.profile.timezone_name != config.timezone_name:
            raise ValueError("profile timezone does not match model configuration")
        self.battery_share = battery_share or BatteryShare()
        self._recent_home_load = EWMAStatistics()
        self._recent_discharge = EWMAStatistics()
        self.last_sample: datetime | None = None

    def update_current_conditions(
        self,
        timestamp: datetime,
        home_load_w: float | None,
        battery_power_w: float | None,
    ) -> bool:
        """Update recent-condition smoothing without changing profile cells.

        Discharge smoothing is scoped to one active discharge session. A valid
        idle or charging reading ends that session so its zero-normalized power
        cannot dilute the first reading when discharge starts again.
        """

        timestamp = _require_aware(timestamp)
        home_load = _source_number(home_load_w, minimum=0.0)
        battery_power = _source_number(battery_power_w)
        if home_load is None or battery_power is None:
            self._recent_home_load = EWMAStatistics()
            self._recent_discharge = EWMAStatistics()
            return False

        self._recent_home_load.update(home_load, timestamp, alpha=CURRENT_EWMA_ALPHA)
        discharge = normalize_discharge_power(
            battery_power, self.config.battery_power_sign
        )
        if discharge > self.config.discharge_threshold_w:
            self._recent_discharge.update(
                discharge, timestamp, alpha=CURRENT_EWMA_ALPHA
            )
        else:
            self._recent_discharge = EWMAStatistics()
        return True

    def observe(
        self,
        timestamp: datetime,
        home_load_w: float | None,
        battery_power_w: float | None,
    ) -> bool:
        """Learn one live or Recorder aggregate observation.

        A valid home-load value is useful even if battery power is unavailable.
        Battery-share learning occurs only above the configured discharge
        threshold. Invalid source values are ignored and never reach aggregates.
        Historical observations never affect transient current-condition state.
        """

        timestamp = _require_aware(timestamp)
        home_load = _source_number(home_load_w, minimum=0.0)
        battery_power = _source_number(battery_power_w)
        if home_load is None:
            return False

        self.profile.update(timestamp, home_load)
        if battery_power is not None:
            discharge = normalize_discharge_power(
                battery_power, self.config.battery_power_sign
            )
            if discharge > self.config.discharge_threshold_w:
                self.battery_share.update(discharge, home_load, timestamp)
        if self.last_sample is None or timestamp > self.last_sample:
            self.last_sample = timestamp
        return True

    def predict(self, inputs: PredictionInput) -> PredictionResult:
        """Predict reserve depletion from validated current source readings."""

        if not isinstance(inputs, PredictionInput):
            raise TypeError("inputs must be PredictionInput")
        now = _require_aware(inputs.timestamp)
        soc = _source_number(inputs.soc_percent, minimum=0.0, maximum=100.0)
        home_load = _source_number(inputs.home_load_w, minimum=0.0)
        battery_power = _source_number(inputs.battery_power_w)
        if soc is None or home_load is None or battery_power is None:
            return self._empty_result(PredictionStatus.SOURCE_UNAVAILABLE)

        discharge = normalize_discharge_power(
            battery_power, self.config.battery_power_sign
        )
        if discharge < -self.config.discharge_threshold_w:
            return self._empty_result(PredictionStatus.CHARGING)
        if discharge <= self.config.discharge_threshold_w:
            return self._empty_result(PredictionStatus.IDLE)

        remaining_energy = available_energy_kwh(
            self.config.capacity_kwh,
            soc,
            self.config.reserve_soc_percent,
        )
        recent_home = self._current_value(self._recent_home_load, home_load)
        recent_discharge = self._current_value(self._recent_discharge, discharge)
        if remaining_energy == 0:
            return PredictionResult(
                status=PredictionStatus.DISCHARGING,
                confidence=PredictionConfidence.LOW,
                depletion_time=now,
                runtime_hours=0.0,
                coverage_percent=0.0,
                average_discharge_w=recent_discharge,
                used_fallback=False,
            )

        return self._profile_forecast(
            now,
            remaining_energy,
            recent_home,
            recent_discharge,
            discharge,
        )

    @staticmethod
    def _empty_result(status: PredictionStatus) -> PredictionResult:
        return PredictionResult(
            status=status,
            confidence=PredictionConfidence.LOW,
            depletion_time=None,
            runtime_hours=None,
            coverage_percent=0.0,
            average_discharge_w=None,
            used_fallback=False,
        )

    @staticmethod
    def _current_value(statistics: EWMAStatistics, current: float) -> float:
        if statistics.sample_count == 0:
            return current
        return (
            1.0 - CURRENT_READING_WEIGHT
        ) * statistics.mean + CURRENT_READING_WEIGHT * current

    def _constant_power_forecast(
        self,
        now: datetime,
        remaining_energy_kwh: float,
        discharge_w: float,
    ) -> PredictionResult:
        runtime = constant_power_runtime_hours(remaining_energy_kwh, discharge_w)
        if runtime is None or runtime > MAX_FORECAST_HOURS:
            average = discharge_w if discharge_w > 0 else None
            return PredictionResult(
                status=PredictionStatus.BEYOND_HORIZON,
                confidence=PredictionConfidence.LOW,
                depletion_time=None,
                runtime_hours=None,
                coverage_percent=self._coverage_for(now, MAX_FORECAST_HOURS),
                average_discharge_w=average,
                used_fallback=True,
            )
        return PredictionResult(
            status=PredictionStatus.LEARNING,
            confidence=PredictionConfidence.LOW,
            depletion_time=_timeline_add(now, timedelta(hours=runtime)),
            runtime_hours=runtime,
            coverage_percent=self._coverage_for(now, runtime),
            average_discharge_w=discharge_w,
            used_fallback=True,
        )

    def _profile_forecast(
        self,
        now: datetime,
        remaining_energy_kwh: float,
        recent_home_w: float,
        recent_discharge_w: float,
        current_discharge_w: float,
    ) -> PredictionResult:
        share = self.battery_share.value
        expected_now = self.profile.estimate_home_load(now)
        if expected_now.home_load_w is not None and expected_now.home_load_w > 0:
            current_adjustment = min(
                max(
                    recent_home_w / expected_now.home_load_w,
                    CURRENT_ADJUSTMENT_MIN,
                ),
                CURRENT_ADJUSTMENT_MAX,
            )
            expected_current_discharge = expected_now.home_load_w * share
        else:
            current_adjustment = 1.0
            expected_current_discharge = recent_discharge_w

        cumulative_energy = 0.0
        elapsed_hours = 0.0
        exact_steps = 0
        total_steps = 0
        effective_weight_sum = 0.0
        variation_sum = 0.0

        while elapsed_hours < MAX_FORECAST_HOURS:
            midpoint_hours = elapsed_hours + FORECAST_STEP_HOURS / 2.0
            midpoint = _timeline_add(now, timedelta(hours=midpoint_hours))
            day_type, _ = self.profile.day_type_and_slot(midpoint)
            if self.profile.observed_cell_count(day_type) < MIN_PROFILE_CELLS:
                return self._constant_power_forecast(
                    now, remaining_energy_kwh, recent_discharge_w
                )
            estimate = self.profile.estimate_home_load(midpoint)
            if estimate.home_load_w is None:
                baseline_discharge = recent_discharge_w
            else:
                adjustment = 1.0 + (current_adjustment - 1.0) * math.exp(
                    -midpoint_hours / CURRENT_ADJUSTMENT_DECAY_HOURS
                )
                baseline_discharge = estimate.home_load_w * share * adjustment

            immediate_weight = max(0.0, 1.0 - midpoint_hours / CURRENT_BLEND_HOURS)
            predicted_discharge = max(
                immediate_weight * recent_discharge_w
                + (1.0 - immediate_weight) * baseline_discharge,
                0.0,
            )
            step_energy = predicted_discharge * FORECAST_STEP_HOURS / 1000.0

            total_steps += 1
            if estimate.exact and estimate.cell is not None:
                exact_steps += 1
                effective_weight_sum += estimate.cell.effective_weight
                if estimate.cell.mean > 0:
                    variation_sum += min(
                        math.sqrt(estimate.cell.variance) / estimate.cell.mean,
                        CONFIDENCE_MAX_COEFFICIENT_OF_VARIATION,
                    )
                elif estimate.cell.variance > 0:
                    variation_sum += CONFIDENCE_MAX_COEFFICIENT_OF_VARIATION

            if (
                step_energy > 0
                and cumulative_energy + step_energy >= remaining_energy_kwh
            ):
                fraction = (remaining_energy_kwh - cumulative_energy) / step_energy
                runtime = elapsed_hours + FORECAST_STEP_HOURS * fraction
                coverage = exact_steps / total_steps
                confidence = self._classify_confidence(
                    coverage,
                    effective_weight_sum / exact_steps if exact_steps else 0.0,
                    variation_sum / exact_steps if exact_steps else 1.0,
                    current_discharge_w,
                    expected_current_discharge,
                )
                return PredictionResult(
                    status=PredictionStatus.DISCHARGING,
                    confidence=confidence,
                    depletion_time=_timeline_add(now, timedelta(hours=runtime)),
                    runtime_hours=runtime,
                    coverage_percent=coverage * 100.0,
                    average_discharge_w=remaining_energy_kwh * 1000.0 / runtime,
                    used_fallback=False,
                )

            cumulative_energy += step_energy
            elapsed_hours += FORECAST_STEP_HOURS

        coverage = exact_steps / total_steps if total_steps else 0.0
        average_discharge = (
            cumulative_energy * 1000.0 / MAX_FORECAST_HOURS
            if cumulative_energy > 0
            else 0.0
        )
        confidence = self._classify_confidence(
            coverage,
            effective_weight_sum / exact_steps if exact_steps else 0.0,
            variation_sum / exact_steps if exact_steps else 1.0,
            current_discharge_w,
            expected_current_discharge,
        )
        return PredictionResult(
            status=PredictionStatus.BEYOND_HORIZON,
            confidence=confidence,
            depletion_time=None,
            runtime_hours=None,
            coverage_percent=coverage * 100.0,
            average_discharge_w=average_discharge,
            used_fallback=False,
        )

    def _coverage_for(self, now: datetime, runtime_hours: float) -> float:
        if runtime_hours <= 0:
            return 0.0
        steps = max(1, math.ceil(runtime_hours / FORECAST_STEP_HOURS))
        exact = 0
        for step in range(steps):
            midpoint = _timeline_add(
                now,
                timedelta(hours=(step + 0.5) * FORECAST_STEP_HOURS),
            )
            exact += self.profile.estimate_home_load(midpoint).exact
        return exact / steps * 100.0

    @staticmethod
    def _classify_confidence(
        coverage: float,
        average_effective_weight: float,
        average_coefficient_of_variation: float,
        current_discharge_w: float,
        expected_discharge_w: float,
    ) -> PredictionConfidence:
        coverage = min(max(coverage, 0.0), 1.0)
        sample_quality = min(
            max(average_effective_weight, 0.0) / CONFIDENCE_FULL_EFFECTIVE_WEIGHT,
            1.0,
        )
        variability_quality = 1.0 - min(
            max(average_coefficient_of_variation, 0.0)
            / CONFIDENCE_MAX_COEFFICIENT_OF_VARIATION,
            1.0,
        )
        denominator = max(current_discharge_w, expected_discharge_w, 1.0)
        agreement_quality = 1.0 - min(
            abs(current_discharge_w - expected_discharge_w) / denominator,
            1.0,
        )
        score = (
            0.40 * coverage
            + 0.25 * sample_quality
            + 0.20 * variability_quality
            + 0.15 * agreement_quality
        )
        if coverage >= CONFIDENCE_HIGH_COVERAGE and score >= CONFIDENCE_HIGH_SCORE:
            return PredictionConfidence.HIGH
        if coverage >= CONFIDENCE_MEDIUM_COVERAGE and score >= CONFIDENCE_MEDIUM_SCORE:
            return PredictionConfidence.MEDIUM
        return PredictionConfidence.LOW

    def to_dict(self) -> dict[str, object]:
        """Serialize learned model aggregates for a versioned Store record."""

        return {
            "version": MODEL_SCHEMA_VERSION,
            "profile": self.profile.to_dict(),
            "battery_share": self.battery_share.to_dict(),
            "last_sample": (
                self.last_sample.isoformat() if self.last_sample is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, config: ModelConfig, data: object) -> BatteryRuntimeModel:
        """Restore learned aggregates, rejecting malformed/incompatible data."""

        record = _require_mapping(data, "model")
        version = record.get("version")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != MODEL_SCHEMA_VERSION
        ):
            raise ValueError("unsupported model schema version")
        profile = ConsumptionProfile.from_dict(record.get("profile"))
        if profile.timezone_name != config.timezone_name:
            raise ValueError("stored profile timezone does not match configuration")
        battery_share = BatteryShare.from_dict(record.get("battery_share"))
        model = cls(config, profile=profile, battery_share=battery_share)
        last_sample_value = record.get("last_sample")
        model.last_sample = (
            None
            if last_sample_value is None
            else _parse_datetime(last_sample_value, "last_sample")
        )
        return model


__all__ = [
    "BATTERY_SHARE_MAX_RATIO",
    "BATTERY_SHARE_MIN_RATIO",
    "CONFIDENCE_HIGH_COVERAGE",
    "CONFIDENCE_HIGH_SCORE",
    "CONFIDENCE_MEDIUM_COVERAGE",
    "CONFIDENCE_MEDIUM_SCORE",
    "MAX_FORECAST_HOURS",
    "MODEL_SCHEMA_VERSION",
    "SLOTS_PER_DAY",
    "BatteryPowerSign",
    "BatteryRuntimeModel",
    "BatteryShare",
    "ConsumptionProfile",
    "DayType",
    "EWMAStatistics",
    "ModelConfig",
    "PredictionConfidence",
    "PredictionInput",
    "PredictionResult",
    "PredictionStatus",
    "ProfileCell",
    "ProfileEstimate",
    "available_energy_kwh",
    "constant_power_runtime_hours",
    "normalize_discharge_power",
    "power_to_watts",
]
