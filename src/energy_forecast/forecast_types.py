from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math


@dataclass(frozen=True, slots=True)
class ForecastPoint:
    start: datetime
    end: datetime
    value: float | None
    estimated: bool = False
    source: str | None = None

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Forecast interval timestamps must be timezone-aware")
        if self.end <= self.start:
            raise ValueError("Forecast interval end must follow start")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("Forecast values must be finite or null")


@dataclass(frozen=True, slots=True)
class ForecastSeries:
    kind: str
    unit: str
    points: tuple[ForecastPoint, ...]
    generated_at: datetime
    site_timezone: str
    source: str
    completeness: float

    def __post_init__(self) -> None:
        if self.generated_at.tzinfo is None:
            raise ValueError("Forecast generation time must be timezone-aware")
        if not 0.0 <= self.completeness <= 1.0:
            raise ValueError("Completeness must be between zero and one")
        previous_end: datetime | None = None
        for point in self.points:
            if previous_end is not None and point.start < previous_end:
                raise ValueError("Forecast intervals must be ordered and non-overlapping")
            previous_end = point.end

    def energy_kwh(self) -> float | None:
        """Integrate interval-average W into kWh; missing slots invalidate the total."""
        total = 0.0
        for point in self.points:
            if point.value is None:
                return None
            hours = (point.end - point.start).total_seconds() / 3600.0
            total += point.value * hours / 1000.0
        return total


@dataclass(frozen=True, slots=True)
class BatteryForecast:
    points: tuple[ForecastPoint, ...]
    battery_surplus_min_kwh: float | None
    battery_minimum_soc_pct: float | None
    battery_minimum_at: datetime | None
    non_free_grid_import_kwh: float | None
    status: str
    error: str | None = None
