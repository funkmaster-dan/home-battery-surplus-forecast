from __future__ import annotations

from datetime import date, time
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, model_validator


EntityId = Annotated[str, Field(pattern=r"^sensor\.[a-z0-9_]+$")]


class SensorSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: EntityId
    unit: str = Field(min_length=1, max_length=32)
    semantics: Literal[
        "instantaneous_power",
        "interval_average_power",
        "interval_energy",
        "cumulative_energy",
    ]
    interval_minutes: int = Field(default=5, ge=1, le=1440)

    @model_validator(mode="after")
    def validate_unit_semantics(self) -> SensorSelection:
        unit = self.unit.strip().lower().replace(" ", "")
        power_units = {"w", "kw"}
        energy_units = {"wh", "kwh", "mwh"}
        if self.semantics in {"instantaneous_power", "interval_average_power"} and unit not in power_units:
            raise ValueError("Power semantics require a W or kW unit")
        if self.semantics in {"interval_energy", "cumulative_energy"} and unit not in energy_units:
            raise ValueError("Energy semantics require a Wh, kWh, or MWh unit")
        return self


class ConsumptionSensor(SensorSelection):
    historical_statistic_id: str = Field(min_length=1, max_length=256)
    direct_whole_home_confirmed: bool = False

    @model_validator(mode="after")
    def require_direct_whole_home_confirmation(self) -> ConsumptionSensor:
        if not self.direct_whole_home_confirmed:
            raise ValueError("Confirm that this is direct whole-home consumption, not grid import/export")
        return self


class PVArrayConfig(SensorSelection):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,48}$")
    name: str = Field(min_length=1, max_length=80)
    tilt_deg: float = Field(ge=0, le=90)
    azimuth_deg: float = Field(ge=-180, le=180)


class BatteryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: EntityId
    unit: str = Field(min_length=1, max_length=32)
    state_type: Literal["soc_percent", "energy_kwh"]
    capacity_kwh: float = Field(gt=0, le=100_000)
    max_charge_kw: float = Field(gt=0, le=100_000)
    max_discharge_kw: float = Field(gt=0, le=100_000)
    charge_efficiency: float = Field(gt=0, le=1)
    discharge_efficiency: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_state_unit(self) -> BatteryConfig:
        unit = self.unit.strip().lower().replace(" ", "")
        if self.state_type == "soc_percent" and unit not in {"%", "percent"}:
            raise ValueError("SOC battery state requires a percent unit")
        if self.state_type == "energy_kwh" and unit not in {"wh", "kwh", "mwh"}:
            raise ValueError("Energy battery state requires a Wh, kWh, or MWh unit")
        return self


class GridImportWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weekday: int = Field(ge=0, le=6, description="Monday is 0; Sunday is 6")
    start: time
    end: time
    target_soc_pct: float = Field(gt=0, le=100)
    grid_charge_kw: float = Field(gt=0, le=100_000)

    @model_validator(mode="after")
    def require_nonempty_window(self) -> GridImportWindow:
        if self.start == self.end:
            raise ValueError("Grid-import window start and end must differ")
        return self


class ForecastConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str = Field(min_length=1, max_length=128)
    elevation_m: float | None = Field(default=None, ge=-500, le=9000)
    calibration_start: date
    calibration_end: date
    horizon_hours: int = Field(default=48, ge=24, le=48)
    consumption: ConsumptionSensor
    battery: BatteryConfig
    solar_arrays: list[PVArrayConfig] = Field(min_length=1)
    temperature_entity_id: EntityId | None = None
    temperature_unit: Literal["°C", "°F"] = "°C"
    grid_import_windows: list[GridImportWindow] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ranges_and_ids(self) -> ForecastConfig:
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be a valid IANA time zone") from exc
        if self.calibration_start >= self.calibration_end:
            raise ValueError("calibration_end must be later than calibration_start")
        array_ids = [array.id for array in self.solar_arrays]
        if len(array_ids) != len(set(array_ids)):
            raise ValueError("PV array IDs must be unique")
        array_sensors = [array.entity_id for array in self.solar_arrays]
        if len(array_sensors) != len(set(array_sensors)):
            raise ValueError("Each PV array must use a distinct production sensor")
        return self

    def entity_ids(self) -> list[str]:
        ids = [self.consumption.entity_id, self.battery.entity_id]
        ids.extend(array.entity_id for array in self.solar_arrays)
        if self.temperature_entity_id:
            ids.append(self.temperature_entity_id)
        return list(dict.fromkeys(ids))
