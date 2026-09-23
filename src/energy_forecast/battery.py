from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
import math
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

from .config import GridImportWindow
from .forecast_types import BatteryForecast, ForecastPoint, ForecastSeries


class BatterySurplusCalculator:
    """Simulate PV, household load, fixed import windows, and battery limits."""

    @staticmethod
    def calculate(
        solar: ForecastSeries,
        load: ForecastSeries,
        current_battery_state: float | dict[str, Any],
        battery_config: Any,
        grid_import_windows: Sequence[GridImportWindow | dict[str, Any]],
        horizon_hours: int,
    ) -> BatteryForecast:
        try:
            return _calculate(solar, load, current_battery_state, battery_config, grid_import_windows, horizon_hours)
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            return _unavailable(f"Invalid battery configuration or state: {exc}")


def _calculate(
    solar: ForecastSeries,
    load: ForecastSeries,
    current_battery_state: float | dict[str, Any],
    battery_config: Any,
    windows: Sequence[GridImportWindow | dict[str, Any]],
    horizon_hours: int,
) -> BatteryForecast:
    capacity = _number(battery_config, "capacity_kwh")
    max_charge_kw = _number(battery_config, "max_charge_kw")
    max_discharge_kw = _number(battery_config, "max_discharge_kw")
    charge_efficiency = _number(battery_config, "charge_efficiency")
    discharge_efficiency = _number(battery_config, "discharge_efficiency")
    timezone_name = solar.site_timezone
    zone = ZoneInfo(timezone_name)
    if not (math.isfinite(capacity) and capacity > 0):
        return _unavailable("Battery capacity must be positive")
    if not (math.isfinite(max_charge_kw) and max_charge_kw > 0 and math.isfinite(max_discharge_kw) and max_discharge_kw > 0):
        return _unavailable("Battery charge and discharge limits must be positive")
    if not (0 < charge_efficiency <= 1 and 0 < discharge_efficiency <= 1):
        return _unavailable("Battery efficiencies must be in (0, 1]")
    energy = _initial_energy(current_battery_state, capacity)
    if energy is None or not math.isfinite(energy) or energy < 0 or energy > capacity:
        return _unavailable("Current battery state is outside configured capacity")
    if horizon_hours <= 0:
        return _unavailable("Forecast horizon must be positive")
    if solar.unit != "W" or load.unit != "W":
        return _unavailable("Battery simulation requires interval-average W forecasts")
    if solar.site_timezone != load.site_timezone:
        return _unavailable("Solar and load time zones do not match")
    solar_points = solar.points
    load_points = load.points
    if not solar_points or len(solar_points) != len(load_points):
        return _unavailable("Solar and load forecasts do not cover the same intervals")
    expected_count = horizon_hours * 12
    if len(solar_points) != expected_count:
        return _unavailable("Forecast interval count does not match the requested horizon")
    step = timedelta(minutes=5)
    previous_end = None
    for pv, home in zip(solar_points, load_points, strict=True):
        if pv.start != home.start or pv.end != home.end or pv.end - pv.start != step:
            return _unavailable("Solar and load forecasts must align to five-minute intervals")
        if previous_end is not None and pv.start != previous_end:
            return _unavailable("Battery forecast contains a missing five-minute interval")
        previous_end = pv.end
        if pv.value is None or home.value is None:
            return _unavailable("Battery forecast unavailable because a required power slot is missing")
        if not math.isfinite(pv.value) or not math.isfinite(home.value) or pv.value < 0 or home.value < 0:
            return _unavailable("Battery forecast contains invalid or negative power")
    normalized_windows = [_window_dict(window) for window in windows]
    for window in normalized_windows:
        if not 0 <= window["weekday"] <= 6 or window["start"] == window["end"]:
            return _unavailable("Grid-import window is invalid")
        if not 0 < window["target_soc_pct"] <= 100 or window["grid_charge_kw"] <= 0:
            return _unavailable("Grid-import windows require a target SOC and positive charge limit")
    hours = step.total_seconds() / 3600.0
    start_energy = energy
    minimum_energy = energy
    minimum_at = solar_points[0].start
    non_free_import_kwh = 0.0
    trajectory: list[ForecastPoint] = []
    for pv_point, load_point in zip(solar_points, load_points, strict=True):
        pv_kw = max(0.0, float(pv_point.value)) / 1000.0
        load_kw = max(0.0, float(load_point.value)) / 1000.0
        local = pv_point.start.astimezone(zone)
        active_windows = [window for window in normalized_windows if _window_active(local, window)]
        free_window = bool(active_windows)
        surplus_kw = max(0.0, pv_kw - load_kw)
        deficit_kw = max(0.0, load_kw - pv_kw)
        charge_input_kw = min(surplus_kw, max_charge_kw)
        if charge_input_kw > 0:
            input_energy = min(charge_input_kw * hours, (capacity - energy) / charge_efficiency)
            energy += max(0.0, input_energy) * charge_efficiency
            charge_input_kw = max(0.0, input_energy / hours)
        if free_window:
            # All load import during the configured window is free for the forecast metric.
            target_soc = max(window["target_soc_pct"] for window in active_windows)
            grid_charge_limit = min(window["grid_charge_kw"] for window in active_windows)
            target_energy = capacity * target_soc / 100.0
            remaining_charge_limit = max(0.0, max_charge_kw - charge_input_kw)
            grid_charge_kw = min(grid_charge_limit, remaining_charge_limit)
            if target_energy > energy and grid_charge_kw > 0:
                required_input_kw = (target_energy - energy) / (charge_efficiency * hours)
                grid_charge_kw = min(grid_charge_kw, required_input_kw)
                energy += grid_charge_kw * charge_efficiency * hours
        elif deficit_kw > 0:
            requested_discharge_kw = min(deficit_kw, max_discharge_kw)
            deliverable_kw = energy * discharge_efficiency / hours
            delivered_kw = min(requested_discharge_kw, deliverable_kw)
            energy -= delivered_kw / discharge_efficiency * hours
            residual_kw = max(0.0, deficit_kw - delivered_kw)
            non_free_import_kwh += residual_kw * hours
        energy = min(capacity, max(0.0, energy))
        interval_end = pv_point.end
        if energy < minimum_energy - 1e-12:
            minimum_energy = energy
            minimum_at = interval_end
        trajectory.append(
            ForecastPoint(interval_end - step, interval_end, energy, False, "battery_simulation")
        )
    minimum_soc = minimum_energy / capacity * 100.0
    return BatteryForecast(
        tuple(trajectory),
        minimum_energy,
        minimum_soc,
        minimum_at,
        non_free_import_kwh,
        "ready",
    )


def _initial_energy(state: float | dict[str, Any], capacity: float) -> float | None:
    if isinstance(state, dict):
        if state.get("energy_kwh") is not None:
            return _finite(state["energy_kwh"])
        if state.get("soc_pct") is not None:
            pct = _finite(state["soc_pct"])
            return None if pct is None else capacity * pct / 100.0
        return None
    return _finite(state)


def _number(obj: Any, key: str) -> float:
    if isinstance(obj, dict):
        return float(obj[key])
    return float(getattr(obj, key))


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _window_dict(window: GridImportWindow | dict[str, Any]) -> dict[str, Any]:
    if isinstance(window, dict):
        weekday = int(window["weekday"])
        start = _as_time(window["start"])
        end = _as_time(window["end"])
        target_soc = float(window["target_soc_pct"])
        grid_charge_kw = float(window["grid_charge_kw"])
    else:
        weekday = window.weekday
        start = window.start
        end = window.end
        target_soc = window.target_soc_pct
        grid_charge_kw = window.grid_charge_kw
    return {"weekday": weekday, "start": start, "end": end,
            "target_soc_pct": target_soc, "grid_charge_kw": grid_charge_kw}


def _as_time(value: Any) -> time:
    if isinstance(value, time):
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        return time.fromisoformat(value)
    raise ValueError("Grid window times must be local wall-clock times")


def _window_active(local: datetime, window: dict[str, Any]) -> bool:
    current_time = local.timetz().replace(tzinfo=None)
    weekday = local.weekday()
    start: time = window["start"]
    end: time = window["end"]
    if start < end:
        return weekday == window["weekday"] and start <= current_time < end
    if weekday == window["weekday"] and current_time >= start:
        return True
    previous_weekday = (window["weekday"] + 1) % 7
    return weekday == previous_weekday and current_time < end


def _unavailable(error: str) -> BatteryForecast:
    return BatteryForecast((), None, None, None, None, "unavailable", error)
