from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import uuid
from typing import Any

import joblib

from .battery import BatterySurplusCalculator
from .config import ForecastConfig
from .consumption import ConsumptionForecaster, ConsumptionModelUnavailable, interpolate_temperature
from .forecast_types import BatteryForecast, ForecastPoint, ForecastSeries
from .ha_client import HomeAssistantClient, HomeAssistantError
from .history import (
    HistoryImporter,
    calibration_bounds,
    finite_number,
    normalize_power_samples,
    normalize_temperature_samples,
    parse_timestamp,
)
from .solar import SolarForecaster, SolarModelUnavailable
from .storage import Storage
from .weather import OpenMeteoWeather, WeatherSeries, WeatherUnavailable


_LOGGER = logging.getLogger(__name__)
REFRESH_SECONDS = 300
WEATHER_MAX_AGE = timedelta(minutes=60)


class ServiceError(RuntimeError):
    pass


class ForecastService:
    def __init__(self, storage: Storage | None = None) -> None:
        self.storage = storage or Storage()
        self.weather = OpenMeteoWeather(self.storage)
        saved = self.storage.get_config()
        if saved is None:
            self.config = None
        else:
            windows = saved.get("grid_import_windows")
            migrated_windows = False
            if isinstance(windows, list):
                for window in windows:
                    if isinstance(window, dict) and "weekday" in window and "weekdays" not in window:
                        window["weekdays"] = [window.pop("weekday")]
                        migrated_windows = True
            self.config = ForecastConfig.model_validate(saved)
            if migrated_windows:
                self.storage.save_config(self.config.model_dump(mode="json"))
        self.solar: SolarForecaster | None = None
        self.consumption: ConsumptionForecaster | None = None
        self.snapshot: dict[str, Any] | None = None
        self.last_calibration: dict[str, Any] | None = None
        self._refresh_task: asyncio.Task[None] | None = None
        self._calibration_task: asyncio.Task[None] | None = None
        self._active_run_id: str | None = None
        self._calibration_lock = asyncio.Lock()
        self._forecast_lock = asyncio.Lock()
        self._load_models()

    def _load_models(self) -> None:
        solar_path = self.storage.model_dir / "solar.joblib"
        load_path = self.storage.model_dir / "consumption.joblib"
        try:
            if solar_path.is_file():
                self.solar = joblib.load(solar_path)
            if load_path.is_file():
                self.consumption = joblib.load(load_path)
        except Exception:
            _LOGGER.exception("Could not load saved forecast models")
            self.solar = None
            self.consumption = None

    async def start(self) -> None:
        if self._refresh_task is None:
            self._refresh_task = asyncio.create_task(self._refresh_loop(), name="forecast-refresh")

    async def stop(self) -> None:
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None
        if self._calibration_task and not self._calibration_task.done():
            self._calibration_task.cancel()
            try:
                await self._calibration_task
            except asyncio.CancelledError:
                pass

    async def _refresh_loop(self) -> None:
        while True:
            if self.config is not None and (self.solar is not None or self.consumption is not None):
                await self.refresh_forecast()
            await asyncio.sleep(REFRESH_SECONDS)

    async def connect_home_assistant(self, base_url: str, access_token: str) -> dict[str, Any]:
        client = HomeAssistantClient(base_url, access_token)
        try:
            api = await client.validate()
            site = await client.get_site_config()
        finally:
            await client.close()
        timezone_name = site.get("time_zone")
        if not isinstance(timezone_name, str) or not timezone_name:
            raise ServiceError("Home Assistant did not provide an IANA time zone")
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(timezone_name)
        except Exception as exc:
            raise ServiceError("Home Assistant returned an invalid IANA time zone") from exc
        safe_site = {
            "latitude": site.get("latitude"),
            "longitude": site.get("longitude"),
            "elevation": site.get("elevation"),
            "time_zone": timezone_name,
            "location_name": site.get("location_name"),
        }
        self.storage.save_ha_connection(base_url, access_token, safe_site)
        return {"connected": True, "site": safe_site, "ha_version": api.get("version")}

    async def entities(self) -> dict[str, Any]:
        connection = self.storage.get_ha_connection()
        if connection is None:
            raise ServiceError("Connect Home Assistant before listing entities")
        client = HomeAssistantClient(connection["base_url"], connection["access_token"])
        try:
            states = await client.get_states()
            try:
                statistic_ids = await client.list_statistic_ids()
                statistics_error = None
            except HomeAssistantError as exc:
                statistic_ids = []
                statistics_error = str(exc)
        finally:
            await client.close()
        configured_start: datetime | None = None
        configured_end: datetime | None = None
        if self.config:
            configured_start, configured_end = calibration_bounds(
                self.config.calibration_start, self.config.calibration_end, self.config.timezone
            )
        entities: list[dict[str, Any]] = []
        for state in states:
            entity_id = state.get("entity_id")
            if not isinstance(entity_id, str) or not entity_id.startswith("sensor."):
                continue
            attributes = state.get("attributes") or {}
            state_text = str(state.get("state", ""))
            entities.append(
                {
                    "entity_id": entity_id,
                    "name": attributes.get("friendly_name", entity_id),
                    "unit": attributes.get("unit_of_measurement"),
                    "device_class": attributes.get("device_class"),
                    "state_class": attributes.get("state_class"),
                    "available": state_text not in {"unknown", "unavailable", "none", ""},
                    "current_state": state_text,
                    "last_updated": state.get("last_updated"),
                    "statistics_available": False,
                    "statistics_coverage": None,
                }
            )
        entity_by_id = {entity["entity_id"]: entity for entity in entities}

        statistics: list[dict[str, Any]] = []
        for item in statistic_ids:
            statistic_id = item.get("statistic_id")
            if not isinstance(statistic_id, str):
                continue
            statistic_entity = entity_by_id.get(statistic_id, {})
            coverage = None
            if configured_start and configured_end:
                coverage = self.storage.history_coverage(statistic_id, configured_start, configured_end)
            elif self.config and statistic_id == self.config.consumption.historical_statistic_id:
                coverage = self.storage.history_coverage(
                    statistic_id,
                    *calibration_bounds(self.config.calibration_start, self.config.calibration_end, self.config.timezone),
                )
            statistics.append(
                {
                    "statistic_id": statistic_id,
                    "name": item.get("name") or statistic_entity.get("name") or statistic_id,
                    "unit": item.get("unit_of_measurement") or item.get("unit") or statistic_entity.get("unit"),
                    "source": item.get("source"),
                    "has_mean": bool(item.get("has_mean")),
                    "has_sum": bool(item.get("has_sum")),
                    "mean_type": item.get("mean_type"),
                    "unit_class": item.get("unit_class"),
                    "coverage": coverage,
                    "selected_for_household_load": bool(
                        self.config and statistic_id == self.config.consumption.historical_statistic_id
                    ),
                }
            )
        stats_by_id = {item["statistic_id"]: item for item in statistics}
        for entity in entities:
            metadata = stats_by_id.get(entity["entity_id"])
            if metadata:
                entity["statistics_available"] = True
                entity["statistics_coverage"] = metadata["coverage"]
        return {"entities": entities, "statistics": statistics, "statistics_error": statistics_error}

    async def _validate_config_inputs(self, config: ForecastConfig) -> None:
        discovery = await self.entities()
        def canonical_unit(value: Any) -> str:
            unit = str(value or "").strip().lower().replace(" ", "")
            if unit in {"%", "percent"}:
                return "%"
            if unit in {"c", "°c", "celsius"}:
                return "°c"
            if unit in {"f", "°f", "fahrenheit"}:
                return "°f"
            return unit
        available_entities = {item["entity_id"]: item for item in discovery["entities"]}
        selected: list[tuple[str, str]] = [
            (config.consumption.entity_id, config.consumption.unit),
            (config.battery.entity_id, config.battery.unit),
            *((array.entity_id, array.unit) for array in config.solar_arrays),
        ]
        if config.temperature_entity_id:
            selected.append((config.temperature_entity_id, config.temperature_unit))
        for entity_id, configured_unit in selected:
            entity = available_entities.get(entity_id)
            if entity is None:
                raise ServiceError(f"Selected sensor {entity_id} is not present in Home Assistant")
            if canonical_unit(entity.get("unit")) != canonical_unit(configured_unit):
                raise ServiceError(f"Selected sensor {entity_id} has a different or unsupported unit")
        load_statistic_id = config.consumption.historical_statistic_id
        load_statistic = next(
            (item for item in discovery["statistics"] if item["statistic_id"] == load_statistic_id),
            None,
        )
        if load_statistic is None:
            if discovery.get("statistics_error"):
                raise ServiceError(f"Home Assistant Long Term Statistics is unavailable; {load_statistic_id} is required")
            raise ServiceError(f"Home Assistant Long Term Statistics statistic_id {load_statistic_id} was not found")
        unit = canonical_unit(load_statistic.get("unit"))
        if unit in {"w", "kw", "wh", "kwh", "mwh"}:
            return
        raise ServiceError(f"{load_statistic_id} must use a supported power or energy unit in Long Term Statistics")

    def setup_status(self) -> dict[str, Any]:
        connection = self.storage.get_ha_connection()
        config = self.storage.get_config()
        if self._active_run_id:
            calibration = self.storage.get_calibration(self._active_run_id)
        else:
            calibration = self.last_calibration
        return {
            "home_assistant_connected": connection is not None,
            "configured": config is not None,
            "calibrated": self.solar is not None or self.consumption is not None,
            "calibration": _safe_calibration(calibration),
            "forecast_available": self.snapshot is not None,
        }

    async def save_config(self, config: ForecastConfig) -> dict[str, Any]:
        if self.storage.get_ha_connection() is None:
            raise ServiceError("Connect Home Assistant before saving configuration")
        await self._validate_config_inputs(config)
        old_payload = self.storage.get_config()
        new_payload = config.model_dump(mode="json")
        training_changed = (
            old_payload is None
            or _training_signature(old_payload) != _training_signature(new_payload)
        )
        should_calibrate = training_changed
        if should_calibrate and self._active_run_id:
            active = self.storage.get_calibration(self._active_run_id)
            if active and active.get("status") in {"queued", "running"}:
                raise ServiceError("Wait for the active calibration before changing training configuration")
        self.storage.save_config(new_payload)
        self.config = config
        if training_changed:
            self.solar = None
            self.consumption = None
            (self.storage.model_dir / "solar.joblib").unlink(missing_ok=True)
            (self.storage.model_dir / "consumption.joblib").unlink(missing_ok=True)
            self.storage.clear_model_metadata()
            if self.snapshot:
                self.snapshot = {
                    **self.snapshot,
                    "status": "stale",
                    "last_error": "Training configuration changed; recalibration is required",
                }
        if should_calibrate:
            run_id = await self.start_calibration()
        else:
            run_id = None
            if self._refresh_task is not None:
                await self.refresh_forecast()
        return {"saved": True, "config": new_payload, "calibration_run_id": run_id}

    async def start_calibration(self) -> str:
        if self.config is None:
            raise ServiceError("Save a complete forecast configuration before calibration")
        if self.storage.get_ha_connection() is None:
            raise ServiceError("Connect Home Assistant before calibration")
        if self._active_run_id:
            active = self.storage.get_calibration(self._active_run_id)
            if active and active.get("status") in {"queued", "running"}:
                raise ServiceError("A calibration is already running")
        run_id = str(uuid.uuid4())
        self.storage.create_calibration(run_id)
        self._active_run_id = run_id
        self._calibration_task = asyncio.create_task(self._calibrate(run_id, self.config), name=f"calibration-{run_id}")
        return run_id

    def calibration_status(self, run_id: str) -> dict[str, Any] | None:
        result = self.storage.get_calibration(run_id)
        return _safe_calibration(result)

    async def _calibrate(self, run_id: str, config: ForecastConfig) -> None:
        async with self._calibration_lock:
            previous_solar = self.solar
            previous_consumption = self.consumption
            coverage: dict[str, Any] = {}
            client: HomeAssistantClient | None = None
            try:
                connection = self.storage.get_ha_connection()
                if connection is None:
                    raise ServiceError("Home Assistant connection was removed")
                self.storage.update_calibration(run_id, "running", 0.01, coverage)
                client = HomeAssistantClient(connection["base_url"], connection["access_token"])
                importer = HistoryImporter(client, self.storage)
                imported = await importer.import_for_config(
                    config, progress=lambda p: self.storage.update_calibration(run_id, "running", p, coverage)
                )
                coverage.update(imported["coverage"])
                warnings = list(imported["warnings"])
                self.storage.update_calibration(run_id, "running", 0.5, coverage)
                start, end = calibration_bounds(config.calibration_start, config.calibration_end, config.timezone)
                orientations = sorted({(array.tilt_deg, array.azimuth_deg) for array in config.solar_arrays})
                start_date = config.calibration_start
                end_date = config.calibration_end - timedelta(days=1)
                archive_by_source: dict[str, dict[tuple[float, float], WeatherSeries]] = {"era5": {}, "ecmwf_ifs": {}}
                archive_notes: list[str] = []
                for orientation in orientations:
                    try:
                        era5, ifs, note = await self.weather.historical_candidates(
                            config.latitude,
                            config.longitude,
                            config.timezone,
                            start_date,
                            end_date,
                            orientation[0],
                            orientation[1],
                            None,
                        )
                        archive_by_source["era5"][orientation] = era5
                        if ifs is not None:
                            archive_by_source["ecmwf_ifs"][orientation] = ifs
                        archive_notes.append(note)
                    except WeatherUnavailable as exc:
                        warnings.append(f"Historical weather unavailable for orientation {orientation}: {exc}")
                if not archive_by_source["era5"]:
                    warnings.append("Historical weather unavailable; PV calibration remains independent of load training")
                history_by_array: dict[str, tuple[ForecastPoint, ...]] = {}
                array_coverage: dict[str, dict[str, Any]] = {}
                for array in config.solar_arrays:
                    stored_samples = self.storage.get_history(array.entity_id, start, end)
                    samples = _current_array_samples(stored_samples, array.semantics, array.interval_minutes)
                    points = normalize_power_samples(samples, start, end, array.interval_minutes)
                    history_by_array[array.id] = points
                    array_coverage[array.entity_id] = _series_coverage(points, samples)
                    coverage[array.entity_id] = array_coverage[array.entity_id]
                candidates: dict[str, tuple[SolarForecaster, dict[str, dict[str, Any]]]] = {}
                for source_name in ("era5", "ecmwf_ifs"):
                    source_weather = archive_by_source[source_name]
                    if not source_weather:
                        continue
                    if source_name == "ecmwf_ifs" and len(source_weather) != len(orientations):
                        continue
                    elevation = next(iter(source_weather.values())).elevation_m
                    forecaster = SolarForecaster(config.latitude, config.longitude, elevation)
                    metrics: dict[str, dict[str, Any]] = {}
                    for array in config.solar_arrays:
                        weather_series = source_weather.get((array.tilt_deg, array.azimuth_deg))
                        if weather_series is None:
                            continue
                        try:
                            metrics[array.id] = await asyncio.to_thread(
                                forecaster.fit,
                                array.id,
                                array.tilt_deg,
                                array.azimuth_deg,
                                history_by_array[array.id],
                                weather_series,
                                config.timezone,
                            )
                        except Exception as exc:
                            warnings.append(f"PV array {array.id} unavailable: {_safe_error(exc)}")
                    candidates[source_name] = (forecaster, metrics)
                scores = {
                    source: _total_array_daily_mae(metrics, [a.id for a in config.solar_arrays])
                    for source, (_, metrics) in candidates.items()
                }
                selected_source = "era5"
                if scores.get("ecmwf_ifs") is not None and (
                    scores.get("era5") is None or scores["ecmwf_ifs"] < scores["era5"]
                ):
                    selected_source = "ecmwf_ifs"
                if selected_source not in candidates and candidates:
                    selected_source = next(iter(candidates))
                if not candidates and previous_solar and previous_solar.models:
                    previous_source = next(iter(previous_solar.models.values())).weather_source
                    if previous_source.startswith("open-meteo:"):
                        selected_source = previous_source.split(":", 1)[1]
                if selected_source in candidates:
                    selected_solar, _ = candidates[selected_source]
                    if previous_solar is not None:
                        expected_source = f"open-meteo:{selected_source}"
                        configured_ids = {array.id for array in config.solar_arrays}
                        for array_id, model in previous_solar.models.items():
                            if (
                                array_id in configured_ids
                                and array_id not in selected_solar.models
                                and model.weather_source == expected_source
                            ):
                                selected_solar.models[array_id] = model
                    self.solar = selected_solar
                solar_metrics = self.solar.metrics() if self.solar else {}
                for array in config.solar_arrays:
                    if self.solar is None or array.id not in self.solar.models:
                        solar_metrics[array.id] = {"status": "unavailable", "error": "No usable historical targets or weather"}
                for array in config.solar_arrays:
                    source_weather = archive_by_source.get(selected_source, {}).get((array.tilt_deg, array.azimuth_deg))
                    if source_weather is None:
                        continue
                    coverage[array.entity_id]["weather_source"] = source_weather.source
                    coverage[array.entity_id]["weather_elevation_m"] = source_weather.elevation_m
                    coverage[array.entity_id]["weather_elevation_source"] = source_weather.elevation_source
                coverage["solar_model_selection"] = {
                    "selected_source": selected_source,
                    "total_array_daily_kwh_mae": scores,
                    "notes": list(dict.fromkeys(archive_notes)),
                }
                self.storage.update_calibration(run_id, "running", 0.72, coverage)

                load_statistic_id = config.consumption.historical_statistic_id
                load_samples = self.storage.get_history(load_statistic_id, start, end)
                load_points = normalize_power_samples(load_samples, start, end, 60)
                coverage[load_statistic_id] = _series_coverage(load_points, load_samples)
                temperature_points: tuple[ForecastPoint, ...] = ()
                if config.temperature_entity_id:
                    temperature_samples = self.storage.get_history(config.temperature_entity_id, start, end)
                    temperature_points = normalize_temperature_samples(temperature_samples, start, end)
                    temperature_source = f"Home Assistant measured {config.temperature_entity_id}"
                    coverage[config.temperature_entity_id] = _series_coverage(temperature_points, temperature_samples)
                elif archive_by_source.get(selected_source):
                    first_weather = next(iter(archive_by_source[selected_source].values()))
                    temperature_points = interpolate_temperature(first_weather.temperature, start, end)
                    temperature_source = "Open-Meteo historical archive"
                else:
                    temperature_source = "Unavailable; fitting the non-weather baseline only"
                load_model = ConsumptionForecaster()
                try:
                    load_metrics = await asyncio.to_thread(
                        load_model.fit, load_points, temperature_points, config.timezone, load_statistic_id
                    )
                    self.consumption = load_model
                    load_metrics["temperature_source"] = temperature_source
                    load_metrics["weather_source"] = selected_source if not config.temperature_entity_id else None
                except Exception as exc:
                    if previous_consumption is not None:
                        self.consumption = previous_consumption
                        load_metrics = {
                            **(previous_consumption.metrics or {}),
                            "status": "retained_previous_model",
                            "recalibration_error": _safe_error(exc),
                        }
                    else:
                        self.consumption = None
                        load_metrics = {
                            "status": "unavailable",
                            "error": _safe_error(exc),
                            "historical_statistic_id": load_statistic_id,
                        }
                    warnings.append(f"Home consumption model unavailable: {_safe_error(exc)}")
                coverage["consumption_model"] = load_metrics
                if self.solar:
                    _atomic_joblib_dump(self.solar, self.storage.model_dir / "solar.joblib")
                    for array_id, metrics in self.solar.metrics().items():
                        self.storage.save_model_metadata(f"solar:{array_id}", selected_source, metrics)
                if self.consumption:
                    _atomic_joblib_dump(self.consumption, self.storage.model_dir / "consumption.joblib")
                    self.storage.save_model_metadata("consumption", load_statistic_id, load_metrics)
                status = "complete" if self.solar and len(self.solar.models) == len(config.solar_arrays) and self.consumption else "partial"
                coverage["warnings"] = warnings
                self.storage.update_calibration(run_id, status, 1.0, coverage)
                self.last_calibration = self.storage.get_calibration(run_id)
                self._active_run_id = None
                if self.solar or self.consumption:
                    await self.refresh_forecast()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _LOGGER.exception("Calibration %s failed", run_id)
                self.storage.update_calibration(run_id, "failed", 1.0, coverage, _safe_error(exc))
                self.last_calibration = self.storage.get_calibration(run_id)
                self._active_run_id = None
            finally:
                if client is not None:
                    await client.close()

    async def refresh_forecast(self) -> dict[str, Any] | None:
        if self.config is None or (self.solar is None and self.consumption is None):
            return self.snapshot
        async with self._forecast_lock:
            now = datetime.now(timezone.utc)
            try:
                return await self._build_forecast(now)
            except Exception as exc:
                _LOGGER.warning("Forecast refresh failed: %s", _safe_error(exc))
                if self.snapshot:
                    self.snapshot = {
                        **self.snapshot,
                        "status": "stale",
                        "last_error": _safe_error(exc),
                        "freshness": {**self.snapshot.get("freshness", {}), "refresh_error": _safe_error(exc)},
                    }
                return self.snapshot

    async def _build_forecast(self, now: datetime) -> dict[str, Any]:
        config = self.config
        if config is None:
            return self.snapshot or {}
        connection = self.storage.get_ha_connection()
        if connection is None:
            raise ServiceError("Home Assistant is not connected")
        client = HomeAssistantClient(connection["base_url"], connection["access_token"])
        try:
            states_results = await asyncio.gather(
                *(client.get_state(entity_id) for entity_id in config.entity_ids()), return_exceptions=True
            )
        finally:
            await client.close()
        states: dict[str, dict[str, Any] | None] = {}
        stale_entities: list[str] = []
        state_ages: dict[str, float | None] = {}
        for entity_id, state in zip(config.entity_ids(), states_results, strict=True):
            if isinstance(state, Exception) or not isinstance(state, dict):
                states[entity_id] = None
                stale_entities.append(entity_id)
                state_ages[entity_id] = None
                continue
            states[entity_id] = state
            timestamp = parse_timestamp(state.get("last_updated") or state.get("last_changed"))
            age = (now - timestamp).total_seconds() if timestamp else None
            state_ages[entity_id] = age / 60.0 if age is not None else None
            state_value = str(state.get("state", ""))
            if state_value in {"unknown", "unavailable", "none", ""}:
                stale_entities.append(entity_id)
        runtime_weather: dict[tuple[float, float], WeatherSeries] = {}
        weather_errors: dict[str, str] = {}
        orientations = sorted({(array.tilt_deg, array.azimuth_deg) for array in config.solar_arrays})
        for orientation in orientations:
            try:
                runtime_weather[orientation] = await self.weather.runtime(
                    config.latitude,
                    config.longitude,
                    config.timezone,
                    orientation[0],
                    orientation[1],
                    config.horizon_hours,
                    None,
                    now,
                    refresh_interval_minutes=config.open_meteo_refresh_minutes,
                )
            except WeatherUnavailable as exc:
                weather_errors[f"{orientation[0]}:{orientation[1]}"] = str(exc)
        start = now
        solar_by_array: dict[str, ForecastSeries | None] = {}
        array_energy: dict[str, float | None] = {}
        array_status: dict[str, Any] = {}
        for array in config.solar_arrays:
            model_ready = self.solar is not None and array.id in self.solar.models
            weather = runtime_weather.get((array.tilt_deg, array.azimuth_deg))
            if not model_ready or weather is None:
                solar_by_array[array.id] = None
                array_energy[array.id] = None
                array_status[array.id] = {"available": False, "error": weather_errors.get(f"{array.tilt_deg}:{array.azimuth_deg}", "Model unavailable")}
                continue
            series = self.solar.predict(array.id, weather, start, config.horizon_hours, now, config.timezone)
            solar_by_array[array.id] = series
            array_energy[array.id] = series.energy_kwh()
            array_status[array.id] = {"available": series.completeness == 1.0, **self.solar.models[array.id].metrics}
        solar_total: ForecastSeries | None = None
        if solar_by_array and all(series is not None for series in solar_by_array.values()):
            array_series = [series for series in solar_by_array.values() if series is not None]
            total_points = []
            for index, point in enumerate(array_series[0].points):
                values = [series.points[index].value for series in array_series]
                value = sum(float(item) for item in values) if all(item is not None for item in values) else None
                total_points.append(
                    ForecastPoint(point.start, point.end, value,
                                  any(series.points[index].estimated for series in array_series),
                                  "+".join(sorted({series.source for series in array_series})))
                )
            completeness = sum(point.value is not None for point in total_points) / len(total_points) if total_points else 0.0
            solar_total = ForecastSeries("solar_power", "W", tuple(total_points), now, config.timezone,
                                         "+".join(sorted({series.source for series in array_series})), completeness)
            array_energy["total"] = solar_total.energy_kwh()
        else:
            array_energy["total"] = None
        load_series: ForecastSeries | None = None
        temperature_points: tuple[ForecastPoint, ...] = ()
        if runtime_weather:
            first_weather = next(iter(runtime_weather.values()))
            temperature_points = interpolate_temperature(
                first_weather.temperature, start, start + timedelta(hours=config.horizon_hours), max_gap=timedelta(hours=2)
            )
        if self.consumption is not None:
            load_series = self.consumption.predict(
                start, config.horizon_hours, now, config.timezone, temperature_points
            )
        battery_state, battery_state_error = _current_battery_energy(states.get(config.battery.entity_id), config.battery)
        battery_fresh = config.battery.entity_id not in stale_entities and battery_state is not None
        battery_result: BatteryForecast | None = None
        if solar_total is not None and load_series is not None and battery_fresh:
            battery_result = BatterySurplusCalculator.calculate(
                solar_total,
                load_series,
                {"energy_kwh": battery_state},
                config.battery,
                config.grid_import_windows,
                config.horizon_hours,
            )
        weather_ages = [(now - series.generated_at).total_seconds() for series in runtime_weather.values()]
        weather_age = max(weather_ages) if weather_ages else None
        weather_stale_limit = max(
            WEATHER_MAX_AGE.total_seconds(),
            config.open_meteo_refresh_minutes * 60 + REFRESH_SECONDS,
        )
        weather_stale = (
            weather_age is None
            or weather_age > weather_stale_limit
            or any("cached fallback" in series.source for series in runtime_weather.values())
        )
        state_stale = bool(stale_entities)
        if weather_stale or state_stale:
            status = "stale"
        elif (
            solar_total is not None
            and solar_total.completeness == 1.0
            and load_series is not None
            and load_series.completeness == 1.0
            and battery_result is not None
            and battery_result.status == "ready"
        ):
            status = "ready"
        else:
            status = "partial"
        solar_generation = array_energy.get("total")
        home_consumption = load_series.energy_kwh() if load_series else None
        if state_stale:
            battery_result = None
        intervals: list[dict[str, Any]] = []
        profile_points = solar_total.points if solar_total else ()
        for index in range(config.horizon_hours * 12):
            start_time = profile_points[index].start if profile_points else start + timedelta(minutes=5 * index)
            end_time = start_time + timedelta(minutes=5)
            solar_value = profile_points[index].value if profile_points else None
            load_value = load_series.points[index].value if load_series and index < len(load_series.points) else None
            battery_value = (
                battery_result.points[index].value
                if battery_result and battery_result.status == "ready" and index < len(battery_result.points)
                else None
            )
            intervals.append(
                {
                    "start": _rfc3339(start_time),
                    "end": _rfc3339(end_time),
                    "solar_power_w": solar_value,
                    "home_load_power_w": load_value,
                    "battery_energy_kwh": battery_value,
                }
            )
        model_metadata = self.storage.get_model_metadata()
        snapshot = {
            "schema_version": 1,
            "generated_at": _rfc3339(now),
            "valid_until": _rfc3339(now + timedelta(minutes=10)),
            "site_timezone": config.timezone,
            "horizon_hours": config.horizon_hours,
            "status": status,
            "solar_generation_kwh": solar_generation,
            "solar_by_array_kwh": {array.id: array_energy.get(array.id) for array in config.solar_arrays},
            "home_consumption_kwh": home_consumption,
            "battery_surplus_min_kwh": battery_result.battery_surplus_min_kwh if battery_result else None,
            "battery_minimum_soc_pct": battery_result.battery_minimum_soc_pct if battery_result else None,
            "battery_minimum_at": _rfc3339(battery_result.battery_minimum_at) if battery_result and battery_result.battery_minimum_at else None,
            "non_free_grid_import_kwh": battery_result.non_free_grid_import_kwh if battery_result else None,
            "intervals_5m": intervals,
            "model_status": {
                "solar_by_array": array_status,
                "consumption": self.consumption.metrics if self.consumption else {"available": False},
                "trained_models": model_metadata,
                "battery": {
                    "available": bool(battery_result and battery_result.status == "ready"),
                    "error": battery_result.error if battery_result and battery_result.error else battery_state_error,
                },
            },
            "freshness": {
                "generated_at": _rfc3339(now),
                "home_assistant_state_age_minutes": state_ages,
                "stale_entities": stale_entities,
                "weather_age_minutes": weather_age / 60.0 if weather_age is not None else None,
                "weather_sources": {f"{key[0]}:{key[1]}": value.source for key, value in runtime_weather.items()},
                "weather_elevation": {f"{key[0]}:{key[1]}": {"meters": value.elevation_m, "source": value.elevation_source}
                                       for key, value in runtime_weather.items()},
                "weather_errors": weather_errors,
            },
            "coverage": self.last_calibration.get("coverage", {}) if self.last_calibration else {},
        }
        self.snapshot = snapshot
        return snapshot


def _current_array_samples(
    samples: list[dict[str, Any]], semantics: str, interval_minutes: int
) -> list[dict[str, Any]]:
    """Exclude cached samples imported under an earlier sensor confirmation."""
    expected_statistics_semantics = (
        "interval_average_power"
        if semantics in {"instantaneous_power", "interval_average_power"}
        else "interval_energy"
    )
    expected_resolution = interval_minutes * 60
    return [
        sample
        for sample in samples
        if (
            sample.get("source") == "raw"
            and sample.get("semantics") == semantics
            and sample.get("resolution_seconds") == expected_resolution
        )
        or (
            sample.get("source") != "raw"
            and sample.get("semantics") == expected_statistics_semantics
        )
    ]


def _training_signature(payload: dict[str, Any]) -> str:
    keys = (
        "latitude", "longitude", "timezone", "calibration_start", "calibration_end",
        "consumption", "temperature_entity_id", "temperature_unit", "solar_arrays",
    )
    value = {key: payload.get(key) for key in keys}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _total_array_daily_mae(metrics: dict[str, dict[str, Any]], expected_arrays: list[str]) -> float | None:
    if not expected_arrays or any(array_id not in metrics for array_id in expected_arrays):
        return None
    actual_maps = [metrics[array_id].get("holdout_daily_actual_kwh", {}) for array_id in expected_arrays]
    predicted_maps = [metrics[array_id].get("holdout_daily_predicted_kwh", {}) for array_id in expected_arrays]
    common_dates = set.intersection(*(set(values) for values in actual_maps)) if actual_maps else set()
    common_dates &= set.intersection(*(set(values) for values in predicted_maps)) if predicted_maps else set()
    if not common_dates:
        return None
    errors = []
    for day in sorted(common_dates):
        actual_total = sum(float(values[day]) for values in actual_maps)
        predicted_total = sum(float(values[day]) for values in predicted_maps)
        errors.append(abs(actual_total - predicted_total))
    return sum(errors) / len(errors)


def _series_coverage(points: tuple[ForecastPoint, ...], samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [point for point in points if point.value is not None]
    sources: dict[str, int] = {}
    for point in valid:
        for source in (point.source or "unknown").split("+"):
            sources[source] = sources.get(source, 0) + 1
    return {
        "records": len(samples),
        "valid_five_minute_slots": len(valid),
        "expected_five_minute_slots": len(points),
        "slot_coverage_pct": round(len(valid) / len(points) * 100, 2) if points else 0.0,
        "first_valid": _rfc3339(valid[0].start) if valid else None,
        "last_valid": _rfc3339(valid[-1].end) if valid else None,
        "sources": sources,
        "estimated_slots": sum(point.estimated for point in valid),
    }


def _current_battery_energy(state: dict[str, Any] | None, config: Any) -> tuple[float | None, str | None]:
    if not state:
        return None, "Current battery state is unavailable"
    raw = finite_number(state.get("state"))
    if raw is None:
        return None, "Current battery state is nonnumeric or unavailable"
    if config.state_type == "soc_percent":
        if not 0 <= raw <= 100:
            return None, "Current battery SOC is outside 0-100%"
        return config.capacity_kwh * raw / 100.0, None
    attributes = state.get("attributes") or {}
    unit = str(attributes.get("unit_of_measurement") or config.unit).strip().lower().replace(" ", "")
    if unit == "wh":
        raw /= 1000.0
    elif unit == "mwh":
        raw *= 1000.0
    elif unit != "kwh":
        return None, "Current battery energy unit is unsupported"
    return raw, None


def _atomic_joblib_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temp, compress=3)
    temp.chmod(0o600)
    os.replace(temp, path)


def _rfc3339(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_calibration(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {key: value.get(key) for key in ("run_id", "status", "progress", "coverage", "error", "created_at", "updated_at")}


def _safe_error(exc: BaseException) -> str:
    message = str(exc).strip()
    if not message:
        return type(exc).__name__
    return message[:500]
