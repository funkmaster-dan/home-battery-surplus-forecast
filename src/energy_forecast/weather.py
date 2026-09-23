from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
import os
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import httpx

from .storage import Storage


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
BOM_MODEL = "bom_access_global"


@dataclass(frozen=True, slots=True)
class HourlyWeatherPoint:
    start: datetime
    temperature_at: datetime
    irradiance_wm2: float | None
    temperature_c: float | None

@dataclass(frozen=True, slots=True)
class WeatherSeries:
    points: tuple[HourlyWeatherPoint, ...]
    source: str
    generated_at: datetime
    elevation_m: float
    elevation_source: str
    request_parameters: dict[str, Any]

    @property
    def irradiance(self) -> dict[datetime, float | None]:
        return {point.start: point.irradiance_wm2 for point in self.points}

    @property
    def temperature(self) -> dict[datetime, float | None]:
        return {point.temperature_at: point.temperature_c for point in self.points}


class WeatherUnavailable(RuntimeError):
    pass


class OpenMeteoWeather:
    def __init__(self, storage: Storage, timeout: float = 45.0) -> None:
        self.storage = storage
        self.timeout = timeout

    @staticmethod
    def _cache_key(parameters: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    async def _request(self, url: str, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout), follow_redirects=False) as client:
                response = await client.get(url, params=parameters)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WeatherUnavailable("Open-Meteo request failed") from exc
        if not isinstance(payload, dict) or payload.get("error"):
            reason = payload.get("reason") if isinstance(payload, dict) else None
            raise WeatherUnavailable(str(reason or "Open-Meteo returned an invalid response"))
        return payload

    async def historical(
        self,
        latitude: float,
        longitude: float,
        timezone_name: str,
        start_date: date,
        end_date: date,
        tilt_deg: float,
        azimuth_deg: float,
        model: str,
        elevation_m: float | None = None,
    ) -> WeatherSeries:
        if model not in {"era5", "ecmwf_ifs"}:
            raise ValueError("Historical model must be era5 or ecmwf_ifs")
        zone = ZoneInfo(timezone_name)
        utc_start = datetime.combine(start_date, time.min, zone).astimezone(timezone.utc)
        utc_end = datetime.combine(end_date + timedelta(days=1), time.min, zone).astimezone(timezone.utc)
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": utc_start.date().isoformat(),
            "end_date": (utc_end - timedelta(microseconds=1)).date().isoformat(),
            "hourly": "global_tilted_irradiance,temperature_2m",
            "tilt": tilt_deg,
            "azimuth": azimuth_deg,
            "timezone": "UTC",
            "timeformat": "unixtime",
            "models": model,
        }
        if elevation_m is not None:
            params["elevation"] = elevation_m
        key_params = {"endpoint": "archive", "site_timezone": timezone_name, **params}
        key = self._cache_key(key_params)
        cached = self.storage.get_weather_cache(key, max_age_seconds=10 * 365 * 24 * 3600)
        if cached:
            return _parse_weather(cached["payload"], model, key_params)
        payload = await self._request(ARCHIVE_URL, params)
        self.storage.save_weather_cache(key, payload)
        return _parse_weather(payload, model, key_params)

    async def historical_candidates(
        self,
        latitude: float,
        longitude: float,
        timezone_name: str,
        start_date: date,
        end_date: date,
        tilt_deg: float,
        azimuth_deg: float,
        elevation_m: float | None = None,
    ) -> tuple[WeatherSeries, WeatherSeries | None, str]:
        """Fetch ERA5 and IFS when common coverage permits; always keep one consistent source."""
        era5 = await self.historical(
            latitude, longitude, timezone_name, start_date, end_date, tilt_deg, azimuth_deg, "era5", elevation_m
        )
        if start_date < date(2017, 1, 1):
            return era5, None, "Selected range predates common ERA5/IFS coverage; used ERA5 consistently"
        try:
            ifs = await self.historical(
                latitude, longitude, timezone_name, start_date, end_date, tilt_deg, azimuth_deg, "ecmwf_ifs", elevation_m
            )
        except WeatherUnavailable:
            return era5, None, "IFS unavailable for selected range; used ERA5 consistently"
        if not _complete_historical_weather(era5) or not _complete_historical_weather(ifs):
            return era5, None, "One archive source lacked required values; used ERA5 consistently"
        return era5, ifs, "Compared ERA5 and ECMWF IFS on the chronological holdout"

    async def runtime(
        self,
        latitude: float,
        longitude: float,
        timezone_name: str,
        tilt_deg: float,
        azimuth_deg: float,
        horizon_hours: int,
        elevation_m: float | None = None,
        now: datetime | None = None,
        runtime_model: str = "auto",
    ) -> WeatherSeries:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        model = runtime_model if runtime_model in {"auto", BOM_MODEL} else "auto"
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "global_tilted_irradiance,temperature_2m",
            "tilt": tilt_deg,
            "azimuth": azimuth_deg,
            "timezone": "UTC",
            "timeformat": "unixtime",
            "forecast_hours": horizon_hours + 2,
        }
        if model != "auto":
            params["models"] = model
        if elevation_m is not None:
            params["elevation"] = elevation_m
        cache_slot = now.replace(minute=(now.minute // 30) * 30, second=0, microsecond=0)
        key_params = {
            "endpoint": "forecast",
            "slot": cache_slot.strftime("%Y-%m-%dT%H:%M"),
            "site_timezone": timezone_name,
            "model": model,
            **params,
        }
        key = self._cache_key(key_params)
        cached = self.storage.get_weather_cache(key, max_age_seconds=1800)
        if cached:
            stored = cached["payload"]
            if isinstance(stored, dict) and "_open_meteo_body" in stored:
                cached_series = _parse_weather(
                    stored["_open_meteo_body"],
                    stored["model"],
                    stored["parameters"],
                    generated_at=_parse_cache_time(cached["fetched_at"]),
                )
                source = stored.get("source")
                if source:
                    return WeatherSeries(cached_series.points, source, cached_series.generated_at,
                                         cached_series.elevation_m, cached_series.elevation_source,
                                         cached_series.request_parameters)
                return cached_series
            return _parse_weather(stored, model, key_params, generated_at=_parse_cache_time(cached["fetched_at"]))
        if model == BOM_MODEL:
            try:
                payload = await self._request(FORECAST_URL, params)
                series = _parse_weather(payload, model, key_params)
                if not _complete_runtime_weather(series, now, horizon_hours):
                    raise WeatherUnavailable("BOM forecast omitted required fields")
                self.storage.save_weather_cache(
                    key, {"_open_meteo_body": payload, "model": model, "parameters": key_params}
                )
                return series
            except WeatherUnavailable:
                fallback_params = {key: value for key, value in params.items() if key != "models"}
                try:
                    payload = await self._request(FORECAST_URL, fallback_params)
                except WeatherUnavailable as exc:
                    raise WeatherUnavailable("BOM and generic Open-Meteo forecasts are unavailable") from exc
                fallback_key_params = {"endpoint": "forecast", "site_timezone": timezone_name, "model": "auto", **fallback_params}
                series = _parse_weather(payload, "auto", fallback_key_params)
                series = WeatherSeries(
                    series.points,
                    "open-meteo:auto (BOM fallback)",
                    series.generated_at,
                    series.elevation_m,
                    series.elevation_source,
                    series.request_parameters,
                )
                self.storage.save_weather_cache(
                    key,
                    {
                        "_open_meteo_body": payload,
                        "model": "auto",
                        "parameters": fallback_key_params,
                        "source": "open-meteo:auto (BOM fallback)",
                    },
                )
                return series
        payload = await self._request(FORECAST_URL, params)
        series = _parse_weather(payload, model, key_params)
        self.storage.save_weather_cache(
            key, {"_open_meteo_body": payload, "model": model, "parameters": key_params}
        )
        return series

    async def runtime_for_arrays(
        self,
        latitude: float,
        longitude: float,
        timezone_name: str,
        arrays: Iterable[Any],
        horizon_hours: int,
        elevation_m: float | None = None,
        now: datetime | None = None,
    ) -> dict[tuple[float, float], WeatherSeries]:
        orientations = sorted({(float(a.tilt_deg), float(a.azimuth_deg)) for a in arrays})
        if not orientations:
            return {}
        runtime_model = os.environ.get("OPEN_METEO_RUNTIME_MODEL", "auto").strip().lower()
        return dict(
            zip(
                orientations,
                await asyncio.gather(
                    *(
                        self.runtime(
                            latitude,
                            longitude,
                            timezone_name,
                            tilt,
                            azimuth,
                            horizon_hours,
                            elevation_m,
                            now,
                            runtime_model,
                        )
                        for tilt, azimuth in orientations
                    )
                ),
                strict=True,
            )
        )


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _parse_weather(
    payload: dict[str, Any],
    model: str,
    parameters: dict[str, Any],
    generated_at: datetime | None = None,
) -> WeatherSeries:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        hourly = {}
    times = hourly.get("time")
    irradiance = hourly.get("global_tilted_irradiance")
    temperature = hourly.get("temperature_2m")
    if not isinstance(times, list):
        times = []
    if not isinstance(irradiance, list):
        irradiance = []
    if not isinstance(temperature, list):
        temperature = []
    points: list[HourlyWeatherPoint] = []
    for index, item in enumerate(times):
        timestamp: datetime | None = None
        if isinstance(item, (float, int)):
            try:
                timestamp = datetime.fromtimestamp(item, timezone.utc)
            except (OverflowError, OSError, ValueError):
                timestamp = None
        elif isinstance(item, str):
            timestamp = _parse_iso_hour(item)
        if timestamp is None:
            continue
        # Tilted irradiance is the mean over the preceding hour; temperature is an instant.
        interval_start = timestamp - timedelta(hours=1)
        irrad = _finite(irradiance[index]) if index < len(irradiance) else None
        temp = _finite(temperature[index]) if index < len(temperature) else None
        points.append(HourlyWeatherPoint(interval_start, timestamp, irrad, temp))
    elevation = _finite(payload.get("elevation"))
    elevation_source = "open-meteo-grid-elevation" if elevation is not None else "sea-level-fallback"
    generated_at = generated_at or datetime.now(timezone.utc)
    return WeatherSeries(
        tuple(points),
        f"open-meteo:{model}",
        generated_at,
        elevation if elevation is not None else 0.0,
        elevation_source,
        parameters,
    )


def _parse_cache_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

def _parse_iso_hour(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _complete_historical_weather(series: WeatherSeries) -> bool:
    if not series.points:
        return False
    count = len(series.points)
    irradiance = sum(point.irradiance_wm2 is not None for point in series.points) / count
    temperature = sum(point.temperature_c is not None for point in series.points) / count
    return irradiance >= 0.8 and temperature >= 0.8


def _complete_runtime_weather(series: WeatherSeries, now: datetime, horizon_hours: int) -> bool:
    end = now + timedelta(hours=horizon_hours)
    relevant = [point for point in series.points if point.start < end and point.start + timedelta(hours=1) > now]
    if not relevant:
        return False
    return all(point.irradiance_wm2 is not None and point.temperature_c is not None for point in relevant)
