from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
import pvlib

from .forecast_types import ForecastPoint, ForecastSeries
from .weather import WeatherSeries


class SolarModelUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class ArrayModel:
    estimator: lgb.LGBMRegressor
    resolution: str
    tilt_deg: float
    azimuth_deg: float
    metrics: dict[str, Any]
    weather_source: str


class SolarForecaster:
    """Chronological, per-array LightGBM production models."""

    def __init__(self, latitude: float, longitude: float, elevation_m: float) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.elevation_m = elevation_m
        self.models: dict[str, ArrayModel] = {}

    def fit(
        self,
        array_id: str,
        tilt_deg: float,
        azimuth_deg: float,
        targets: Sequence[ForecastPoint],
        weather: WeatherSeries,
        site_timezone: str,
    ) -> dict[str, Any]:
        valid = [point for point in targets if point.value is not None and math.isfinite(point.value) and point.value >= 0]
        if len(valid) < 48:
            raise SolarModelUnavailable(f"Array {array_id} has fewer than 48 valid history intervals")
        zone = ZoneInfo(site_timezone)
        dates = sorted({point.start.astimezone(zone).date() for point in valid})
        if len(dates) < 2:
            raise SolarModelUnavailable(f"Array {array_id} has no chronological holdout dates")
        split_at = max(1, int(len(dates) * 0.8))
        if split_at >= len(dates):
            split_at = len(dates) - 1
        train_dates, holdout_dates = set(dates[:split_at]), set(dates[split_at:])
        expected_slots = max(1, len(targets))
        learned_5m = [
            point for point in valid
            if not point.estimated and not (point.source and "statistics_hour" in point.source)
        ]
        resolution = "five_minute" if len(learned_5m) / expected_slots >= 0.8 else "hourly"
        irr_map = weather.irradiance
        if resolution == "five_minute":
            rows = self._five_minute_rows(learned_5m, irr_map, tilt_deg, azimuth_deg, site_timezone)
        else:
            rows = self._hourly_rows(valid, irr_map, tilt_deg, azimuth_deg, site_timezone)
        train_rows = [row for row in rows if row["date"] in train_dates]
        holdout_rows = [row for row in rows if row["date"] in holdout_dates]
        if len(train_rows) < 24 or len(holdout_rows) < 12:
            raise SolarModelUnavailable(f"Array {array_id} lacks usable training or holdout weather/target samples")
        candidate = _new_regressor()
        candidate.fit(np.asarray([row["features"] for row in train_rows]), np.asarray([row["target"] for row in train_rows]))
        holdout_predictions = np.maximum(
            0.0, candidate.predict(np.asarray([row["features"] for row in holdout_rows]))
        )
        holdout_truth = np.asarray([row["target"] for row in holdout_rows])
        metrics = _solar_metrics(holdout_rows, holdout_predictions, site_timezone)
        final_model = _new_regressor()
        final_model.fit(np.asarray([row["features"] for row in rows]), np.asarray([row["target"] for row in rows]))
        metrics.update(
            {
                "array_id": array_id,
                "model": "LightGBM",
                "resolution": resolution,
                "training_samples": len(rows),
                "holdout_samples": len(holdout_rows),
                "train_dates": [min(train_dates).isoformat(), max(train_dates).isoformat()],
                "holdout_dates": [min(holdout_dates).isoformat(), max(holdout_dates).isoformat()],
                "holdout_w_mae": float(np.mean(np.abs(holdout_truth - holdout_predictions))),
                "weather_source": weather.source,
                "weather_elevation_m": weather.elevation_m,
                "weather_elevation_source": weather.elevation_source,
            }
        )
        self.models[array_id] = ArrayModel(final_model, resolution, tilt_deg, azimuth_deg, metrics, weather.source)
        return metrics

    def predict(
        self,
        array_id: str,
        weather: WeatherSeries,
        start: datetime,
        horizon_hours: int,
        generated_at: datetime,
        site_timezone: str,
    ) -> ForecastSeries:
        model = self.models.get(array_id)
        if model is None:
            raise SolarModelUnavailable(f"No trained model is available for array {array_id}")
        start = start.astimezone(timezone.utc)
        end = start + timedelta(hours=horizon_hours)
        slots: list[datetime] = []
        cursor = start
        while cursor < end:
            slots.append(cursor)
            cursor += timedelta(minutes=5)
        irr_map = weather.irradiance
        shaped = _shape_hourly_irradiance(
            irr_map, model.tilt_deg, model.azimuth_deg, self.latitude, self.longitude, weather.elevation_m
        )
        points: list[ForecastPoint] = []
        if model.resolution == "five_minute":
            feature_times: list[datetime] = []
            feature_irradiance: list[float] = []
            valid_indices: list[int] = []
            values: list[float | None] = [None] * len(slots)
            for index, slot in enumerate(slots):
                weather_slot = slot.replace(minute=slot.minute // 5 * 5, second=0, microsecond=0)
                irradiance = shaped.get(weather_slot)
                if irradiance is None:
                    continue
                feature_times.append(slot + timedelta(minutes=2, seconds=30))
                feature_irradiance.append(irradiance)
                valid_indices.append(index)
            if feature_times:
                features = _solar_features(
                    feature_times, feature_irradiance, self.latitude, self.longitude, weather.elevation_m
                )
                predictions = np.maximum(0.0, model.estimator.predict(np.asarray(features)))
                for index, value in zip(valid_indices, predictions, strict=True):
                    values[index] = float(value)
            points = [
                ForecastPoint(slot, slot + timedelta(minutes=5), values[index], True, weather.source)
                for index, slot in enumerate(slots)
            ]
        else:
            hourly_means: dict[datetime, float | None] = {}
            hourly_centers: dict[datetime, list[float]] = {}
            for slot in slots:
                hour = slot.replace(minute=0, second=0, microsecond=0)
                value = irr_map.get(hour)
                hourly_means.setdefault(hour, value)
                hourly_centers.setdefault(hour, [])
            hours = sorted(hourly_means)
            valid_hours = [hour for hour in hours if hourly_means[hour] is not None]
            if valid_hours:
                features = _solar_features(
                    [hour + timedelta(minutes=30) for hour in valid_hours],
                    [float(hourly_means[hour]) for hour in valid_hours],
                    self.latitude,
                    self.longitude,
                    weather.elevation_m,
                )
                means = np.maximum(0.0, model.estimator.predict(np.asarray(features)))
                for hour, value in zip(valid_hours, means, strict=True):
                    hourly_centers[hour] = [float(value)]
            for slot in slots:
                hour = slot.replace(minute=0, second=0, microsecond=0)
                predicted_mean = hourly_centers[hour][0] if hourly_centers[hour] else None
                weather_slot = slot.replace(minute=slot.minute // 5 * 5, second=0, microsecond=0)
                shaped_feature = shaped.get(weather_slot)
                irradiance_mean = irr_map.get(hour)
                if predicted_mean is None or shaped_feature is None or irradiance_mean is None:
                    value = None
                else:
                    slot_factor = shaped_feature / irradiance_mean if irradiance_mean > 0 else 0.0
                    value = predicted_mean * slot_factor
                points.append(ForecastPoint(slot, slot + timedelta(minutes=5), value, True, weather.source))
        valid_count = sum(point.value is not None for point in points)
        return ForecastSeries(
            "solar_power",
            "W",
            tuple(points),
            generated_at,
            site_timezone,
            weather.source,
            valid_count / len(points) if points else 0.0,
        )

    def metrics(self) -> dict[str, dict[str, Any]]:
        return {array_id: model.metrics for array_id, model in self.models.items()}

    def _five_minute_rows(
        self,
        targets: Sequence[ForecastPoint],
        irradiance: dict[datetime, float | None],
        tilt_deg: float,
        azimuth_deg: float,
        site_timezone: str,
    ) -> list[dict[str, Any]]:
        shaped = _shape_hourly_irradiance(
            irradiance, tilt_deg, azimuth_deg, self.latitude, self.longitude, self.elevation_m
        )
        valid_points = [point for point in targets if point.value is not None]
        if not valid_points:
            return []
        feature_points: list[ForecastPoint] = []
        feature_times: list[datetime] = []
        feature_irradiance: list[float] = []
        for point in valid_points:
            value = shaped.get(point.start.astimezone(timezone.utc))
            if value is None:
                continue
            feature_points.append(point)
            feature_times.append(point.start + (point.end - point.start) / 2)
            feature_irradiance.append(value)
        if not feature_points:
            return []
        feature_rows = _solar_features(
            feature_times, feature_irradiance, self.latitude, self.longitude, self.elevation_m
        )
        return [
            {
                "features": feature,
                "target": float(point.value),
                "date": point.start.astimezone(ZoneInfo(site_timezone)).date(),
                "start": point.start,
                "duration_hours": (point.end - point.start).total_seconds() / 3600,
            }
            for point, feature in zip(feature_points, feature_rows, strict=True)
        ]

    def _hourly_rows(
        self,
        targets: Sequence[ForecastPoint],
        irradiance: dict[datetime, float | None],
        tilt_deg: float,
        azimuth_deg: float,
        site_timezone: str,
    ) -> list[dict[str, Any]]:
        buckets: dict[datetime, list[tuple[float, float, ForecastPoint]]] = {}
        for point in targets:
            if point.value is None:
                continue
            hour = point.start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
            duration = (point.end - point.start).total_seconds()
            buckets.setdefault(hour, []).append((float(point.value), duration, point))
        hourly_targets: list[tuple[datetime, float, float]] = []
        for hour, values in sorted(buckets.items()):
            covered_seconds = sum(item[1] for item in values)
            if covered_seconds < 1800:
                continue
            irr = irradiance.get(hour)
            if irr is None or not math.isfinite(irr):
                continue
            target = sum(value * duration for value, duration, _ in values) / covered_seconds
            hourly_targets.append((hour, target, float(irr)))
        if not hourly_targets:
            return []
        features = _solar_features(
            [hour + timedelta(minutes=30) for hour, _, _ in hourly_targets],
            [irr for _, _, irr in hourly_targets],
            self.latitude,
            self.longitude,
            self.elevation_m,
        )
        return [
            {
                "features": feature,
                "target": target,
                "date": hour.astimezone(ZoneInfo(site_timezone)).date(),
                "start": hour,
                "duration_hours": 1.0,
            }
            for (hour, target, _), feature in zip(hourly_targets, features, strict=True)
        ]


def _new_regressor() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )


def _solar_features(
    timestamps: Sequence[datetime],
    irradiance: Sequence[float],
    latitude: float,
    longitude: float,
    elevation_m: float,
) -> list[list[float]]:
    if not timestamps:
        return []
    index = pd.DatetimeIndex([ts.astimezone(timezone.utc) for ts in timestamps])
    position = pvlib.solarposition.get_solarposition(index, latitude, longitude, altitude=elevation_m)
    azimuth = np.deg2rad(position["azimuth"].to_numpy(dtype=float))
    elevation = position["elevation"].to_numpy(dtype=float)
    return [
        [float(max(0.0, irrad)), float(math.sin(az)), float(math.cos(az)), float(elev)]
        for irrad, az, elev in zip(irradiance, azimuth, elevation, strict=True)
    ]


def _shape_hourly_irradiance(
    hourly: dict[datetime, float | None],
    tilt_deg: float,
    azimuth_deg: float,
    latitude: float,
    longitude: float,
    elevation_m: float,
) -> dict[datetime, float | None]:
    output: dict[datetime, float | None] = {}
    work: list[tuple[datetime, datetime, float]] = []
    for hour, mean in sorted(hourly.items()):
        hour = hour.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        if mean is None or not math.isfinite(mean):
            for offset in range(12):
                slot = hour + timedelta(minutes=5 * offset)
                output[slot] = None
            continue
        for offset in range(12):
            slot = hour + timedelta(minutes=5 * offset)
            center = slot + timedelta(minutes=2, seconds=30)
            work.append((hour, slot, max(0.0, float(mean))))
    if not work:
        return output
    centers = [slot + timedelta(minutes=2, seconds=30) for _, slot, _ in work]
    index = pd.DatetimeIndex(centers)
    location = pvlib.location.Location(latitude, longitude, tz="UTC", altitude=elevation_m)
    clearsky = location.get_clearsky(index)
    position = pvlib.solarposition.get_solarposition(index, latitude, longitude, altitude=elevation_m)
    total = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=(azimuth_deg + 180.0) % 360.0,
        solar_zenith=position["apparent_zenith"],
        solar_azimuth=position["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
    )["poa_global"].fillna(0.0).to_numpy(dtype=float)
    by_hour: dict[datetime, list[tuple[datetime, float, float]]] = {}
    for (hour, slot, mean), clear in zip(work, total, strict=True):
        by_hour.setdefault(hour, []).append((slot, mean, max(0.0, float(clear))))
    for hour, items in by_hour.items():
        average_clear = sum(item[2] for item in items) / len(items)
        for slot, mean, clear in items:
            output[slot] = mean * clear / average_clear if average_clear > 0 else 0.0
    return output


def _solar_metrics(rows: Sequence[dict[str, Any]], predictions: np.ndarray, site_timezone: str) -> dict[str, Any]:
    zone = ZoneInfo(site_timezone)
    truth_daily: dict[Any, float] = {}
    pred_daily: dict[Any, float] = {}
    for row, prediction in zip(rows, predictions, strict=True):
        day = row["start"].astimezone(zone).date()
        duration = float(row["duration_hours"])
        truth_daily[day] = truth_daily.get(day, 0.0) + float(row["target"]) * duration / 1000.0
        pred_daily[day] = pred_daily.get(day, 0.0) + float(prediction) * duration / 1000.0
    days = sorted(truth_daily)
    daily_mae = float(np.mean([abs(truth_daily[day] - pred_daily.get(day, 0.0)) for day in days])) if days else None
    return {
        "daily_kwh_mae": daily_mae,
        "holdout_days": len(days),
        "holdout_daily_actual_kwh": {day.isoformat(): truth_daily[day] for day in days},
        "holdout_daily_predicted_kwh": {day.isoformat(): pred_daily[day] for day in days},
    }
