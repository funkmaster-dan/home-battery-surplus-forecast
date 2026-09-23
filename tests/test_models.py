from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

from energy_forecast.consumption import ConsumptionForecaster
from energy_forecast.forecast_types import ForecastPoint
from energy_forecast.service import _total_array_daily_mae
from energy_forecast.solar import SolarForecaster
from energy_forecast.weather import HourlyWeatherPoint, WeatherSeries


UTC = timezone.utc


def test_solar_model_uses_measured_five_minute_targets_and_predicts_full_profile() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    hourly = []
    irradiance_by_hour = {}
    for index in range(7 * 24):
        hour = start + timedelta(hours=index)
        irradiance = 300.0 if 7 <= hour.hour < 18 else 0.0
        irradiance_by_hour[hour] = irradiance
        hourly.append(HourlyWeatherPoint(hour, hour + timedelta(hours=1), irradiance, 18.0))
    history_points = []
    for index in range(7 * 24 * 12):
        point_start = start + timedelta(minutes=5 * index)
        irradiance = irradiance_by_hour[point_start.replace(minute=0, second=0, microsecond=0)]
        target = 250.0 if irradiance > 0 else 0.0
        history_points.append(
            ForecastPoint(point_start, point_start + timedelta(minutes=5), target, False, "raw")
        )
    historical_weather = WeatherSeries(
        tuple(hourly), "open-meteo:era5", datetime.now(UTC), 35.0,
        "open-meteo-grid-elevation", {"model": "era5"},
    )
    forecaster = SolarForecaster(51.5, -0.12, historical_weather.elevation_m)

    metrics = forecaster.fit(
        "array_one", 25, 0, tuple(history_points), historical_weather, "UTC"
    )
    future_start = start + timedelta(days=7)
    future_hourly = tuple(
        HourlyWeatherPoint(
            future_start + timedelta(hours=index),
            future_start + timedelta(hours=index + 1),
            300.0 if 7 <= (future_start.hour + index) % 24 < 18 else 0.0,
            19.0,
        )
        for index in range(26)
    )
    future_weather = WeatherSeries(
        future_hourly, "open-meteo:auto", datetime.now(UTC), 35.0,
        "open-meteo-grid-elevation", {"model": "auto"},
    )
    forecast = forecaster.predict("array_one", future_weather, future_start, 24, datetime.now(UTC), "UTC")

    assert metrics["model"] == "LightGBM"
    assert metrics["resolution"] == "five_minute"
    assert metrics["holdout_samples"] > 0
    assert len(forecast.points) == 24 * 12
    assert forecast.completeness == 1.0
    assert all(point.value is not None and point.value >= 0 for point in forecast.points)


def test_household_tree_tie_keeps_time_of_week_baseline() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    targets = []
    temperatures = []
    for index in range(14 * 24 * 12):
        point_start = start + timedelta(minutes=5 * index)
        point_end = point_start + timedelta(minutes=5)
        targets.append(ForecastPoint(point_start, point_end, 1000.0))
        temperatures.append(ForecastPoint(point_start, point_end, 10.0 + (index % 24) / 10))
    forecaster = ConsumptionForecaster()

    metrics = forecaster.fit(tuple(targets), tuple(temperatures), "UTC", "home_load")

    assert metrics["selected_model"] == "time_of_week_median"
    assert metrics["baseline_hourly_kwh_mae"] == 0.0
    assert metrics["tree_hourly_kwh_mae"] == 0.0
    assert metrics["history_coverage_pct"] == 100.0


def test_weather_source_comparison_uses_total_array_daily_error() -> None:
    metrics = {
        "array_one": {
            "holdout_daily_actual_kwh": {"2024-01-01": 10.0},
            "holdout_daily_predicted_kwh": {"2024-01-01": 9.0},
        },
        "array_two": {
            "holdout_daily_actual_kwh": {"2024-01-01": 10.0},
            "holdout_daily_predicted_kwh": {"2024-01-01": 11.0},
        },
    }

    assert _total_array_daily_mae(metrics, ["array_one", "array_two"]) == 0.0
    assert math.fabs(_total_array_daily_mae(metrics, ["array_one"]) - 1.0) < 1e-12
