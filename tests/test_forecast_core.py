from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import math

import pytest
from pydantic import ValidationError

from energy_forecast.battery import BatterySurplusCalculator
from energy_forecast.config import BatteryConfig, ForecastConfig, GridImportWindow
from energy_forecast.forecast_types import ForecastPoint, ForecastSeries
from energy_forecast.history import (
    HistoryImporter,
    calibration_bounds,
    normalize_power_samples,
)
from energy_forecast.consumption import ConsumptionForecaster
from energy_forecast.service import ForecastService
from energy_forecast.storage import Storage


UTC = timezone.utc


def valid_config() -> ForecastConfig:
    return ForecastConfig.model_validate(
        {
            "latitude": 51.5,
            "longitude": -0.12,
            "timezone": "UTC",
            "calibration_start": "2024-01-01",
            "calibration_end": "2024-02-01",
            "horizon_hours": 48,
            "consumption": {
                "entity_id": "sensor.house_load_live",
                "unit": "W",
                "semantics": "interval_average_power",
                "interval_minutes": 5,
                "historical_statistic_id": "home_load",
                "direct_whole_home_confirmed": True,
            },
            "battery": {
                "entity_id": "sensor.battery_state",
                "unit": "%",
                "state_type": "soc_percent",
                "capacity_kwh": 10,
                "max_charge_kw": 2,
                "max_discharge_kw": 2,
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.95,
            },
            "solar_arrays": [
                {
                    "id": "array_one",
                    "name": "Array one",
                    "entity_id": "sensor.pv_array_one_power",
                    "unit": "W",
                    "semantics": "interval_average_power",
                    "interval_minutes": 5,
                    "tilt_deg": 25,
                    "azimuth_deg": -90,
                }
            ],
            "temperature_entity_id": None,
            "temperature_unit": "°C",
            "grid_import_windows": [],
        }
    )


def power_series(start: datetime, hours: int, value: float, kind: str) -> ForecastSeries:
    points = tuple(
        ForecastPoint(
            start + timedelta(minutes=5 * index),
            start + timedelta(minutes=5 * (index + 1)),
            value,
        )
        for index in range(hours * 12)
    )
    return ForecastSeries(kind, "W", points, start, "UTC", "fixture", 1.0)


def battery_config() -> BatteryConfig:
    return BatteryConfig(
        entity_id="sensor.battery_soc",
        unit="%",
        state_type="soc_percent",
        capacity_kwh=10,
        max_charge_kw=2,
        max_discharge_kw=2,
        charge_efficiency=1,
        discharge_efficiency=1,
    )


def test_free_import_window_recharges_before_later_nonfree_period() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)  # Monday
    solar = power_series(start, 3, 0, "solar_power")
    load = power_series(start, 3, 1000, "home_load_power")
    window = GridImportWindow(
        weekdays=[0],
        start=time(1, 0),
        end=time(2, 0),
        target_soc_pct=80,
        grid_charge_kw=2,
    )

    result = BatterySurplusCalculator.calculate(solar, load, 6.0, battery_config(), [window], 3)

    assert result.status == "ready"
    assert result.battery_surplus_min_kwh == pytest.approx(5.0)
    assert result.battery_minimum_soc_pct == pytest.approx(50.0)
    assert result.non_free_grid_import_kwh == pytest.approx(0.0)
    assert result.battery_minimum_at == start + timedelta(hours=1)

def test_multi_day_import_window_applies_on_each_selected_weekday() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)  # Monday
    config = BatteryConfig(
        entity_id="sensor.battery_soc", unit="%", state_type="soc_percent", capacity_kwh=10,
        max_charge_kw=1, max_discharge_kw=0.1, charge_efficiency=1, discharge_efficiency=1,
    )
    window = GridImportWindow(
        weekdays=[0, 1], start=time(0, 0), end=time(0, 30), target_soc_pct=1, grid_charge_kw=0.001
    )

    result = BatterySurplusCalculator.calculate(
        power_series(start, 48, 0, "solar_power"),
        power_series(start, 48, 1000, "home_load_power"),
        10.0,
        config,
        [window],
        48,
    )

    assert result.status == "ready"
    assert result.non_free_grid_import_kwh == pytest.approx(42.3)


def test_grid_import_days_must_be_nonempty_unique_and_valid() -> None:
    values = {"start": time(0, 0), "end": time(1, 0), "target_soc_pct": 80, "grid_charge_kw": 2}

    with pytest.raises(ValidationError):
        GridImportWindow(weekdays=[], **values)
    with pytest.raises(ValidationError, match="weekdays must be unique"):
        GridImportWindow(weekdays=[1, 1], **values)
    with pytest.raises(ValidationError):
        GridImportWindow(weekdays=[7], **values)


def test_legacy_saved_weekday_migrates_to_weekdays(tmp_path) -> None:
    storage = Storage(tmp_path / "service-data")
    saved = valid_config().model_dump(mode="json")
    saved["grid_import_windows"] = [
        {"weekday": 2, "start": "23:00", "end": "00:30", "target_soc_pct": 80, "grid_charge_kw": 2}
    ]
    storage.save_config(saved)

    service = ForecastService(storage)

    migrated = storage.get_config()["grid_import_windows"][0]
    assert service.config.grid_import_windows[0].weekdays == [2]
    assert migrated["weekdays"] == [2]
    assert "weekday" not in migrated


def test_battery_without_import_window_reaches_three_kwh_minimum() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    result = BatterySurplusCalculator.calculate(
        power_series(start, 3, 0, "solar_power"),
        power_series(start, 3, 1000, "home_load_power"),
        6.0,
        battery_config(),
        [],
        3,
    )

    assert result.battery_surplus_min_kwh == pytest.approx(3.0)
    assert result.battery_minimum_soc_pct == pytest.approx(30.0)
    assert result.non_free_grid_import_kwh == pytest.approx(0.0)


def test_missing_required_power_slot_disables_battery_summary() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    valid = list(power_series(start, 3, 1000, "home_load_power").points)
    valid[7] = ForecastPoint(valid[7].start, valid[7].end, None)
    load = ForecastSeries("home_load_power", "W", tuple(valid), start, "UTC", "fixture", 35 / 36)

    result = BatterySurplusCalculator.calculate(
        power_series(start, 3, 0, "solar_power"), load, 6.0, battery_config(), [], 3
    )

    assert result.status == "unavailable"
    assert result.battery_surplus_min_kwh is None
    assert result.non_free_grid_import_kwh is None


def test_overnight_import_window_applies_after_midnight() -> None:
    start = datetime(2024, 1, 7, 23, tzinfo=UTC)  # Sunday
    config = BatteryConfig(
        entity_id="sensor.battery_soc", unit="%", state_type="soc_percent", capacity_kwh=10,
        max_charge_kw=1, max_discharge_kw=0.5, charge_efficiency=1, discharge_efficiency=1,
    )
    window = GridImportWindow(
        weekdays=[6], start=time(23, 30), end=time(0, 30), target_soc_pct=1, grid_charge_kw=0.001
    )
    result = BatterySurplusCalculator.calculate(
        power_series(start, 2, 0, "solar_power"), power_series(start, 2, 1000, "home_load_power"),
        10.0, config, [window], 2,
    )

    assert result.non_free_grid_import_kwh == pytest.approx(0.5)


def test_both_repeated_dst_wall_clock_windows_are_included() -> None:
    start = datetime(2024, 11, 3, 4, tzinfo=UTC)  # 00:00 before the New York fall-back fold
    config = BatteryConfig(
        entity_id="sensor.battery_soc", unit="%", state_type="soc_percent", capacity_kwh=10,
        max_charge_kw=1, max_discharge_kw=0.5, charge_efficiency=1, discharge_efficiency=1,
    )
    window = GridImportWindow(
        weekdays=[6], start=time(1, 0), end=time(1, 30), target_soc_pct=1, grid_charge_kw=0.001
    )
    solar = ForecastSeries("solar_power", "W", power_series(start, 4, 0, "solar_power").points, start,
                           "America/New_York", "fixture", 1.0)
    load = ForecastSeries("home_load_power", "W", power_series(start, 4, 1000, "home_load_power").points, start,
                          "America/New_York", "fixture", 1.0)

    result = BatterySurplusCalculator.calculate(solar, load, 10.0, config, [window], 4)

    assert result.non_free_grid_import_kwh == pytest.approx(1.5)


def test_cumulative_energy_delta_converts_to_watts_and_reset_is_a_gap() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [
        {"timestamp": start.isoformat(), "value": 5.0, "unit": "kWh", "source": "raw",
         "resolution_seconds": 300, "semantics": "cumulative_energy", "timestamp_is_end": True},
        {"timestamp": (start + timedelta(minutes=5)).isoformat(), "value": 5.1, "unit": "kWh", "source": "raw",
         "resolution_seconds": 300, "semantics": "cumulative_energy", "timestamp_is_end": True},
        {"timestamp": (start + timedelta(minutes=10)).isoformat(), "value": 0.2, "unit": "kWh", "source": "raw",
         "resolution_seconds": 300, "semantics": "cumulative_energy", "timestamp_is_end": True},
        {"timestamp": (start + timedelta(minutes=15)).isoformat(), "value": 0.3, "unit": "kWh", "source": "raw",
         "resolution_seconds": 300, "semantics": "cumulative_energy", "timestamp_is_end": True},
    ]

    points = normalize_power_samples(samples, start, start + timedelta(minutes=15), interval_minutes=5)

    assert points[0].value == pytest.approx(1200.0)
    assert points[1].value is None  # reset/decrease is a gap, not zero or negative energy
    assert points[2].value == pytest.approx(1200.0)


def test_raw_history_overrides_overlapping_hourly_statistics() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    samples = [
        {"timestamp": start.isoformat(), "value": 1000, "unit": "W", "source": "statistics_hour",
         "resolution_seconds": 3600, "semantics": "interval_average_power", "timestamp_is_end": False, "estimated": True},
        {"timestamp": (start + timedelta(minutes=30)).isoformat(), "value": 200, "unit": "W", "source": "raw",
         "resolution_seconds": 300, "semantics": "interval_average_power", "timestamp_is_end": False},
        {"timestamp": (start + timedelta(hours=1)).isoformat(), "value": 2000, "unit": "W", "source": "statistics_hour",
         "resolution_seconds": 3600, "semantics": "interval_average_power", "timestamp_is_end": False, "estimated": True},
    ]

    points = normalize_power_samples(samples, start, start + timedelta(hours=2))

    assert len(points) == 24
    assert points[0].value == pytest.approx(1000.0)
    assert points[0].estimated is True
    assert points[6].value == pytest.approx(200.0)
    assert points[6].estimated is False
    assert points[6].source == "raw"
    assert points[7].value == pytest.approx(1000.0)
    assert points[7].estimated is True
    assert points[12].value == pytest.approx(2000.0)
    assert points[12].estimated is True
    assert all(point.value is not None for point in points)


def test_selected_long_term_statistic_is_the_historical_load_source() -> None:
    config = valid_config()
    statistic_id = config.consumption.historical_statistic_id
    descriptors = HistoryImporter._statistic_descriptors(
        config,
        [{"statistic_id": statistic_id, "unit_of_measurement": "kWh", "has_mean": False, "has_sum": True}],
    )

    assert descriptors[statistic_id] == {"unit": "kWh", "semantics": "interval_energy"}
    assert config.consumption.entity_id != statistic_id
    power_descriptors = HistoryImporter._statistic_descriptors(
        config,
        [{"statistic_id": statistic_id, "unit_of_measurement": None, "unit_class": "power"}],
        "W",
    )
    assert power_descriptors[statistic_id] == {"unit": "W", "semantics": "interval_average_power"}


@pytest.mark.asyncio
async def test_statistics_import_accepts_epoch_millisecond_timestamps(tmp_path) -> None:
    class FakeClient:
        async def statistics_many(self, requests: list[dict]) -> list[dict]:
            return [
                {"sensor.home_pv_2": [{"start": 1781308800000, "mean": 456.0}]}
                if request["period"] == "hour"
                else {}
                for request in requests
            ]

    importer = HistoryImporter(FakeClient(), Storage(tmp_path / "data"))
    rows = await importer._fetch_statistics(
        {"sensor.home_pv_2": {"unit": "W", "semantics": "interval_average_power"}},
        datetime(2026, 6, 13, tzinfo=UTC),
        datetime(2026, 6, 14, tzinfo=UTC),
        [],
    )

    assert rows == [
        {
            "entity_id": "sensor.home_pv_2",
            "timestamp": "2026-06-13T00:00:00+00:00",
            "value": 456.0,
            "unit": "W",
            "source": "statistics_hour",
            "resolution_seconds": 3600,
            "semantics": "interval_average_power",
            "timestamp_is_end": False,
            "estimated": True,
        }
    ]


def test_dst_calibration_bounds_preserve_repeated_local_hour() -> None:
    start, end = calibration_bounds(date(2024, 11, 3), date(2024, 11, 4), "America/New_York")

    assert end - start == timedelta(hours=25)


def test_unconfirmed_sensor_cannot_be_configured_as_whole_home_load() -> None:
    payload = valid_config().model_dump(mode="json")
    payload["consumption"]["direct_whole_home_confirmed"] = False

    with pytest.raises(ValidationError, match="Confirm that this is direct whole-home consumption"):
        ForecastConfig.model_validate(payload)

def test_partial_home_load_history_trains_only_valid_samples_and_reports_coverage() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    targets = tuple(
        ForecastPoint(
            start + timedelta(minutes=5 * index),
            start + timedelta(minutes=5 * (index + 1)),
            1000.0 if index % 2 == 0 else None,
        )
        for index in range(10 * 24 * 12)
    )
    forecaster = ConsumptionForecaster()

    metrics = forecaster.fit(targets, actual_temperature=None, site_timezone="UTC",
                             historical_statistic_id="home_load")
    prediction = forecaster.predict(start + timedelta(days=10), 24, start, "UTC")

    assert metrics["historical_statistic_id"] == "home_load"
    assert metrics["expected_history_intervals"] == len(targets)
    assert metrics["valid_history_intervals"] == len(targets) // 2
    assert metrics["history_coverage_pct"] == pytest.approx(50.0)
    assert metrics["training_samples"] == len(targets) // 2
    assert all(point.value == pytest.approx(1000.0) for point in prediction.points)

