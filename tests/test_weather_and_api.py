from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import os

import pytest
from fastapi.testclient import TestClient

from energy_forecast.config import ForecastConfig
from energy_forecast.storage import Storage
from energy_forecast.weather import BOM_MODEL, OpenMeteoWeather, WeatherUnavailable


UTC = timezone.utc


def _hourly_payload(times: list[int], irradiance: list[float | None], temperature: list[float | None], elevation=18) -> dict:
    return {
        "elevation": elevation,
        "hourly": {
            "time": times,
            "global_tilted_irradiance": irradiance,
            "temperature_2m": temperature,
        },
    }


@pytest.mark.asyncio
async def test_bom_null_irradiance_falls_back_to_generic_open_meteo(tmp_path, monkeypatch) -> None:
    storage = Storage(tmp_path / "data")
    weather = OpenMeteoWeather(storage)
    now = datetime(2025, 1, 1, 12, 10, tzinfo=UTC)
    hour = now.replace(minute=0, second=0, microsecond=0)
    times = [int((hour + timedelta(hours=index)).timestamp()) for index in range(30)]
    calls: list[str] = []
    request_timezones: list[str] = []
    requested_models: list[str | None] = []

    async def fixture_request(url: str, parameters: dict) -> dict:
        model = parameters.get("models", "auto")
        requested_models.append(parameters.get("models"))
        calls.append(model)
        request_timezones.append(parameters["timezone"])
        if model == BOM_MODEL:
            return _hourly_payload(times, [None] * len(times), [None] * len(times))
        return _hourly_payload(times, [150.0] * len(times), [23.0] * len(times), elevation=42)

    monkeypatch.setattr(weather, "_request", fixture_request)

    result = await weather.runtime(
        latitude=51.5,
        longitude=-0.12,
        timezone_name="Australia/Adelaide",
        tilt_deg=25,
        azimuth_deg=0,
        horizon_hours=24,
        now=now,
        runtime_model=BOM_MODEL,
    )

    assert calls == [BOM_MODEL, "auto"]
    assert request_timezones == ["UTC", "UTC"]
    assert requested_models == [BOM_MODEL, None]
    assert result.source == "open-meteo:auto (BOM fallback)"
    assert result.elevation_m == 42
    assert result.elevation_source == "open-meteo-grid-elevation"
    assert any(point.irradiance_wm2 == 150 for point in result.points)
    cached = await weather.runtime(
        51.5, -0.12, "Australia/Adelaide", 25, 0, 24, now=now + timedelta(minutes=5),
        runtime_model=BOM_MODEL,
    )
    assert cached.source == "open-meteo:auto (BOM fallback)"
    assert calls == [BOM_MODEL, "auto"]


@pytest.mark.asyncio
async def test_historical_weather_uses_utc_hour_bins_for_adelaide(tmp_path, monkeypatch) -> None:
    weather = OpenMeteoWeather(Storage(tmp_path / "data"))
    start_hour = datetime(2024, 1, 1, tzinfo=UTC)
    times = [int((start_hour + timedelta(hours=index + 1)).timestamp()) for index in range(3)]
    requests: list[dict] = []

    async def fixture_request(url: str, parameters: dict) -> dict:
        requests.append(dict(parameters))
        return _hourly_payload(times, [100.0] * len(times), [21.0] * len(times))

    monkeypatch.setattr(weather, "_request", fixture_request)
    series = await weather.historical(
        latitude=-34.85,
        longitude=138.52,
        timezone_name="Australia/Adelaide",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        tilt_deg=25,
        azimuth_deg=0,
        model="era5",
    )

    assert requests[0]["timezone"] == "UTC"
    assert requests[0]["start_date"] == "2023-12-31"
    assert requests[0]["end_date"] == "2024-01-02"
    assert list(series.irradiance) == [start_hour + timedelta(hours=index) for index in range(3)]


@pytest.mark.asyncio
async def test_runtime_cache_interval_and_last_success_fallback(tmp_path, monkeypatch) -> None:
    storage = Storage(tmp_path / "data")
    weather = OpenMeteoWeather(storage)
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    hour = now.replace(minute=0)
    times = [int((hour + timedelta(hours=index + 1)).timestamp()) for index in range(40)]
    requests: list[dict] = []

    async def fixture_request(url: str, parameters: dict) -> dict:
        requests.append(dict(parameters))
        if len(requests) > 1:
            raise WeatherUnavailable("simulated Open-Meteo outage")
        return _hourly_payload(times, [150.0] * len(times), [22.0] * len(times))

    monkeypatch.setattr(weather, "_request", fixture_request)
    first = await weather.runtime(
        51.5, -0.12, "Australia/Adelaide", 25, 0, 24,
        now=now, refresh_interval_minutes=120,
    )
    assert "models" not in requests[0]

    with storage._connection() as connection:
        connection.execute(
            "UPDATE weather_cache SET fetched_at=?",
            ((datetime.now(UTC) - timedelta(minutes=90)).isoformat(),),
        )
    cached = await weather.runtime(
        51.5, -0.12, "Australia/Adelaide", 25, 0, 24,
        now=now + timedelta(minutes=90), refresh_interval_minutes=120,
    )
    assert len(requests) == 1
    assert cached.points == first.points

    with storage._connection() as connection:
        connection.execute(
            "UPDATE weather_cache SET fetched_at=?",
            ((datetime.now(UTC) - timedelta(minutes=121)).isoformat(),),
        )
    fallback = await weather.runtime(
        51.5, -0.12, "Australia/Adelaide", 25, 0, 24,
        now=now + timedelta(minutes=121), refresh_interval_minutes=120,
    )
    retry = await weather.runtime(
        51.5, -0.12, "Australia/Adelaide", 25, 0, 24,
        now=now + timedelta(minutes=130), refresh_interval_minutes=120,
    )

    assert len(requests) == 2
    assert fallback.source.endswith("(cached fallback)")
    assert fallback.points == first.points
    assert retry.points == first.points

@pytest.mark.asyncio
async def test_unitless_statistic_uses_matching_entity_unit(tmp_path, monkeypatch) -> None:
    from energy_forecast.service import ForecastService

    service = ForecastService(Storage(tmp_path / "data"))
    service.storage.save_ha_connection("http://ha.example:8123", "fixture-token", {})

    class FakeHomeAssistantClient:
        def __init__(self, base_url: str, access_token: str) -> None:
            assert base_url == "http://ha.example:8123"

        async def get_states(self) -> list[dict]:
            return [{
                "entity_id": "sensor.home_pv_2",
                "state": "unavailable",
                "attributes": {
                    "friendly_name": "home_load",
                    "unit_of_measurement": "W",
                    "device_class": "power",
                },
            }]

        async def list_statistic_ids(self) -> list[dict]:
            return [{
                "statistic_id": "sensor.home_pv_2",
                "unit_of_measurement": None,
                "unit": None,
                "source": "recorder",
                "has_mean": True,
                "has_sum": False,
                "mean_type": 1,
                "unit_class": "power",
            }]

        async def close(self) -> None:
            return None

    monkeypatch.setattr("energy_forecast.service.HomeAssistantClient", FakeHomeAssistantClient)
    discovery = await service.entities()

    statistic = discovery["statistics"][0]
    assert statistic["statistic_id"] == "sensor.home_pv_2"
    assert statistic["name"] == "home_load"
    assert statistic["unit"] == "W"
    assert discovery["entities"][0]["statistics_available"] is True


def _valid_config() -> dict:
    return {
        "latitude": 51.5,
        "longitude": -0.12,
        "timezone": "UTC",
        "calibration_start": "2024-01-01",
        "calibration_end": "2024-02-01",
        "horizon_hours": 48,
        "consumption": {
            "entity_id": "sensor.house_power",
            "unit": "W",
            "semantics": "interval_average_power",
            "interval_minutes": 5,
            "historical_statistic_id": "sensor.house_power",
            "direct_whole_home_confirmed": True,
        },
        "battery": {
            "entity_id": "sensor.battery_soc",
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
                "id": "pv1",
                "name": "Array one",
                "entity_id": "sensor.pv_array_one_power",
                "unit": "W",
                "semantics": "interval_average_power",
                "interval_minutes": 5,
                "tilt_deg": 25,
                "azimuth_deg": 0,
            }
        ],
        "temperature_entity_id": None,
        "temperature_unit": "°C",
        "grid_import_windows": [],
    }


def test_llm_and_hacs_machine_tokens_are_distinct_and_config_is_atomic(tmp_path, monkeypatch) -> None:
    llm_token = "llm-machine-secret-value-123456"
    hacs_token = "hacs-machine-secret-value-123456"
    ha_token = "home-assistant-secret-token-123456"
    monkeypatch.setenv("LLM_API_TOKEN", llm_token)
    monkeypatch.setenv("HA_INTEGRATION_TOKEN", hacs_token)
    monkeypatch.setenv("ENERGY_FORECAST_DATA_DIR", str(tmp_path / "module-data"))

    # Import after setting the data directory: the module also exposes a default ASGI app.
    from energy_forecast.web import create_app
    from energy_forecast.service import ForecastService

    service = ForecastService(Storage(tmp_path / "service-data"))
    original = _valid_config()
    service.storage.save_config(original)
    service.config = ForecastConfig.model_validate(original)
    assert service.config.open_meteo_refresh_minutes == 60
    service.storage.save_ha_connection("http://ha.example:8123", ha_token, {"time_zone": "UTC"})
    service.solar = object()  # the request only exercises atomic configuration persistence
    service.consumption = object()

    async def fake_connect(base_url: str, access_token: str) -> dict:
        service.storage.save_ha_connection(base_url, access_token, {"time_zone": "UTC"})
        return {"connected": True, "site": {"time_zone": "UTC"}, "ha_version": "fixture"}

    monkeypatch.setattr(service, "connect_home_assistant", fake_connect)

    async def fake_entities() -> dict:
        return {
            "entities": [
                {"entity_id": "sensor.house_power", "unit": "W"},
                {"entity_id": "sensor.battery_soc", "unit": "%"},
                {"entity_id": "sensor.pv_array_one_power", "unit": "W"},
            ],
            "statistics": [
                {
                    "statistic_id": "sensor.house_power",
                    "unit": "W",
                    "has_mean": False,
                    "has_sum": False,
                    "mean_type": "power",
                    "unit_class": "power",
                }
            ],
            "statistics_error": None,
        }

    monkeypatch.setattr(service, "entities", fake_entities)
    app = create_app(service)
    client = TestClient(app)

    assert client.get("/api/v1/setup/status").status_code == 401
    assert client.get("/api/v1/setup/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/v1/setup/status", headers={"Authorization": f"Bearer {llm_token}"}).status_code == 200

    connect_response = client.put(
        "/api/v1/home-assistant",
        headers={"Authorization": f"Bearer {llm_token}"},
        json={"base_url": "http://ha.example:8123", "access_token": ha_token},
    )
    assert connect_response.status_code == 200
    assert ha_token not in json.dumps(connect_response.json())

    updated = _valid_config()
    updated["battery"]["capacity_kwh"] = 12
    updated["open_meteo_refresh_minutes"] = 120
    rejected = client.put(
        "/api/v1/config",
        headers={"Authorization": f"Bearer {hacs_token}"},
        json=updated,
    )
    assert rejected.status_code == 401

    async def missing_home_load() -> dict:
        return {**(await fake_entities()), "statistics": []}

    monkeypatch.setattr(service, "entities", missing_home_load)
    missing_stat = client.put(
        "/api/v1/config",
        headers={"Authorization": f"Bearer {llm_token}"},
        json=updated,
    )
    assert missing_stat.status_code == 409
    assert service.storage.get_config()["battery"]["capacity_kwh"] == 10
    monkeypatch.setattr(service, "entities", fake_entities)
    assert service.storage.get_config()["battery"]["capacity_kwh"] == 10

    accepted = client.put(
        "/api/v1/config",
        headers={"Authorization": f"Bearer {llm_token}"},
        json=updated,
    )
    assert accepted.status_code == 200
    assert accepted.json()["saved"] is True
    assert service.storage.get_config()["battery"]["capacity_kwh"] == 12

    invalid = {**updated, "latitude": 91}
    response = client.put(
        "/api/v1/config",
        headers={"Authorization": f"Bearer {llm_token}"},
        json=invalid,
    )
    assert response.status_code == 422
    assert service.storage.get_config()["battery"]["capacity_kwh"] == 12

    config_response = client.get("/api/v1/config", headers={"Authorization": f"Bearer {llm_token}"})
    assert config_response.status_code == 200
    assert ha_token not in json.dumps(config_response.json())
    assert config_response.json()["config"]["open_meteo_refresh_minutes"] == 120


def test_hacs_read_token_cannot_access_llm_configuration(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_TOKEN", "llm-machine-secret-value-123456")
    monkeypatch.setenv("HA_INTEGRATION_TOKEN", "hacs-machine-secret-value-123456")
    monkeypatch.setenv("ENERGY_FORECAST_DATA_DIR", str(tmp_path / "module-data"))
    from energy_forecast.web import create_app
    from energy_forecast.service import ForecastService

    service = ForecastService(Storage(tmp_path / "service-data"))
    client = TestClient(create_app(service))

    now = datetime.now(UTC)
    service.snapshot = {
        "schema_version": 1,
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "valid_until": (now + timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
        "site_timezone": "UTC",
        "horizon_hours": 24,
        "status": "ready",
        "solar_generation_kwh": 1.0,
        "solar_by_array_kwh": {"pv1": 1.0},
        "home_consumption_kwh": 2.0,
        "battery_surplus_min_kwh": 3.0,
        "battery_minimum_soc_pct": 30.0,
        "battery_minimum_at": None,
        "non_free_grid_import_kwh": 0.0,
        "intervals_5m": [],
    }
    forecast_response = client.get(
        "/api/v1/forecast",
        headers={"Authorization": "Bearer hacs-machine-secret-value-123456"},
    )
    assert forecast_response.status_code == 200
    assert forecast_response.json()["schema_version"] == 1
    assert client.get(
        "/api/v1/forecast",
        headers={"Authorization": "Bearer llm-machine-secret-value-123456"},
    ).status_code == 401

    response = client.get(
        "/api/v1/config",
        headers={"Authorization": "Bearer hacs-machine-secret-value-123456"},
    )

    assert response.status_code == 401
    assert service.storage.get_config() is None
