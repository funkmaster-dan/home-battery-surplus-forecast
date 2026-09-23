from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hmac
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from .api import HomeAssistantConnectionRequest, router as api_router
from .config import ForecastConfig
from .ha_client import HomeAssistantAuthenticationError, HomeAssistantError
from .service import ForecastService, ServiceError


PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))


def _verify_separate_machine_tokens() -> None:
    llm_token = os.environ.get("LLM_API_TOKEN")
    integration_token = os.environ.get("HA_INTEGRATION_TOKEN")
    if llm_token and integration_token and hmac.compare_digest(llm_token, integration_token):
        raise RuntimeError("LLM_API_TOKEN and HA_INTEGRATION_TOKEN must be different secrets")


def _service(request: Request) -> ForecastService:
    return request.app.state.forecast_service


def _service_exception(exc: Exception) -> None:
    if isinstance(exc, HomeAssistantAuthenticationError):
        raise HTTPException(status_code=400, detail="Home Assistant rejected the access token") from exc
    if isinstance(exc, HomeAssistantError):
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if isinstance(exc, ServiceError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail="Request could not be completed") from exc


def _forecast_for_display(service: ForecastService) -> dict[str, Any]:
    if service.snapshot is None:
        raise HTTPException(status_code=503, detail="No forecast has been computed")
    result = dict(service.snapshot)
    try:
        valid_until = datetime.fromisoformat(str(result["valid_until"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=503, detail="Forecast freshness metadata is invalid")
    if valid_until.tzinfo is None or valid_until <= datetime.now(timezone.utc):
        result["status"] = "stale"
    return result


def create_app(forecast_service: ForecastService | None = None) -> FastAPI:
    service = forecast_service or ForecastService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _verify_separate_machine_tokens()
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(
        title="Home Battery Surplus Forecast API",
        version="1.0.0",
        description="Read-only solar, home-load, and battery surplus forecasts plus machine-authenticated configuration endpoints.",
        lifespan=lifespan,
    )
    app.state.forecast_service = service
    app.include_router(api_router)
    app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        status_data = _service(request).setup_status()
        return {"status": "ok", "configured": status_data["configured"], "forecast_available": status_data["forecast_available"]}

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        active = _service(request)
        if active.config is not None and (active.solar is not None or active.consumption is not None):
            return TEMPLATES.TemplateResponse(request=request, name="dashboard.html", context={})
        return TEMPLATES.TemplateResponse(request=request, name="setup.html", context={})

    @app.get("/setup", response_class=HTMLResponse)
    async def setup_page(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request=request, name="setup.html", context={"edit_mode": False})

    @app.get("/configuration", response_class=HTMLResponse)
    async def configuration_page(request: Request) -> HTMLResponse:
        active = _service(request)
        edit_mode = active.config is not None and active.storage.get_ha_connection() is not None
        return TEMPLATES.TemplateResponse(
            request=request, name="setup.html", context={"edit_mode": edit_mode}
        )

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_page(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request=request, name="dashboard.html", context={})

    @app.get("/ui/api/status")
    async def ui_status(request: Request) -> dict[str, Any]:
        active = _service(request)
        result = active.setup_status()
        connection = active.storage.get_ha_connection()
        result["base_url"] = connection.get("base_url") if connection else ""
        return result

    @app.post("/ui/api/home-assistant")
    async def ui_connect(payload: HomeAssistantConnectionRequest, request: Request) -> dict[str, Any]:
        try:
            return await _service(request).connect_home_assistant(
                payload.base_url, payload.access_token.get_secret_value()
            )
        except (HomeAssistantError, ServiceError, ValueError) as exc:
            _service_exception(exc)

    @app.get("/ui/api/entities")
    async def ui_entities(request: Request) -> dict[str, Any]:
        try:
            return await _service(request).entities()
        except (HomeAssistantError, ServiceError) as exc:
            _service_exception(exc)

    @app.get("/ui/api/config")
    async def ui_get_config(request: Request) -> dict[str, Any]:
        active = _service(request)
        connection = active.storage.get_ha_connection()
        config = active.config.model_dump(mode="json") if active.config else None
        return {
            "configured": config is not None,
            "config": config,
            "site": connection.get("site_config") if connection else None,
            "base_url": connection.get("base_url") if connection else "",
        }

    @app.put("/ui/api/config")
    async def ui_put_config(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        try:
            config = ForecastConfig.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_input=False)) from exc
        try:
            return await _service(request).save_config(config)
        except ServiceError as exc:
            _service_exception(exc)


    @app.get("/ui/api/hacs-token")
    async def ui_hacs_token() -> JSONResponse:
        token = os.environ.get("HA_INTEGRATION_TOKEN")
        if not token:
            raise HTTPException(status_code=503, detail="HA integration token is not configured")
        return JSONResponse(content={"token": token}, headers={"Cache-Control": "no-store"})
    @app.post("/ui/api/calibration", status_code=status.HTTP_202_ACCEPTED)
    async def ui_start_calibration(request: Request) -> JSONResponse:
        try:
            run_id = await _service(request).start_calibration()
        except ServiceError as exc:
            _service_exception(exc)
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content={"run_id": run_id, "status": "queued"})

    @app.get("/ui/api/calibration/{run_id}")
    async def ui_calibration_status(run_id: str, request: Request) -> dict[str, Any]:
        result = _service(request).calibration_status(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Calibration run not found")
        return result


    @app.post("/ui/api/forecast/refresh")
    async def ui_refresh_forecast(request: Request) -> JSONResponse:
        active = _service(request)
        await active.refresh_forecast()
        return JSONResponse(content=_forecast_for_display(active), headers={"Cache-Control": "no-store"})

    @app.get("/ui/api/forecast")
    async def ui_forecast(request: Request) -> JSONResponse:
        return JSONResponse(content=_forecast_for_display(_service(request)), headers={"Cache-Control": "no-store"})

    return app


app = create_app()
