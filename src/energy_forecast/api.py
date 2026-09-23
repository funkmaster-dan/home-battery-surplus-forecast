from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from starlette.responses import JSONResponse

from .config import ForecastConfig
from .ha_client import HomeAssistantAuthenticationError, HomeAssistantError
from .service import ForecastService, ServiceError


router = APIRouter(prefix="/api/v1")


class HomeAssistantConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(min_length=8, max_length=512)
    access_token: SecretStr = Field(max_length=4096)


def _service(request: Request) -> ForecastService:
    return request.app.state.forecast_service


def _require_machine_token(request: Request, variable: str) -> None:
    expected = os.environ.get(variable)
    if not expected:
        raise HTTPException(status_code=503, detail=f"{variable} is not configured")
    authorization = request.headers.get("authorization", "")
    scheme, separator, provided = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_llm_token(request: Request) -> None:
    _require_machine_token(request, "LLM_API_TOKEN")


def require_hacs_token(request: Request) -> None:
    _require_machine_token(request, "HA_INTEGRATION_TOKEN")


def _raise_service_error(exc: Exception) -> None:
    if isinstance(exc, HomeAssistantAuthenticationError):
        raise HTTPException(status_code=400, detail="Home Assistant rejected the access token") from exc
    if isinstance(exc, HomeAssistantError):
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if isinstance(exc, ServiceError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail="Request could not be completed") from exc


@router.get("/setup/status", dependencies=[Depends(require_llm_token)])
async def setup_status(request: Request) -> dict[str, Any]:
    return _service(request).setup_status()


@router.put("/home-assistant", dependencies=[Depends(require_llm_token)])
async def connect_home_assistant(
    payload: HomeAssistantConnectionRequest, request: Request
) -> dict[str, Any]:
    try:
        return await _service(request).connect_home_assistant(
            payload.base_url, payload.access_token.get_secret_value()
        )
    except (HomeAssistantError, ServiceError, ValueError) as exc:
        _raise_service_error(exc)


@router.get("/entities", dependencies=[Depends(require_llm_token)])
async def list_entities(request: Request) -> dict[str, Any]:
    try:
        return await _service(request).entities()
    except (HomeAssistantError, ServiceError) as exc:
        _raise_service_error(exc)


@router.get("/config", dependencies=[Depends(require_llm_token)])
async def get_config(request: Request) -> dict[str, Any]:
    active = _service(request)
    config = active.config.model_dump(mode="json") if active.config else None
    return {"configured": config is not None, "config": config}


@router.put("/config", dependencies=[Depends(require_llm_token)])
async def put_config(payload: ForecastConfig, request: Request) -> dict[str, Any]:
    try:
        return await _service(request).save_config(payload)
    except ServiceError as exc:
        _raise_service_error(exc)


@router.post("/calibration", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_llm_token)])
async def start_calibration(request: Request) -> JSONResponse:
    try:
        run_id = await _service(request).start_calibration()
    except ServiceError as exc:
        _raise_service_error(exc)
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content={"run_id": run_id, "status": "queued"})


@router.get("/calibration/{run_id}", dependencies=[Depends(require_llm_token)])
async def calibration_status(run_id: str, request: Request) -> dict[str, Any]:
    result = _service(request).calibration_status(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Calibration run not found")
    return result


@router.get("/forecast", dependencies=[Depends(require_hacs_token)])
async def forecast(request: Request) -> JSONResponse:
    service = _service(request)
    snapshot = service.snapshot
    if snapshot is None:
        raise HTTPException(status_code=503, detail="No forecast has been computed")
    result = dict(snapshot)
    result.pop("weather_forecast_hourly", None)
    try:
        from datetime import datetime, timezone
        valid_until = datetime.fromisoformat(str(result["valid_until"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=503, detail="Forecast freshness metadata is invalid")
    if valid_until.tzinfo is None or valid_until <= datetime.now(timezone.utc):
        result["status"] = "stale"
    return JSONResponse(content=result)
