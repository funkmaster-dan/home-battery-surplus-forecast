from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from typing import Any, AsyncIterator, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx
import websockets


class HomeAssistantError(RuntimeError):
    pass


class HomeAssistantAuthenticationError(HomeAssistantError):
    pass


class HomeAssistantClient:
    """Small read-only Home Assistant REST and Recorder WebSocket client."""

    def __init__(self, base_url: str, access_token: str, timeout: float = 30.0) -> None:
        parsed = urlsplit(base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Home Assistant URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Home Assistant URL must not contain credentials, query, or fragment")
        if access_token.strip() != access_token or not access_token:
            raise ValueError("Home Assistant access token is required")
        path = parsed.path.rstrip("/")
        self.base_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")
        self.access_token = access_token
        self.timeout = timeout
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=False,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            response = await self._http.get(f"{self.base_url}/api/{path.lstrip('/')}", params=params)
        except httpx.RequestError as exc:
            raise HomeAssistantError("Home Assistant is unreachable") from exc
        if response.status_code in {401, 403}:
            raise HomeAssistantAuthenticationError("Home Assistant rejected the access token")
        if response.status_code >= 400:
            raise HomeAssistantError(f"Home Assistant returned HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise HomeAssistantError("Home Assistant returned invalid JSON") from exc

    async def validate(self) -> dict[str, Any]:
        response = await self._get("/")
        if not isinstance(response, dict) or response.get("message") != "API running.":
            raise HomeAssistantError("Home Assistant API response was not recognized")
        return response

    async def get_site_config(self) -> dict[str, Any]:
        response = await self._get("config")
        if not isinstance(response, dict):
            raise HomeAssistantError("Home Assistant configuration response was invalid")
        return {
            key: response.get(key)
            for key in ("latitude", "longitude", "elevation", "time_zone", "location_name")
        }

    async def get_states(self) -> list[dict[str, Any]]:
        response = await self._get("states")
        if not isinstance(response, list):
            raise HomeAssistantError("Home Assistant state list was invalid")
        return [state for state in response if isinstance(state, dict)]

    async def get_state(self, entity_id: str) -> dict[str, Any] | None:
        try:
            response = await self._get(f"states/{entity_id}")
        except HomeAssistantError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise
        return response if isinstance(response, dict) else None

    async def history_period(
        self, entity_ids: Sequence[str], start: datetime, end: datetime
    ) -> list[list[dict[str, Any]]]:
        if not entity_ids:
            return []
        start_utc = start.astimezone(timezone.utc).isoformat()
        end_utc = end.astimezone(timezone.utc).isoformat()
        return await self._get(
            f"history/period/{start_utc}",
            params={
                "filter_entity_id": ",".join(entity_ids),
                "end_time": end_utc,
                "minimal_response": "1",
                "no_attributes": "1",
                "significant_changes_only": "0",
            },
        )

    def _websocket_url(self) -> str:
        parsed = urlsplit(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunsplit((scheme, parsed.netloc, f"{parsed.path.rstrip('/')}/api/websocket", "", ""))

    @asynccontextmanager
    async def _authenticated_websocket(self) -> AsyncIterator[Any]:
        try:
            async with websockets.connect(
                self._websocket_url(),
                open_timeout=10,
                close_timeout=5,
                max_size=32 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
            ) as socket:
                first = json.loads(await socket.recv())
                if first.get("type") != "auth_required":
                    raise HomeAssistantError("Home Assistant WebSocket handshake was invalid")
                await socket.send(json.dumps({"type": "auth", "access_token": self.access_token}))
                auth = json.loads(await socket.recv())
                if auth.get("type") == "auth_invalid":
                    raise HomeAssistantAuthenticationError("Home Assistant rejected the access token")
                if auth.get("type") != "auth_ok":
                    raise HomeAssistantError("Home Assistant WebSocket authentication failed")
                yield socket
        except HomeAssistantError:
            raise
        except Exception as exc:
            raise HomeAssistantError("Home Assistant Recorder WebSocket request failed") from exc

    @staticmethod
    async def _command(socket: Any, command_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        await socket.send(json.dumps({"id": command_id, **payload}))
        while True:
            response = json.loads(await socket.recv())
            if response.get("id") != command_id:
                continue
            if response.get("type") != "result":
                raise HomeAssistantError("Home Assistant returned an unexpected WebSocket response")
            if not response.get("success"):
                error = response.get("error") or {}
                message = error.get("message") if isinstance(error, dict) else None
                raise HomeAssistantError(message or "Home Assistant Recorder command failed")
            return response.get("result")

    async def list_statistic_ids(self) -> list[dict[str, Any]]:
        async with self._authenticated_websocket() as socket:
            result = await self._command(socket, 1, {"type": "recorder/list_statistic_ids"})
        if not isinstance(result, list):
            raise HomeAssistantError("Home Assistant returned invalid statistic metadata")
        return [item for item in result if isinstance(item, dict)]

    async def statistics_many(self, requests: Sequence[dict[str, Any]]) -> list[Any | None]:
        """Run read-only statistics commands over one authenticated WebSocket."""
        if not requests:
            return []
        results: list[Any | None] = []
        async with self._authenticated_websocket() as socket:
            for index, request in enumerate(requests, start=1):
                try:
                    results.append(
                        await self._command(
                            socket,
                            index,
                            {
                                "type": "recorder/statistics_during_period",
                                "start_time": _iso(request["start"]),
                                "end_time": _iso(request["end"]),
                                "statistic_ids": request["statistic_ids"],
                                "period": request["period"],
                                "types": request["types"],
                            },
                        )
                    )
                except HomeAssistantError:
                    results.append(None)
        return results


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()
