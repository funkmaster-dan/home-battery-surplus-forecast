from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator


class Storage:
    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir or os.environ.get("ENERGY_FORECAST_DATA_DIR", "/data"))
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.data_dir.chmod(0o700)
        except OSError:
            pass
        self.db_path = self.data_dir / "energy_forecast.sqlite3"
        self.model_dir = self.data_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ha_connection (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    base_url TEXT NOT NULL,
                    access_token TEXT NOT NULL,
                    site_config TEXT NOT NULL,
                    saved_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS app_config (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    payload TEXT NOT NULL,
                    saved_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS history_samples (
                    entity_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    source TEXT NOT NULL,
                    resolution_seconds INTEGER NOT NULL,
                    semantics TEXT NOT NULL,
                    timestamp_is_end INTEGER NOT NULL,
                    estimated INTEGER NOT NULL,
                    PRIMARY KEY (entity_id, timestamp, source, resolution_seconds, semantics)
                );
                CREATE INDEX IF NOT EXISTS idx_history_entity_time
                    ON history_samples(entity_id, timestamp);
                CREATE TABLE IF NOT EXISTS weather_cache (
                    cache_key TEXT PRIMARY KEY,
                    fetched_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS calibration_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    coverage TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_metadata (
                    name TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    trained_at TEXT NOT NULL,
                    metrics TEXT NOT NULL
                );
                """
            )
        try:
            self.db_path.chmod(0o600)
        except OSError:
            pass

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def save_ha_connection(self, base_url: str, access_token: str, site_config: dict[str, Any]) -> None:
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO ha_connection(singleton, base_url, access_token, site_config, saved_at)
                   VALUES(1, ?, ?, ?, ?)
                   ON CONFLICT(singleton) DO UPDATE SET base_url=excluded.base_url,
                     access_token=excluded.access_token, site_config=excluded.site_config,
                     saved_at=excluded.saved_at""",
                (base_url.rstrip("/"), access_token, json.dumps(site_config), self._now()),
            )

    def get_ha_connection(self) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT base_url, access_token, site_config, saved_at FROM ha_connection WHERE singleton=1"
            ).fetchone()
        if row is None:
            return None
        return {
            "base_url": row["base_url"],
            "access_token": row["access_token"],
            "site_config": json.loads(row["site_config"]),
            "saved_at": row["saved_at"],
        }

    def clear_ha_connection(self) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM ha_connection WHERE singleton=1")

    def save_config(self, payload: dict[str, Any]) -> None:
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO app_config(singleton, payload, saved_at) VALUES(1, ?, ?)
                   ON CONFLICT(singleton) DO UPDATE SET payload=excluded.payload, saved_at=excluded.saved_at""",
                (json.dumps(payload, separators=(",", ":")), self._now()),
            )

    def get_config(self) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT payload FROM app_config WHERE singleton=1").fetchone()
        return json.loads(row["payload"]) if row else None

    def insert_history_samples(self, samples: Iterable[dict[str, Any]]) -> int:
        rows = [
            (
                sample["entity_id"],
                sample["timestamp"],
                float(sample["value"]),
                sample["unit"],
                sample["source"],
                int(sample["resolution_seconds"]),
                sample["semantics"],
                int(bool(sample.get("timestamp_is_end", False))),
                int(bool(sample.get("estimated", False))),
            )
            for sample in samples
        ]
        if not rows:
            return 0
        with self._connection() as conn:
            conn.executemany(
                """INSERT INTO history_samples(entity_id, timestamp, value, unit, source,
                       resolution_seconds, semantics, timestamp_is_end, estimated)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(entity_id, timestamp, source, resolution_seconds, semantics)
                   DO UPDATE SET value=excluded.value, unit=excluded.unit,
                     timestamp_is_end=excluded.timestamp_is_end, estimated=excluded.estimated""",
                rows,
            )
        return len(rows)

    def get_history(self, entity_id: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT entity_id, timestamp, value, unit, source, resolution_seconds,
                          semantics, timestamp_is_end, estimated
                   FROM history_samples WHERE entity_id=? AND timestamp>=? AND timestamp<?
                   ORDER BY timestamp, CASE source WHEN 'raw' THEN 0 ELSE 1 END""",
                (entity_id, start.astimezone(timezone.utc).isoformat(), end.astimezone(timezone.utc).isoformat()),
            ).fetchall()
        return [dict(row) for row in rows]

    def history_coverage(self, entity_id: str, start: datetime, end: datetime) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS count, MIN(timestamp) AS first, MAX(timestamp) AS last
                   FROM history_samples WHERE entity_id=? AND timestamp>=? AND timestamp<?""",
                (entity_id, start.astimezone(timezone.utc).isoformat(), end.astimezone(timezone.utc).isoformat()),
            ).fetchone()
            sources = conn.execute(
                """SELECT source, COUNT(*) AS count FROM history_samples
                   WHERE entity_id=? AND timestamp>=? AND timestamp<? GROUP BY source""",
                (entity_id, start.astimezone(timezone.utc).isoformat(), end.astimezone(timezone.utc).isoformat()),
            ).fetchall()
        return {"count": row["count"], "first": row["first"], "last": row["last"],
                "sources": {item["source"]: item["count"] for item in sources}}

    def get_weather_cache(self, cache_key: str, max_age_seconds: int) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT fetched_at, payload FROM weather_cache WHERE cache_key=?", (cache_key,)
            ).fetchone()
        if not row:
            return None
        try:
            fetched_at = datetime.fromisoformat(row["fetched_at"])
            age = (datetime.now(timezone.utc) - fetched_at.astimezone(timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            return None
        if age < 0 or age > max_age_seconds:
            return None
        return {"fetched_at": row["fetched_at"], "payload": json.loads(row["payload"])}

    def save_weather_cache(self, cache_key: str, payload: dict[str, Any]) -> None:
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO weather_cache(cache_key, fetched_at, payload) VALUES(?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET fetched_at=excluded.fetched_at,
                     payload=excluded.payload""",
                (cache_key, self._now(), json.dumps(payload, separators=(",", ":"))),
            )

    def create_calibration(self, run_id: str) -> None:
        now = self._now()
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO calibration_runs(run_id,status,progress,coverage,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (run_id, "queued", 0.0, "{}", now, now),
            )

    def update_calibration(
        self, run_id: str, status: str, progress: float, coverage: dict[str, Any], error: str | None = None
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """UPDATE calibration_runs SET status=?, progress=?, coverage=?, error=?, updated_at=?
                   WHERE run_id=?""",
                (status, progress, json.dumps(coverage), error, self._now(), run_id),
            )

    def get_calibration(self, run_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM calibration_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["coverage"] = json.loads(result["coverage"])
        return result

    def save_model_metadata(self, name: str, source: str, metrics: dict[str, Any]) -> None:
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO model_metadata(name,source,trained_at,metrics) VALUES(?,?,?,?)
                   ON CONFLICT(name) DO UPDATE SET source=excluded.source,
                     trained_at=excluded.trained_at, metrics=excluded.metrics""",
                (name, source, self._now(), json.dumps(metrics, allow_nan=False)),
            )

    def get_model_metadata(self) -> dict[str, dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute("SELECT name, source, trained_at, metrics FROM model_metadata").fetchall()
        return {row["name"]: {"source": row["source"], "trained_at": row["trained_at"],
                              "metrics": json.loads(row["metrics"])} for row in rows}

    def clear_model_metadata(self) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM model_metadata")
