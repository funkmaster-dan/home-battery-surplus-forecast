from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import math
from typing import Any, Callable, Iterable, Sequence
from zoneinfo import ZoneInfo

from .config import ForecastConfig, SensorSelection
from .forecast_types import ForecastPoint
from .ha_client import HomeAssistantClient, HomeAssistantError
from .storage import Storage


STEP = timedelta(minutes=5)
RAW_CHUNK = timedelta(days=7)


def parse_timestamp(value: str | int | float | datetime) -> datetime | None:
    try:
        if isinstance(value, datetime):
            result = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            # Recorder statistics return Unix timestamps in milliseconds.
            result = datetime.fromtimestamp(value / 1000, timezone.utc)
        elif isinstance(value, str):
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            return None
    except (OverflowError, OSError, TypeError, ValueError):
        return None
    if result.tzinfo is None:
        return result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _power_w(value: float, unit: str) -> float | None:
    unit = unit.strip().lower().replace(" ", "")
    if unit == "w":
        return value
    if unit == "kw":
        return value * 1000.0
    return None


def _energy_kwh(value: float, unit: str) -> float | None:
    unit = unit.strip().lower().replace(" ", "")
    if unit == "wh":
        return value / 1000.0
    if unit == "kwh":
        return value
    if unit == "mwh":
        return value * 1000.0
    return None


def _slot_starts(start: datetime, end: datetime, step: timedelta = STEP) -> list[datetime]:
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    slots: list[datetime] = []
    cursor = start_utc
    while cursor < end_utc:
        slots.append(cursor)
        cursor += step
    return slots


def _source_priority(source: str) -> int:
    if source == "raw":
        return 0
    if source.endswith("5minute"):
        return 1
    if source.endswith("hour"):
        return 2
    return 3


def _interpolate_instants(
    rows: Sequence[dict[str, Any]],
    slots: Sequence[datetime],
    step: timedelta,
    max_gap: timedelta,
) -> list[ForecastPoint]:
    samples: list[tuple[datetime, float, str, bool]] = []
    for row in rows:
        ts = parse_timestamp(row["timestamp"])
        value = finite_number(row["value"])
        watts = _power_w(value, row["unit"]) if value is not None else None
        if ts is not None and watts is not None and watts >= 0:
            samples.append((ts, watts, row["source"], bool(row.get("estimated"))))
    samples.sort(key=lambda item: item[0])
    output: list[ForecastPoint] = []
    cursor = 0
    for slot_start in slots:
        slot_end = slot_start + step
        center = slot_start + step / 2
        while cursor + 1 < len(samples) and samples[cursor + 1][0] < center:
            cursor += 1
        left = samples[cursor] if cursor < len(samples) and samples[cursor][0] <= center else None
        right = samples[cursor + 1] if cursor + 1 < len(samples) else None
        value: float | None = None
        source: str | None = None
        estimated = False
        if left and right and right[0] >= center and right[0] - left[0] <= max_gap:
            span = (right[0] - left[0]).total_seconds()
            ratio = 0.0 if span == 0 else (center - left[0]).total_seconds() / span
            value = left[1] + ratio * (right[1] - left[1])
            source = left[2] if left[2] == right[2] else f"{left[2]}+{right[2]}"
            estimated = left[3] or right[3]
        elif left and left[0] == center:
            value, source, estimated = left[1], left[2], left[3]
        output.append(ForecastPoint(slot_start, slot_end, value, estimated, source))
    return output


def _normalize_intervals(
    rows: Sequence[dict[str, Any]],
    slots: Sequence[datetime],
    step: timedelta,
    default_interval_minutes: int,
) -> list[ForecastPoint]:
    slot_seconds = step.total_seconds()
    energy: list[float] = [0.0 for _ in slots]
    covered: list[float] = [0.0 for _ in slots]
    source_rank: list[int] = [99 for _ in slots]
    sources: list[set[str]] = [set() for _ in slots]
    estimated: list[bool] = [False for _ in slots]
    for row in rows:
        value = finite_number(row["value"])
        timestamp = parse_timestamp(row["timestamp"])
        if value is None or timestamp is None:
            continue
        semantics = row.get("semantics", "")
        unit = row["unit"]
        resolution = max(1, int(row.get("resolution_seconds") or default_interval_minutes * 60))
        is_end = bool(row.get("timestamp_is_end"))
        if semantics in {"interval_average_power", "instantaneous_power"}:
            watts = _power_w(value, unit)
            interval_start = timestamp - timedelta(seconds=resolution) if is_end else timestamp
            interval_end = timestamp if is_end else timestamp + timedelta(seconds=resolution)
            if watts is None or interval_end <= interval_start:
                continue
            interval_energy_kwh = watts * resolution / 3_600_000.0
        elif semantics in {"interval_energy", "cumulative_energy"}:
            kwh = _energy_kwh(value, unit)
            if kwh is None:
                continue
            interval_start = timestamp - timedelta(seconds=resolution) if is_end else timestamp
            interval_end = timestamp if is_end else timestamp + timedelta(seconds=resolution)
            if interval_end <= interval_start:
                continue
            interval_energy_kwh = kwh
        else:
            continue
        rank = _source_priority(str(row.get("source", "")))
        interval_seconds = (interval_end - interval_start).total_seconds()
        interval_source = str(row.get("source", "unknown"))
        first_index = max(0, int((interval_start - slots[0]).total_seconds() // slot_seconds)) if slots else 0
        last_index = min(len(slots) - 1, int((interval_end - slots[0]).total_seconds() // slot_seconds)) if slots else -1
        for index in range(first_index, last_index + 1):
            slot_start = slots[index]
            slot_end = slot_start + step
            overlap = max(0.0, (min(slot_end, interval_end) - max(slot_start, interval_start)).total_seconds())
            if not overlap:
                continue
            if rank > source_rank[index]:
                continue
            if rank < source_rank[index]:
                energy[index] = 0.0
                covered[index] = 0.0
                sources[index].clear()
                estimated[index] = False
                source_rank[index] = rank
            energy[index] += interval_energy_kwh * overlap / interval_seconds
            covered[index] += overlap
            sources[index].add(interval_source)
            estimated[index] = estimated[index] or bool(row.get("estimated")) or resolution > int(slot_seconds)
    output: list[ForecastPoint] = []
    for index, slot_start in enumerate(slots):
        value = None
        if covered[index] >= slot_seconds - 1e-6:
            value = energy[index] * 3_600_000.0 / slot_seconds
        source = "+".join(sorted(sources[index])) if sources[index] else None
        output.append(ForecastPoint(slot_start, slot_start + step, value, estimated[index], source))
    return output


def normalize_power_samples(
    samples: Sequence[dict[str, Any]],
    start: datetime,
    end: datetime,
    interval_minutes: int = 5,
    step: timedelta = STEP,
    max_interpolation_gap: timedelta = timedelta(minutes=15),
) -> tuple[ForecastPoint, ...]:
    """Normalize raw and statistic readings to five-minute interval-average watts."""
    slots = _slot_starts(start, end, step)
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        groups.setdefault(str(sample.get("source", "unknown")), []).append(sample)
    normalized: list[list[ForecastPoint]] = []
    for source in sorted(groups, key=_source_priority):
        rows = groups[source]
        semantics = {row.get("semantics") for row in rows}
        if semantics <= {"instantaneous_power"}:
            points = _interpolate_instants(rows, slots, step, max_interpolation_gap)
        else:
            interval_rows = [row for row in rows if row.get("semantics") != "instantaneous_power"]
            point_map: dict[datetime, ForecastPoint] = {}
            for semantic in {row.get("semantics") for row in interval_rows}:
                selected = [row for row in interval_rows if row.get("semantics") == semantic]
                if semantic == "cumulative_energy":
                    selected = _cumulative_as_intervals(selected)
                point_map.update({p.start: p for p in _normalize_intervals(selected, slots, step, interval_minutes)})
            points = [point_map.get(slot, ForecastPoint(slot, slot + step, None)) for slot in slots]
        normalized.append(points)
    merged: list[ForecastPoint] = []
    for index, slot in enumerate(slots):
        selected = next((group[index] for group in normalized if group[index].value is not None), None)
        merged.append(selected or ForecastPoint(slot, slot + step, None))
    return tuple(merged)


def _cumulative_as_intervals(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: parse_timestamp(row["timestamp"]) or datetime.min.replace(tzinfo=timezone.utc))
    intervals: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for row in ordered:
        if previous is not None:
            current_time = parse_timestamp(row["timestamp"])
            previous_time = parse_timestamp(previous["timestamp"])
            current = finite_number(row["value"])
            before = finite_number(previous["value"])
            if current_time and previous_time and current is not None and before is not None:
                elapsed = (current_time - previous_time).total_seconds()
                delta = current - before
                if elapsed > 0 and delta >= 0:
                    intervals.append(
                        {
                            **row,
                            "value": delta,
                            "timestamp": current_time.isoformat(),
                            "resolution_seconds": int(elapsed),
                            "semantics": "interval_energy",
                            "timestamp_is_end": True,
                            "estimated": bool(row.get("estimated")) or elapsed > 300,
                        }
                    )
        previous = row
    return intervals


def normalize_temperature_samples(
    samples: Sequence[dict[str, Any]], start: datetime, end: datetime, step: timedelta = STEP
) -> tuple[ForecastPoint, ...]:
    slots = _slot_starts(start, end, step)
    rows: list[tuple[datetime, float, str]] = []
    for sample in samples:
        timestamp = parse_timestamp(sample["timestamp"])
        value = finite_number(sample["value"])
        if timestamp is None or value is None:
            continue
        unit = str(sample.get("unit", "°C")).strip().lower().replace(" ", "")
        if unit in {"°f", "f", "fahrenheit"}:
            value = (value - 32.0) * 5.0 / 9.0
        elif unit not in {"°c", "c", "celsius"}:
            continue
        rows.append((timestamp, value, str(sample.get("source", "unknown"))))
    rows.sort(key=lambda item: item[0])
    output: list[ForecastPoint] = []
    cursor = 0
    max_gap = timedelta(hours=3)
    for slot_start in slots:
        center = slot_start + step / 2
        while cursor + 1 < len(rows) and rows[cursor + 1][0] < center:
            cursor += 1
        left = rows[cursor] if cursor < len(rows) and rows[cursor][0] <= center else None
        right = rows[cursor + 1] if cursor + 1 < len(rows) else None
        value = None
        source = None
        if left and right and right[0] >= center and right[0] - left[0] <= max_gap:
            span = (right[0] - left[0]).total_seconds()
            ratio = 0.0 if span == 0 else (center - left[0]).total_seconds() / span
            value = left[1] + ratio * (right[1] - left[1])
            source = left[2] if left[2] == right[2] else f"{left[2]}+{right[2]}"
        elif left and left[0] == center:
            value, source = left[1], left[2]
        output.append(ForecastPoint(slot_start, slot_start + step, value, value is not None, source))
    return tuple(output)


class HistoryImporter:
    def __init__(self, client: HomeAssistantClient, storage: Storage) -> None:
        self.client = client
        self.storage = storage

    async def import_for_config(
        self, config: ForecastConfig, progress: Callable[[float], None] | None = None
    ) -> dict[str, Any]:
        start, end = calibration_bounds(config.calibration_start, config.calibration_end, config.timezone)
        load_statistic_id = config.consumption.historical_statistic_id
        warnings: list[str] = []
        raw_entities: dict[str, tuple[str, str, int]] = {
            config.battery.entity_id: (config.battery.unit, "battery_state", 300),
            **{
                array.entity_id: (array.unit, array.semantics, array.interval_minutes * 60)
                for array in config.solar_arrays
            },
        }
        if config.temperature_entity_id:
            raw_entities[config.temperature_entity_id] = (config.temperature_unit, "temperature", 300)
        raw_ids = list(raw_entities)
        if raw_ids:
            cursor = start
            chunk_count = max(1, math.ceil((end - start) / RAW_CHUNK))
            chunk_index = 0
            while cursor < end:
                chunk_end = min(cursor + RAW_CHUNK, end)
                try:
                    history = await self.client.history_period(raw_ids, cursor, chunk_end)
                    rows_to_save: list[dict[str, Any]] = []
                    for entity_states in history:
                        for state in entity_states:
                            entity_id = state.get("entity_id")
                            if entity_id not in raw_entities:
                                continue
                            timestamp = parse_timestamp(state.get("last_changed") or state.get("last_updated"))
                            value = finite_number(state.get("state"))
                            if timestamp is None or value is None:
                                continue
                            unit, semantics, resolution = raw_entities[entity_id]
                            rows_to_save.append(
                                {
                                    "entity_id": entity_id,
                                    "timestamp": timestamp.isoformat(),
                                    "value": value,
                                    "unit": unit,
                                    "source": "raw",
                                    "resolution_seconds": resolution,
                                    "semantics": semantics,
                                    "timestamp_is_end": semantics in {"interval_energy", "cumulative_energy"},
                                }
                            )
                    self.storage.insert_history_samples(rows_to_save)
                except HomeAssistantError as exc:
                    warnings.append(f"Raw history unavailable for one chunk: {exc}")
                cursor = chunk_end
                chunk_index += 1
                if progress:
                    progress(0.05 + 0.25 * chunk_index / chunk_count)

        try:
            metadata = await self.client.list_statistic_ids()
        except HomeAssistantError as exc:
            metadata = []
            warnings.append(f"Long-term statistics metadata unavailable: {exc}")
        load_statistic = next(
            (item for item in metadata if item.get("statistic_id") == load_statistic_id),
            None,
        )
        load_statistic_unit = (
            load_statistic.get("unit_of_measurement") or load_statistic.get("unit")
            if load_statistic
            else None
        )
        if load_statistic and not load_statistic_unit:
            try:
                state = await self.client.get_state(load_statistic_id)
            except HomeAssistantError as exc:
                warnings.append(f"Could not read selected household-load unit: {exc}")
            else:
                load_statistic_unit = ((state or {}).get("attributes") or {}).get("unit_of_measurement")
        descriptors = self._statistic_descriptors(config, metadata, load_statistic_unit)
        if load_statistic_id not in descriptors:
            warnings.append(f"Selected household-load statistic '{load_statistic_id}' is not available")
        statistic_rows = await self._fetch_statistics(descriptors, start, end, warnings)
        if statistic_rows:
            self.storage.insert_history_samples(statistic_rows)
        if progress:
            progress(0.45)
        coverage: dict[str, Any] = {}
        keys = {array.entity_id for array in config.solar_arrays} | {config.battery.entity_id}
        if config.temperature_entity_id:
            keys.add(config.temperature_entity_id)
        keys.add(load_statistic_id)
        for entity_id in sorted(keys):
            data = self.storage.history_coverage(entity_id, start, end)
            expected_slots = max(1, math.ceil((end - start).total_seconds() / STEP.total_seconds()))
            coverage[entity_id] = {
                **data,
                "slot_coverage_pct": round(min(100.0, data["count"] / expected_slots * 100.0), 2),
                "requested_start": start.isoformat(),
                "requested_end": end.isoformat(),
            }
        if progress:
            progress(0.5)
        return {"coverage": coverage, "warnings": warnings, "start": start.isoformat(), "end": end.isoformat()}

    @staticmethod
    def _statistic_descriptors(
        config: ForecastConfig,
        metadata: Sequence[dict[str, Any]],
        load_statistic_unit: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        metadata_by_id = {
            str(item.get("statistic_id")): item
            for item in metadata
            if isinstance(item.get("statistic_id"), str)
        }
        requested: dict[str, tuple[str, str]] = {
            array.entity_id: (array.unit, array.semantics) for array in config.solar_arrays
        }
        requested[config.battery.entity_id] = (config.battery.unit, "battery_state")
        if config.temperature_entity_id:
            requested[config.temperature_entity_id] = (config.temperature_unit, "temperature")
        descriptors: dict[str, dict[str, Any]] = {}
        for statistic_id, (fallback_unit, semantics) in requested.items():
            item = metadata_by_id.get(statistic_id)
            if item:
                descriptors[statistic_id] = {
                    "unit": str(item.get("unit_of_measurement") or fallback_unit),
                    "semantics": semantics,
                }
        load_statistic_id = config.consumption.historical_statistic_id
        load_statistic = metadata_by_id.get(load_statistic_id)
        if load_statistic:
            unit = (
                load_statistic.get("unit_of_measurement")
                or load_statistic.get("unit")
                or load_statistic_unit
                or (config.consumption.unit if load_statistic_id == config.consumption.entity_id else None)
            )
            if unit:
                unit = str(unit)
                normalized_unit = unit.strip().lower().replace(" ", "")
                if normalized_unit in {"w", "kw"}:
                    semantics = "interval_average_power"
                elif normalized_unit in {"wh", "kwh", "mwh"}:
                    semantics = "interval_energy"
                else:
                    semantics = None
                if semantics:
                    descriptors[load_statistic_id] = {"unit": unit, "semantics": semantics}
        return descriptors

    async def _fetch_statistics(
        self,
        descriptors: dict[str, dict[str, Any]],
        start: datetime,
        end: datetime,
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        if not descriptors:
            return []
        requests: list[dict[str, Any]] = []
        for month_start, month_end in calendar_month_chunks(start, end):
            for period in ("5minute", "hour"):
                for kind in ("power", "energy"):
                    ids = []
                    for statistic_id, desc in descriptors.items():
                        semantics = desc["semantics"]
                        is_energy = semantics in {"interval_energy", "cumulative_energy"}
                        if semantics == "battery_state":
                            unit = str(desc["unit"]).strip().lower().replace(" ", "")
                            is_energy = unit in {"wh", "kwh", "mwh"}
                        if semantics == "temperature":
                            is_energy = False
                        if (kind == "energy") == is_energy:
                            ids.append(statistic_id)
                    if ids:
                        requests.append(
                            {
                                "start": month_start,
                                "end": month_end,
                                "statistic_ids": ids,
                                "period": period,
                                "types": ["change", "sum"] if kind == "energy" else ["mean"],
                                "kind": kind,
                            }
                        )
        results = await self.client.statistics_many(requests)
        if any(result is None for result in results):
            warnings.append("Some Home Assistant long-term statistics periods were unavailable")
        raw_by_key: dict[tuple[str, str], list[tuple[datetime, dict[str, Any]]]] = {}
        for request, result in zip(requests, results, strict=False):
            if not isinstance(result, dict):
                continue
            period = request["period"]
            for statistic_id in request["statistic_ids"]:
                records = result.get(statistic_id)
                if not isinstance(records, list):
                    continue
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    timestamp = parse_timestamp(record.get("start"))
                    if timestamp and start <= timestamp < end:
                        raw_by_key.setdefault((statistic_id, period), []).append((timestamp, record))
        output: list[dict[str, Any]] = []
        for (statistic_id, period), records in raw_by_key.items():
            descriptor = descriptors[statistic_id]
            unit = descriptor["unit"]
            semantics = descriptor["semantics"]
            is_energy = semantics in {"interval_energy", "cumulative_energy"} or (
                semantics == "battery_state" and unit.strip().lower().replace(" ", "") in {"wh", "kwh", "mwh"}
            )
            records.sort(key=lambda item: item[0])
            previous_sum: float | None = None
            previous_time: datetime | None = None
            resolution = 300 if period == "5minute" else 3600
            for timestamp, record in records:
                if is_energy:
                    value = finite_number(record.get("change"))
                    if value is None:
                        cumulative = finite_number(record.get("sum"))
                        if cumulative is not None and previous_sum is not None and previous_time is not None:
                            value = cumulative - previous_sum
                            if value < 0:
                                value = None
                        if cumulative is not None:
                            previous_sum = cumulative
                            previous_time = timestamp
                    else:
                        cumulative = finite_number(record.get("sum"))
                        if cumulative is not None:
                            previous_sum = cumulative
                            previous_time = timestamp
                    if value is None or value < 0:
                        continue
                    row_semantics = "interval_energy"
                else:
                    value = finite_number(record.get("mean"))
                    if value is None:
                        continue
                    row_semantics = (
                        "battery_state" if semantics == "battery_state"
                        else "interval_average_power" if semantics != "temperature"
                        else "temperature"
                    )
                output.append(
                    {
                        "entity_id": statistic_id,
                        "timestamp": timestamp.isoformat(),
                        "value": value,
                        "unit": unit,
                        "source": f"statistics_{period}",
                        "resolution_seconds": resolution,
                        "semantics": row_semantics,
                        "timestamp_is_end": row_semantics == "interval_energy",
                        "estimated": period == "hour" and row_semantics in {"interval_average_power", "interval_energy"},
                    }
                )
        return output


def calibration_bounds(start_date: date, end_date: date, timezone_name: str) -> tuple[datetime, datetime]:
    zone = ZoneInfo(timezone_name)
    start = datetime.combine(start_date, time.min, zone).astimezone(timezone.utc)
    end = datetime.combine(end_date, time.min, zone).astimezone(timezone.utc)
    return start, end


def calendar_month_chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    while cursor < end_utc:
        if cursor.month == 12:
            next_month = cursor.replace(year=cursor.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            next_month = cursor.replace(month=cursor.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
        chunk_end = min(next_month, end_utc)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks
