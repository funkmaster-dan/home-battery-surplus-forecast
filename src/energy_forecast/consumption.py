from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np

from .forecast_types import ForecastPoint, ForecastSeries


class ConsumptionModelUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class LoadModel:
    method: str
    estimator: lgb.LGBMRegressor | None
    weekday_slot_medians: dict[tuple[int, int], float]
    slot_medians: dict[int, float]
    global_median: float
    metrics: dict[str, Any]
    site_timezone: str
    historical_statistic_id: str

class ConsumptionForecaster:
    """Time-of-week median baseline compared with a temperature-aware LightGBM."""

    def __init__(self) -> None:
        self.model: LoadModel | None = None

    def fit(
        self,
        targets: Sequence[ForecastPoint],
        actual_temperature: Sequence[ForecastPoint] | None,
        site_timezone: str,
        historical_statistic_id: str,
    ) -> dict[str, Any]:
        valid = [point for point in targets if point.value is not None and math.isfinite(point.value) and point.value >= 0]
        if len(valid) < 48:
            raise ConsumptionModelUnavailable(f"{historical_statistic_id} has fewer than 48 valid five-minute intervals")
        zone = ZoneInfo(site_timezone)
        dates = sorted({point.start.astimezone(zone).date() for point in valid})
        if len(dates) < 2:
            raise ConsumptionModelUnavailable(f"{historical_statistic_id} has no chronological holdout dates")
        split_at = max(1, int(len(dates) * 0.8))
        if split_at >= len(dates):
            split_at = len(dates) - 1
        train_dates = set(dates[:split_at])
        holdout_dates = set(dates[split_at:])
        temperatures = {
            point.start.astimezone(timezone.utc): float(point.value)
            for point in (actual_temperature or [])
            if point.value is not None and math.isfinite(point.value)
        }
        rows = [
            {
                "point": point,
                "date": point.start.astimezone(zone).date(),
                "target": float(point.value),
                "weekday": point.start.astimezone(zone).weekday(),
                "slot": point.start.astimezone(zone).hour * 12 + point.start.astimezone(zone).minute // 5,
                "features": _features(point.start, site_timezone, temperatures.get(point.start.astimezone(timezone.utc))),
            }
            for point in valid
        ]
        train_rows = [row for row in rows if row["date"] in train_dates]
        holdout_rows = [row for row in rows if row["date"] in holdout_dates]
        if len(train_rows) < 24 or len(holdout_rows) < 12:
            raise ConsumptionModelUnavailable(f"{historical_statistic_id} lacks usable training or chronological holdout samples")
        baseline = _fit_baseline(train_rows)
        baseline_predictions = np.asarray([_baseline_predict(row, baseline) for row in holdout_rows])
        baseline_metrics = _load_metrics(holdout_rows, baseline_predictions, site_timezone)

        train_with_temp = [row for row in train_rows if row["features"] is not None]
        holdout_with_temp = [row for row in holdout_rows if row["features"] is not None]
        selected_method = "time_of_week_median"
        estimator: lgb.LGBMRegressor | None = None
        tree_metrics: dict[str, Any] | None = None
        tree_predictions: np.ndarray | None = None
        if len(train_with_temp) >= 100 and len(holdout_with_temp) >= 24:
            candidate = _new_regressor()
            candidate.fit(
                np.asarray([row["features"] for row in train_with_temp]),
                np.asarray([row["target"] for row in train_with_temp]),
            )
            tree_predictions = np.maximum(
                0.0, candidate.predict(np.asarray([row["features"] for row in holdout_with_temp]))
            )
            tree_metrics = _load_metrics(holdout_with_temp, tree_predictions, site_timezone)
            baseline_comparable = np.asarray([_baseline_predict(row, baseline) for row in holdout_with_temp])
            baseline_comparable_mae = _load_metrics(
                holdout_with_temp, baseline_comparable, site_timezone
            )["hourly_kwh_mae"]
            if (
                tree_metrics["hourly_kwh_mae"] is not None
                and baseline_comparable_mae is not None
                and tree_metrics["hourly_kwh_mae"] < baseline_comparable_mae
            ):
                selected_method = "lightgbm"
        final_baseline = _fit_baseline(rows)
        if selected_method == "lightgbm":
            final_rows = [row for row in rows if row["features"] is not None]
            final_tree = _new_regressor()
            final_tree.fit(
                np.asarray([row["features"] for row in final_rows]),
                np.asarray([row["target"] for row in final_rows]),
            )
            estimator = final_tree
            selected_holdout_rows = holdout_with_temp
            selected_holdout_predictions = tree_predictions
        else:
            selected_holdout_rows = holdout_rows
            selected_holdout_predictions = baseline_predictions
        selected_metrics = _load_metrics(selected_holdout_rows, selected_holdout_predictions, site_timezone)
        metrics = {
            **selected_metrics,
            "selected_model": selected_method,
            "baseline_hourly_kwh_mae": baseline_metrics["hourly_kwh_mae"],
            "tree_hourly_kwh_mae": tree_metrics["hourly_kwh_mae"] if tree_metrics else None,
            "training_samples": len(rows),
            "valid_history_intervals": len(valid),
            "expected_history_intervals": len(targets),
            "history_coverage_pct": round(len(valid) / len(targets) * 100, 2) if targets else 0.0,
            "holdout_samples": len(holdout_rows),
            "train_dates": [min(train_dates).isoformat(), max(train_dates).isoformat()],
            "holdout_dates": [min(holdout_dates).isoformat(), max(holdout_dates).isoformat()],
            "historical_statistic_id": historical_statistic_id,
            "temperature_samples": len(temperatures),
        }
        self.model = LoadModel(
            selected_method,
            estimator,
            final_baseline[0],
            final_baseline[1],
            final_baseline[2],
            metrics,
            site_timezone,
            historical_statistic_id,
        )
        return metrics

    def predict(
        self,
        start: datetime,
        horizon_hours: int,
        generated_at: datetime,
        site_timezone: str,
        forecast_temperature: Sequence[ForecastPoint] | None = None,
    ) -> ForecastSeries:
        model = self.model
        if model is None:
            raise ConsumptionModelUnavailable("No trained household-load model is available")
        temperatures = {
            point.start.astimezone(timezone.utc): float(point.value)
            for point in (forecast_temperature or [])
            if point.value is not None and math.isfinite(point.value)
        }
        end = start.astimezone(timezone.utc) + timedelta(hours=horizon_hours)
        cursor = start.astimezone(timezone.utc)
        slots: list[datetime] = []
        values: list[float | None] = []
        tree_rows: list[list[float]] = []
        tree_indices: list[int] = []
        while cursor < end:
            local = cursor.astimezone(ZoneInfo(site_timezone))
            row = {"weekday": local.weekday(), "slot": local.hour * 12 + local.minute // 5}
            slots.append(cursor)
            if model.method == "lightgbm":
                features = _features(cursor, site_timezone, temperatures.get(cursor))
                values.append(None)
                if features is not None and model.estimator is not None:
                    tree_rows.append(features)
                    tree_indices.append(len(slots) - 1)
            else:
                values.append(
                    _baseline_predict(row, (model.weekday_slot_medians, model.slot_medians, model.global_median))
                )
            cursor += timedelta(minutes=5)
        if tree_rows and model.estimator is not None:
            predictions = np.maximum(0.0, model.estimator.predict(np.asarray(tree_rows)))
            for index, value in zip(tree_indices, predictions, strict=True):
                values[index] = float(value)
        source = (
            f"lightgbm:{model.historical_statistic_id}"
            if model.method == "lightgbm"
            else f"time_of_week_median:{model.historical_statistic_id}"
        )
        completeness = sum(value is not None for value in values) / len(values) if values else 0.0
        points = [
            ForecastPoint(slot, slot + timedelta(minutes=5), value, False, source)
            for slot, value in zip(slots, values, strict=True)
        ]
        return ForecastSeries(
            "household_consumption_power",
            "W",
            tuple(points),
            generated_at,
            site_timezone,
            model.method,
            completeness,
        )

    @property
    def metrics(self) -> dict[str, Any] | None:
        return self.model.metrics if self.model else None


def interpolate_temperature(
    observations: dict[datetime, float | None],
    start: datetime,
    end: datetime,
    max_gap: timedelta = timedelta(hours=3),
) -> tuple[ForecastPoint, ...]:
    values = sorted(
        (timestamp.astimezone(timezone.utc), float(value))
        for timestamp, value in observations.items()
        if value is not None and math.isfinite(value)
    )
    if not values:
        return ()
    cursor = start.astimezone(timezone.utc)
    limit = end.astimezone(timezone.utc)
    output: list[ForecastPoint] = []
    index = 0
    step = timedelta(minutes=5)
    while cursor < limit:
        center = cursor + step / 2
        while index + 1 < len(values) and values[index + 1][0] < center:
            index += 1
        left = values[index] if index < len(values) and values[index][0] <= center else None
        right = values[index + 1] if index + 1 < len(values) else None
        value = None
        if left and right and right[0] >= center and right[0] - left[0] <= max_gap:
            span = (right[0] - left[0]).total_seconds()
            ratio = 0.0 if span == 0 else (center - left[0]).total_seconds() / span
            value = left[1] + ratio * (right[1] - left[1])
        elif left and left[0] == center:
            value = left[1]
        output.append(ForecastPoint(cursor, cursor + step, value, value is not None, "weather_temperature"))
        cursor += step
    return tuple(output)


def _features(timestamp: datetime, site_timezone: str, temperature_c: float | None) -> list[float] | None:
    if temperature_c is None or not math.isfinite(temperature_c):
        return None
    local = timestamp.astimezone(ZoneInfo(site_timezone))
    minute_of_day = local.hour * 60 + local.minute
    weekday = local.weekday()
    day_of_year = local.timetuple().tm_yday
    return [
        math.sin(2 * math.pi * minute_of_day / 1440),
        math.cos(2 * math.pi * minute_of_day / 1440),
        math.sin(2 * math.pi * weekday / 7),
        math.cos(2 * math.pi * weekday / 7),
        1.0 if weekday >= 5 else 0.0,
        math.sin(2 * math.pi * day_of_year / 365.2425),
        math.cos(2 * math.pi * day_of_year / 365.2425),
        float(temperature_c),
    ]


def _fit_baseline(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[tuple[int, int], float], dict[int, float], float]:
    by_weekday_slot: dict[tuple[int, int], list[float]] = {}
    by_slot: dict[int, list[float]] = {}
    values: list[float] = []
    for row in rows:
        value = float(row["target"])
        by_weekday_slot.setdefault((row["weekday"], row["slot"]), []).append(value)
        by_slot.setdefault(row["slot"], []).append(value)
        values.append(value)
    if not values:
        raise ConsumptionModelUnavailable("No training values for household-load baseline")
    weekday_medians = {key: float(np.median(items)) for key, items in by_weekday_slot.items()}
    slot_medians = {key: float(np.median(items)) for key, items in by_slot.items()}
    return weekday_medians, slot_medians, float(np.median(values))


def _baseline_predict(
    row: dict[str, Any],
    baseline: tuple[dict[tuple[int, int], float], dict[int, float], float],
) -> float:
    weekday_medians, slot_medians, global_median = baseline
    key = (row["weekday"], row["slot"])
    return weekday_medians.get(key, slot_medians.get(row["slot"], global_median))


def _new_regressor() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=250,
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


def _load_metrics(rows: Sequence[dict[str, Any]], predictions: np.ndarray, site_timezone: str) -> dict[str, Any]:
    zone = ZoneInfo(site_timezone)
    hourly_truth: dict[tuple[Any, int], float] = {}
    hourly_pred: dict[tuple[Any, int], float] = {}
    hourly_coverage: dict[tuple[Any, int], float] = {}
    daily_truth: dict[Any, float] = {}
    daily_pred: dict[Any, float] = {}
    five_minute_errors: list[float] = []
    for row, prediction in zip(rows, predictions, strict=True):
        point = row["point"]
        local = point.start.astimezone(zone)
        duration = (point.end - point.start).total_seconds() / 3600.0
        energy_scale = duration / 1000.0
        hour_key = (local.date(), local.hour)
        hourly_truth[hour_key] = hourly_truth.get(hour_key, 0.0) + row["target"] * energy_scale
        hourly_pred[hour_key] = hourly_pred.get(hour_key, 0.0) + float(prediction) * energy_scale
        hourly_coverage[hour_key] = hourly_coverage.get(hour_key, 0.0) + duration
        daily_truth[local.date()] = daily_truth.get(local.date(), 0.0) + row["target"] * energy_scale
        daily_pred[local.date()] = daily_pred.get(local.date(), 0.0) + float(prediction) * energy_scale
        five_minute_errors.append(abs(row["target"] - float(prediction)))
    valid_hours = [key for key, covered in hourly_coverage.items() if covered >= 0.5]
    hourly_mae = float(np.mean([abs(hourly_truth[key] - hourly_pred[key]) for key in valid_hours])) if valid_hours else None
    valid_days = sorted(daily_truth)
    daily_mae = float(np.mean([abs(daily_truth[day] - daily_pred.get(day, 0.0)) for day in valid_days])) if valid_days else None
    return {
        "hourly_kwh_mae": hourly_mae,
        "daily_kwh_mae": daily_mae,
        "five_minute_w_mae": float(np.mean(five_minute_errors)) if five_minute_errors else None,
        "holdout_hours": len(valid_hours),
        "holdout_days": len(valid_days),
    }
