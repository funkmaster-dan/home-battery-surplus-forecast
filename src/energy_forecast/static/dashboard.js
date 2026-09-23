const $ = (id) => document.getElementById(id);
let lastSnapshot;

async function requestJSON(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

function fmt(value, unit, digits = 1) {
  return value === null || value === undefined || !Number.isFinite(Number(value))
    ? "Unavailable"
    : `${Number(value).toFixed(digits)} ${unit}`;
}

function localTime(value, timezone) {
  if (!value) return "—";
  const date = new Date(value);
  return new Intl.DateTimeFormat(undefined, {
    timeZone: timezone,
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function detailRow(label, value) {
  const row = document.createElement("div");
  row.className = "detail-row";
  const name = document.createElement("span");
  name.textContent = label;
  const content = document.createElement("strong");
  content.textContent = value ?? "Unavailable";
  row.append(name, content);
  return row;
}

function drawChart(canvas, series, unit, color, timezone, maxOverride = null) {
  const context = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, rect.width);
  const height = Number(canvas.getAttribute("height")) || 240;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.height = `${height}px`;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  const pad = { left: 55, right: 18, top: 16, bottom: 34 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const values = series.map((item) => item.value).filter((value) => value !== null && Number.isFinite(value));
  if (!values.length) {
    context.fillStyle = "#73877d";
    context.font = "13px DM Sans, sans-serif";
    context.fillText("No complete forecast profile is available", pad.left, pad.top + 24);
    return;
  }
  const maxValue = maxOverride || Math.max(1, ...values) * 1.08;
  context.strokeStyle = "#e4ece7";
  context.fillStyle = "#788980";
  context.font = "10px DM Sans, sans-serif";
  context.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick++) {
    const y = pad.top + plotHeight * tick / 4;
    const labelValue = maxValue * (1 - tick / 4);
    context.beginPath();
    context.moveTo(pad.left, y);
    context.lineTo(width - pad.right, y);
    context.stroke();
    context.textAlign = "right";
    context.fillText(`${labelValue.toFixed(0)} ${unit}`, pad.left - 8, y + 3);
  }
  const xLabelCount = Math.min(8, series.length);
  for (let tick = 0; tick < xLabelCount; tick++) {
    const index = Math.round(tick * (series.length - 1) / Math.max(1, xLabelCount - 1));
    const x = pad.left + plotWidth * index / Math.max(1, series.length - 1);
    context.textAlign = tick === 0 ? "left" : tick === xLabelCount - 1 ? "right" : "center";
    context.fillText(localTime(series[index].start, timezone), x, height - 10);
  }
  context.strokeStyle = color;
  context.lineWidth = 2.2;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.beginPath();
  let drawing = false;
  series.forEach((item, index) => {
    if (item.value === null || !Number.isFinite(item.value)) {
      drawing = false;
      return;
    }
    const x = pad.left + plotWidth * index / Math.max(1, series.length - 1);
    const y = pad.top + plotHeight * (1 - Math.max(0, item.value) / maxValue);
    if (drawing) context.lineTo(x, y);
    else { context.moveTo(x, y); drawing = true; }
  });
  context.stroke();
}

function showModelDetails(snapshot) {
  const host = $("model-status");
  host.replaceChildren();
  const models = snapshot.model_status || {};
  const load = models.consumption || {};
  const selectedLoad = load.selected_model || load.method || (load.available ? "trained" : "unavailable");
  const statisticId = load.historical_statistic_id || "Not reported";
  host.append(detailRow("Historical load source", `${statisticId} · Long Term Statistics`));
  host.append(detailRow("Load model", selectedLoad));
  host.append(detailRow("Load hourly energy MAE", fmt(load.hourly_kwh_mae, "kWh")));
  host.append(detailRow("Load five-minute power MAE", fmt(load.five_minute_w_mae, "W", 0)));
  host.append(detailRow("Load temperature source", load.temperature_source || "Archive or selected HA sensor"));
  for (const [arrayId, info] of Object.entries(models.solar_by_array || {})) {
    const label = info.model ? `${info.model} · ${info.resolution || ""}` : (info.error || "unavailable");
    host.append(detailRow(`PV ${arrayId}`, label));
    if (info.daily_kwh_mae !== undefined) host.append(detailRow(`${arrayId} daily energy MAE`, fmt(info.daily_kwh_mae, "kWh")));
  }
  const coverage = snapshot.coverage || {};
  const loadCoverage = coverage[statisticId];
  host.append(detailRow(
    `${statisticId} five-minute coverage`,
    loadCoverage ? `${loadCoverage.slot_coverage_pct ?? "—"}% · ${loadCoverage.valid_five_minute_slots ?? 0} valid slots` : "Not reported",
  ));
  const battery = models.battery || {};
  host.append(detailRow("Battery baseline", battery.available ? "Available" : battery.error || "Unavailable"));
}

function showFreshness(snapshot) {
  const host = $("freshness-details");
  host.replaceChildren();
  const freshness = snapshot.freshness || {};
  host.append(detailRow("Last calculation", localTime(snapshot.generated_at, snapshot.site_timezone)));
  host.append(detailRow("Weather age", fmt(freshness.weather_age_minutes, "min", 0)));
  host.append(detailRow("Weather provider", Object.values(freshness.weather_sources || {}).join(", ") || "Unavailable"));
  const elevation = Object.values(freshness.weather_elevation || {})[0];
  host.append(detailRow("Solar elevation input", elevation ? `${elevation.meters} m · ${elevation.source}` : "Unavailable"));
  const warning = $("freshness-warning");
  const errors = Object.values(freshness.weather_errors || {});
  if (snapshot.status === "stale" || errors.length || snapshot.last_error) {
    warning.hidden = false;
    warning.textContent = snapshot.last_error || (errors.length ? errors.join(" · ") : `Data is stale; generated ${snapshot.generated_at}.`);
  } else {
    warning.hidden = true;
    warning.textContent = "";
  }
}

function render(snapshot) {
  lastSnapshot = snapshot;
  const status = $("forecast-status");
  status.textContent = snapshot.status || "unknown";
  status.className = `status-pill ${snapshot.status === "ready" ? "success" : snapshot.status === "stale" ? "danger" : ""}`;
  $("forecast-subtitle").textContent = `${snapshot.horizon_hours} hours · ${snapshot.site_timezone} · valid until ${localTime(snapshot.valid_until, snapshot.site_timezone)}`;
  $("metric-solar").textContent = fmt(snapshot.solar_generation_kwh, "kWh");
  $("metric-load").textContent = fmt(snapshot.home_consumption_kwh, "kWh");
  $("metric-surplus").textContent = fmt(snapshot.battery_surplus_min_kwh, "kWh");
  $("metric-soc").textContent = fmt(snapshot.battery_minimum_soc_pct, "%");
  $("metric-minimum-at").textContent = snapshot.battery_minimum_at
    ? `at ${localTime(snapshot.battery_minimum_at, snapshot.site_timezone)}`
    : "projected minimum";
  $("metric-grid").textContent = fmt(snapshot.non_free_grid_import_kwh, "kWh");
  const intervals = snapshot.intervals_5m || [];
  $("chart-range").textContent = intervals.length ? `${localTime(intervals[0].start, snapshot.site_timezone)} — ${localTime(intervals[intervals.length - 1].end, snapshot.site_timezone)}` : "No intervals";
  // Draw both interval-average power series against the same axis.
  drawPowerChart(intervals, snapshot.site_timezone);
  drawWeatherChart(snapshot.weather_forecast_hourly || [], snapshot.site_timezone);
  drawChart($("battery-chart"), intervals.map((point) => ({ start: point.start, value: point.battery_energy_kwh })), "kWh", "#287a55", snapshot.site_timezone);
  showModelDetails(snapshot);
  showFreshness(snapshot);
}

function drawPowerChart(intervals, timezone) {
  const canvas = $("power-chart");
  const context = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, rect.width);
  const height = Number(canvas.getAttribute("height")) || 260;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.height = `${height}px`;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  const pad = { left: 55, right: 18, top: 16, bottom: 34 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const values = intervals.flatMap((point) => [point.solar_power_w, point.home_load_power_w]).filter((value) => value !== null && Number.isFinite(value));
  if (!values.length) {
    context.fillStyle = "#73877d";
    context.font = "13px DM Sans, sans-serif";
    context.fillText("No complete solar/load profile is available", pad.left, pad.top + 24);
    return;
  }
  const maxValue = Math.max(1, ...values) * 1.08;
  context.strokeStyle = "#e4ece7";
  context.fillStyle = "#788980";
  context.font = "10px DM Sans, sans-serif";
  for (let tick = 0; tick <= 4; tick++) {
    const y = pad.top + plotHeight * tick / 4;
    context.beginPath(); context.moveTo(pad.left, y); context.lineTo(width - pad.right, y); context.stroke();
    context.textAlign = "right";
    context.fillText(`${(maxValue * (1 - tick / 4)).toFixed(0)} W`, pad.left - 8, y + 3);
  }
  const labelCount = Math.min(8, intervals.length);
  for (let tick = 0; tick < labelCount; tick++) {
    const index = Math.round(tick * (intervals.length - 1) / Math.max(1, labelCount - 1));
    const x = pad.left + plotWidth * index / Math.max(1, intervals.length - 1);
    context.textAlign = tick === 0 ? "left" : tick === labelCount - 1 ? "right" : "center";
    context.fillText(localTime(intervals[index].start, timezone), x, height - 10);
  }
  for (const [key, color] of [["solar_power_w", "#e6a63e"], ["home_load_power_w", "#3486a0"]]) {
    context.strokeStyle = color;
    context.lineWidth = 2.2;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.beginPath();
    let drawing = false;
    intervals.forEach((point, index) => {
      const value = point[key];
      if (value === null || !Number.isFinite(value)) { drawing = false; return; }
      const x = pad.left + plotWidth * index / Math.max(1, intervals.length - 1);
      const y = pad.top + plotHeight * (1 - Math.max(0, value) / maxValue);
      if (drawing) context.lineTo(x, y); else { context.moveTo(x, y); drawing = true; }
    });
    context.stroke();
  }
}


function drawWeatherChart(weather, timezone) {
  const canvas = $("weather-chart");
  const legend = $("weather-legend");
  const context = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, rect.width);
  const height = Number(canvas.getAttribute("height")) || 260;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.height = `${height}px`;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  legend.replaceChildren();

  const orientations = [...new Set(weather.flatMap((hour) => Object.keys(hour.irradiance_wm2_by_orientation || {})))].sort();
  const temperatureColor = "#3486a0";
  const orientationColors = new Map(orientations.map((name, index) => [name, `hsl(${(38 + index * 67) % 360} 62% 46%)`]));
  const addLegendItem = (label, color) => {
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.backgroundColor = color;
    item.append(swatch, document.createTextNode(label));
    legend.append(item);
  };

  const temperatures = weather
    .map((hour) => ({ time: Date.parse(hour.end), value: hour.temperature_c }))
    .filter((point) => Number.isFinite(point.time) && point.value !== null && Number.isFinite(Number(point.value)))
    .map((point) => ({ ...point, value: Number(point.value) }));
  const irradianceValues = orientations.flatMap((orientation) =>
    weather.map((hour) => hour.irradiance_wm2_by_orientation?.[orientation])
  ).filter((value) => value !== null && value !== undefined && Number.isFinite(Number(value))).map(Number);
  if (temperatures.length) addLegendItem("Temperature (°C)", temperatureColor);
  for (const orientation of orientations) {
    const hasValues = weather.some((hour) => {
      const value = hour.irradiance_wm2_by_orientation?.[orientation];
      return value !== null && value !== undefined && Number.isFinite(Number(value));
    });
    if (hasValues) addLegendItem(`${orientation} irradiance`, orientationColors.get(orientation));
  }

  const timeValues = weather.flatMap((hour) => [Date.parse(hour.start), Date.parse(hour.end)]).filter(Number.isFinite);
  if (!timeValues.length || (!temperatures.length && !irradianceValues.length)) {
    context.fillStyle = "#73877d";
    context.font = "13px DM Sans, sans-serif";
    context.fillText("No hourly weather forecast is available", 58, 42);
    return;
  }

  const minTime = Math.min(...timeValues);
  const maxTime = Math.max(...timeValues);
  const timeRange = Math.max(1, maxTime - minTime);
  const rawTempMin = temperatures.length ? Math.min(...temperatures.map((point) => point.value)) : 0;
  const rawTempMax = temperatures.length ? Math.max(...temperatures.map((point) => point.value)) : 1;
  const tempPadding = Math.max(1, (rawTempMax - rawTempMin) * 0.08);
  const tempMin = rawTempMin - tempPadding;
  const tempMax = rawTempMax + tempPadding;
  const irradianceMax = Math.max(1, ...irradianceValues) * 1.08;
  const pad = { left: 66, right: 66, top: 16, bottom: 36 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  context.strokeStyle = "#e4ece7";
  context.fillStyle = "#788980";
  context.font = "10px DM Sans, sans-serif";
  context.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick++) {
    const ratio = tick / 4;
    const y = pad.top + plotHeight * ratio;
    context.beginPath();
    context.moveTo(pad.left, y);
    context.lineTo(width - pad.right, y);
    context.stroke();
    if (irradianceValues.length) {
      context.textAlign = "right";
      context.fillText(`${(irradianceMax * (1 - ratio)).toFixed(0)} W/m²`, pad.left - 8, y + 3);
    }
    if (temperatures.length) {
      context.textAlign = "left";
      context.fillText(`${(tempMax - (tempMax - tempMin) * ratio).toFixed(1)} °C`, width - pad.right + 8, y + 3);
    }
  }
  const labelCount = Math.min(8, weather.length);
  for (let tick = 0; tick < labelCount; tick++) {
    const ratio = tick / Math.max(1, labelCount - 1);
    const x = pad.left + plotWidth * ratio;
    context.textAlign = tick === 0 ? "left" : tick === labelCount - 1 ? "right" : "center";
    context.fillText(localTime(new Date(minTime + timeRange * ratio).toISOString(), timezone), x, height - 10);
  }

  const drawLine = (series, color, valueToY) => {
    context.strokeStyle = color;
    context.lineWidth = 2.2;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.beginPath();
    let drawing = false;
    for (const point of series) {
      if (!Number.isFinite(point.time) || point.value === null || point.value === undefined || !Number.isFinite(Number(point.value))) {
        drawing = false;
        continue;
      }
      const x = pad.left + plotWidth * (point.time - minTime) / timeRange;
      const y = valueToY(Number(point.value));
      if (drawing) context.lineTo(x, y);
      else { context.moveTo(x, y); drawing = true; }
    }
    context.stroke();
  };
  for (const orientation of orientations) {
    const series = weather.map((hour) => {
      const start = Date.parse(hour.start);
      const end = Date.parse(hour.end);
      return { time: (start + end) / 2, value: hour.irradiance_wm2_by_orientation?.[orientation] };
    });
    drawLine(series, orientationColors.get(orientation), (value) => pad.top + plotHeight * (1 - value / irradianceMax));
  }
  drawLine(temperatures, temperatureColor, (value) => pad.top + plotHeight * (1 - (value - tempMin) / (tempMax - tempMin)));
}
async function loadForecast() {
  try {
    const snapshot = await requestJSON("/ui/api/forecast");
    render(snapshot);
  } catch (error) {
    $("forecast-status").textContent = "waiting";
    $("forecast-status").className = "status-pill";
    $("freshness-warning").hidden = false;
    $("freshness-warning").textContent = error.message;
  }
}


async function recomputeForecast() {
  const button = $("recompute-forecast");
  button.disabled = true;
  button.textContent = "Recomputing…";
  try {
    render(await requestJSON("/ui/api/forecast/refresh", { method: "POST" }));
  } catch (error) {
    const warning = $("freshness-warning");
    warning.hidden = false;
    warning.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Recompute forecast";
  }
}

$("recompute-forecast").addEventListener("click", recomputeForecast);

$("recalibrate").addEventListener("click", async () => {
  const button = $("recalibrate");
  button.disabled = true;
  try {
    const result = await requestJSON("/ui/api/calibration", { method: "POST" });
    $("calibration-card").hidden = false;
    await watchCalibration(result.run_id);
  } catch (error) {
    $("freshness-warning").hidden = false;
    $("freshness-warning").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

async function watchCalibration(runId) {
  const pill = $("calibration-pill");
  const progress = $("calibration-progress");
  const details = $("calibration-details");
  const card = $("calibration-card");
  card.hidden = false;
  while (true) {
    const result = await requestJSON(`/ui/api/calibration/${encodeURIComponent(runId)}`);
    pill.textContent = result.status;
    pill.className = `status-pill ${result.status === "complete" ? "success" : result.status === "failed" ? "danger" : ""}`;
    progress.style.width = `${Math.max(0, Math.min(100, Number(result.progress || 0) * 100))}%`;
    details.textContent = JSON.stringify({ coverage: result.coverage, error: result.error }, null, 2);
    if (["complete", "partial"].includes(result.status)) {
      await loadForecast();
      return;
    }
    if (result.status === "failed") return;
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

window.addEventListener("resize", () => { if (lastSnapshot) render(lastSnapshot); });
loadForecast();
setInterval(loadForecast, 60_000);
