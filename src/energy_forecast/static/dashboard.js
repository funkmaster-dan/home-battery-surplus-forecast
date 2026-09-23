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
  const staleEntities = freshness.stale_entities || [];
  host.append(detailRow("Stale Home Assistant entities", staleEntities.length ? staleEntities.join(", ") : "None"));
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
