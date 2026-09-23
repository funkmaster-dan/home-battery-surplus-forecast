const $ = (id) => document.getElementById(id);
const POWER_SEMANTICS = [
  ["instantaneous_power", "Instantaneous power"],
  ["interval_average_power", "Interval-average power"],
  ["interval_energy", "Energy per interval"],
  ["cumulative_energy", "Cumulative energy counter"],
];
let sensors = [];
let statistics = [];
let statisticById = new Map();
let statisticsError = "";
let sensorById = new Map();
let map;
let marker;
let arrayCounter = 1;
let windowCounter = 1;

function setMessage(element, text, kind = "") {
  element.textContent = text || "";
  element.className = `form-message ${kind}`;
}

function formatResponseError(detail, status) {
  if (Array.isArray(detail)) {
    return detail.map((issue) => {
      const message = typeof issue === "string" ? issue : issue?.msg || issue?.message || JSON.stringify(issue);
      const location = Array.isArray(issue?.loc)
        ? issue.loc.filter((part) => part !== "body").join(".")
        : "";
      return location ? `${location}: ${message}` : message;
    }).join("; ");
  }
  if (detail && typeof detail === "object") return detail.message || JSON.stringify(detail);
  return detail ? String(detail) : `Request failed (${status})`;
}

async function requestJSON(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(formatResponseError(payload.detail, response.status));
  return payload;
}

function safeNumber(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function makeOption(select, value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  select.append(option);
  return option;
}

function entityLabel(entity) {
  return `${entity.name || entity.entity_id} · ${entity.unit || "unit not reported"} · ${entity.entity_id}`;
}


function fillSensorSelect(select, { placeholder, filter = () => true, selected = "" } = {}) {
  select.replaceChildren();
  makeOption(select, "", placeholder || "Choose a sensor");
  for (const entity of sensors) {
    if (!filter(entity)) continue;
    makeOption(select, entity.entity_id, entityLabel(entity));
  }
  if (selected) select.value = selected;
}

function setEntityMeta(output, entityId) {
  const entity = sensorById.get(entityId);
  output.textContent = entity
    ? `${entity.unit || "unit not reported"} · ${entity.device_class || "no device class"}${entity.state_class ? ` · ${entity.state_class}` : ""}`
    : "Select a sensor";
}

function statisticAggregationLabel(statistic) {
  const reported = [statistic.has_mean ? "mean" : "", statistic.has_sum ? "sum" : ""]
    .filter(Boolean).join(", ");
  if (reported) return reported;
  const unit = String(statistic.unit || "").trim().toLowerCase();
  const statisticType = String(statistic.mean_type || statistic.unit_class || "").toLowerCase();
  if (statisticType === "power" || ["w", "kw"].includes(unit)) return "power mean (inferred from unit)";
  if (statisticType === "energy" || ["wh", "kwh", "mwh"].includes(unit)) return "energy values (inferred from unit)";
  return "aggregation not reported";
}

function updateHistoricalStatistic(statisticId) {
  const statistic = statisticById.get(statisticId);
  const meta = $("historical-statistic-meta");
  const coverage = $("historical-statistic-coverage");
  if (!statistic) {
    meta.textContent = "Select a statistic";
    coverage.textContent = statisticsError || "Choose the Long Term Statistics ID for direct household use.";
    return;
  }
  const aggregations = statisticAggregationLabel(statistic);
  meta.textContent = `${statistic.unit || "unit not reported"} · ${aggregations} · ${statistic.name || statistic.statistic_id}`;
  coverage.textContent = statistic.coverage
    ? `${statistic.statistic_id} · ${statistic.coverage.count} cached statistics records; calibration reports valid five-minute coverage.`
    : `Home Assistant reports ${statistic.statistic_id}; exact date-range coverage is measured during calibration.`;
}

function addLabeledControl(parent, title, control, className = "") {
  const label = document.createElement("label");
  label.textContent = title;
  if (className) label.className = className;
  label.append(control);
  parent.append(label);
  return control;
}

function input(type, value, attributes = {}) {
  const control = document.createElement("input");
  control.type = type;
  if (value !== undefined && value !== null) control.value = value;
  for (const [key, val] of Object.entries(attributes)) control.setAttribute(key, String(val));
  return control;
}

function selectControl(options, selected = "") {
  const control = document.createElement("select");
  for (const [value, label] of options) makeOption(control, value, label);
  control.value = selected;
  return control;
}

function semanticsOptions() {
  return [["", "Confirm sensor semantics"], ...POWER_SEMANTICS];
}

function addArrayRow(saved = null) {
  const id = saved?.id || `pv${arrayCounter++}`;
  const row = document.createElement("article");
  row.className = "repeat-row array-row";
  row.dataset.arrayId = id;
  const head = document.createElement("div");
  head.className = "repeat-row-head";
  const title = document.createElement("h3");
  title.textContent = saved?.name || `PV array ${id}`;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "button danger small";
  remove.textContent = "Remove";
  remove.addEventListener("click", () => row.remove());
  head.append(title, remove);
  row.append(head);
  const grid = document.createElement("div");
  grid.className = "field-grid three";
  const name = input("text", saved?.name || `PV array ${id}`, { maxlength: 80, required: true });
  name.dataset.role = "name";
  const entity = selectControl([["", "Choose PV production sensor"]], "");
  entity.dataset.role = "entity";
  fillSensorSelect(entity, { placeholder: "Choose PV production sensor", selected: saved?.entity_id || "" });
  const meta = document.createElement("output");
  meta.className = "entity-meta";
  meta.textContent = "Select a sensor";
  const semantics = selectControl(semanticsOptions(), saved?.semantics || "");
  semantics.dataset.role = "semantics";
  const interval = input("number", saved?.interval_minutes ?? 5, { min: 1, max: 1440, required: true });
  interval.dataset.role = "interval";
  const tilt = input("number", saved?.tilt_deg ?? 25, { min: 0, max: 90, step: "any", required: true });
  tilt.dataset.role = "tilt";
  const azimuth = input("number", saved?.azimuth_deg ?? 0, { min: -180, max: 180, step: "any", required: true });
  azimuth.dataset.role = "azimuth";
  addLabeledControl(grid, "Array name", name);
  addLabeledControl(grid, "PV production sensor", entity);
  addLabeledControl(grid, "Sensor unit and class", meta);
  addLabeledControl(grid, "Value semantics", semantics);
  addLabeledControl(grid, "Sampling interval (minutes)", interval);
  addLabeledControl(grid, "Tilt (°)", tilt);
  addLabeledControl(grid, "Azimuth (°)", azimuth);
  row.append(grid);
  entity.addEventListener("change", () => setEntityMeta(meta, entity.value));
  name.addEventListener("input", () => { title.textContent = name.value || `PV array ${id}`; });
  if (saved?.entity_id) setEntityMeta(meta, saved.entity_id);
  $("solar-arrays").append(row);
  return row;
}

function addGridWindow(saved = null) {
  const row = document.createElement("article");
  row.className = "repeat-row window-row";
  const head = document.createElement("div");
  head.className = "repeat-row-head";
  const title = document.createElement("h3");
  title.textContent = `Import window ${windowCounter++}`;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "button danger small";
  remove.textContent = "Remove";
  remove.addEventListener("click", () => row.remove());
  head.append(title, remove);
  row.append(head);

  const selectedWeekdays = new Set((saved?.weekdays ?? [0, 1, 2, 3, 4, 5, 6]).map(Number));
  const weekdayField = document.createElement("fieldset");
  weekdayField.className = "weekday-field";
  const legend = document.createElement("legend");
  legend.textContent = "Days";
  const picker = document.createElement("div");
  picker.className = "weekday-picker";
  const weekdayNames = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
  weekdayNames.forEach((name, weekday) => {
    const option = document.createElement("label");
    option.className = "weekday-option";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = String(weekday);
    checkbox.dataset.role = "weekday";
    checkbox.checked = selectedWeekdays.has(weekday);
    const text = document.createElement("span");
    text.textContent = name;
    option.append(checkbox, text);
    picker.append(option);
  });
  weekdayField.append(legend, picker);
  row.append(weekdayField);

  const grid = document.createElement("div");
  grid.className = "field-grid two window-grid";
  const start = input("time", saved?.start || "00:00", { required: true });
  start.dataset.role = "start";
  const end = input("time", saved?.end || "06:00", { required: true });
  end.dataset.role = "end";
  const target = input("number", saved?.target_soc_pct ?? 80, { min: 1, max: 100, step: "any", required: true });
  target.dataset.role = "target";
  const charge = input("number", saved?.grid_charge_kw ?? 2, { min: 0.01, step: "any", required: true });
  charge.dataset.role = "charge";
  addLabeledControl(grid, "Start time", start);
  addLabeledControl(grid, "End time", end);
  addLabeledControl(grid, "Target SOC (%)", target);
  addLabeledControl(grid, "Grid-charge limit (kW)", charge);
  row.append(grid);
  $("grid-windows").append(row);
}

function initMap(latitude, longitude) {
  const lat = safeNumber(latitude, 0);
  const lon = safeNumber(longitude, 0);
  $("latitude").value = lat;
  $("longitude").value = lon;
  if (!window.L) {
    $("setup-map").textContent = "Map tiles could not load. Enter latitude and longitude above.";
    return;
  }
  if (!map) {
    map = L.map("setup-map", { scrollWheelZoom: false }).setView([lat, lon], lat === 0 && lon === 0 ? 2 : 13);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "© OpenStreetMap contributors",
    }).addTo(map);
    marker = L.marker([lat, lon], { draggable: true }).addTo(map);
    map.on("click", (event) => moveMarker(event.latlng.lat, event.latlng.lng));
    marker.on("dragend", () => {
      const point = marker.getLatLng();
      moveMarker(point.lat, point.lng);
    });
  } else {
    map.setView([lat, lon], lat === 0 && lon === 0 ? 2 : 13);
    marker.setLatLng([lat, lon]);
    setTimeout(() => map.invalidateSize(), 50);
  }
}

function moveMarker(latitude, longitude) {
  $("latitude").value = latitude.toFixed(6);
  $("longitude").value = longitude.toFixed(6);
  if (marker) marker.setLatLng([latitude, longitude]);
}

function loadEntities(payload) {
  sensors = payload.entities || [];
  statistics = payload.statistics || [];
  statisticsError = payload.statistics_error || "";
  sensorById = new Map(sensors.map((entity) => [entity.entity_id, entity]));
  statisticById = new Map(statistics.map((statistic) => [statistic.statistic_id, statistic]));
  fillSensorSelect($("consumption-entity"), { placeholder: "Choose a direct whole-home sensor" });
  fillSensorSelect($("battery-entity"), {
    placeholder: "Choose current battery state",
    filter: (entity) => ["%", "percent", "Wh", "kWh", "MWh"].includes(entity.unit),
  });
  fillSensorSelect($("temperature-entity"), {
    placeholder: "Use Open-Meteo archive temperature",
    filter: (entity) => ["°C", "°F", "C", "F"].includes(entity.unit),
  });
  const statisticSelect = $("historical-statistic");
  statisticSelect.replaceChildren();
  makeOption(statisticSelect, "", "Choose a Long Term Statistics ID");
  for (const statistic of statistics) {
    const unit = String(statistic.unit || "").trim().toLowerCase();
    if (!["w", "kw", "wh", "kwh", "mwh"].includes(unit)) continue;
    const aggregates = statisticAggregationLabel(statistic);
    makeOption(
      statisticSelect,
      statistic.statistic_id,
      `${statistic.name || statistic.statistic_id} · ${statistic.unit} · ${aggregates} · ${statistic.statistic_id}`,
    );
  }
  statisticSelect.onchange = () => updateHistoricalStatistic(statisticSelect.value);
  for (const row of document.querySelectorAll(".array-row")) {
    const entity = row.querySelector("select");
    const selected = entity.value;
    fillSensorSelect(entity, {
      placeholder: "Choose PV production sensor",
      filter: (candidate) => ["W", "kW", "Wh", "kWh", "MWh"].includes(candidate.unit),
      selected,
    });
    setEntityMeta(row.querySelector("output"), selected);
  }
  if (statisticsError) {
    $("historical-statistic-coverage").textContent = `Long Term Statistics metadata could not be read: ${statisticsError}`;
  } else {
    updateHistoricalStatistic(statisticSelect.value);
  }
  $("consumption-entity").addEventListener("change", () => setEntityMeta($("consumption-meta"), $("consumption-entity").value));
  $("battery-entity").addEventListener("change", () => setEntityMeta($("battery-meta"), $("battery-entity").value));
  $("temperature-entity").addEventListener("change", () => {
    const selected = sensorById.get($("temperature-entity").value);
    if (selected && ["°C", "°F"].includes(selected.unit)) $("temperature-unit").value = selected.unit;
  });
}

function localDateInput(date, timezone) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: timezone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}

function fillConfig(config, site) {
  const home = site || {};
  $("latitude").value = config?.latitude ?? home.latitude ?? 0;
  $("longitude").value = config?.longitude ?? home.longitude ?? 0;
  $("timezone").value = config?.timezone || home.time_zone || "UTC";
  $("horizon-hours").value = String(config?.horizon_hours ?? 48);
  $("open-meteo-refresh-minutes").value = String(config?.open_meteo_refresh_minutes ?? 60);
  const today = new Date();
  const timezone = config?.timezone || home.time_zone || Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  const endDefault = localDateInput(today, timezone);
  const startDefault = localDateInput(new Date(today.getTime() - 365 * 86400000), timezone);
  $("calibration-start").value = config?.calibration_start || startDefault;
  $("calibration-end").value = config?.calibration_end || endDefault;
  $("temperature-unit").value = config?.temperature_unit || "°C";
  $("solar-arrays").replaceChildren();
  arrayCounter = 1;
  for (const array of config?.solar_arrays || []) {
    addArrayRow(array);
    const numeric = Number(String(array.id).replace(/^pv/i, ""));
    if (Number.isInteger(numeric)) arrayCounter = Math.max(arrayCounter, numeric + 1);
  }
  if (!config?.solar_arrays?.length) addArrayRow();
  $("grid-windows").replaceChildren();
  windowCounter = 1;
  for (const window of config?.grid_import_windows || []) addGridWindow(window);
  if (config?.consumption) {
    $("consumption-entity").value = config.consumption.entity_id;
    $("consumption-semantics").value = config.consumption.semantics;
    $("consumption-interval").value = config.consumption.interval_minutes;
    $("historical-statistic").value = config.consumption.historical_statistic_id;
    setEntityMeta($("consumption-meta"), config.consumption.entity_id);
    updateHistoricalStatistic(config.consumption.historical_statistic_id);
    $("direct-whole-home-confirmed").checked = Boolean(config.consumption.direct_whole_home_confirmed);
  }
  if (config?.battery) {
    $("battery-entity").value = config.battery.entity_id;
    $("battery-state-type").value = config.battery.state_type;
    $("capacity-kwh").value = config.battery.capacity_kwh;
    $("max-charge-kw").value = config.battery.max_charge_kw;
    $("max-discharge-kw").value = config.battery.max_discharge_kw;
    $("charge-efficiency").value = config.battery.charge_efficiency;
    $("discharge-efficiency").value = config.battery.discharge_efficiency;
    setEntityMeta($("battery-meta"), config.battery.entity_id);
  }
  if (config?.temperature_entity_id) $("temperature-entity").value = config.temperature_entity_id;
  initMap($("latitude").value, $("longitude").value);
  $("config-form").hidden = false;
}

async function loadExisting() {
  try {
    const state = await requestJSON("/ui/api/status");
    if (state.home_assistant_connected) {
      $("connection-status").textContent = "Connected";
      $("connection-status").className = "status-pill success";
      const entities = await requestJSON("/ui/api/entities");
      loadEntities(entities);
      const saved = await requestJSON("/ui/api/config");
      $("ha-url").value = saved.base_url || "";
      fillConfig(saved.config, saved.site);
      if (state.calibration?.run_id && ["queued", "running"].includes(state.calibration.status)) {
        watchCalibration(state.calibration.run_id);
      }
    }
  } catch (error) {
    setMessage($("connection-message"), error.message, "error");
  }
}

$("connect-button").addEventListener("click", async () => {
  const button = $("connect-button");
  const message = $("connection-message");
  button.disabled = true;
  setMessage(message, "Checking the read-only Home Assistant API…");
  const accessToken = $("ha-token").value;
  try {
    const response = await requestJSON("/ui/api/home-assistant", {
      method: "POST",
      body: JSON.stringify({ base_url: $("ha-url").value.trim(), access_token: accessToken }),
    });
    $("ha-token").value = "";
    $("connection-status").textContent = "Connected";
    $("connection-status").className = "status-pill success";
    setMessage(message, `Connected · ${response.site.time_zone}`, "success");
    const entities = await requestJSON("/ui/api/entities");
    loadEntities(entities);
    const saved = await requestJSON("/ui/api/config");
    fillConfig(saved.config, response.site);
  } catch (error) {
    $("ha-token").value = "";
    setMessage(message, error.message, "error");
  } finally {
    button.disabled = false;
  }
});

$("add-array").addEventListener("click", () => addArrayRow());
$("add-window").addEventListener("click", () => addGridWindow());
$("consumption-entity").addEventListener("change", () => setEntityMeta($("consumption-meta"), $("consumption-entity").value));
$("battery-entity").addEventListener("change", () => setEntityMeta($("battery-meta"), $("battery-entity").value));
$("temperature-entity").addEventListener("change", () => {
  const entity = sensorById.get($("temperature-entity").value);
  if (entity && ["°C", "°F"].includes(entity.unit)) $("temperature-unit").value = entity.unit;
});
$("latitude").addEventListener("change", () => initMap($("latitude").value, $("longitude").value));
$("longitude").addEventListener("change", () => initMap($("latitude").value, $("longitude").value));

$("config-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter || document.querySelector("#config-form button[type=submit]");
  const message = $("config-message");
  button.disabled = true;
  setMessage(message, document.body.dataset.editMode === "true" ? "Saving configuration…" : "Saving validated configuration and starting calibration…");
  try {
    const consumptionEntity = sensorById.get($("consumption-entity").value);
    const batteryEntity = sensorById.get($("battery-entity").value);
    if (!consumptionEntity || !batteryEntity) throw new Error("Choose valid consumption and battery sensors.");
    const arrays = [...document.querySelectorAll(".array-row")].map((row) => {
      const name = row.querySelector('[data-role="name"]').value.trim();
      const entityId = row.querySelector('[data-role="entity"]').value;
      const selected = sensorById.get(entityId);
      if (!selected) throw new Error(`Choose a PV production sensor for ${name || row.dataset.arrayId}.`);
      return {
        id: row.dataset.arrayId,
        name,
        entity_id: entityId,
        unit: selected.unit,
        semantics: row.querySelector('[data-role="semantics"]').value,
        interval_minutes: Number(row.querySelector('[data-role="interval"]').value),
        tilt_deg: Number(row.querySelector('[data-role="tilt"]').value),
        azimuth_deg: Number(row.querySelector('[data-role="azimuth"]').value),
      };
    });
    const windows = [...document.querySelectorAll(".window-row")].map((row) => {
      const weekdays = [...row.querySelectorAll('[data-role="weekday"]:checked')].map((input) => Number(input.value));
      if (weekdays.length === 0) throw new Error("Choose one or more days for each grid-import window.");
      return {
        weekdays,
        start: row.querySelector('[data-role="start"]').value,
        end: row.querySelector('[data-role="end"]').value,
        target_soc_pct: Number(row.querySelector('[data-role="target"]').value),
        grid_charge_kw: Number(row.querySelector('[data-role="charge"]').value),
      };
    });
    const temperatureEntity = $("temperature-entity").value || null;
    const config = {
      latitude: Number($("latitude").value),
      longitude: Number($("longitude").value),
      timezone: $("timezone").value.trim(),
      calibration_start: $("calibration-start").value,
      calibration_end: $("calibration-end").value,
      horizon_hours: Number($("horizon-hours").value),
      open_meteo_refresh_minutes: Number($("open-meteo-refresh-minutes").value),
      consumption: {
        entity_id: consumptionEntity.entity_id,
        unit: consumptionEntity.unit,
        semantics: $("consumption-semantics").value,
        interval_minutes: Number($("consumption-interval").value),
        historical_statistic_id: $("historical-statistic").value,
        direct_whole_home_confirmed: $("direct-whole-home-confirmed").checked,
      },
      battery: {
        entity_id: batteryEntity.entity_id,
        unit: batteryEntity.unit,
        state_type: $("battery-state-type").value,
        capacity_kwh: Number($("capacity-kwh").value),
        max_charge_kw: Number($("max-charge-kw").value),
        max_discharge_kw: Number($("max-discharge-kw").value),
        charge_efficiency: Number($("charge-efficiency").value),
        discharge_efficiency: Number($("discharge-efficiency").value),
      },
      solar_arrays: arrays,
      temperature_entity_id: temperatureEntity,
      temperature_unit: $("temperature-unit").value,
      grid_import_windows: windows,
    };
    const result = await requestJSON("/ui/api/config", { method: "PUT", body: JSON.stringify(config) });
    if (result.calibration_run_id) {
      $("calibration-card").hidden = false;
      watchCalibration(result.calibration_run_id);
      $("calibration-card").scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
      setMessage(message, "Configuration saved. Models are current.", "success");
    }
  } catch (error) {
    setMessage(message, error.message, "error");
  } finally {
    button.disabled = false;
  }
});


const hacsTokenField = $("hacs-integration-token");
if (hacsTokenField) {
  const revealButton = $("reveal-hacs-token");
  const copyButton = $("copy-hacs-token");
  const message = $("hacs-token-message");
  revealButton.addEventListener("click", async () => {
    if (hacsTokenField.type === "text") {
      hacsTokenField.value = "";
      hacsTokenField.type = "password";
      copyButton.disabled = true;
      revealButton.textContent = "Reveal token";
      setMessage(message, "");
      return;
    }
    revealButton.disabled = true;
    try {
      const result = await requestJSON("/ui/api/hacs-token");
      hacsTokenField.value = result.token;
      hacsTokenField.type = "text";
      copyButton.disabled = false;
      revealButton.textContent = "Hide token";
      setMessage(message, "Use this value in the HACS setup flow.", "success");
    } catch (error) {
      setMessage(message, error.message, "error");
    } finally {
      revealButton.disabled = false;
    }
  });
  copyButton.addEventListener("click", async () => {
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard access is unavailable");
      await navigator.clipboard.writeText(hacsTokenField.value);
      setMessage(message, "Token copied to clipboard.", "success");
    } catch {
      hacsTokenField.focus();
      hacsTokenField.select();
      setMessage(message, "Token selected; copy it manually if clipboard access is unavailable.");
    }
  });
}

async function watchCalibration(runId) {
  const card = $("calibration-card");
  if (!card) return;
  card.hidden = false;
  const pill = $("calibration-pill");
  const progress = $("calibration-progress");
  const details = $("calibration-details");
  const link = $("dashboard-link");
  while (true) {
    try {
      const result = await requestJSON(`/ui/api/calibration/${encodeURIComponent(runId)}`);
      const status = result.status || "running";
      pill.textContent = status;
      pill.className = `status-pill ${status === "complete" ? "success" : status === "failed" ? "danger" : ""}`;
      progress.style.width = `${Math.max(0, Math.min(100, Number(result.progress || 0) * 100))}%`;
      details.textContent = JSON.stringify({ coverage: result.coverage, error: result.error }, null, 2);
      if (["complete", "partial"].includes(status)) {
        link.hidden = false;
        return;
      }
      if (status === "failed") return;
    } catch (error) {
      details.textContent = error.message;
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

loadExisting();
