const state = {
  source: null,
  dataset: "",
  fileBase64: "",
  fileName: "",
  headers: [],
  inspection: null,
  datasetMetadata: null,
  uploadError: "",
  result: null,
  running: false,
};

const SCENARIOS = {
  "edge-failure": {
    question: "Which sensor readings help predict a device failure?",
    description:
      "This simulated dataset contains 720 device snapshots. Each snapshot has 18 sensor readings, and the goal is to predict whether that device fails within the next 24 hours.",
    rows: "720",
    rowsLabel: "device snapshots",
    columns: "18",
    columnsLabel: "source inputs",
    inputs: "18",
    inputsLabel: "prepared inputs",
    target: "Yes / no",
    targetLabel: "target type",
    format: "Complete synthetic CSV",
  },
  "cloud-cost": {
    question: "Which workload signals help explain hourly cloud cost?",
    description:
      "This simulated dataset contains 680 hourly workload records. Each record has 16 service signals, and the goal is to estimate its hourly cost in US dollars.",
    rows: "680",
    rowsLabel: "hourly workloads",
    columns: "16",
    columnsLabel: "source inputs",
    inputs: "16",
    inputsLabel: "prepared inputs",
    target: "Number",
    targetLabel: "target type",
    format: "Complete synthetic CSV",
  },
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const elements = {
  budget: $("#feature-budget"),
  budgetValue: $("#feature-budget-value"),
  redundancy: $("#redundancy-weight"),
  redundancyValue: $("#redundancy-value"),
  run: $("#run-button"),
  runLabel: $("#run-button-label"),
  resultStatus: $("#result-status"),
  resultTitle: $("#result-title"),
  resultQuestion: $("#result-question"),
  progressShell: $("#progress-shell"),
  progressBar: $("#progress-bar"),
  progressLabel: $("#progress-label"),
  progressValue: $("#progress-value"),
  content: $("#results-content"),
  file: $("#csv-file"),
  uploadLabel: $("#upload-label"),
  uploadInspection: $("#upload-inspection"),
  targetControl: $("#target-control"),
  target: $("#target-column"),
  results: $(".results-panel"),
  budgetHelp: $("#budget-help"),
  scenarioQuestion: $("#scenario-question"),
  scenarioDescription: $("#scenario-description"),
  scenarioRows: $("#scenario-rows"),
  scenarioRowsLabel: $("#scenario-rows-label"),
  scenarioInputs: $("#scenario-inputs"),
  scenarioInputsLabel: $("#scenario-inputs-label"),
  scenarioTarget: $("#scenario-target"),
  scenarioTargetLabel: $("#scenario-target-label"),
  resultDetails: $("#result-details"),
  workbench: $("#workbench"),
  setupStatus: $("#setup-status"),
  scenarioGuide: $("#scenario-guide"),
  scenarioColumns: $("#scenario-columns"),
  scenarioColumnsLabel: $("#scenario-columns-label"),
  budgetFieldset: $("#budget-fieldset"),
  selectedDataDownload: $("#selected-data-download"),
  previewFormat: $("#preview-format"),
  previewColumns: $("#preview-columns"),
  previewPrepared: $("#preview-prepared"),
  previewHead: $("#preview-head"),
  previewBody: $("#preview-body"),
  previewNote: $("#preview-note"),
  advancedSettings: $("#advanced-settings"),
  uploadTab: $("#upload-tab"),
  sampleTab: $("#sample-tab"),
  uploadPanel: $("#upload-panel"),
  samplePanel: $("#sample-panel"),
  sampleSelect: $("#sample-dataset"),
  analysisControls: $("#analysis-controls"),
  labEmpty: $("#lab-empty"),
};

function setRangeProgress(input) {
  const value = Number(input.value);
  const min = Number(input.min);
  const max = Number(input.max);
  const percentage = ((value - min) / Math.max(1, max - min)) * 100;
  input.style.setProperty("--range-progress", `${percentage}%`);
}

function updateBudgetLabel() {
  if (!state.source) {
    elements.budgetValue.textContent = "—";
    elements.budgetHelp.textContent =
      "Choose a dataset first. We will explain how many usable inputs it contains.";
    updateRunLabel();
    return;
  }
  const kept = Number(elements.budget.value);
  const total = Number(elements.budget.max);
  elements.budgetValue.textContent = `${kept} / ${total}`;
  elements.budgetHelp.textContent =
    kept === total
      ? `Keep all ${total} inputs. Lower the slider to test a smaller set.`
      : `Keep ${kept} of ${total} inputs. QUBOLens will try to remove ${
          total - kept
        } without losing useful signal.`;
  updateRunLabel();
  setRangeProgress(elements.budget);
}

function updateRunLabel() {
  if (state.running) return;
  const kept = Number(elements.budget.value);
  if (!state.source) {
    elements.runLabel.textContent = "Choose data to continue";
  } else if (state.source === "upload" && state.uploadError) {
    elements.runLabel.textContent = "Choose a different target";
  } else {
    elements.runLabel.textContent =
      state.source === "upload"
        ? `Analyze the best ${kept} inputs`
        : `Show me the best ${kept} inputs`;
  }
}

function updateScenario(scenario) {
  elements.scenarioQuestion.textContent = scenario.question;
  elements.scenarioDescription.textContent = scenario.description;
  elements.scenarioRows.textContent = scenario.rows;
  elements.scenarioRowsLabel.textContent = scenario.rowsLabel;
  elements.scenarioColumns.textContent = scenario.columns;
  elements.scenarioColumnsLabel.textContent = scenario.columnsLabel;
  elements.scenarioInputs.textContent = scenario.inputs;
  elements.scenarioInputsLabel.textContent = scenario.inputsLabel;
  elements.scenarioTarget.textContent = scenario.target;
  elements.scenarioTargetLabel.textContent = scenario.targetLabel;
}

function invalidateResult() {
  if (state.running) return;
  state.result = null;
  elements.results.classList.add("hidden");
  elements.workbench.classList.remove("has-results");
  $("#download-result").disabled = true;
  $("#download-qubo").disabled = true;
}

function markSettingsChanged() {
  invalidateResult();
}

function syncSetupState() {
  const ready = Boolean(state.source) && !state.uploadError;
  elements.budgetFieldset.disabled = !ready;
  elements.run.disabled = !ready || state.running;
  elements.analysisControls.classList.toggle("hidden", !ready);
  elements.labEmpty.classList.toggle("hidden", Boolean(state.source));
  elements.uploadTab.disabled = state.running;
  elements.sampleTab.disabled = state.running;
  elements.setupStatus.textContent = ready
    ? state.source === "upload"
      ? "Your data is ready"
      : "Sample ready"
    : state.uploadError
      ? "Needs attention"
      : "Waiting for data";
  updateRunLabel();
}

function revealResults() {
  elements.workbench.classList.add("has-results");
  elements.results.classList.remove("hidden");
}

function showSetupError(message) {
  state.uploadError = message;
  elements.uploadInspection.textContent = message;
  elements.uploadInspection.classList.remove("hidden");
  elements.uploadInspection.classList.add("warning");
  syncSetupState();
}

function resetDataSelection() {
  if (state.running) return;
  invalidateResult();
  state.source = null;
  state.dataset = "";
  state.fileBase64 = "";
  state.fileName = "";
  state.headers = [];
  state.inspection = null;
  state.datasetMetadata = null;
  state.uploadError = "";
  elements.file.value = "";
  elements.sampleSelect.value = "";
  elements.uploadLabel.textContent = "Choose a data file";
  elements.uploadInspection.classList.add("hidden");
  elements.uploadInspection.classList.remove("warning");
  elements.targetControl.classList.add("hidden");
  elements.target.replaceChildren();
  elements.scenarioGuide.classList.add("hidden");
  elements.selectedDataDownload.classList.add("hidden");
  syncSetupState();
}

function setSourceMode(mode) {
  if (state.running) return;
  const uploadMode = mode === "upload";
  if (
    (uploadMode && state.source === "demo") ||
    (!uploadMode && state.source === "upload")
  ) {
    resetDataSelection();
  }
  elements.uploadTab.classList.toggle("active", uploadMode);
  elements.sampleTab.classList.toggle("active", !uploadMode);
  elements.uploadTab.setAttribute("aria-selected", String(uploadMode));
  elements.sampleTab.setAttribute("aria-selected", String(!uploadMode));
  elements.uploadPanel.classList.toggle("hidden", !uploadMode);
  elements.samplePanel.classList.toggle("hidden", uploadMode);
}

function updateRedundancyLabel() {
  elements.redundancyValue.textContent = Number(elements.redundancy.value).toFixed(2);
  setRangeProgress(elements.redundancy);
  markSettingsChanged();
}

function renderDataPreview({
  columns,
  preview,
  target,
  format,
  preparedColumns = [],
  downloadUrl = "",
  totalRows,
}) {
  const sourceColumns = columns.filter((column) => column !== target);
  const shownColumns = sourceColumns.slice(0, 4);
  if (target && !shownColumns.includes(target)) shownColumns.push(target);
  elements.previewFormat.textContent = format;
  elements.previewColumns.textContent =
    `${sourceColumns.length} source inputs before preparation: ` +
    `${sourceColumns.slice(0, 8).map(humanizeFeatureName).join(", ")}` +
    `${sourceColumns.length > 8 ? `, and ${sourceColumns.length - 8} more` : ""}.`;
  const prepared = preparedColumns.length ? preparedColumns : sourceColumns;
  elements.previewPrepared.textContent =
    `${prepared.length} model-ready inputs: ` +
    `${prepared.slice(0, 10).map(humanizeFeatureName).join(", ")}` +
    `${prepared.length > 10 ? `, and ${prepared.length - 10} more` : ""}.`;
  elements.previewHead.replaceChildren();
  const headRow = document.createElement("tr");
  shownColumns.forEach((column) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent =
      column === target
        ? `${humanizeFeatureName(column)} · target`
        : humanizeFeatureName(column);
    headRow.append(cell);
  });
  elements.previewHead.append(headRow);
  elements.previewBody.replaceChildren();
  preview.slice(0, 4).forEach((record) => {
    const row = document.createElement("tr");
    shownColumns.forEach((column) => {
      const cell = document.createElement("td");
      const value = record[column];
      cell.textContent =
        typeof value === "number" ? Number(value).toFixed(3) : String(value ?? "—");
      row.append(cell);
    });
    elements.previewBody.append(row);
  });
  elements.previewNote.textContent = downloadUrl
    ? `Showing 4 of ${Number(totalRows).toLocaleString()} raw rows. Download the CSV to inspect every value.`
    : `Showing up to 4 raw rows from your file. Your original file remains unchanged and is not saved.`;
  if (downloadUrl) {
    elements.selectedDataDownload.href = downloadUrl;
    elements.selectedDataDownload.download =
      `qubolens-${state.dataset || "sample"}.csv`;
    elements.selectedDataDownload.classList.remove("hidden");
  } else {
    elements.selectedDataDownload.classList.add("hidden");
    elements.selectedDataDownload.removeAttribute("href");
  }
}

async function loadDemoMetadata(slug) {
  const response = await fetch(`/api/datasets/${slug}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "The sample preview could not be loaded.");
  }
  if (state.source !== "demo" || state.dataset !== slug) return;
  state.datasetMetadata = payload;
  renderDataPreview({
    columns: [...payload.feature_names, payload.target],
    preview: payload.preview,
    target: payload.target,
    format: SCENARIOS[slug].format,
    preparedColumns: payload.feature_names,
    downloadUrl: payload.download_url,
    totalRows: payload.samples,
  });
}

async function selectDemo(slug) {
  if (!slug) {
    resetDataSelection();
    return;
  }
  invalidateResult();
  state.source = "demo";
  state.dataset = slug;
  state.fileBase64 = "";
  state.fileName = "";
  state.headers = [];
  state.inspection = null;
  state.datasetMetadata = null;
  state.uploadError = "";
  elements.file.value = "";
  elements.uploadLabel.textContent = "Choose a data file";
  elements.uploadInspection.classList.add("hidden");
  elements.targetControl.classList.add("hidden");
  elements.budget.max = Number(SCENARIOS[state.dataset].inputs);
  elements.budget.value = Math.min(6, Number(elements.budget.max));
  updateScenario(SCENARIOS[state.dataset]);
  elements.scenarioGuide.classList.remove("hidden");
  elements.previewFormat.textContent = "Loading sample preview…";
  elements.previewColumns.textContent = "";
  elements.previewPrepared.textContent = "Loading prepared input names…";
  elements.previewHead.replaceChildren();
  elements.previewBody.replaceChildren();
  updateBudgetLabel();
  syncSetupState();
  try {
    await loadDemoMetadata(state.dataset);
  } catch (error) {
    elements.previewNote.textContent =
      error.message || "The sample preview could not be loaded.";
  }
}

function updateUploadedScenario() {
  if (state.source !== "upload" || !state.inspection) return;
  const inspection = state.inspection;
  const target = elements.target.value || inspection.target;
  const prepared = Number(inspection.prepared_features || 0);
  const taskLabel =
    inspection.task === "classification"
      ? "Yes / no"
      : inspection.task === "regression"
        ? "Number"
        : "Check target";
  updateScenario({
    question: `Which inputs best predict ${humanizeFeatureName(target)}?`,
    description:
      inspection.description ||
      `QUBOLens detected ${Number(inspection.rows).toLocaleString()} rows and will prepare numbers, dates, categories, and text automatically.`,
    rows: Number(inspection.rows).toLocaleString(),
    rowsLabel: "data rows",
    columns: String(inspection.source_columns),
    columnsLabel: "source inputs",
    inputs: String(prepared),
    inputsLabel: "prepared inputs",
    target: taskLabel,
    targetLabel: "target type",
  });
  elements.scenarioGuide.classList.remove("hidden");
  renderDataPreview({
    columns: inspection.columns,
    preview: inspection.preview || [],
    target,
    format: inspection.format,
    preparedColumns: inspection.prepared_feature_names || [],
    totalRows: inspection.rows,
  });
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
    reader.onerror = () => reject(new Error("The file could not be read."));
    reader.readAsDataURL(file);
  });
}

async function inspectCurrentUpload(target = "") {
  const response = await fetch("/api/inspect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      file: state.fileBase64,
      filename: state.fileName,
      target,
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "The file could not be inspected.");
  }
  state.inspection = payload;
  state.uploadError = payload.target_error || "";
  state.headers = payload.columns;
  elements.budget.max = Math.max(1, Number(payload.prepared_features));
  elements.budget.value = Math.min(
    Number(elements.budget.value),
    Number(elements.budget.max),
    6,
  );
  elements.uploadInspection.textContent = state.uploadError
    ? `${payload.format} detected · ${Number(payload.rows).toLocaleString()} rows · ${state.uploadError}`
    : `${payload.format} detected · ${Number(payload.rows).toLocaleString()} rows · ${
        payload.source_columns
      } source inputs → ${Number(payload.prepared_features)} prepared inputs`;
  elements.uploadInspection.classList.toggle("warning", Boolean(state.uploadError));
  elements.uploadInspection.classList.remove("hidden");
  updateBudgetLabel();
  updateUploadedScenario();
  syncSetupState();
}

async function handleFile(file) {
  if (!file) return;
  invalidateResult();
  state.source = null;
  state.dataset = "";
  state.fileBase64 = "";
  state.fileName = "";
  state.headers = [];
  state.inspection = null;
  state.datasetMetadata = null;
  state.uploadError = "";
  elements.sampleSelect.value = "";
  elements.scenarioGuide.classList.add("hidden");
  elements.targetControl.classList.add("hidden");
  elements.target.replaceChildren();
  elements.uploadLabel.textContent = "Choose a data file";
  syncSetupState();
  if (file.size > 20_000_000) {
    showSetupError("That file is larger than the 20 MB interactive limit.");
    elements.file.value = "";
    return;
  }
  elements.uploadLabel.textContent = `Reading ${file.name}…`;
  elements.uploadInspection.textContent = "Detecting rows, columns, and data types…";
  elements.uploadInspection.classList.remove("hidden", "warning");
  elements.run.disabled = true;
  try {
    const encoded = await fileToBase64(file);
    state.fileBase64 = encoded;
    state.fileName = file.name;
    state.source = "upload";
    elements.budget.value = 6;
    await inspectCurrentUpload();
  } catch (error) {
    state.source = null;
    state.fileBase64 = "";
    state.fileName = "";
    state.inspection = null;
    elements.file.value = "";
    elements.uploadLabel.textContent = "Choose a data file";
    showSetupError(error.message || "The file could not be read.");
    return;
  } finally {
    syncSetupState();
  }
  elements.uploadLabel.textContent = file.name;
  elements.target.replaceChildren();
  state.headers.forEach((header) => {
    const option = document.createElement("option");
    option.value = header;
    option.textContent = header;
    if (header === state.inspection.target) option.selected = true;
    elements.target.append(option);
  });
  elements.targetControl.classList.remove("hidden");
  syncSetupState();
}

function selectedQuality() {
  return $('input[name="quality"]:checked').value;
}

function buildPayload() {
  const common = {
    k: Number(elements.budget.value),
    redundancy_weight: Number(elements.redundancy.value),
    quality: selectedQuality(),
    seed: 42,
  };
  if (state.source === "upload") {
    return {
      ...common,
      source: "upload",
      file: state.fileBase64,
      filename: state.fileName,
      name: state.fileName.replace(/\.[^.]+$/, "") || "Uploaded data",
      target: elements.target.value,
      task: "auto",
    };
  }
  return { ...common, source: "demo", dataset: state.dataset };
}

function startProgress() {
  const stages = [
    [11, "Reading the dataset…"],
    [26, "Measuring each feature…"],
    [43, "Looking for repeated signal…"],
    [61, "Trying different feature mixes…"],
    [77, "Comparing with a simple ranking…"],
    [88, "Building the charts…"],
  ];
  let position = 0;
  elements.progressShell.classList.remove("hidden");
  elements.progressBar.style.width = `${stages[0][0]}%`;
  elements.progressLabel.textContent = stages[0][1];
  elements.progressValue.textContent = `${stages[0][0]}%`;
  return window.setInterval(() => {
    position = Math.min(position + 1, stages.length - 1);
    const [progress, label] = stages[position];
    elements.progressBar.style.width = `${progress}%`;
    elements.progressLabel.textContent = label;
    elements.progressValue.textContent = `${progress}%`;
  }, 480);
}

async function runExperiment() {
  if (state.running) return;
  if (!state.source) {
    showSetupError("Choose your data file or one of the complete samples first.");
    return;
  }
  if (state.source === "upload" && (!state.fileBase64 || !elements.target.value)) {
    showSetupError("Choose a data file and prediction target first.");
    return;
  }
  if (state.source === "upload" && state.uploadError) {
    showSetupError(state.uploadError);
    return;
  }
  revealResults();
  state.running = true;
  elements.results.setAttribute("aria-busy", "true");
  elements.run.disabled = true;
  elements.runLabel.textContent = "Searching…";
  elements.content.classList.add("hidden");
  elements.resultStatus.innerHTML =
    '<span class="pulse-dot" aria-hidden="true"></span>Finding feature mixes';
  elements.resultTitle.textContent = elements.scenarioQuestion.textContent;
  elements.resultQuestion.textContent =
    "The search is measuring useful signal, repeated information, and the exact input limit.";
  const progressTimer = startProgress();
  const controller = new AbortController();
  const requestTimeout = window.setTimeout(() => controller.abort(), 30_000);
  try {
    const response = await fetch("/api/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
      signal: controller.signal,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Something went wrong while testing the feature set.");
    }
    state.result = payload;
    window.clearInterval(progressTimer);
    elements.progressBar.style.width = "100%";
    elements.progressLabel.textContent = "Results ready";
    elements.progressValue.textContent = "100%";
    renderResult(payload);
    window.setTimeout(() => elements.progressShell.classList.add("hidden"), 550);
  } catch (error) {
    showError(
      error.name === "AbortError"
        ? "The run took too long and was stopped. Try Quick search or a smaller dataset."
        : error.message || "The experiment failed.",
    );
  } finally {
    window.clearTimeout(requestTimeout);
    window.clearInterval(progressTimer);
    state.running = false;
    syncSetupState();
  }
}

function showError(message) {
  elements.results.setAttribute("aria-busy", "false");
  elements.progressShell.classList.add("hidden");
  elements.content.classList.add("hidden");
  elements.resultStatus.innerHTML =
    '<span class="pulse-dot" aria-hidden="true"></span>Needs attention';
  elements.resultTitle.textContent = "The run could not finish.";
  elements.resultQuestion.textContent = message;
}

function formatScore(value) {
  return Number(value).toFixed(3);
}

function formatSignedPoints(value) {
  const points = Number(value) * 100;
  return `${points >= 0 ? "+" : ""}${points.toFixed(1)} pts`;
}

function formatComparison(value) {
  if (Math.abs(Number(value)) < 0.005) return "≈ same";
  return formatSignedPoints(value);
}

function formatRuntime(milliseconds) {
  return milliseconds < 1000
    ? `${Math.round(milliseconds)} ms`
    : `${(milliseconds / 1000).toFixed(2)} s`;
}

function humanizeFeatureName(name) {
  const acronyms = new Map([
    ["cpu", "CPU"],
    ["gb", "GB"],
    ["iops", "IOPS"],
    ["kb", "KB"],
    ["p95", "P95"],
    ["rms", "RMS"],
    ["usd", "USD"],
  ]);
  const words = String(name || "")
    .replace(/[_-]+/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => acronyms.get(word.toLowerCase()) || word.toLowerCase());
  if (!words.length) return "Unnamed input";
  const label = words.join(" ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function overlapDescription(value) {
  if (value <= 0.15) return "low overlap";
  if (value <= 0.35) return "some overlap";
  return "high overlap";
}

function setText(selector, value) {
  const node = $(selector);
  if (node) node.textContent = value;
}

function renderResult(result) {
  elements.results.setAttribute("aria-busy", "false");
  elements.content.classList.remove("hidden", "loading-soft");
  elements.resultStatus.innerHTML =
    '<span class="pulse-dot" aria-hidden="true"></span>Results ready';
  elements.resultTitle.textContent = result.dataset.question || result.dataset.name;
  elements.resultQuestion.textContent =
    result.dataset.description ||
    `Finding a smaller input set for ${humanizeFeatureName(result.dataset.target)}.`;
  setText("#metric-features", `${result.selection.k} of ${result.dataset.features}`);
  setText(
    "#metric-reduction",
    `${result.dataset.features - result.selection.k} inputs removed`,
  );
  setText(
    "#metric-score-label",
    `${result.benchmark.qubo.score_label} exploration score`,
  );
  setText("#metric-score", formatScore(result.benchmark.qubo.score));
  setText(
    "#metric-score-note",
    result.dataset.task === "classification"
      ? "0.50 random · 1.00 perfect"
      : "0.00 average guess · 1.00 perfect",
  );
  setText(
    "#metric-score-delta",
    formatComparison(result.insight.score_delta_vs_greedy),
  );
  setText("#metric-space", result.selection.search_space_label);
  setText("#metric-runtime", formatRuntime(result.runtime.total_ms));
  const fullScoreDifference =
    Number(result.benchmark.qubo.score) -
    Number(result.benchmark.all_features.score);
  setText("#metric-full-delta", formatComparison(fullScoreDifference));
  const qualityMessage =
    Math.abs(fullScoreDifference) < 0.015
      ? "Quality stayed close to using every input."
      : fullScoreDifference > 0
        ? "Quality was stronger than using every input."
        : "There is a visible quality trade-off at this smaller limit.";
  setText(
    "#finding-title",
    `Keep ${result.selection.k} inputs. ${qualityMessage}`,
  );
  setText("#finding-copy", result.insight.finding);
  setText("#score-explainer", result.insight.score_explanation);
  setText(
    "#anneal-stat",
    `${result.annealing.reads} attempts · ${result.annealing.sweeps} steps`,
  );
  setText(
    "#constraint-check",
    `${result.selection.names.length} = ${result.selection.k} ✓`,
  );
  setText(
    "#feasible-stat",
    `${result.annealing.feasible_reads} / ${result.annealing.reads}`,
  );
  setText("#qubo-energy", Number(result.selection.energy).toFixed(3));
  setText("#qubo-formula", result.qubo.formula);
  setText("#validation-stat", `${result.dataset.samples.toLocaleString()} rows`);
  setText(
    "#cv-stat",
    `${result.benchmark.qubo.cv_folds}-fold · same splits`,
  );
  setText("#energy-value", `${result.selection.k} selected`);
  setText(
    "#selection-title",
    `${result.selection.k} ${
      result.selection.k === 1 ? "input" : "inputs"
    } worth keeping`,
  );
  setText("#avg-relevance", result.selection.average_relevance.toFixed(2));
  setText("#avg-redundancy", result.selection.average_redundancy.toFixed(2));
  setText("#proxy-ops", `${result.selection.feature_reduction}%`);
  setText("#matrix-stat", `${result.dataset.features} inputs`);

  const cloud = $("#feature-cloud");
  cloud.replaceChildren();
  result.selection.names.forEach((name) => {
    const feature = result.features.find((item) => item.name === name);
    const card = document.createElement("article");
    card.className = "feature-choice";
    const heading = document.createElement("strong");
    heading.textContent = humanizeFeatureName(name);
    const detail = document.createElement("span");
    detail.textContent = feature
      ? `Target link ${Number(feature.relevance).toFixed(2)} · ${overlapDescription(
          Number(feature.selected_redundancy),
        )}`
      : "Chosen for this feature mix";
    card.append(heading, detail);
    cloud.append(card);
  });

  setText("#benchmark-score-label", result.benchmark.qubo.score_label);
  setText("#bench-qubo-k", result.selection.k);
  setText("#bench-greedy-k", result.selection.k);
  setText("#bench-all-k", result.dataset.features);
  setText("#bench-qubo-score", formatScore(result.benchmark.qubo.score));
  setText("#bench-greedy-score", formatScore(result.benchmark.greedy.score));
  setText("#bench-all-score", formatScore(result.benchmark.all_features.score));
  const selectedWork = `${Math.round(
    (result.selection.k / result.dataset.features) * 100,
  )}%`;
  setText("#bench-qubo-ops", selectedWork);
  setText("#bench-greedy-ops", selectedWork);
  setText("#bench-all-ops", "100%");
  setText("#caveat", result.caveat);

  $("#download-result").disabled = false;
  $("#download-qubo").disabled = false;
  drawPareto(result);
  if (elements.resultDetails.open) {
    drawEnergy(result);
    drawQubo(result);
  }
}

function prepareCanvas(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(260, canvas.clientWidth);
  const height = Math.max(150, canvas.clientHeight);
  const pixelWidth = Math.round(width * ratio);
  const pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

function drawGrid(context, width, height, padding, rows = 4) {
  context.strokeStyle = "rgba(244,240,230,0.08)";
  context.lineWidth = 1;
  for (let row = 0; row <= rows; row += 1) {
    const y = padding.top + ((height - padding.top - padding.bottom) * row) / rows;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
  }
}

function drawPareto(result) {
  const canvas = $("#pareto-chart");
  const { context, width, height } = prepareCanvas(canvas);
  context.clearRect(0, 0, width, height);
  const padding = { left: 36, right: 12, top: 16, bottom: 27 };
  drawGrid(context, width, height, padding);
  const series = ["QUBO", "Top relevance"];
  const colors = { QUBO: "#d9ff70", "Top relevance": "#9f92ff" };
  const allScores = result.frontier.map((point) => point.score);
  let minimum = Math.min(...allScores);
  let maximum = Math.max(...allScores);
  const span = Math.max(0.04, maximum - minimum);
  minimum -= span * 0.17;
  maximum += span * 0.12;
  const maxK = Math.max(...result.frontier.map((point) => point.k));
  const x = (k) =>
    padding.left + ((k - 1) / Math.max(1, maxK - 1)) * (width - padding.left - padding.right);
  const y = (score) =>
    padding.top +
    ((maximum - score) / (maximum - minimum)) * (height - padding.top - padding.bottom);

  context.font = "9px SFMono-Regular, monospace";
  context.fillStyle = "rgba(244,240,230,0.35)";
  context.textAlign = "right";
  context.fillText(maximum.toFixed(2), padding.left - 7, padding.top + 3);
  context.fillText(minimum.toFixed(2), padding.left - 7, height - padding.bottom);
  context.textAlign = "center";
  const uniqueK = [...new Set(result.frontier.map((point) => point.k))];
  uniqueK.forEach((k) => {
    context.fillText(k, x(k), height - 8);
  });

  series.forEach((method) => {
    const points = result.frontier.filter((point) => point.method === method);
    context.strokeStyle = colors[method];
    context.lineWidth = method === "QUBO" ? 2.3 : 1.4;
    context.setLineDash(method === "QUBO" ? [] : [5, 5]);
    context.beginPath();
    points.forEach((point, index) => {
      if (index === 0) context.moveTo(x(point.k), y(point.score));
      else context.lineTo(x(point.k), y(point.score));
    });
    context.stroke();
    context.setLineDash([]);
    points.forEach((point) => {
      context.fillStyle = colors[method];
      context.beginPath();
      context.arc(x(point.k), y(point.score), method === "QUBO" ? 3.5 : 2.5, 0, Math.PI * 2);
      context.fill();
      if (point.k === result.selection.k && method === "QUBO") {
        context.strokeStyle = colors[method];
        context.lineWidth = 1;
        context.beginPath();
        context.arc(x(point.k), y(point.score), 8, 0, Math.PI * 2);
        context.stroke();
      }
    });
  });
}

function drawEnergy(result) {
  const canvas = $("#energy-chart");
  const { context, width, height } = prepareCanvas(canvas);
  context.clearRect(0, 0, width, height);
  const padding = { left: 32, right: 10, top: 16, bottom: 27 };
  drawGrid(context, width, height, padding);
  const points = result.annealing.curve;
  const energies = points.map((point) => point.energy);
  let minimum = Math.min(...energies);
  let maximum = Math.max(...energies);
  const span = Math.max(0.1, maximum - minimum);
  minimum -= span * 0.1;
  maximum += span * 0.1;
  const maxSweep = Math.max(...points.map((point) => point.sweep));
  const x = (sweep) =>
    padding.left +
    (sweep / maxSweep) * (width - padding.left - padding.right);
  const y = (energy) =>
    padding.top +
    ((maximum - energy) / (maximum - minimum)) * (height - padding.top - padding.bottom);

  const gradient = context.createLinearGradient(0, padding.top, 0, height - padding.bottom);
  gradient.addColorStop(0, "rgba(217,255,112,0.28)");
  gradient.addColorStop(1, "rgba(217,255,112,0)");
  context.beginPath();
  points.forEach((point, index) => {
    if (index === 0) context.moveTo(x(point.sweep), y(point.energy));
    else context.lineTo(x(point.sweep), y(point.energy));
  });
  context.lineTo(x(maxSweep), height - padding.bottom);
  context.lineTo(padding.left, height - padding.bottom);
  context.closePath();
  context.fillStyle = gradient;
  context.fill();

  context.beginPath();
  points.forEach((point, index) => {
    if (index === 0) context.moveTo(x(point.sweep), y(point.energy));
    else context.lineTo(x(point.sweep), y(point.energy));
  });
  context.strokeStyle = "#d9ff70";
  context.lineWidth = 2;
  context.stroke();
  const finalPoint = points[points.length - 1];
  context.fillStyle = "#d9ff70";
  context.beginPath();
  context.arc(x(finalPoint.sweep), y(finalPoint.energy), 4, 0, Math.PI * 2);
  context.fill();

  context.fillStyle = "rgba(244,240,230,0.35)";
  context.font = "9px SFMono-Regular, monospace";
  context.textAlign = "left";
  context.fillText("0", padding.left, height - 8);
  context.textAlign = "right";
  context.fillText(maxSweep, width - padding.right, height - 8);
}

function drawQubo(result) {
  const canvas = $("#qubo-chart");
  const { context, width, height } = prepareCanvas(canvas);
  context.clearRect(0, 0, width, height);
  const matrix = result.qubo.matrix;
  const n = matrix.length;
  const size = Math.min(width, height) - 8;
  const cell = size / n;
  const offsetX = (width - size) / 2;
  const offsetY = (height - size) / 2;
  const maximum = Math.max(...matrix.flat().map((value) => Math.abs(value)), 1e-9);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const value = matrix[i][j];
      const strength = Math.min(1, Math.abs(value) / maximum);
      const selected = result.selection.indices.includes(i) && result.selection.indices.includes(j);
      const color = value >= 0 ? [159, 146, 255] : [217, 255, 112];
      const alpha = 0.07 + strength * 0.65;
      context.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
      context.fillRect(offsetX + j * cell, offsetY + i * cell, cell - 0.6, cell - 0.6);
      if (selected) {
        context.strokeStyle = "rgba(255,141,114,0.5)";
        context.lineWidth = 0.7;
        context.strokeRect(offsetX + j * cell, offsetY + i * cell, cell - 0.6, cell - 0.6);
      }
    }
  }
}

function downloadJSON(filename, payload) {
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], {
    type: "application/json",
  });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}

function drawHero(time = 0) {
  const canvas = $("#hero-canvas");
  if (!canvas) return;
  const { context, width, height } = prepareCanvas(canvas);
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) * 0.38;
  context.clearRect(0, 0, width, height);
  for (let ring = 0; ring < 4; ring += 1) {
    context.beginPath();
    const steps = 110;
    for (let step = 0; step <= steps; step += 1) {
      const angle = (step / steps) * Math.PI * 2;
      const wobble =
        Math.sin(angle * (3 + ring) + time * 0.00035 + ring) * (7 + ring * 2);
      const localRadius = radius * (0.32 + ring * 0.19) + wobble;
      const x = centerX + Math.cos(angle) * localRadius;
      const y = centerY + Math.sin(angle) * localRadius * 0.64;
      if (step === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.strokeStyle =
      ring === 2 ? "rgba(217,255,112,0.34)" : "rgba(159,146,255,0.15)";
    context.lineWidth = ring === 2 ? 1.4 : 0.8;
    context.stroke();
  }
  const nodes = 14;
  for (let index = 0; index < nodes; index += 1) {
    const angle = (index / nodes) * Math.PI * 2 + time * 0.00008;
    const nodeRadius = radius * (0.38 + (index % 4) * 0.16);
    const x = centerX + Math.cos(angle) * nodeRadius;
    const y = centerY + Math.sin(angle) * nodeRadius * 0.64;
    context.fillStyle = index % 5 === 0 ? "#d9ff70" : "rgba(244,240,230,0.45)";
    context.beginPath();
    context.arc(x, y, index % 5 === 0 ? 3 : 1.7, 0, Math.PI * 2);
    context.fill();
  }
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    window.requestAnimationFrame(drawHero);
  }
}

function initializePointerGlow() {
  const glow = $("#pointer-glow");
  if (!glow) return;
  let frame = 0;
  let x = 0;
  let y = 0;
  let positioned = false;
  const positionGlow = () => {
    glow.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%)`;
    frame = 0;
  };
  window.addEventListener(
    "pointermove",
    (event) => {
      x = event.clientX;
      y = event.clientY;
      if (!positioned) {
        positionGlow();
        positioned = true;
      }
      glow.classList.add("visible");
      glow.classList.toggle(
        "interactive",
        event.target instanceof Element &&
          Boolean(event.target.closest("button, a, input, select, summary, label")),
      );
      if (!frame) frame = window.requestAnimationFrame(positionGlow);
    },
    { passive: true },
  );
  window.addEventListener(
    "pointerdown",
    (event) => {
      x = event.clientX;
      y = event.clientY;
      positionGlow();
      glow.classList.add("visible", "pressed");
    },
    { passive: true },
  );
  window.addEventListener(
    "pointerup",
    () => glow.classList.remove("pressed"),
    { passive: true },
  );
  document.documentElement.addEventListener("mouseleave", () =>
    glow.classList.remove("visible"),
  );
}

elements.budget.addEventListener("input", () => {
  updateBudgetLabel();
  markSettingsChanged();
});
elements.redundancy.addEventListener("input", updateRedundancyLabel);
elements.run.addEventListener("click", runExperiment);
elements.file.addEventListener("change", (event) => handleFile(event.target.files[0]));
elements.uploadTab.addEventListener("click", () => setSourceMode("upload"));
elements.sampleTab.addEventListener("click", () => setSourceMode("sample"));
elements.sampleSelect.addEventListener("change", (event) =>
  selectDemo(event.target.value),
);
elements.target.addEventListener("change", async () => {
  invalidateResult();
  elements.run.disabled = true;
  elements.uploadInspection.textContent = "Checking this prediction target…";
  elements.uploadInspection.classList.remove("warning");
  try {
    await inspectCurrentUpload(elements.target.value);
    markSettingsChanged("Target changed · run to update");
  } catch (error) {
    state.uploadError = error.message || "This prediction target could not be checked.";
    showSetupError(state.uploadError);
  } finally {
    syncSetupState();
  }
});
$$('input[name="quality"]').forEach((input) =>
  input.addEventListener("change", markSettingsChanged),
);
$("#download-result").addEventListener("click", () => {
  if (state.result) downloadJSON("qubolens-result.json", state.result);
});
$("#download-qubo").addEventListener("click", () => {
  if (state.result) downloadJSON("qubolens-qubo.json", state.result.qubo.export);
});
elements.resultDetails.addEventListener("toggle", () => {
  if (elements.resultDetails.open && state.result) {
    window.requestAnimationFrame(() => {
      drawEnergy(state.result);
      drawQubo(state.result);
    });
  }
});

let resizeTimer;
window.addEventListener("resize", () => {
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => {
    if (state.result) {
      drawPareto(state.result);
      if (elements.resultDetails.open) {
        drawEnergy(state.result);
        drawQubo(state.result);
      }
    }
  }, 120);
});

updateBudgetLabel();
updateRedundancyLabel();
syncSetupState();
initializePointerGlow();
window.requestAnimationFrame(drawHero);
