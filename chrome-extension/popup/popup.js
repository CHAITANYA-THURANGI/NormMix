const DEFAULT_API_BASE = "http://127.0.0.1:8000";

async function getApiBase() {
  try {
    const { apiBase } = await chrome.storage.sync.get("apiBase");
    return (apiBase || DEFAULT_API_BASE).replace(/\/$/, "");
  } catch (e) {
    return DEFAULT_API_BASE;
  }
}

const $ = (id) => document.getElementById(id);
let lastResponse = null;
let currentView = "normalized_code_mixed";

const VIEW_LABELS = {
  "normalized_code_mixed": "Normalized Code-Mixed",
  "english_translation": "English Translation",
  "pure_telugu": "Pure Telugu (అచ్చ తెలుగు)",
  "all_telugu_script": "Telugu Script (తెలుగు లిపి)",
  "all_romanized_tanglish": "Tanglish (రోమనైజ్డ్)",
  "english_gloss": "Word Meaning (పదార్థం)"
};

async function init() {
  const apiBase = await getApiBase();
  $("apiBase").value = apiBase;
  try {
    const { lastText, lastEngine } = await chrome.storage.local.get(["lastText", "lastEngine"]);
    if (lastText) $("input").value = lastText;
    if (lastEngine) $("engine").value = lastEngine;
  } catch(e) {}
}

function switchView(viewName) {
  currentView = viewName;
  ["tabNorm", "tabEn", "tabPureTe", "tabTe", "tabRom", "tabGloss"].forEach(id => {
    const el = $(id);
    if (el) el.classList.remove("active");
  });
  if (viewName === "normalized_code_mixed") $("tabNorm")?.classList.add("active");
  else if (viewName === "english_translation") $("tabEn")?.classList.add("active");
  else if (viewName === "pure_telugu") $("tabPureTe")?.classList.add("active");
  else if (viewName === "all_telugu_script") $("tabTe")?.classList.add("active");
  else if (viewName === "all_romanized_tanglish") $("tabRom")?.classList.add("active");
  else if (viewName === "english_gloss") $("tabGloss")?.classList.add("active");

  const badge = $("outputTypeBadge");
  if (badge) badge.textContent = VIEW_LABELS[viewName] || "Output";

  if (lastResponse && lastResponse.options) {
    const text = lastResponse.options[viewName] || "";
    $("outputText").textContent = text;

    // Display Prescribed Meaning in meaning card
    const meaning = lastResponse.options.prescribed_meaning || lastResponse.options.english_translation || "";
    if (meaning) {
      $("meaningCard").style.display = "block";
      $("meaningText").textContent = meaning;
    } else {
      $("meaningCard").style.display = "none";
    }
  }
}

async function run() {
  const text = $("input").value.trim();
  if (!text) {
    $("status").textContent = "Type or select sample text first.";
    $("status").className = "status err";
    return;
  }
  $("run").disabled = true;
  $("status").className = "status";
  $("status").textContent = "Processing auto-detection & neural inference...";
  $("output").style.display = "none";
  $("meaningCard").style.display = "none";
  $("detectBadge").style.display = "none";
  $("viewTabs").style.display = "none";

  const apiBase = await getApiBase();
  const engine = $("engine").value;
  try {
    await chrome.storage.local.set({ lastText: text, lastEngine: engine });
  } catch(e) {}

  try {
    const res = await fetch(`${apiBase}/omni/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, engine, n_best: 1, beam_size: 4 }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${res.status}`);
    }
    const data = await res.json();
    lastResponse = data;

    // Show Detect Badge
    $("detectBadge").style.display = "block";
    $("detectBadge").textContent = `✨ ${data.detected.label} (${Math.round(data.detected.confidence * 100)}% conf)`;

    // Show View Tabs & Output Card
    $("viewTabs").style.display = "flex";
    $("output").style.display = "block";
    $("metaLabel").textContent = `${data.latency_ms}ms · ${data.recommended.engine} · ${data.recommended.model}`;
    switchView(currentView);

    $("status").textContent = "";
  } catch (e) {
    $("status").className = "status err";
    $("status").textContent = `Error: ${e.message}. Is server running at ${apiBase}?`;
  } finally {
    $("run").disabled = false;
  }
}

// Attach event listeners cleanly
$("tabNorm").addEventListener("click", () => switchView("normalized_code_mixed"));
$("tabEn").addEventListener("click", () => switchView("english_translation"));
$("tabPureTe").addEventListener("click", () => switchView("pure_telugu"));
$("tabTe").addEventListener("click", () => switchView("all_telugu_script"));
$("tabRom").addEventListener("click", () => switchView("all_romanized_tanglish"));
$("tabGloss").addEventListener("click", () => switchView("english_gloss"));

$("run").addEventListener("click", run);

$("copyBtn").addEventListener("click", () => {
  const text = $("outputText").textContent;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    $("copyBtn").textContent = "Copied!";
    setTimeout(() => { $("copyBtn").textContent = "Copy"; }, 1500);
  });
});

$("copyMeaningBtn").addEventListener("click", () => {
  const text = $("meaningText").textContent;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    $("copyMeaningBtn").textContent = "Copied!";
    setTimeout(() => { $("copyMeaningBtn").textContent = "Copy Meaning"; }, 1500);
  });
});

document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    const sample = chip.getAttribute("data-sample");
    if (sample) {
      $("input").value = sample;
      run();
    }
  });
});

$("save").addEventListener("click", async () => {
  try {
    await chrome.storage.sync.set({ apiBase: $("apiBase").value.trim() || DEFAULT_API_BASE });
    $("status").className = "status";
    $("status").textContent = "API URL saved.";
  } catch(e) {}
});

$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
});

init();

