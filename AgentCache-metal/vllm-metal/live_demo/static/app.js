const els = {
  subtitle: document.querySelector("#subtitle"),
  baselineStatus: document.querySelector("#baseline-status"),
  centroidStatus: document.querySelector("#centroid-status"),
  conversation: document.querySelector("#conversation"),
  maxTokens: document.querySelector("#max-tokens"),
  resetBtn: document.querySelector("#reset-btn"),
  prepareBtn: document.querySelector("#prepare-btn"),
  runBtn: document.querySelector("#run-btn"),
  advanceBtn: document.querySelector("#advance-btn"),
  runSingleBtn: document.querySelector("#run-single-btn"),
  singlePrompt: document.querySelector("#single-prompt"),
  turnMeta: document.querySelector("#turn-meta"),
  turnPrompt: document.querySelector("#turn-prompt"),
  turnNext: document.querySelector("#turn-next"),
  speedupRatio: document.querySelector("#speedup-ratio"),
  speedupSaved: document.querySelector("#speedup-saved"),
  tokensSaved: document.querySelector("#tokens-saved"),
  baselineTtft: document.querySelector("#baseline-ttft"),
  baselineTotal: document.querySelector("#baseline-total"),
  baselineTokens: document.querySelector("#baseline-tokens"),
  baselineHistory: document.querySelector("#baseline-history"),
  baselineOutput: document.querySelector("#baseline-output"),
  centroidTtft: document.querySelector("#centroid-ttft"),
  centroidTotal: document.querySelector("#centroid-total"),
  centroidTokens: document.querySelector("#centroid-tokens"),
  centroidHistory: document.querySelector("#centroid-history"),
  centroidOutput: document.querySelector("#centroid-output"),
  timelineBody: document.querySelector("#timeline-body"),
  eventLog: document.querySelector("#event-log"),
};

let config = null;
let current = null;
let busy = false;

function logEvent(message) {
  const item = document.createElement("li");
  item.textContent = `${new Date().toLocaleTimeString()} ${message}`;
  els.eventLog.prepend(item);
}

function fmtMs(value) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)} ms` : "--";
}

function fmtInt(value) {
  return Number.isFinite(Number(value)) ? `${Math.round(Number(value))}` : "--";
}

function fmtRatio(value) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(2)}x` : "--";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function setBusy(value) {
  busy = value;
  for (const button of [els.resetBtn, els.prepareBtn, els.runBtn, els.advanceBtn, els.runSingleBtn]) {
    button.disabled = busy;
  }
}

function setStatus(el, label, health) {
  const ready = Boolean(health?.ready);
  const alive = Boolean(health?.alive);
  const error = health?.startup_error || health?.last_error;
  el.classList.toggle("ready", ready && !error);
  el.classList.toggle("error", Boolean(error) || !alive);
  if (error) {
    el.textContent = `${label}: error`;
  } else if (ready) {
    el.textContent = `${label}: ready`;
  } else if (alive) {
    el.textContent = `${label}: loading`;
  } else {
    el.textContent = `${label}: stopped`;
  }
}

async function pollHealth() {
  try {
    const health = await api("/api/health");
    setStatus(els.baselineStatus, "Baseline", health.workers?.baseline);
    setStatus(els.centroidStatus, "Centroid", health.workers?.centroid);
  } catch (error) {
    els.baselineStatus.textContent = "Baseline: unknown";
    els.centroidStatus.textContent = "Centroid: unknown";
  }
}

function renderCurrent(payload) {
  current = payload;
  els.turnMeta.textContent = `Turn ${payload.turn} of ${payload.total_turns} · ${payload.conversation}`;
  els.turnPrompt.textContent = payload.user;
  els.turnNext.textContent = payload.next ? `Next: ${payload.next}` : "Last scripted turn.";
  renderTimeline(payload.timeline || []);
}

function renderConfig(payload) {
  config = payload;
  const ttftMode = payload.measurement_warmup ? "warmed TTFT" : "cold TTFT";
  const memory = payload.metal_memory_fraction || "auto";
  els.subtitle.textContent = `${payload.model} · N=${payload.n} · ${payload.execution_mode || "sequential"} · ${ttftMode} · Metal memory=${memory} · ${payload.default_conversation}`;
  els.maxTokens.value = payload.default_max_tokens;
  els.conversation.innerHTML = "";
  for (const [name, info] of Object.entries(payload.conversations)) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `${name} (${info.turns} turns)`;
    els.conversation.append(option);
  }
  if (payload.conversations[payload.default_conversation]) {
    els.conversation.value = payload.default_conversation;
  }
  if (payload.canned_prompts?.length) {
    els.singlePrompt.value = payload.canned_prompts[0];
  }
}

function renderResult(payload) {
  const {baseline, centroid, speedup} = payload;
  els.baselineTtft.textContent = fmtMs(baseline?.ttft_ms);
  els.baselineTotal.textContent = fmtMs(baseline?.total_ms);
  els.baselineTokens.textContent = fmtInt(baseline?.prompt_tokens);
  els.baselineHistory.textContent = fmtInt(baseline?.history_turns);
  els.baselineOutput.textContent = baseline?.output || baseline?.error || "";

  els.centroidTtft.textContent = fmtMs(centroid?.ttft_ms);
  els.centroidTotal.textContent = fmtMs(centroid?.total_ms);
  els.centroidTokens.textContent = fmtInt(centroid?.prompt_tokens);
  els.centroidHistory.textContent = fmtInt(centroid?.history_turns);
  els.centroidOutput.textContent = centroid?.output || centroid?.error || "";

  els.speedupRatio.textContent = fmtRatio(speedup?.ttft_ratio);
  els.speedupSaved.textContent = fmtMs(speedup?.ttft_saved_ms);
  els.tokensSaved.textContent = fmtInt(speedup?.prompt_tokens_saved);
  renderTimeline(payload.timeline || current?.timeline || []);
}

function renderTimeline(rows) {
  els.timelineBody.innerHTML = "";
  if (!rows.length) {
    const row = document.createElement("tr");
    row.innerHTML = `<td colspan="6">No turns run yet.</td>`;
    els.timelineBody.append(row);
    return;
  }
  for (const item of rows) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${item.turn}</td>
      <td>${fmtMs(item.baseline_ttft_ms)}</td>
      <td>${fmtMs(item.centroid_ttft_ms)}</td>
      <td>${fmtInt(item.baseline_prompt_tokens)}</td>
      <td>${fmtInt(item.centroid_prompt_tokens)}</td>
      <td>${fmtRatio(item.ttft_ratio)}</td>
    `;
    els.timelineBody.append(row);
  }
}

async function loadInitial() {
  const payload = await api("/api/config");
  renderConfig(payload);
  const currentPayload = await api("/api/turn/current");
  renderCurrent(currentPayload);
  logEvent("dashboard loaded");
}

async function resetConversation() {
  setBusy(true);
  try {
    const payload = await api("/api/session/reset", {
      method: "POST",
      body: JSON.stringify({
        conversation: els.conversation.value,
        max_tokens: Number(els.maxTokens.value),
      }),
    });
    renderCurrent(payload);
    renderResult({baseline: {}, centroid: {}, speedup: {}, timeline: []});
    logEvent(`reset conversation ${payload.conversation}`);
  } catch (error) {
    logEvent(`reset failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function preparePrefix() {
  setBusy(true);
  try {
    const payload = await api("/api/prefix/prepare", {method: "POST", body: "{}"});
    renderCurrent(payload);
    logEvent(`both modes preflighted (${config?.execution_mode || "parallel"})`);
  } catch (error) {
    logEvent(`prepare failed: ${error.message}`);
  } finally {
    setBusy(false);
    pollHealth();
  }
}

async function runCurrentTurn() {
  setBusy(true);
  try {
    const payload = await api("/api/turn/run", {
      method: "POST",
      body: JSON.stringify({
        turn: current?.turn,
        max_tokens: Number(els.maxTokens.value),
      }),
    });
    renderResult(payload);
    logEvent(`turn ${payload.turn} completed (${config?.execution_mode || "parallel"})`);
    pollHealth();
  } catch (error) {
    logEvent(`run failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function advanceTurn() {
  setBusy(true);
  try {
    const payload = await api("/api/turn/advance", {method: "POST", body: "{}"});
    renderCurrent(payload);
    if (payload.at_end) {
      logEvent("conversation is at the final turn");
    } else {
      logEvent(`advanced to turn ${payload.turn}`);
    }
  } catch (error) {
    logEvent(`advance failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function runSinglePrompt() {
  setBusy(true);
  try {
    const payload = await api("/api/run_single", {
      method: "POST",
      body: JSON.stringify({
        prompt: els.singlePrompt.value,
        max_tokens: Number(els.maxTokens.value),
      }),
    });
    renderResult(payload);
    logEvent(`single prompt completed (${config?.execution_mode || "parallel"})`);
    pollHealth();
  } catch (error) {
    logEvent(`single prompt failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

els.resetBtn.addEventListener("click", resetConversation);
els.prepareBtn.addEventListener("click", preparePrefix);
els.runBtn.addEventListener("click", runCurrentTurn);
els.advanceBtn.addEventListener("click", advanceTurn);
els.runSingleBtn.addEventListener("click", runSinglePrompt);
els.conversation.addEventListener("change", resetConversation);

loadInitial().catch((error) => logEvent(`startup failed: ${error.message}`));
pollHealth();
setInterval(pollHealth, 2500);
