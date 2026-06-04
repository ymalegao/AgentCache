/* AgentCache Live Demo — v2 frontend, driven by the real server.py API. */

const {
  Pill, Panel, Btn, Toggle, Metric, KV, StatusCell,
  OutputBox, fmtMs, unitFor, fmtInt,
} = window;

const { useState, useEffect, useRef, useMemo, useCallback } = React;

const HEALTH_POLL_MS = 2500;

/* ---------- helpers ---------- */
function pad(n) { return String(n).padStart(2, "0"); }
function nowTs() {
  const t = new Date();
  return `${pad(t.getHours())}:${pad(t.getMinutes())}:${pad(t.getSeconds())}`;
}
function summarize(prompt, max = 56) {
  if (!prompt) return "";
  const s = prompt.replace(/\s+/g, " ").trim();
  return s.length > max ? s.slice(0, max - 1) + "…" : s;
}
function workerStatus(w) {
  if (!w) return { tone: "idle", label: "Unknown" };
  if (w.startup_error) return { tone: "err", label: "Error" };
  if (!w.alive) return { tone: "idle", label: "Unloaded" };
  if (!w.ready) return { tone: "warn", label: "Loading" };
  if (w.last_error) return { tone: "err", label: "Error" };
  return { tone: "ok", label: "Ready" };
}
async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const text = await r.text();
  let data;
  try { data = text ? JSON.parse(text) : null; }
  catch { throw new Error(`${path}: bad JSON response`); }
  if (!r.ok) {
    const msg = data && data.error ? data.error : `${path}: HTTP ${r.status}`;
    throw new Error(msg);
  }
  return data;
}

/* =========================================================
   APP
   ========================================================= */
function App() {
  /* ----- server state ----- */
  const [config, setConfig] = useState(null);
  const [health, setHealth] = useState(null);
  const [currentPayload, setCurrentPayload] = useState(null);
  const [turnResults, setTurnResults] = useState({}); // {turnNum: {baseline, centroid, speedup}}
  const [timeline, setTimeline] = useState([]);
  const [error, setError] = useState(null);

  /* ----- UI state ----- */
  const [maxTokens, setMaxTokens] = useState(192);
  const [showEventLog, setShowEventLog] = useState(true);
  const [pauseSeconds, setPauseSeconds] = useState(5); // wait between turns in a full-sequence run

  /* ----- run state ----- */
  const [phase, setPhase] = useState("idle"); // idle | preparing | running | done
  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false); // any action in flight — disables buttons
  const [turnInFlight, setTurnInFlight] = useState(false); // true only during /api/turn/run
  const [pauseRemainingMs, setPauseRemainingMs] = useState(null); // ms remaining in inter-turn pause
  const stopRef = useRef(false);

  /* ----- log ----- */
  const [logLines, setLogLines] = useState([]);
  const terminalRef = useRef(null);
  const log = useCallback((tag, text) => {
    setLogLines((lines) => {
      const next = lines.length > 240 ? lines.slice(-200) : lines;
      return [...next, { ts: nowTs(), tag, text }];
    });
  }, []);
  useEffect(() => {
    if (terminalRef.current) terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
  }, [logLines]);

  /* ----- initial load ----- */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [cfg, hp, cur] = await Promise.all([
          api("/api/config"),
          api("/api/health"),
          api("/api/turn/current"),
        ]);
        if (cancelled) return;
        setConfig(cfg);
        setHealth(hp);
        setCurrentPayload(cur);
        setTimeline(cur.timeline || []);
        setMaxTokens(cfg.default_max_tokens || 192);
        log("server", `connected · model=${cfg.model} · N=${cfg.n} · execution_mode=${cfg.execution_mode}`);
        if (hp.mock) log("server", "mock mode is ENABLED (synthetic timings, no real model)");
      } catch (e) {
        if (!cancelled) {
          setError(String(e.message || e));
          log("err", `initial load failed: ${e.message || e}`);
        }
      }
    })();
    return () => { cancelled = true; };
  }, [log]);

  /* ----- health polling ----- */
  useEffect(() => {
    let cancelled = false;
    let timer = null;
    const tick = async () => {
      try {
        const hp = await api("/api/health");
        if (cancelled) return;
        setHealth(hp);
      } catch (e) {
        if (!cancelled) log("err", `health poll failed: ${e.message || e}`);
      }
      if (!cancelled) timer = setTimeout(tick, HEALTH_POLL_MS);
    };
    timer = setTimeout(tick, HEALTH_POLL_MS);
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [log]);

  /* ----- derived ----- */
  const conversations = config?.conversations || {};
  const conversationKey = currentPayload?.conversation || "csv_cli";
  const totalTurns = currentPayload?.total_turns || 0;
  const turnNumber = currentPayload?.turn || 0; // 1-based
  const turnHasRun = !!currentPayload?.turn_has_run;
  const prepared = !!currentPayload?.prepared;
  const baseWorker = health?.workers?.baseline;
  const centWorker = health?.workers?.centroid;
  const baseState = workerStatus(baseWorker);
  const centState = workerStatus(centWorker);
  const serverOk = health?.server === "ok";
  const modelLoaded = !!(baseWorker?.ready && centWorker?.ready);
  const centroidArmed = !!(centWorker?.last_event?.centroid_armed);
  const mockMode = !!health?.mock;
  const currentResult = turnResults[turnNumber] || null;

  /* ----- actions ----- */
  async function handleReset() {
    if (busy) return;
    setBusy(true);
    stopRef.current = false;
    try {
      const payload = await api("/api/session/reset", {
        method: "POST",
        body: JSON.stringify({ conversation: conversationKey, max_tokens: maxTokens }),
      });
      setCurrentPayload(payload);
      setTimeline(payload.timeline || []);
      setTurnResults({});
      setPhase("idle");
      log("server", `reset · conversation=${payload.conversation} · turns=${payload.total_turns}`);
    } catch (e) {
      log("err", `reset failed: ${e.message || e}`);
    } finally {
      setBusy(false);
    }
  }

  async function handleConversationChange(key) {
    if (busy || running) return;
    setBusy(true);
    stopRef.current = false;
    try {
      const payload = await api("/api/session/reset", {
        method: "POST",
        body: JSON.stringify({ conversation: key, max_tokens: maxTokens }),
      });
      setCurrentPayload(payload);
      setTimeline(payload.timeline || []);
      setTurnResults({});
      setPhase("idle");
      log("server", `switched conversation: ${key} (${payload.total_turns} turns)`);
    } catch (e) {
      log("err", `conversation change failed: ${e.message || e}`);
    } finally {
      setBusy(false);
    }
  }

  async function handlePrepare() {
    if (busy || running) return;
    setBusy(true);
    setPhase("preparing");
    log("server", "preparing workers · loading model + centroid into memory");
    try {
      const payload = await api("/api/prefix/prepare", {
        method: "POST",
        body: JSON.stringify({}),
      });
      setCurrentPayload({ ...currentPayload, ...payload, timeline: payload.timeline || currentPayload?.timeline || [] });
      setTimeline(payload.timeline || currentPayload?.timeline || []);
      const b = payload.workers?.baseline || {};
      const c = payload.workers?.centroid || {};
      log("baseline", `prepare ok · ttft=${b.ttft_ms ?? "—"} ms · prompt_tokens=${b.prompt_tokens ?? "—"}`);
      log("centroid", `prepare ok · ttft=${c.ttft_ms ?? "—"} ms · prompt_tokens=${c.prompt_tokens ?? "—"} · synthetic=${c.synthetic_tokens ?? 0}`);
      setPhase("done");
    } catch (e) {
      log("err", `prepare failed: ${e.message || e}`);
      setPhase("idle");
    } finally {
      setBusy(false);
    }
  }

  async function pauseBetweenTurns(seconds) {
    const totalMs = Math.max(0, Math.floor(seconds * 1000));
    if (totalMs === 0) return;
    const tickMs = 100;
    const startedAt = Date.now();
    setPauseRemainingMs(totalMs);
    try {
      while (Date.now() - startedAt < totalMs) {
        if (stopRef.current) return;
        await new Promise(r => setTimeout(r, tickMs));
        setPauseRemainingMs(Math.max(0, totalMs - (Date.now() - startedAt)));
      }
    } finally {
      setPauseRemainingMs(null);
    }
  }

  async function runCurrentTurnOnce() {
    const turn = currentPayload?.turn;
    if (!turn) throw new Error("no current turn");
    log("turn", `turn ${turn}: running both workers · ${summarize(currentPayload.user, 80)}`);
    setTurnInFlight(true);
    let payload;
    try {
      payload = await api("/api/turn/run", {
        method: "POST",
        body: JSON.stringify({ turn, max_tokens: maxTokens }),
      });
    } finally {
      setTurnInFlight(false);
    }
    const b = payload.baseline || {};
    const c = payload.centroid || {};
    setTurnResults((prev) => ({
      ...prev,
      [turn]: { baseline: b, centroid: c, speedup: payload.speedup || {} },
    }));
    setTimeline(payload.timeline || []);
    setCurrentPayload((prev) => prev ? { ...prev, turn_has_run: true, timeline: payload.timeline || prev.timeline } : prev);
    const sp = payload.speedup || {};
    const ratio = sp.ttft_ratio != null ? `${sp.ttft_ratio}×` : "—";
    log("baseline", `  TTFT=${b.ttft_ms ?? "—"} ms · prompt_tokens=${b.prompt_tokens ?? "—"} · total=${b.total_ms ?? "—"} ms`);
    log("centroid", `  TTFT=${c.ttft_ms ?? "—"} ms · prompt_tokens=${c.prompt_tokens ?? "—"} · total=${c.total_ms ?? "—"} ms · prefix_injected=${c.prefix_injected ? "yes" : "no"}`);
    log("turn", `  speedup=${ratio} · tokens_saved=${sp.prompt_tokens_saved ?? "—"}`);
    return payload;
  }

  async function handleRunCurrentTurn() {
    if (busy || running) return;
    setBusy(true);
    setPhase("running");
    try {
      await runCurrentTurnOnce();
      setPhase("done");
    } catch (e) {
      log("err", `turn run failed: ${e.message || e}`);
      setPhase("idle");
    } finally {
      setBusy(false);
    }
  }

  async function handleAdvance() {
    if (busy || running) return;
    if (!turnHasRun) {
      log("warn", "advance ignored: current turn has not been run yet");
      return;
    }
    setBusy(true);
    try {
      const payload = await api("/api/turn/advance", {
        method: "POST",
        body: JSON.stringify({}),
      });
      if (payload.at_end) {
        log("server", "reached end of conversation");
      } else {
        log("server", `advanced to turn ${payload.turn} / ${payload.total_turns}`);
      }
      setCurrentPayload(payload);
      setTimeline(payload.timeline || []);
    } catch (e) {
      log("err", `advance failed: ${e.message || e}`);
    } finally {
      setBusy(false);
    }
  }

  async function handleRunFullSequence() {
    if (busy || running) return;
    setRunning(true);
    setBusy(true);
    stopRef.current = false;
    log("server", `running full sequence · ${totalTurns} turns · ${pauseSeconds}s pause between turns`);
    try {
      // Reset first so we start clean from turn 1.
      const resetPayload = await api("/api/session/reset", {
        method: "POST",
        body: JSON.stringify({ conversation: conversationKey, max_tokens: maxTokens }),
      });
      setCurrentPayload(resetPayload);
      setTimeline([]);
      setTurnResults({});
      setPhase("running");

      let payload = resetPayload;
      while (!stopRef.current) {
        const turn = payload.turn;
        const total = payload.total_turns;
        log("turn", `turn ${turn}/${total}: ${summarize(payload.user, 80)}`);
        setTurnInFlight(true);
        let runPayload;
        try {
          runPayload = await api("/api/turn/run", {
            method: "POST",
            body: JSON.stringify({ turn, max_tokens: maxTokens }),
          });
        } finally {
          setTurnInFlight(false);
        }
        if (stopRef.current) break;
        const b = runPayload.baseline || {};
        const c = runPayload.centroid || {};
        setTurnResults((prev) => ({
          ...prev,
          [turn]: { baseline: b, centroid: c, speedup: runPayload.speedup || {} },
        }));
        setTimeline(runPayload.timeline || []);
        setCurrentPayload((prev) => prev ? { ...prev, turn_has_run: true, timeline: runPayload.timeline || prev.timeline } : prev);
        const sp = runPayload.speedup || {};
        const ratio = sp.ttft_ratio != null ? `${sp.ttft_ratio}×` : "—";
        log("turn", `  speedup=${ratio} · b_ttft=${b.ttft_ms ?? "—"} ms · c_ttft=${c.ttft_ms ?? "—"} ms`);

        if (turn >= total) {
          payload = { ...payload, turn_has_run: true };
          break;
        }

        // Pause so the user can read the result before advancing.
        await pauseBetweenTurns(pauseSeconds);
        if (stopRef.current) break;

        const next = await api("/api/turn/advance", {
          method: "POST",
          body: JSON.stringify({}),
        });
        if (next.at_end) {
          setCurrentPayload(next);
          break;
        }
        setCurrentPayload(next);
        setTimeline(next.timeline || []);
        payload = next;
      }
      if (stopRef.current) {
        log("warn", "sequence interrupted by user");
      } else {
        log("server", "sequence complete");
      }
      setPhase("done");
    } catch (e) {
      log("err", `sequence failed: ${e.message || e}`);
      setPhase("idle");
    } finally {
      setRunning(false);
      setBusy(false);
      stopRef.current = false;
    }
  }

  function handleStop() {
    if (!running && !busy) return;
    stopRef.current = true;
    log("warn", "stop requested · will halt after the current request completes");
  }

  /* ----- aggregates ----- */
  const summary = useMemo(() => {
    if (!currentResult) return null;
    const b = currentResult.baseline, c = currentResult.centroid, s = currentResult.speedup || {};
    if (!b?.ttft_ms || !c?.ttft_ms) return null;
    return {
      speedup: s.ttft_ratio ?? (b.ttft_ms / c.ttft_ms).toFixed(2),
      ttftSaved: Math.round(s.ttft_saved_ms ?? (b.ttft_ms - c.ttft_ms)),
      tokensSaved: s.prompt_tokens_saved ?? ((b.prompt_tokens || 0) - (c.prompt_tokens || 0)),
    };
  }, [currentResult]);

  const aggregate = useMemo(() => {
    const rows = (timeline || []).filter(r => r.baseline_ttft_ms && r.centroid_ttft_ms);
    if (rows.length === 0) return null;
    const meanB = rows.reduce((s, r) => s + r.baseline_ttft_ms, 0) / rows.length;
    const meanC = rows.reduce((s, r) => s + r.centroid_ttft_ms, 0) / rows.length;
    const tokensSaved = rows.reduce((s, r) => s + ((r.baseline_prompt_tokens || 0) - (r.centroid_prompt_tokens || 0)), 0);
    return {
      n: rows.length,
      meanBaseline: Math.round(meanB),
      meanCentroid: Math.round(meanC),
      meanSpeedup: meanC > 0 ? (meanB / meanC).toFixed(2) : "—",
      tokensSaved,
    };
  }, [timeline]);

  /* ----- render ----- */
  return (
    <div className="app">
      <Header
        serverOk={serverOk}
        baseState={baseState}
        centState={centState}
        modelLoaded={modelLoaded}
        centroidArmed={centroidArmed}
        mockMode={mockMode}
      />

      <MetaBar
        config={config}
        conversationKey={conversationKey}
        totalTurns={totalTurns}
      />

      <ControlsPanel
        config={config}
        conversations={conversations}
        conversationKey={conversationKey}
        maxTokens={maxTokens}
        onMaxTokensChange={setMaxTokens}
        pauseSeconds={pauseSeconds}
        onPauseSecondsChange={setPauseSeconds}
        showEventLog={showEventLog}
        onShowEventLog={setShowEventLog}
        running={running}
        busy={busy}
        turnHasRun={turnHasRun}
        prepared={prepared}
        onConversationChange={handleConversationChange}
        onPrepare={handlePrepare}
        onRunCurrentTurn={handleRunCurrentTurn}
        onAdvance={handleAdvance}
        onRunFull={handleRunFullSequence}
        onReset={handleReset}
        onStop={handleStop}
      />

      <PhaseBanner
        phase={phase}
        running={running}
        busy={busy}
        turnNumber={turnNumber}
        totalTurns={totalTurns}
        turnHasRun={turnHasRun}
        prepared={prepared}
        timeline={timeline}
        pauseRemainingMs={pauseRemainingMs}
      />

      <PromptPanel
        conversationKey={conversationKey}
        turnNumber={turnNumber}
        totalTurns={totalTurns}
        prev={currentPayload?.previous}
        current={currentPayload?.user}
        next={currentPayload?.next}
      />

      <SummaryCard summary={summary} aggregate={aggregate} turnNumber={turnNumber} />

      <ComparisonSection
        result={currentResult}
        busy={turnInFlight}
        turnHasRun={turnHasRun}
      />

      <TimelineTable
        conversations={conversations}
        conversationKey={conversationKey}
        totalTurns={totalTurns}
        currentTurn={turnNumber}
        timeline={timeline}
        running={running}
      />

      <div className="bottom-grid">
        <BaselinesTable />
        {showEventLog ? (
          <TerminalPanel logLines={logLines} terminalRef={terminalRef} />
        ) : (
          <Panel title="Event log" right={<span style={{ color: "var(--fg-4)", fontSize: 11, fontFamily: "var(--mono)" }}>hidden</span>}>
            <div style={{ padding: "20px", color: "var(--fg-4)", fontFamily: "var(--mono)", fontSize: 11, textAlign: "center" }}>
              Toggle "Show event log" in controls to view streamed events.
            </div>
          </Panel>
        )}
      </div>

      <Footer mockMode={mockMode} error={error} />
    </div>
  );
}

/* ====================================================================
   SUB-COMPONENTS
   ==================================================================== */

function Header({ serverOk, baseState, centState, modelLoaded, centroidArmed, mockMode }) {
  return (
    <header className="header">
      <div className="brand">
        <div className="brand-mark">AC</div>
        <div>
          <h1>AgentCache Live Demo</h1>
          <div className="sub">Local vLLM-Metal inference · per-turn parallel comparison{mockMode ? " · MOCK MODE" : ""}</div>
        </div>
      </div>
      <div className="header-right">
        <Pill tone={serverOk ? "ok" : "err"} label="server" value={serverOk ? "Online" : "Offline"} />
        <Pill tone={baseState.tone} label="baseline vllm" value={baseState.label} />
        <Pill tone={centState.tone} label="centroid vllm" value={centState.label} />
        <Pill tone={modelLoaded ? "ok" : "idle"} label="model in mem" value={modelLoaded ? "Yes" : "No"} />
        <Pill tone={centroidArmed ? "ok" : "idle"} label="centroid armed" value={centroidArmed ? "Yes" : "No"} />
      </div>
    </header>
  );
}

function MetaCell({ k, v, mono, muted }) {
  return (
    <div className="meta-cell">
      <div className="k">{k}</div>
      <div className={`v ${mono ? "mono" : ""} ${muted ? "muted" : ""}`}>{v}</div>
    </div>
  );
}

function MetaBar({ config, conversationKey, totalTurns }) {
  const model = config?.model || "—";
  const n = config?.n != null ? `N = ${config.n}` : "—";
  const mode = config?.execution_mode || "—";
  const warmup = config?.measurement_warmup ? "on" : "off";
  return (
    <div className="meta-bar">
      <MetaCell k="hardware" v="M3 Max · 96 GB" />
      <MetaCell k="backend" v="vLLM-Metal" mono />
      <MetaCell k="model" v={model} mono />
      <MetaCell k="system prompt" v="Python agent · 2000 tokens" />
      <MetaCell k="centroid" v={n} mono />
      <MetaCell k="conversation" v={`${conversationKey} · ${totalTurns} turns`} mono />
      <MetaCell k="warmup" v={warmup} mono />
      <MetaCell k="run mode" v={mode} mono />
    </div>
  );
}

function ControlsPanel(props) {
  const {
    conversations, conversationKey, maxTokens, onMaxTokensChange,
    pauseSeconds, onPauseSecondsChange,
    showEventLog, onShowEventLog,
    running, busy, turnHasRun, prepared,
    onConversationChange, onPrepare, onRunCurrentTurn, onAdvance, onRunFull, onReset, onStop,
  } = props;
  const convKeys = Object.keys(conversations || {});
  const disabledAll = busy || running;
  return (
    <section className="panel">
      <div className="controls">
        <Btn kind="primary" onClick={onRunFull} disabled={disabledAll}>▶ Run Full Sequence</Btn>
        <span className="ctl-divider"></span>
        <Btn onClick={onPrepare} disabled={disabledAll} title="POST /api/prefix/prepare — load workers + run one-token warmup">
          Prepare Workers
        </Btn>
        <Btn onClick={onRunCurrentTurn} disabled={disabledAll} title="POST /api/turn/run for the current turn">
          Run Current Turn
        </Btn>
        <Btn onClick={onAdvance} disabled={disabledAll || !turnHasRun} title="POST /api/turn/advance">
          Continue to Next Turn
        </Btn>
        <span className="ctl-divider"></span>
        <Btn onClick={onReset} disabled={disabledAll}>Reset</Btn>
        <Btn kind="danger" onClick={onStop} disabled={!running && !busy}>■ Stop</Btn>

        <span className="ctl-divider"></span>

        <span className="ctl-group">
          <label>conv</label>
          <select
            value={conversationKey}
            onChange={e => onConversationChange(e.target.value)}
            disabled={disabledAll || convKeys.length === 0}
          >
            {convKeys.length === 0 ? <option value={conversationKey}>{conversationKey}</option> : null}
            {convKeys.map(k => <option key={k} value={k}>{k}</option>)}
          </select>
        </span>

        <span className="ctl-group">
          <label>max_tokens</label>
          <input
            type="number" value={maxTokens} min={16} max={4096} step={16}
            onChange={e => onMaxTokensChange(parseInt(e.target.value || 0))}
            disabled={disabledAll}
          />
        </span>

        <span className="ctl-group" title="Seconds to pause between turns during a full-sequence run, so the result stays on screen long enough to read.">
          <label>pause (s)</label>
          <input
            type="number" value={pauseSeconds} min={0} max={60} step={1}
            onChange={e => onPauseSecondsChange(Math.max(0, parseInt(e.target.value || 0)))}
            disabled={running}
          />
        </span>

        <span style={{ flex: 1 }}></span>

        <Toggle on={showEventLog} onChange={onShowEventLog}>Show event log</Toggle>

        <span className={`pill ${prepared ? "ok" : "idle"}`} style={{ marginLeft: 4 }}>
          <span className="dot"></span>
          <span className="label">prepared</span>
          <span className="val">{prepared ? "Yes" : "No"}</span>
        </span>
      </div>
    </section>
  );
}

function PhaseBanner({ phase, running, busy, turnNumber, totalTurns, turnHasRun, prepared, timeline, pauseRemainingMs }) {
  let phaseClass = "";
  let phaseNum = "·";
  let phaseLabel = "ready";
  let statusText = null;
  const pausing = pauseRemainingMs != null && pauseRemainingMs > 0;

  if (pausing) {
    phaseClass = "centroid";
    phaseNum = String(turnNumber || 1);
    phaseLabel = `turn ${turnNumber}/${totalTurns}`;
    statusText = <><span className="phase-side">Showing turn {turnNumber} result</span> <span className="sub">advancing to next turn in {Math.ceil(pauseRemainingMs / 1000)}s… (press Stop to halt)</span></>;
  } else if (phase === "preparing") {
    phaseClass = "centroid";
    phaseNum = "·";
    phaseLabel = "prepare";
    statusText = <><span className="phase-side">Preparing workers</span> <span className="sub">warming both vLLM workers — first-load can take ~30 s</span></>;
  } else if (phase === "running") {
    phaseClass = "centroid";
    phaseNum = String(turnNumber || 1);
    phaseLabel = `turn ${turnNumber}/${totalTurns}`;
    statusText = <><span className="phase-side">Running turn {turnNumber}</span> <span className="sub">baseline + centroid in parallel · awaiting response</span></>;
  } else if (phase === "done") {
    phaseClass = "centroid";
    phaseNum = "✓";
    phaseLabel = "complete";
    if (turnHasRun) {
      statusText = <>Turn {turnNumber} complete <span className="sub">results below · advance to continue</span></>;
    } else {
      statusText = <>Workers ready <span className="sub">prepared · run current turn or full sequence</span></>;
    }
  } else {
    phaseClass = "";
    phaseNum = "·";
    phaseLabel = "ready";
    statusText = <>Idle <span className="sub">click ▶ Run Full Sequence or Run Current Turn to begin</span></>;
  }

  const ticks = Array.from({ length: totalTurns || 0 }, (_, i) => {
    const turn = i + 1;
    const row = (timeline || []).find(t => t.turn === turn);
    const done = !!row;
    const isCurrent = (phase === "running" || phase === "preparing") && turn === turnNumber;
    return { done, running: isCurrent && phase === "running" };
  });

  const completed = (timeline || []).length;
  const stateLabel =
    phase === "preparing" ? `preparing workers` :
    phase === "running"   ? `running turn ${turnNumber}/${totalTurns}` :
    phase === "done" && completed >= (totalTurns || 0) && totalTurns > 0 ? `✓ ${completed} turns swept` :
    phase === "done"      ? `${completed}/${totalTurns || "?"} turns complete` :
    "queued";

  return (
    <section className={`phase-banner ${phaseClass}`}>
      <div className="phase-label">
        <span className="phase-number">{phaseNum}</span>
        <span>{phaseLabel}</span>
      </div>
      <div className="phase-status">{statusText}</div>
      <div className="phase-progress">
        {ticks.map((t, i) => (
          <span key={i} className={`turn-tick ${t.done ? "done centroid" : ""} ${t.running ? "running" : ""}`}></span>
        ))}
      </div>

      <div className="phase-strip" style={{ gridColumn: "1 / -1" }}>
        <div className={`phase-block centroid ${phase === "preparing" || phase === "running" ? "active" : (completed > 0 ? "done" : "queued")}`}>
          <span className="swatch"></span>
          <span className="nm">Baseline + AgentCache · parallel workers</span>
          <span className="stage">{stateLabel}</span>
        </div>
        <div className={`phase-block centroid ${prepared ? "done" : "queued"}`}>
          <span className="swatch"></span>
          <span className="nm">Prefix preparation</span>
          <span className="stage">{prepared ? "✓ prepared · system prefix warmed" : "not prepared yet"}</span>
        </div>
      </div>
    </section>
  );
}

function PromptPanel({ conversationKey, turnNumber, totalTurns, prev, current, next }) {
  return (
    <Panel
      title={`Active prompt · ${conversationKey}`}
      right={<span style={{ color: "var(--fg-4)", fontSize: 11, fontFamily: "var(--mono)" }}>turn {turnNumber || 0} / {totalTurns || 0}</span>}
    >
      <div className="prompt-grid">
        <div className="prompt-main">
          {prev ? (
            <div className="prev-prompt">
              <span className="tag">prev · turn {Math.max(1, (turnNumber || 1) - 1)}</span>
              <span>{prev}</span>
            </div>
          ) : (
            <div className="prev-prompt">
              <span className="tag">prev</span>
              <span style={{ color: "var(--fg-4)" }}>— start of conversation —</span>
            </div>
          )}

          <div className="current-prompt">
            <span className="tag">current · turn {turnNumber || 0}</span>
            {current || "—"}
          </div>

          {next ? (
            <div className="next-prompt">
              <span className="tag">next · turn {(turnNumber || 0) + 1}</span>
              <span>{next}</span>
            </div>
          ) : (
            <div className="next-prompt">
              <span className="tag">next</span>
              <span style={{ color: "var(--fg-4)" }}>— end of conversation —</span>
            </div>
          )}
        </div>

        <div className="prompt-aside">
          <div className="badge-card baseline">
            <div className="label">Baseline</div>
            <div className="name">
              <span style={{ width: 6, height: 6, borderRadius: 99, background: "var(--baseline)", display: "inline-block" }}></span>
              Full prompt
            </div>
            <div className="desc">Full 2000-token system prompt re-sent every turn.</div>
          </div>
          <div className="badge-card centroid">
            <div className="label">AgentCache</div>
            <div className="name">
              <span style={{ width: 6, height: 6, borderRadius: 99, background: "var(--centroid)", display: "inline-block" }}></span>
              Centroid N=128
            </div>
            <div className="desc">System prompt omitted; trained KV centroid injected as prefix.</div>
          </div>
        </div>
      </div>
    </Panel>
  );
}

function SummaryCard({ summary, aggregate, turnNumber }) {
  return (
    <div className="summary">
      <div className="summary-cell">
        <div className="k">TTFT speedup · turn {turnNumber || 0}</div>
        <div className={`v ${summary ? "" : "pending"}`}>
          {summary ? <>{summary.speedup}<span className="unit">×</span></> : null}
        </div>
        <div className="delta">
          {summary ? <><span className="ok">centroid faster</span> · time-to-first-token</> : "awaiting current turn result"}
        </div>
      </div>
      <div className="summary-cell">
        <div className="k">TTFT saved · this turn</div>
        <div className={`v ${summary ? "" : "pending"}`}>
          {summary ? <>{summary.ttftSaved}<span className="unit">ms</span></> : null}
        </div>
        <div className="delta">{summary ? <>tokens saved: <span className="ok">{fmtInt(summary.tokensSaved)}</span></> : "—"}</div>
      </div>
      <div className="summary-cell">
        <div className="k">Mean speedup · {aggregate ? aggregate.n : 0} turns</div>
        <div className={`v ${aggregate ? "" : "pending"}`}>
          {aggregate ? <>{aggregate.meanSpeedup}<span className="unit">×</span></> : null}
        </div>
        <div className="delta">
          {aggregate ? <>tokens saved across run: <span className="ok">{fmtInt(aggregate.tokensSaved)}</span></> : "computed across completed turns"}
        </div>
      </div>
    </div>
  );
}

function ComparisonSection({ result, busy, turnHasRun }) {
  const b = result?.baseline;
  const c = result?.centroid;
  const baselineDone = !!(b && b.total_ms != null);
  const centroidDone = !!(c && c.total_ms != null);

  function statusFor(done, running) {
    if (running) return "running";
    if (done) return "ready";
    return "pending";
  }

  function panelStatus(side) {
    if (busy) return "running";
    if (side === "baseline" && baselineDone) return "ready";
    if (side === "centroid" && centroidDone) return "ready";
    return "pending";
  }

  function outputStatus(side) {
    if (busy) return "running";
    if (side === "baseline" && baselineDone) return "done";
    if (side === "centroid" && centroidDone) return "done";
    return "pending";
  }

  return (
    <div className="comparison">
      {/* ---- BASELINE ---- */}
      <section className={`cmp-col baseline panel`}>
        <header className="panel-h">
          <div className="title-with-dot">
            <span className="stripe"></span>
            <span className="title">BASELINE · vLLM (full system prompt)</span>
          </div>
          <div className="right"><StatusCell status={statusFor(baselineDone, busy)} /></div>
        </header>
        <div className="panel-body">
          <div className="metrics-grid">
            <Metric k="TTFT" v={b?.ttft_ms != null ? fmtMs(b.ttft_ms) : null} unit={b?.ttft_ms != null ? unitFor(b.ttft_ms) : null} tone="baseline" pending={!b?.ttft_ms} />
            <Metric k="Total latency" v={b?.total_ms != null ? fmtMs(b.total_ms) : null} unit={b?.total_ms != null ? unitFor(b.total_ms) : null} tone="baseline" pending={!b?.total_ms} />
            <Metric k="Prompt tokens" v={b?.prompt_tokens != null ? fmtInt(b.prompt_tokens) : null} tone="baseline" pending={!b?.prompt_tokens} />
          </div>
          <KV k="System prompt sent" v={b?.system_prompt_sent ? "Yes" : (b ? "No" : "—")} tone={b?.system_prompt_sent ? "yes" : "no"} />
          <KV k="Prefix injected" v={b?.prefix_injected ? "Yes" : (b ? "No" : "—")} tone={b?.prefix_injected ? "yes" : "no"} />
          <KV k="History turns" v={b?.history_turns != null ? fmtInt(b.history_turns) : "—"} />
          <KV k="Warmup TTFT" v={b?.warmup_ms != null ? `${fmtMs(b.warmup_ms)} ${unitFor(b.warmup_ms)}` : "—"} />
          <OutputBox
            tone="baseline"
            status={outputStatus("baseline")}
            text={b?.output || ""}
            generatedTokens={null}
          />
        </div>
      </section>

      {/* ---- CENTROID ---- */}
      <section className={`cmp-col centroid panel`}>
        <header className="panel-h">
          <div className="title-with-dot">
            <span className="stripe"></span>
            <span className="title">AGENTCACHE · Centroid N=128</span>
          </div>
          <div className="right"><StatusCell status={statusFor(centroidDone, busy)} /></div>
        </header>
        <div className="panel-body">
          <div className="metrics-grid">
            <Metric k="TTFT" v={c?.ttft_ms != null ? fmtMs(c.ttft_ms) : null} unit={c?.ttft_ms != null ? unitFor(c.ttft_ms) : null} tone="centroid" pending={!c?.ttft_ms} />
            <Metric k="Total latency" v={c?.total_ms != null ? fmtMs(c.total_ms) : null} unit={c?.total_ms != null ? unitFor(c.total_ms) : null} tone="centroid" pending={!c?.total_ms} />
            <Metric k="Prompt tokens" v={c?.prompt_tokens != null ? fmtInt(c.prompt_tokens) : null} tone="centroid" pending={!c?.prompt_tokens} />
          </div>
          <KV k="System prompt sent" v={c?.system_prompt_sent ? "Yes" : (c ? "No" : "—")} tone={c?.system_prompt_sent ? "yes" : "no"} />
          <KV k="Prefix injected" v={c?.prefix_injected ? `Yes · N=${c.synthetic_tokens || 128}` : (c ? "No" : "—")} tone={c?.prefix_injected ? "yes" : "no"} />
          <KV k="History turns" v={c?.history_turns != null ? fmtInt(c.history_turns) : "—"} />
          <KV k="Warmup TTFT" v={c?.warmup_ms != null ? `${fmtMs(c.warmup_ms)} ${unitFor(c.warmup_ms)}` : "—"} />
          <OutputBox
            tone="centroid"
            status={outputStatus("centroid")}
            text={c?.output || ""}
            generatedTokens={null}
          />
        </div>
      </section>
    </div>
  );
}

function TimelineTable({ conversations, conversationKey, totalTurns, currentTurn, timeline, running }) {
  const prompts = conversations?.[conversationKey]?.prompts || [];
  const byTurn = {};
  (timeline || []).forEach(row => { byTurn[row.turn] = row; });
  const rows = [];
  for (let i = 1; i <= (totalTurns || 0); i++) {
    rows.push({ turn: i, summary: summarize(prompts[i - 1] || "", 56), data: byTurn[i] || null });
  }
  const completed = (timeline || []).length;
  return (
    <Panel
      title="Per-turn timeline"
      right={
        <span style={{ display: "flex", gap: 14, fontSize: 11, fontFamily: "var(--mono)" }}>
          <span style={{ color: "var(--centroid)" }}>completed {completed}/{totalTurns || 0}</span>
        </span>
      }
      bodyClass="tight"
    >
      <table className="tbl">
        <thead>
          <tr>
            <th>Turn</th>
            <th>Prompt summary</th>
            <th className="right">Baseline TTFT</th>
            <th className="right">AgentCache TTFT</th>
            <th className="right">Speedup</th>
            <th className="right">Baseline tok</th>
            <th className="right">AgentCache tok</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ turn, summary, data }) => {
            const isCurrent = turn === currentTurn;
            const speedup = data?.ttft_ratio != null ? `${data.ttft_ratio}×` : "—";
            const status = data ? "ready" : (isCurrent && running ? "running" : "pending");
            const isPending = !data;
            return (
              <tr key={turn} className={`${isCurrent ? "current" : ""} ${isPending ? "pending" : ""}`}>
                <td className="turn">{String(turn).padStart(2, "0")}</td>
                <td className="summary">{summary || "—"}</td>
                <td className="right baseline">{data?.baseline_ttft_ms != null ? `${data.baseline_ttft_ms.toFixed ? data.baseline_ttft_ms.toFixed(0) : data.baseline_ttft_ms} ms` : "—"}</td>
                <td className="right centroid">{data?.centroid_ttft_ms != null ? `${data.centroid_ttft_ms.toFixed ? data.centroid_ttft_ms.toFixed(0) : data.centroid_ttft_ms} ms` : "—"}</td>
                <td className="right speedup">{speedup}</td>
                <td className="right">{data?.baseline_prompt_tokens != null ? fmtInt(data.baseline_prompt_tokens) : "—"}</td>
                <td className="right">{data?.centroid_prompt_tokens != null ? fmtInt(data.centroid_prompt_tokens) : "—"}</td>
                <td><StatusCell status={status} /></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Panel>
  );
}

function BaselinesTable() {
  const rows = [
    ["Cold vLLM",                          "yes",      "Supported"],
    ["vLLM prefix cache",                  "partial",  "Limited on Metal"],
    ["AgentCache centroid injection",      "yes",      "Supported"],
    ["AgentCache centroid + prefix cache", "optional", "Experimental"],
    ["LMCache",                            "no",       "Not supported on Metal"],
    ["LMCache + AgentCache",               "no",       "Not supported on Metal"],
  ];
  return (
    <Panel title="Supported baselines" right={<span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-4)" }}>this run</span>} bodyClass="tight">
      <table className="tbl baselines-tbl">
        <thead>
          <tr>
            <th>Method</th>
            <th>Status</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, supp, note]) => (
            <tr key={label}>
              <td className="label">{label}</td>
              <td className={`support ${supp}`}>
                {supp === "yes" && "● Supported"}
                {supp === "partial" && "◐ Partial"}
                {supp === "optional" && "◌ Experimental"}
                {supp === "no" && "○ Unsupported"}
              </td>
              <td className="muted">{note}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="note">
        <strong>Note:</strong> LMCache is a stronger KV-cache baseline on CUDA systems; this live demo focuses on what runs locally on Mac with vLLM-Metal. Both workers stay resident in parallel mode — the comparison is per-turn, not per-phase.
      </div>
    </Panel>
  );
}

function TerminalPanel({ logLines, terminalRef }) {
  return (
    <section className="panel">
      <header className="panel-h">
        <div className="title-with-dot">
          <span className="dot" style={{ background: "var(--ok)" }}></span>
          <span>Event log · 127.0.0.1:8765</span>
        </div>
        <div className="right" style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--fg-4)" }}>
          {logLines.length} events
        </div>
      </header>
      <div className="terminal" ref={terminalRef}>
        {logLines.length === 0 ? (
          <div className="terminal-line">
            <span className="ts">{nowTs()}</span>
            <span className="tag server">[server]</span>
            <span> waiting for first event…</span>
          </div>
        ) : null}
        {logLines.map((line, i) => (
          <div key={i} className="terminal-line">
            <span className="ts">{line.ts}</span>
            <span className={`tag ${line.tag}`}>[{line.tag}]</span>
            <span> {line.text}</span>
          </div>
        ))}
        <span className="terminal-cursor"></span>
      </div>
    </section>
  );
}

function Footer({ mockMode, error }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "10px 4px 4px", color: "var(--fg-4)", fontFamily: "var(--mono)",
      fontSize: 10.5, borderTop: "1px solid var(--border-soft)", marginTop: 4, gap: 12,
    }}>
      <span>agentcache · live demo · v2 · per-turn parallel mode</span>
      <span>
        {error ? <span style={{ color: "var(--err)" }}>{error}</span> : null}
        {error ? " · " : null}
        mode: <span style={{ color: mockMode ? "var(--warn)" : "var(--ok)" }}>
          {mockMode ? "MOCK (synthetic)" : "REAL (live inference)"}
        </span>
        {" · backend: vllm-metal"}
      </span>
    </div>
  );
}

/* ============ MOUNT ============ */
ReactDOM.createRoot(document.getElementById("root")).render(<App />);
