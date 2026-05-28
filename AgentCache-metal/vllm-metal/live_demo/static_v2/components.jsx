/* AgentCache demo — reusable UI components (v2) */

const { useState, useEffect, useRef, useMemo, useCallback } = React;

/* ------- Status pill ------- */
function Pill({ tone, label, value, children }) {
  return (
    <span className={`pill ${tone || ""}`}>
      <span className="dot"></span>
      {label ? <span className="label">{label}</span> : null}
      {value ? <span className="val">{value}</span> : null}
      {children}
    </span>
  );
}

/* ------- Panel wrapper ------- */
function Panel({ title, right, dotTone, children, bodyClass }) {
  return (
    <section className="panel">
      {title || right ? (
        <header className="panel-h">
          <div className="title-with-dot">
            {dotTone ? <span className="dot" style={{ background: `var(--${dotTone})` }}></span> : null}
            <span>{title}</span>
          </div>
          {right ? <div className="right">{right}</div> : null}
        </header>
      ) : null}
      <div className={`panel-body ${bodyClass || ""}`}>{children}</div>
    </section>
  );
}

/* ------- Button ------- */
function Btn({ children, onClick, kind, disabled, title }) {
  return (
    <button className={`btn ${kind || ""}`} onClick={onClick} disabled={disabled} title={title}>
      {children}
    </button>
  );
}

/* ------- Toggle switch ------- */
function Toggle({ on, onChange, children }) {
  return (
    <span className={`toggle ${on ? "on" : ""}`} onClick={() => onChange(!on)}>
      <span className="sw"></span>
      <span>{children}</span>
    </span>
  );
}

/* ------- Metric card ------- */
function Metric({ k, v, unit, tone, pending }) {
  return (
    <div className={`metric ${pending ? "pending" : ""}`}>
      <div className="k">{k}</div>
      <div className={`v ${tone || ""}`}>
        {!pending && v != null ? (
          <>
            {v}
            {unit ? <span className="unit">{unit}</span> : null}
          </>
        ) : null}
      </div>
    </div>
  );
}

/* ------- Key/value row ------- */
function KV({ k, v, tone }) {
  return (
    <div className="kv-row">
      <span className="k">{k}</span>
      <span className={`v ${tone || ""}`}>{v}</span>
    </div>
  );
}

/* ------- Status cell for tables ------- */
function StatusCell({ status }) {
  const label = {
    ready: "Ready", running: "Running", pending: "Pending", error: "Error",
  }[status] || status;
  return (
    <span className={`status-cell ${status}`}>
      <span className="dot"></span>
      {label}
    </span>
  );
}

/* ------- Number formatting helpers ------- */
function fmtMs(ms) {
  if (ms == null) return null;
  if (ms < 1000) return ms.toFixed(0);
  return (ms / 1000).toFixed(2);
}
function unitFor(ms) { return ms < 1000 ? "ms" : "s"; }
function fmtInt(n) {
  if (n == null) return null;
  return n.toLocaleString("en-US");
}
function fmtSpeedup(b, c) {
  if (!b || !c) return null;
  return (b / c).toFixed(2) + "x";
}

/* ------- Output box for generation ------- */
function OutputBox({ tone, status, text, generatedTokens }) {
  if (status === "pending") {
    return <div className={`output-box ${tone || ""}`}><span className="empty">// no output yet — run the current turn to generate</span></div>;
  }
  if (status === "running") {
    return (
      <div className={`output-box ${tone || ""}`}>
        <div className="running-prefix">{tone === "baseline" ? "[baseline]" : "[centroid]"} generating…</div>
        {"\n"}{text}
        <span className="caret"></span>
      </div>
    );
  }
  return (
    <div className={`output-box ${tone || ""}`}>
      {text}
      {generatedTokens ? `\n\n— ${generatedTokens} tokens generated —` : ""}
    </div>
  );
}

Object.assign(window, {
  Pill, Panel, Btn, Toggle, Metric, KV, StatusCell,
  OutputBox,
  fmtMs, unitFor, fmtInt, fmtSpeedup,
});
