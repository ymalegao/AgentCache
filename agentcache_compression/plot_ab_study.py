"""
AB study: What drives TTFT improvement — model size or centroid size?

Layout (2×3):
  [A] Turn-by-turn: Qwen-1B   [B] Turn-by-turn: Qwen-7B   [C] Turn-by-turn: GPT-20B
  [D] Speedup vs cold          [E] N ablation               [F] Turn-1 vs mid-conv speedup
"""

import json
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

FILES = {
    "Qwen-1B": "agentcache_compression/results/nogb_csv_cli_2048withcachehitbenchmark.jsonl",
    "Qwen-7B": "agentcache_compression/results/qwen7b.jsonl",
    "GPT-20B": "agentcache_compression/results/gptmulti_turn_benchmark.jsonl",
}
MODELS = list(FILES.keys())

C = {
    "cold":     "#e15759",
    "warm_apc": "#f28e2b",
    "synth":    "#4e79a7",
    "n64":      "#76b7b2",
    "n128":     "#4e79a7",
    "n256":     "#1a4f72",
}


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def group_by(records, *keys):
    d = defaultdict(list)
    for r in records:
        k = tuple(r.get(k) for k in keys)
        d[k].append(r["ttft_s"])
    return d


all_records = {m: load(p) for m, p in FILES.items()}

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
axes_bot = [fig.add_subplot(gs[1, i]) for i in range(3)]

fig.suptitle(
    "AgentCache AB Study: What Drives TTFT? Model Size vs Centroid Size vs Warm APC",
    fontsize=13, fontweight="bold", y=1.01,
)

# ── Panels A–C: Turn-by-turn for each model ───────────────────────────────────
# Shared y-axis limit so panels are visually comparable
y_max_global = 1.15

for ax, model in zip(axes_top, MODELS):
    records = all_records[model]
    by_mode_turn = group_by(records, "mode", "turn")

    # Aggregate synthetic across all N values
    synth_by_turn = defaultdict(list)
    for r in records:
        if r["mode"] == "synthetic":
            synth_by_turn[r["turn"]].append(r["ttft_s"])

    for mode_key, color, label in [
        ("cold",     C["cold"],     "Cold"),
        ("warm_apc", C["warm_apc"], "Warm APC"),
    ]:
        turn_means = {}
        for t in range(1, 11):
            vals = by_mode_turn.get((mode_key, t), [])
            if vals:
                turn_means[t] = statistics.mean(vals)
        if turn_means:
            ts = sorted(turn_means)
            ax.plot(ts, [turn_means[t] for t in ts], "-o", color=color,
                    label=label, linewidth=1.8, markersize=5)

    # Synth (all N pooled)
    synth_turns = sorted(synth_by_turn)
    synth_vals  = [statistics.mean(synth_by_turn[t]) for t in synth_turns]
    ax.plot(synth_turns, synth_vals, "-^", color=C["synth"],
            label="Synth (all N)", linewidth=1.8, markersize=5)

    ax.set_title(model, fontweight="bold")
    ax.set_xlabel("Conversation Turn")
    ax.set_ylabel("TTFT (s)" if model == "Qwen-1B" else "")
    ax.set_ylim(0, y_max_global)
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.25)
    if model == "Qwen-1B":
        ax.legend(fontsize=8)

# shared label
axes_top[1].set_title("Qwen-7B", fontweight="bold")
fig.text(0.5, 0.985, "Turn-by-turn TTFT — finding: model size determines how much synthetic helps",
         ha="center", fontsize=9, style="italic", color="#444")

# ── Panel D: Speedup over cold by model & config ─────────────────────────────
ax_d = axes_bot[0]

bar_w = 0.25
x = np.arange(len(MODELS))

for i, (config, color, label) in enumerate([
    ("warm_apc",  C["warm_apc"], "Warm APC"),
    ("synthetic", C["synth"],    "Synth (all N)"),
]):
    speedups = []
    for model in MODELS:
        records = all_records[model]
        cold_vals  = [r["ttft_s"] for r in records if r["mode"] == "cold"]
        config_vals = [r["ttft_s"] for r in records if r["mode"] == config]
        if cold_vals and config_vals:
            speedups.append(statistics.mean(cold_vals) / statistics.mean(config_vals))
        else:
            speedups.append(float("nan"))
    bars = ax_d.bar(x + (i - 0.5) * bar_w, speedups, bar_w,
                    color=color, label=label, zorder=3)
    for bar, val in zip(bars, speedups):
        if not np.isnan(val):
            ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                      f"{val:.2f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax_d.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5, zorder=2)
ax_d.set_xticks(x)
ax_d.set_xticklabels(MODELS)
ax_d.set_ylabel("Speedup over Cold (higher = better)")
ax_d.set_title("D: Speedup vs Cold\n(model size drives gains)", fontweight="bold")
ax_d.legend(fontsize=9)
ax_d.set_ylim(0.5, 2.2)
ax_d.grid(axis="y", alpha=0.25, zorder=1)
ax_d.text(0.5, -0.22, "Finding: 1B gains <1x, 7B +49%, 20B +80%",
          transform=ax_d.transAxes, ha="center", fontsize=8, style="italic", color="#555")

# ── Panel E: N ablation across models ─────────────────────────────────────────
ax_e = axes_bot[1]

n_vals = [64, 128, 256]
n_colors = [C["n64"], C["n128"], C["n256"]]
bar_w_e = 0.22
x_e = np.arange(len(MODELS))

for i, (n, color) in enumerate(zip(n_vals, n_colors)):
    means, errs = [], []
    for model in MODELS:
        vals = [r["ttft_s"] for r in all_records[model]
                if r["mode"] == "synthetic" and r.get("N") == n]
        if vals:
            means.append(statistics.mean(vals))
            errs.append(statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0)
        else:
            means.append(float("nan"))
            errs.append(0)
    ax_e.bar(x_e + (i - 1) * bar_w_e, means, bar_w_e,
             yerr=errs, capsize=3, color=color, label=f"N={n}", zorder=3,
             error_kw={"elinewidth": 1})

ax_e.set_xticks(x_e)
ax_e.set_xticklabels(MODELS)
ax_e.set_ylabel("Mean TTFT (s)")
ax_e.set_title("E: Centroid Size (N) Ablation\n(N barely matters)", fontweight="bold")
ax_e.legend(fontsize=9)
ax_e.set_ylim(0)
ax_e.grid(axis="y", alpha=0.25, zorder=1)
ax_e.text(0.5, -0.22, "Finding: N=64 ≈ N=256 at 1B & 7B; slight gain at 20B",
          transform=ax_e.transAxes, ha="center", fontsize=8, style="italic", color="#555")

# ── Panel F: Turn-1 vs mid-conversation (turns 2–5) speedup ──────────────────
# This separates cold-start benefit from sustained benefit
ax_f = axes_bot[2]

bar_w_f = 0.22
x_f = np.arange(len(MODELS))

synth_t1_speedup, synth_mid_speedup = [], []
warm_t1_speedup, warm_mid_speedup = [], []

for model in MODELS:
    records = all_records[model]

    def _mean_ttft(mode_filter, turns):
        vals = [r["ttft_s"] for r in records
                if r["mode"] == mode_filter and r["turn"] in turns]
        return statistics.mean(vals) if vals else float("nan")

    cold_t1  = _mean_ttft("cold",     {1})
    cold_mid = _mean_ttft("cold",     set(range(2, 6)))

    s_t1  = _mean_ttft("synthetic", {1})
    s_mid = _mean_ttft("synthetic", set(range(2, 6)))

    w_t1  = _mean_ttft("warm_apc",  {1})
    w_mid = _mean_ttft("warm_apc",  set(range(2, 6)))

    synth_t1_speedup.append(cold_t1  / s_t1  if s_t1  else float("nan"))
    synth_mid_speedup.append(cold_mid / s_mid if s_mid else float("nan"))
    warm_t1_speedup.append(cold_t1  / w_t1  if w_t1  else float("nan"))
    warm_mid_speedup.append(cold_mid / w_mid if w_mid else float("nan"))

offsets_f = [-1.5, -0.5, 0.5, 1.5]
data_f = [
    (synth_t1_speedup,  C["synth"],    "Synth  — Turn 1"),
    (synth_mid_speedup, "#a8c8e8",     "Synth  — Turns 2–5"),
    (warm_t1_speedup,   C["warm_apc"], "Warm APC — Turn 1"),
    (warm_mid_speedup,  "#fcd6a1",     "Warm APC — Turns 2–5"),
]

for (vals, color, label), off in zip(data_f, offsets_f):
    bars = ax_f.bar(x_f + off * bar_w_f / 2, vals, bar_w_f / 2,
                    color=color, label=label, zorder=3)
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax_f.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                      f"{val:.2f}x", ha="center", va="bottom", fontsize=7)

ax_f.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5, zorder=2)
ax_f.set_xticks(x_f)
ax_f.set_xticklabels(MODELS)
ax_f.set_ylabel("Speedup over Cold")
ax_f.set_title("F: Cold-Start vs Sustained Speedup\n(where does benefit come from?)", fontweight="bold")
ax_f.legend(fontsize=7, loc="upper left")
ax_f.set_ylim(0, 5.5)
ax_f.grid(axis="y", alpha=0.25, zorder=1)
ax_f.text(0.5, -0.22, "Finding: synthetic helps most on turn 1; warm APC hurts turns 2–5",
          transform=ax_f.transAxes, ha="center", fontsize=8, style="italic", color="#555")

out_path = "agentcache_compression/results/ab_study.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
