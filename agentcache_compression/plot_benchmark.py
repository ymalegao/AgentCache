"""
Plot TTFT benchmark results across Qwen-1B, Qwen-7B, and GPT-20B.
Produces a 3-panel figure saved to results/ttft_comparison.png.
"""

import json
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FILES = {
    "Qwen-1B": "agentcache_compression/results/nogb_csv_cli_2048withcachehitbenchmark.jsonl",
    "Qwen-7B": "agentcache_compression/results/qwen7b.jsonl",
    "GPT-20B": "agentcache_compression/results/gptmulti_turn_benchmark.jsonl",
}

MODE_LABEL = {
    "cold": "Cold",
    "warm_apc": "Warm APC",
    "synthetic": "Synth",
}

CONFIG_ORDER = ["Cold", "Warm APC", "Synth-64", "Synth-128", "Synth-256"]
CONFIG_COLORS = {
    "Cold":      "#e15759",
    "Warm APC":  "#f28e2b",
    "Synth-64":  "#4e79a7",
    "Synth-128": "#59a14f",
    "Synth-256": "#9c755f",
}


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def config_key(r):
    mode = r["mode"]
    n = r.get("N", 0)
    if mode == "synthetic":
        return f"Synth-{n}"
    return MODE_LABEL.get(mode, mode)


def mean_ttft_by_config(records):
    groups = defaultdict(list)
    for r in records:
        groups[config_key(r)].append(r["ttft_s"])
    return {k: statistics.mean(v) for k, v in groups.items()}


def stderr_ttft_by_config(records):
    groups = defaultdict(list)
    for r in records:
        groups[config_key(r)].append(r["ttft_s"])
    return {k: statistics.stdev(v) / len(v) ** 0.5 for k, v in groups.items() if len(v) > 1}


def mean_ttft_by_turn(records, mode_filter=None):
    """Return {turn: mean_ttft} averaged over all N values for a mode."""
    groups = defaultdict(list)
    for r in records:
        if mode_filter and r["mode"] != mode_filter:
            continue
        groups[r["turn"]].append(r["ttft_s"])
    return {t: statistics.mean(v) for t, v in sorted(groups.items())}


# ── Load data ──────────────────────────────────────────────────────────────────
all_records = {name: load_records(path) for name, path in FILES.items()}

# ── Panel 1: Grouped bar chart — mean TTFT per config per model ─────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("TTFT Benchmark: AgentCache KV Compression", fontsize=14, fontweight="bold", y=1.02)

# --- Panel 1: grouped bars ---
ax1 = axes[0]
models = list(FILES.keys())
x = np.arange(len(models))
n_configs = len(CONFIG_ORDER)
bar_width = 0.14
offsets = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * bar_width

means_per_model = {m: mean_ttft_by_config(all_records[m]) for m in models}
errs_per_model  = {m: stderr_ttft_by_config(all_records[m]) for m in models}

for i, cfg in enumerate(CONFIG_ORDER):
    vals = [means_per_model[m].get(cfg, float("nan")) for m in models]
    errs = [errs_per_model[m].get(cfg, 0) for m in models]
    ax1.bar(x + offsets[i], vals, bar_width,
            label=cfg, color=CONFIG_COLORS[cfg], yerr=errs,
            capsize=3, error_kw={"elinewidth": 1})

ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylabel("Mean TTFT (s)")
ax1.set_title("Mean TTFT by Config & Model")
ax1.legend(fontsize=8, loc="upper left")
ax1.set_ylim(bottom=0)
ax1.grid(axis="y", alpha=0.3)

# --- Panel 2: Turn-by-turn TTFT for GPT-20B ---
ax2 = axes[1]
gpt_records = all_records["GPT-20B"]

turn_lines = {
    "Cold":     mean_ttft_by_turn(gpt_records, "cold"),
    "Warm APC": mean_ttft_by_turn(gpt_records, "warm_apc"),
    "Synth":    mean_ttft_by_turn(gpt_records, "synthetic"),
}

line_colors = {"Cold": "#e15759", "Warm APC": "#f28e2b", "Synth": "#4e79a7"}
line_styles  = {"Cold": "-o",    "Warm APC": "-s",       "Synth": "-^"}

for label, turn_dict in turn_lines.items():
    turns = sorted(turn_dict.keys())
    vals  = [turn_dict[t] for t in turns]
    ax2.plot(turns, vals, line_styles[label], color=line_colors[label],
             label=label, linewidth=1.8, markersize=5)

ax2.set_xlabel("Conversation Turn")
ax2.set_ylabel("Mean TTFT (s)")
ax2.set_title("GPT-20B: TTFT per Turn")
ax2.legend(fontsize=9)
ax2.set_xticks(range(1, 11))
ax2.set_ylim(bottom=0)
ax2.grid(alpha=0.3)

# --- Panel 3: Turn-1 TTFT comparison (cold-start cost) ---
ax3 = axes[2]
turn1_data = {}
for model, records in all_records.items():
    t1 = [r["ttft_s"] for r in records if r["turn"] == 1]
    t1_by_mode = defaultdict(list)
    for r in records:
        if r["turn"] == 1:
            t1_by_mode[config_key(r)].append(r["ttft_s"])
    turn1_data[model] = {k: statistics.mean(v) for k, v in t1_by_mode.items()}

configs_t1 = [c for c in CONFIG_ORDER if any(c in turn1_data[m] for m in models)]
x3 = np.arange(len(models))
n3 = len(configs_t1)
offsets3 = np.linspace(-(n3 - 1) / 2, (n3 - 1) / 2, n3) * bar_width

for i, cfg in enumerate(configs_t1):
    vals = [turn1_data[m].get(cfg, float("nan")) for m in models]
    ax3.bar(x3 + offsets3[i], vals, bar_width,
            label=cfg, color=CONFIG_COLORS[cfg])

ax3.set_xticks(x3)
ax3.set_xticklabels(models)
ax3.set_ylabel("Turn-1 TTFT (s)")
ax3.set_title("Turn-1 TTFT (Cold-Start Cost)")
ax3.legend(fontsize=8, loc="upper left")
ax3.set_ylim(bottom=0)
ax3.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_path = "agentcache_compression/results/ttft_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
