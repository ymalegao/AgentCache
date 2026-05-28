"""
TTFT vs physical_prompt_tokens — scatter + regression per mode per model.

Key insight: synthetic modes have fewer physical tokens because the system
prompt is replaced by synthetic KV vectors (not counted as prefill tokens).
The x-axis shift between cold and synthetic IS the compression story.
"""

import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

FILES = {
    "Qwen-1B": "agentcache_compression/results/nogb_csv_cli_2048withcachehitbenchmark.jsonl",
    "Qwen-7B": "agentcache_compression/results/qwen7b.jsonl",
    "GPT-20B": "agentcache_compression/results/gptmulti_turn_benchmark.jsonl",
}
MODELS = list(FILES.keys())

SERIES = [
    ("cold",      None,  "#e15759", "o",  "Cold"),
    ("warm_apc",  None,  "#f28e2b", "s",  "Warm APC"),
    ("synthetic", 64,    "#76b7b2", "^",  "Synth-64"),
    ("synthetic", 128,   "#4e79a7", "^",  "Synth-128"),
    ("synthetic", 256,   "#1a4f72", "^",  "Synth-256"),
]


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def get_points(records, mode, n):
    return [
        (r["physical_prompt_tokens"], r["ttft_s"])
        for r in records
        if r["mode"] == mode and (n is None or r.get("N") == n)
    ]


all_records = {m: load(p) for m, p in FILES.items()}

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig.suptitle(
    "TTFT vs Context Length (physical tokens)\n"
    "Synthetic modes prefill fewer tokens because the system prompt is encoded as KV — "
    "x-axis shift = compression benefit",
    fontsize=12, fontweight="bold",
)

for ax, model in zip(axes, MODELS):
    records = all_records[model]

    for mode, n, color, marker, label in SERIES:
        pts = get_points(records, mode, n)
        if not pts:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])

        ax.scatter(xs, ys, color=color, marker=marker, s=55, alpha=0.75,
                   label=label, zorder=3, edgecolors="white", linewidths=0.4)

        # regression line if enough points
        if len(pts) >= 3:
            slope, intercept, r, _, _ = stats.linregress(xs, ys)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color=color,
                    linewidth=1.4, linestyle="--", alpha=0.7, zorder=2)
            # annotate slope in µs/token
            mid_x = xs.mean()
            mid_y = slope * mid_x + intercept
            ax.annotate(
                f"{slope * 1e6:.1f} µs/tok",
                xy=(mid_x, mid_y),
                xytext=(6, 6), textcoords="offset points",
                fontsize=7, color=color, fontweight="bold",
            )

    ax.set_title(model, fontweight="bold")
    ax.set_xlabel("Physical Prompt Tokens")
    ax.set_ylabel("TTFT (s)" if model == "Qwen-1B" else "")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)

axes[0].legend(fontsize=8, loc="upper left")

# shared annotation
fig.text(
    0.5, -0.03,
    "Cold/Warm APC: physical tokens include full system prompt (~2200 tokens baseline)  |  "
    "Synthetic: physical tokens = user turns only (system prompt → synthetic KV, not prefilled)",
    ha="center", fontsize=8, style="italic", color="#555",
)

plt.tight_layout()
out = "agentcache_compression/results/ttft_vs_tokens.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
