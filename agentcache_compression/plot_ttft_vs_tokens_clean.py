"""
TTFT vs physical_prompt_tokens — connected trajectory per turn.

Each point = one turn. Lines connect T1 → T10 in order.
Cold shows the hockey-stick (spike at T1, then slow climb).
Synth shows a smooth upward slope from a lower baseline.
"""

import json
from collections import defaultdict
import statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FILES = {
    "Qwen-1B":  "agentcache_compression/results/nogb_csv_cli_2048withcachehitbenchmark.jsonl",
    "Qwen-7B":  "agentcache_compression/results/qwen7b.jsonl",
    "GPT-20B":  "agentcache_compression/results/gptmulti_turn_benchmark.jsonl",
}
MODELS = list(FILES.keys())

COLD_COLOR  = "#d62728"
SYNTH_COLOR = "#1f77b4"


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def trajectory(records, mode):
    """Return list of (avg_tokens, avg_ttft) for turns 1–10 in order."""
    by_turn = defaultdict(list)
    for r in records:
        if r["mode"] == mode:
            by_turn[r["turn"]].append((r["physical_prompt_tokens"], r["ttft_s"]))
    result = []
    for t in sorted(by_turn):
        pts = by_turn[t]
        result.append((
            t,
            statistics.mean(p[0] for p in pts),
            statistics.mean(p[1] for p in pts),
        ))
    return result  # [(turn, avg_tokens, avg_ttft), ...]


all_records = {m: load(p) for m, p in FILES.items()}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "TTFT vs Conversation Length (Physical Prompt Tokens)",
    fontsize=14, fontweight="bold", y=1.02,
)

for ax, model in zip(axes, MODELS):
    records = all_records[model]
    cold_traj  = trajectory(records, "cold")
    synth_traj = trajectory(records, "synthetic")

    cold_tok  = [p[1] for p in cold_traj]
    cold_ttft = [p[2] for p in cold_traj]
    cold_turn = [p[0] for p in cold_traj]

    synth_tok  = [p[1] for p in synth_traj]
    synth_ttft = [p[2] for p in synth_traj]
    synth_turn = [p[0] for p in synth_traj]

    # -- lines --
    ax.plot(cold_tok,  cold_ttft,  "-o", color=COLD_COLOR,
            linewidth=2, markersize=7, label="Cold", zorder=4)
    ax.plot(synth_tok, synth_ttft, "-o", color=SYNTH_COLOR,
            linewidth=2, markersize=7, label="Synth (avg N)", zorder=4)

    # -- shade area between lines at overlapping x range --
    # Interpolate onto shared x grid for shading
    x_min = max(min(cold_tok), min(synth_tok))
    x_max = min(max(cold_tok), max(synth_tok))
    if x_min < x_max:
        x_grid = np.linspace(x_min, x_max, 300)
        cold_interp  = np.interp(x_grid, cold_tok,  cold_ttft)
        synth_interp = np.interp(x_grid, synth_tok, synth_ttft)
        ax.fill_between(x_grid, synth_interp, cold_interp,
                        where=cold_interp > synth_interp,
                        alpha=0.12, color=SYNTH_COLOR, label="Synth saves")
        ax.fill_between(x_grid, cold_interp, synth_interp,
                        where=synth_interp > cold_interp,
                        alpha=0.12, color=COLD_COLOR, label="Cold saves")

    # -- annotate every turn with T-label --
    label_every = {1, 5, 10}
    for t, tok, ttft in zip(cold_turn, cold_tok, cold_ttft):
        ax.scatter(tok, ttft, color=COLD_COLOR, s=55, zorder=5)
        if t in label_every:
            ax.annotate(f"T{t}", (tok, ttft),
                        textcoords="offset points", xytext=(-14, 5),
                        fontsize=8, color=COLD_COLOR, fontweight="bold")

    for t, tok, ttft in zip(synth_turn, synth_tok, synth_ttft):
        ax.scatter(tok, ttft, color=SYNTH_COLOR, s=55, zorder=5)
        if t in label_every:
            va = "bottom" if t == 1 else "top"
            ax.annotate(f"T{t}", (tok, ttft),
                        textcoords="offset points", xytext=(4, 5 if va == "bottom" else -12),
                        fontsize=8, color=SYNTH_COLOR, fontweight="bold")

    ax.set_title(model, fontsize=13, fontweight="bold")
    ax.set_xlabel("Physical Prompt Tokens", fontsize=11)
    ax.set_ylabel("TTFT (s)" if model == "Qwen-1B" else "", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
    ))
    ax.grid(alpha=0.2, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    if model == "Qwen-1B":
        ax.legend(fontsize=9, loc="upper left",
                  handles=[
                      plt.Line2D([0], [0], color=COLD_COLOR,  linewidth=2, label="Cold"),
                      plt.Line2D([0], [0], color=SYNTH_COLOR, linewidth=2, label="Synth (avg N)"),
                  ])

# caption
fig.text(
    0.5, -0.04,
    "Cold tokens include full system prompt (~2.2k baseline). "
    "Synth tokens = N synthetic vectors + user turns only — "
    "x-offset between Cold T1 and Synth T1 represents the compressed system prompt.",
    ha="center", fontsize=9, color="#555", style="italic",
)

plt.tight_layout()
out = "agentcache_compression/results/ttft_vs_tokens_clean.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
