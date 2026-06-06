#!/usr/bin/env python3
"""
Analyze multi-turn cache benchmark results.

Usage:
    python analyze_multi_turn.py <path_to_jsonl>
    python analyze_multi_turn.py results/multi_turn_benchmark.jsonl
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from rouge_score import rouge_scorer as _rouge
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mode_label(r: dict) -> str:
    if r["mode"] == "synthetic":
        return f"Synthetic N={r['N']}"
    elif r["mode"] == "warm_apc":
        return "Warm APC"
    return "Cold"


MODE_ORDER = ["Cold", "Warm APC", "Synthetic N=64", "Synthetic N=128", "Synthetic N=256"]

PLOT_COLORS = {
    "Cold": "#4c566a",
    "Warm APC": "#d08770",
    "Synthetic N=64": "#5e81ac",
    "Synthetic N=128": "#88c0d0",
    "Synthetic N=256": "#a3be8c",
}


def load(path: str) -> dict[str, dict[int, list[dict]]]:
    """Returns data[label][turn] = [records...]."""
    data: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            data[mode_label(r)][r["turn"]].append(r)
    return data


def mean(vals):
    return statistics.mean(vals) if vals else float("nan")


def print_section(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


def speedup(base: float, cand: float) -> str:
    if cand == 0:
        return "  inf"
    return f"{base / cand:5.2f}x"


def has_code(text: str) -> bool:
    return "```" in text or "def " in text or "import " in text


def default_plot_path(input_path: str) -> Path:
    src = Path(input_path)
    return src.with_name(f"{src.stem}_ttft_by_turn.png")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_per_turn_table(data, labels, turns, title, get_val, fmt="{:7.4f}"):
    print_section(title)
    col_w = 14
    header = f"  {'Turn':>4}" + "".join(f"  {lbl:>{col_w}}" for lbl in labels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t in turns:
        row = f"  {t:>4}"
        for lbl in labels:
            recs = data.get(lbl, {}).get(t, [])
            if recs:
                val = mean([get_val(r) for r in recs])
                row += f"  {fmt.format(val):>{col_w}}"
            else:
                row += f"  {'—':>{col_w}}"
        print(row)


def plot_ttft_by_turn(data, labels, turns, out_path: Path):
    if not _MATPLOTLIB_AVAILABLE:
        print("\n  (Skipping plot: install matplotlib to generate PNG output)")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    plotted = False
    for lbl in labels:
        xs = []
        ys_ms = []
        for t in turns:
            recs = data.get(lbl, {}).get(t, [])
            if not recs:
                continue
            xs.append(t)
            ys_ms.append(mean([r["ttft_s"] for r in recs]) * 1000)
        if not xs:
            continue
        plotted = True
        ax.plot(
            xs,
            ys_ms,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=PLOT_COLORS.get(lbl),
            label=lbl,
        )

    if not plotted:
        print("\n  (Skipping plot: no turn-level TTFT data found)")
        plt.close(fig)
        return

    ax.set_title("TTFT by Turn")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean TTFT (ms)")
    ax.set_xticks(turns)
    ax.legend(frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"\nPlot : {out_path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(path: str, plot_out: Path | None = None):
    data = load(path)
    labels = [l for l in MODE_ORDER if l in data]
    turns = sorted({t for lbl in labels for t in data[lbl]})

    total = sum(len(recs) for lbl in labels for recs in data[lbl].values())
    print(f"\nFile  : {path}")
    print(f"Records: {total}")
    for lbl in labels:
        n = sum(len(v) for v in data[lbl].values())
        convs = len({r["conversation_id"] for recs in data[lbl].values() for r in recs})
        print(f"  {lbl:<22}  {n:>4} records  ({convs} conversations × {len(turns)} turns)")

    # TTFT per turn
    print_per_turn_table(
        data, labels, turns,
        "TTFT (seconds) — mean across conversations",
        lambda r: r["ttft_s"],
        fmt="{:7.4f}",
    )

    # Speedup vs cold per turn
    if "Cold" in data:
        print_section("Speedup vs Cold — per turn")
        col_w = 14
        cmp_labels = [l for l in labels if l != "Cold"]
        header = f"  {'Turn':>4}" + "".join(f"  {lbl:>{col_w}}" for lbl in cmp_labels)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for t in turns:
            cold_recs = data["Cold"].get(t, [])
            if not cold_recs:
                continue
            cold_ttft = mean([r["ttft_s"] for r in cold_recs])
            row = f"  {t:>4}"
            for lbl in cmp_labels:
                recs = data.get(lbl, {}).get(t, [])
                if recs:
                    cand_ttft = mean([r["ttft_s"] for r in recs])
                    row += f"  {speedup(cold_ttft, cand_ttft):>{col_w}}"
                else:
                    row += f"  {'—':>{col_w}}"
            print(row)

    # KV cache hit rate per turn
    # Prefer kv_cache_hits/queries when non-zero.
    # Fall back to apc_cached_tokens/physical_prompt_tokens.
    def kv_hit_rate_for_records(recs: list[dict]) -> float:
        if not recs:
            return float("nan")
        prom_hits = sum(r.get("kv_cache_hits", 0) for r in recs)
        prom_queries = sum(r.get("kv_cache_queries", 0) for r in recs)
        if prom_queries > 0:
            return prom_hits / prom_queries * 100
        total_hits = sum(r["apc_cached_tokens"] for r in recs)
        total_queries = sum(r["physical_prompt_tokens"] for r in recs)
        return (total_hits / total_queries * 100) if total_queries else 0.0

    print_section("KV Cache Hit Rate (%) — hits / queries × 100  [Prometheus-style aggregate]")
    col_w = 14
    header = f"  {'Turn':>4}" + "".join(f"  {lbl:>{col_w}}" for lbl in labels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t in turns:
        row = f"  {t:>4}"
        for lbl in labels:
            recs = data.get(lbl, {}).get(t, [])
            val = kv_hit_rate_for_records(recs)
            row += f"  {'{:7.1f}%'.format(val) if recs else '—':>{col_w}}"
        print(row)

    # Aggregate KV cache hit rate
    print_section("Aggregate KV Cache Hit Rate (%) — across all turns and conversations")
    header = f"  {'Mode':<22} {'Hit Rate':>10} {'Total Hits':>12} {'Total Queries':>15}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for lbl in labels:
        all_recs = [r for recs in data[lbl].values() for r in recs]
        if not all_recs:
            continue
        prom_hits = sum(r.get("kv_cache_hits", 0) for r in all_recs)
        prom_queries = sum(r.get("kv_cache_queries", 0) for r in all_recs)
        if prom_queries > 0:
            total_hits, total_queries = prom_hits, prom_queries
        else:
            total_hits = sum(r["apc_cached_tokens"] for r in all_recs)
            total_queries = sum(r["physical_prompt_tokens"] for r in all_recs)
        rate = (total_hits / total_queries * 100) if total_queries else 0.0
        print(f"  {lbl:<22} {rate:>9.1f}%  {total_hits:>12.0f}  {total_queries:>15.0f}")

    # Physical prompt tokens per turn
    print_per_turn_table(
        data, labels, turns,
        "Physical Prompt Tokens — mean across conversations",
        lambda r: r["physical_prompt_tokens"],
        fmt="{:7.0f}",
    )

    # Centroid tokens saved per turn
    synth_labels = [l for l in labels if l.startswith("Synthetic")]
    if synth_labels:
        print_section("Centroid Tokens Saved (synthetic modes only — constant = N)")
        for lbl in synth_labels:
            all_recs = [r for recs in data[lbl].values() for r in recs]
            n_val = all_recs[0]["N"] if all_recs else "?"
            print(f"  {lbl:<22}  {n_val} tokens saved per turn (gap injection)")

    # Aggregate TTFT stats
    print_section("Aggregate TTFT — across all turns and conversations")
    header = f"  {'Mode':<22} {'Mean':>8} {'Median':>8} {'Stdev':>8} {'Min':>8} {'Max':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for lbl in labels:
        vals = [r["ttft_s"] for recs in data[lbl].values() for r in recs]
        print(
            f"  {lbl:<22} "
            f"{mean(vals):>8.4f} "
            f"{statistics.median(vals):>8.4f} "
            f"{(statistics.stdev(vals) if len(vals)>1 else 0):>8.4f} "
            f"{min(vals):>8.4f} "
            f"{max(vals):>8.4f}"
        )

    # Quality
    print_section("Response Quality (records with 'response' field)")
    has_response = {
        lbl: [r for recs in data[lbl].values() for r in recs if "response" in r]
        for lbl in labels
    }
    any_quality = any(has_response[lbl] for lbl in labels)
    if not any_quality:
        print("  No 'response' field found in records. Re-run benchmark to collect quality data.")
    else:
        header = f"  {'Mode':<22} {'N':>4} {'Avg words':>10} {'Has code %':>11} {'Truncated %':>12} {'GOODBYE %':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for lbl in labels:
            recs = has_response[lbl]
            if not recs:
                print(f"  {lbl:<22} {'—':>4}")
                continue
            word_counts = [len(r["response"].split()) for r in recs]
            code_pct = 100 * sum(has_code(r["response"]) for r in recs) / len(recs)
            trunc_pct = 100 * sum(
                not r["response"].rstrip().endswith((".", "```", "GOODBYE", "!", "?", '"', "'"))
                for r in recs
            ) / len(recs)
            # Use an "ends with" check (not substring contains). Otherwise code blocks,
            # docstrings, or quoted text can inflate the metric.
            goodbye_pct = 100 * sum(r["response"].strip().endswith("GOODBYE") for r in recs) / len(recs)
            print(
                f"  {lbl:<22} {len(recs):>4} {mean(word_counts):>10.0f} "
                f"{code_pct:>10.1f}% {trunc_pct:>11.1f}% {goodbye_pct:>9.1f}%"
            )

        # ROUGE-L vs cold baseline
        if "Cold" not in data or not has_response.get("Cold"):
            print("\n  (Skipping ROUGE-L: no Cold baseline responses found)")
        elif not _ROUGE_AVAILABLE:
            print("\n  (Skipping ROUGE-L: install rouge-score  →  pip install rouge-score)")
        else:
            scorer = _rouge.RougeScorer(["rougeL"], use_stemmer=False)
            # Build cold lookup: (conversation_id, turn) → response text
            cold_lookup: dict[tuple, str] = {}
            for t, recs in data["Cold"].items():
                for r in recs:
                    if "response" in r:
                        cold_lookup[(r["conversation_id"], r["turn"])] = r["response"]

            cmp_labels = [l for l in labels if l != "Cold" and has_response.get(l)]
            if cmp_labels:
                print_section("ROUGE-L vs Cold Baseline — per mode (higher = more similar to cold)")
                header = f"  {'Mode':<22} {'Mean RL':>10} {'Median RL':>11} {'Min RL':>8} {'Max RL':>8} {'N pairs':>8}"
                print(header)
                print("  " + "-" * (len(header) - 2))
                for lbl in cmp_labels:
                    scores = []
                    for r in has_response[lbl]:
                        cold_ref = cold_lookup.get((r["conversation_id"], r["turn"]))
                        if cold_ref:
                            s = scorer.score(cold_ref, r["response"])
                            scores.append(s["rougeL"].fmeasure)
                    if not scores:
                        print(f"  {lbl:<22} {'—':>10}")
                        continue
                    print(
                        f"  {lbl:<22} {mean(scores):>10.3f} "
                        f"{statistics.median(scores):>11.3f} "
                        f"{min(scores):>8.3f} "
                        f"{max(scores):>8.3f} "
                        f"{len(scores):>8}"
                    )

                # Per-turn breakdown for the synthetic modes
                print_section("ROUGE-L vs Cold — per turn (synthetic modes only)")
                synth_cmp = [l for l in cmp_labels if l.startswith("Synthetic")]
                if synth_cmp:
                    col_w = 16
                    header = f"  {'Turn':>4}" + "".join(f"  {lbl:>{col_w}}" for lbl in synth_cmp)
                    print(header)
                    print("  " + "-" * (len(header) - 2))
                    for t in turns:
                        row = f"  {t:>4}"
                        for lbl in synth_cmp:
                            recs = [r for r in data.get(lbl, {}).get(t, []) if "response" in r]
                            turn_scores = []
                            for r in recs:
                                cold_ref = cold_lookup.get((r["conversation_id"], r["turn"]))
                                if cold_ref:
                                    s = scorer.score(cold_ref, r["response"])
                                    turn_scores.append(s["rougeL"].fmeasure)
                            if turn_scores:
                                row += f"  {'{:.3f}'.format(mean(turn_scores)):>{col_w}}"
                            else:
                                row += f"  {'—':>{col_w}}"
                        print(row)

        # Sample: show one conversation from the first synthetic mode
        first_synth = next((l for l in labels if l.startswith("Synthetic")), None)
        if first_synth:
            print_section(f"Sample Conversation — {first_synth} conv_id=0")
            conv0 = sorted(
                [r for recs in data[first_synth].values() for r in recs if r["conversation_id"] == 0],
                key=lambda r: r["turn"],
            )
            for r in conv0:
                if "user" not in r or "response" not in r:
                    continue
                print(f"\n  [Turn {r['turn']}] {r['task_id']}  ttft={r['ttft_s']:.4f}s  "
                      f"phys_tokens={r['physical_prompt_tokens']}  apc_cached={r['apc_cached_tokens']}")
                print(f"  USER: {r['user'][:100]}")
                print(f"  RESP: {r['response'][:200].replace(chr(10), ' ')}")

    if plot_out is not None:
        plot_ttft_by_turn(data, labels, turns, plot_out)

    print()


def parse_args() -> argparse.Namespace:
    default_input = Path(__file__).parent / "results" / "multi_turn_benchmark.jsonl"
    p = argparse.ArgumentParser(description="Analyze multi-turn cache benchmark results.")
    p.add_argument("path", nargs="?", default=str(default_input))
    p.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Optional PNG output path for a TTFT-by-turn plot. Defaults next to the input JSONL.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Print tables only and skip PNG generation.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_out = None if args.no_plot else (args.plot_out or default_plot_path(args.path))
    analyze(args.path, plot_out=plot_out)
