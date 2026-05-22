#!/usr/bin/env python3
"""
Analyze multi-turn cache benchmark results.

Usage:
    python analyze_multi_turn.py <path_to_jsonl>
    python analyze_multi_turn.py results/multi_turn_benchmark.jsonl
"""

import json
import sys
import statistics
from collections import defaultdict
from pathlib import Path


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


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(path: str):
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

    # ── TTFT per turn ─────────────────────────────────────────────────────────
    print_per_turn_table(
        data, labels, turns,
        "TTFT (seconds) — mean across conversations",
        lambda r: r["ttft_s"],
        fmt="{:7.4f}",
    )

    # ── Speedup vs cold per turn ───────────────────────────────────────────────
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

    # ── Cache hit rate per turn ────────────────────────────────────────────────
    print_per_turn_table(
        data, labels, turns,
        "APC Cache Hit Rate — apc_cached_tokens / physical_prompt_tokens",
        lambda r: r["apc_cached_tokens"] / r["physical_prompt_tokens"] if r["physical_prompt_tokens"] else 0.0,
        fmt="{:7.3f}",
    )

    # ── Physical prompt tokens per turn ───────────────────────────────────────
    print_per_turn_table(
        data, labels, turns,
        "Physical Prompt Tokens — mean across conversations",
        lambda r: r["physical_prompt_tokens"],
        fmt="{:7.0f}",
    )

    # ── Centroid savings (synthetic modes only) ────────────────────────────────
    synth_labels = [l for l in labels if l.startswith("Synthetic")]
    if synth_labels:
        print_section("Centroid Tokens Saved (synthetic modes only — constant = N)")
        for lbl in synth_labels:
            all_recs = [r for recs in data[lbl].values() for r in recs]
            n_val = all_recs[0]["N"] if all_recs else "?"
            print(f"  {lbl:<22}  {n_val} tokens saved per turn (gap injection)")

    # ── Aggregate TTFT stats ───────────────────────────────────────────────────
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

    # ── Quality ───────────────────────────────────────────────────────────────
    print_section("Response Quality (records with 'response' field)")
    has_response = {
        lbl: [r for recs in data[lbl].values() for r in recs if "response" in r]
        for lbl in labels
    }
    any_quality = any(has_response[lbl] for lbl in labels)
    if not any_quality:
        print("  No 'response' field found in records. Re-run benchmark to collect quality data.")
    else:
        header = f"  {'Mode':<22} {'N records':>10} {'Avg words':>10} {'Has code %':>12} {'Truncated %':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for lbl in labels:
            recs = has_response[lbl]
            if not recs:
                print(f"  {lbl:<22} {'—':>10}")
                continue
            word_counts = [len(r["response"].split()) for r in recs]
            code_pct = 100 * sum(has_code(r["response"]) for r in recs) / len(recs)
            # A response is likely truncated if it doesn't end with a sentence-ending char
            trunc_pct = 100 * sum(
                not r["response"].rstrip().endswith((".", "```", "GOODBYE", "!", "?", '"', "'"))
                for r in recs
            ) / len(recs)
            print(
                f"  {lbl:<22} {len(recs):>10} {mean(word_counts):>10.0f} "
                f"{code_pct:>11.1f}% {trunc_pct:>11.1f}%"
            )

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

    print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent / "results" / "multi_turn_benchmark.jsonl"
    )
    analyze(path)
