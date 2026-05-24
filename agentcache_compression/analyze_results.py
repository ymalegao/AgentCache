#!/usr/bin/env python3
"""
Analyze pipeline comparison results from a JSONL file.

Usage:
    python analyze_results.py <path_to_jsonl>
    python analyze_results.py results/64_2000_comparison.jsonl
"""

import json
import sys
import statistics
from collections import defaultdict
from pathlib import Path

MODE_LABELS = {
    "cold_no_synthetic": "Cold (no cache)",
    "warm_apc":          "Warm APC",
    "synthetic_compression": "Synthetic Compression",
}

def load(path: str) -> dict[str, list[dict]]:
    modes: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            modes[r["mode"]].append(r)
    return dict(modes)


def pct_pass(records: list[dict], key: str) -> float:
    return 100 * sum(r["checks"][key] for r in records) / len(records)


def stats(values: list[float]) -> dict:
    return {
        "mean":   statistics.mean(values),
        "median": statistics.median(values),
        "stdev":  statistics.stdev(values) if len(values) > 1 else 0.0,
        "min":    min(values),
        "max":    max(values),
    }


def speedup(baseline: float, candidate: float) -> str:
    ratio = baseline / candidate if candidate else float("inf")
    return f"{ratio:.2f}x"


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def analyze(path: str):
    data = load(path)
    modes = sorted(data.keys())

    print(f"\nFile : {path}")
    print(f"Total records : {sum(len(v) for v in data.values())}")
    for m in modes:
        print(f"  {MODE_LABELS.get(m, m):30s} : {len(data[m])} records")

    # ── TTFT ──────────────────────────────────────────────────────────────────
    print_section("Time-to-First-Token (TTFT) — seconds")
    ttft: dict[str, dict] = {}
    for m in modes:
        vals = [r["ttft_mean_s"] for r in data[m]]
        ttft[m] = stats(vals)

    header = f"{'Mode':<30} {'Mean':>8} {'Median':>8} {'StdDev':>8} {'Min':>8} {'Max':>8}"
    print(header)
    print("-" * len(header))
    for m in modes:
        s = ttft[m]
        label = MODE_LABELS.get(m, m)
        print(f"{label:<30} {s['mean']:>8.4f} {s['median']:>8.4f} {s['stdev']:>8.4f} {s['min']:>8.4f} {s['max']:>8.4f}")

    baseline_mode = "cold_no_synthetic"
    if baseline_mode in ttft:
        print(f"\n  Speedup vs '{MODE_LABELS[baseline_mode]}':")
        for m in modes:
            if m == baseline_mode:
                continue
            su = speedup(ttft[baseline_mode]["mean"], ttft[m]["mean"])
            label = MODE_LABELS.get(m, m)
            print(f"    {label:<30} {su}")

    # ── TOKEN COUNTS ──────────────────────────────────────────────────────────
    print_section("Physical Prompt Tokens")
    header = f"{'Mode':<30} {'Mean':>8} {'Min':>8} {'Max':>8}"
    print(header)
    print("-" * len(header))
    for m in modes:
        vals = [r["physical_prompt_tokens"] for r in data[m]]
        s = stats(vals)
        label = MODE_LABELS.get(m, m)
        print(f"{label:<30} {s['mean']:>8.1f} {s['min']:>8} {s['max']:>8}")

    # Pad tokens (only non-zero modes are interesting)
    has_pad = {m for m in modes if any(r["pad_tokens"] for r in data[m])}
    if has_pad:
        print_section("Pad Tokens (synthetic compression overhead)")
        for m in has_pad:
            vals = [r["pad_tokens"] for r in data[m]]
            label = MODE_LABELS.get(m, m)
            print(f"  {label:<30} mean={statistics.mean(vals):.1f}  min={min(vals)}  max={max(vals)}")

    # User tokens (only where not null)
    has_user = {m for m in modes if any(r["user_tokens"] is not None for r in data[m])}
    if has_user:
        print_section("User Tokens (KV cache footprint)")
        for m in has_user:
            vals = [r["user_tokens"] for r in data[m] if r["user_tokens"] is not None]
            label = MODE_LABELS.get(m, m)
            print(f"  {label:<30} mean={statistics.mean(vals):.1f}  min={min(vals)}  max={max(vals)}")

    # ── QUALITY CHECKS ────────────────────────────────────────────────────────
    print_section("Quality Checks (% pass)")
    check_keys = sorted({k for m in modes for r in data[m] for k in r["checks"]})
    header = f"{'Mode':<30}" + "".join(f" {k:>20}" for k in check_keys)
    print(header)
    print("-" * len(header))
    for m in modes:
        label = MODE_LABELS.get(m, m)
        row = f"{label:<30}"
        for k in check_keys:
            row += f" {pct_pass(data[m], k):>19.1f}%"
        print(row)

    # ── PER-EXAMPLE TTFT COMPARISON ───────────────────────────────────────────
    if len(modes) >= 2 and all(m in data for m in [baseline_mode, "synthetic_compression"]):
        print_section("Per-example TTFT: Cold vs Synthetic Compression")
        cold_by_id = {r["id"]: r["ttft_mean_s"] for r in data[baseline_mode]}
        comp_by_id = {r["id"]: r["ttft_mean_s"] for r in data["synthetic_compression"]}
        common = sorted(set(cold_by_id) & set(comp_by_id))
        deltas = [cold_by_id[i] - comp_by_id[i] for i in common]
        pct_faster = 100 * sum(d > 0 for d in deltas) / len(deltas) if deltas else 0
        print(f"  Matched examples   : {len(common)}")
        if deltas:
            print(f"  Mean delta (cold - comp) : {statistics.mean(deltas):+.4f} s")
            print(f"  Comp faster in     : {pct_faster:.1f}% of cases")
            # top 5 biggest gains
            top = sorted(zip(common, deltas), key=lambda x: -x[1])[:5]
            print(f"\n  Top 5 speedups (by example):")
            print(f"    {'id':<12} {'cold_ttft':>10} {'comp_ttft':>10} {'delta':>10}")
            for eid, d in top:
                print(f"    {eid:<12} {cold_by_id[eid]:>10.4f} {comp_by_id[eid]:>10.4f} {d:>+10.4f}")

    print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/64_2000_comparison.jsonl"
    analyze(path)
