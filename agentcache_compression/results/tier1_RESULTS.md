# Tier 1 — measured corrected centroid speedup (timing only, dummy centroid)

**Date:** 2026-06-08 · **GPU:** H100 80GB · **vLLM:** 0.20.0 · 5 repeats/config.
Method: real harness with warmup + true-cold + clean sampling. Synthetic uses a **dummy**
GPT-OSS centroid `(24, N, 512)` — valid for TIMING (TTFT is independent of centroid
contents), not for quality. Raw data: `results/tier1/`, cold baseline `results/tc/` + sanity.

## Result: turn-1 cold-start speedup (clean; no history at turn 1)
| N | synthetic turn-1 | cold turn-1 | speedup |
|---|--:|--:|--:|
| 64  | 22.8 ms (±2.0) | 120.1 ms | **5.3×** |
| 128 | 24.4 ms (±1.0) | 120.1 ms | **4.9×** |
| 256 | 24.2 ms (±2.0) | 120.1 ms | **5.0×** |

Synthetic turn-1 phys tokens: 184/248/376 (= N + ~120 user) vs cold 2299 → system-prompt
prefill genuinely skipped. Cold sanity this batch: 118/126 ms (no drift). N-independent,
consistent with the paper's "centroid length has small effect."

## Interpretation
- **The cold-start speedup is REAL, ≈5×** — slightly above the paper's 4.3×, not below.
- An earlier estimate of ~1.5× (in `gptoss_corrected_baseline_NOTE.md`) was WRONG: it assumed
  ~64 ms fixed overhead; measured synthetic turn-1 is ~24 ms, so fixed cost is ~20 ms and
  skipping the 2000-token prefill saves the rest.
- **Why the paper's 4.3× survived the flawed methodology:** missing warm-up inflated BOTH
  baselines by the same one-time compile (cold 120→1078, synthetic 24→248), so the ratio was
  preserved. The methodology was sloppy (no warmup, "cold" with APC on, single sample,
  corrupted text) and the ABSOLUTE numbers are ~9× too high, but the SPEEDUP held.

## NOT confirmed here
- **Mean speedup (paper 1.80×):** dummy gives ~1.9–2.1× but it's unreliable (garbage
  responses → non-representative turns 2–10). Only turn-1 is a clean measurement. Needs a
  real centroid (Tier 2).
- **Quality:** dummy = timing only. Needs real centroid + `judge_multi_turn.py` (Tier 2).

## Caveats
- H100, not the paper's Blackwell — ratio should transfer (prefill-token-count driven),
  absolute ms differ.
- Timing valid because injection cost is shape-determined and prefill is token-count-
  determined; a real centroid has identical turn-1 TTFT, differing only in output quality.

## Corrected headline for the paper
cold turn-1 1078 ms → **120 ms**; synthetic turn-1 248 ms → **24 ms**; turn-1 speedup
4.3× → **~5×**. The cold-start claim reproduces; fix the methodology + absolute numbers, and
validate the mean + quality with Tier 2.
