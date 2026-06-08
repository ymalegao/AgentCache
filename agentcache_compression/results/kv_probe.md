# Experiment C — KV-norm probe (centroid vs real system-prompt KV)

System prompt tokens analyzed: **2183**. Base: models/Qwen2.5-7B-Instruct.
Norm = per-token L2 over the flattened (heads×head_dim=512) vector, computed in fp32.
**Caveat:** L2 norm is RoPE-invariant, so comparing the pre-RoPE centroid to the
post-RoPE prompt cache by norm is valid; this probes *magnitude/scale*, not direction.

## Verdict

**OFF-SCALE detected** → the degeneration is likely a fixable export/scale (or RoPE/position) bug, not an intrinsic floor.

**Keys:**
- N=256: median ratio 0.35x (range 0.01–0.50x across layers), outlier tokens >prompt-p99: 18 (0.3%) → OFF-SCALE
- N=64: median ratio 0.38x (range 0.02–0.54x across layers), outlier tokens >prompt-p99: 17 (0.9%) → OFF-SCALE

**Values:**
- N=256: median ratio 0.82x (range 0.21–2.51x across layers), outlier tokens >prompt-p99: 1288 (18.0%) → in-distribution
- N=64: median ratio 0.96x (range 0.26–3.34x across layers), outlier tokens >prompt-p99: 545 (30.4%) → in-distribution

## Refined interpretation (supersedes the coarse auto-verdict above)

**Finding: the centroid Keys are systematically under-scaled.** The model's real
prompt keys vary by layer (median ~35–72 in the bulk, with massive-activation spikes of
**604 at layer 0** and **922 at the final layer 27**). The centroid instead produces a
nearly **flat ~10–19 key norm at every layer** — so it is ~0.35× the prompt in the bulk
and only **0.01–0.02× at layers 0 and 27**. It does not reproduce the model's per-layer
norm structure at all. **Values** are roughly in-scale (median ~0.8–1.0×) but heavy-tailed.

**Why this matters mechanically.** Attention logits are Q·K/√d. Uniformly smaller prefix
keys ⇒ smaller prefix logits ⇒ the prefix receives **less softmax mass** than the real
prompt would — essentially ignored at layers 0/27 and weakly attended elsewhere.
Crucially, a weakly-attended fixed prefix gets **diluted further as multi-turn context
grows**, a clean mechanism for the late-turn degradation and the turn-7/9 degeneration
seen at N=256.

**Attribution: a training/calibration artifact, not an exporter bug.** The export is a
faithful read of `PeftModel.get_prompt()`; the prefix-projection MLP was never constrained
to match the prompt's KV scale, so it lands at a flat, small norm. (Not grossly broken —
N=256 still reached parity on many turns.)

**UPDATE — rescale test run, hypothesis REFUTED.** We rebuilt N=256 with keys rescaled
to the prompt's norm (global ×2.9 and per-layer median-match), keeping V fixed, and
re-ran synthetic_N256 on csv_cli:

| | syntax-valid | degenerate |
|---|---|---|
| baseline N=256 | 8/10 | 2/10 |
| keys ×2.9 (global) | **0/10** | **10/10** |
| keys ×per-layer | **0/10** | **10/10** |

Both collapsed into total token-salad (`filtro BY Zusstile ESingleOrDefault strugg…`).
Mechanism: multiplying K multiplies every prefix attention *logit*, so softmax mass
collapses onto the centroid — the model attends almost entirely to the 256-token prefix
and ignores the actual conversation. **The small key norms are therefore load-bearing
calibration learned in training, not a defect; naive rescaling is the wrong lever.**

**Corrected conclusion:** the C norm-gap is real but is *not* a fixable scale bug. The
baseline degeneration (turns 7/9) is a **dynamic** effect — a fixed, deliberately
weakly-attended prefix being diluted by growing multi-turn context — which points to the
**single-turn-train / multi-turn-serve mismatch** (and the token-CE objective) as the
root cause. The right levers are **multi-turn / KL distillation** and **capacity**, not
KV calibration. (Negative result: the cheap fix is ruled out — don't repeat it.)

## Per-layer K norms

| layer | prompt_med | N256_med | N256/prompt | N256_outl>p99 | N64_med | N64/prompt | N64_outl>p99 |
|------:|-----------:|---------:|------------:|--------------:|--------:|-----------:|-------------:|
| 0 | 604.30 | 9.04 | 0.01x | 0 | 11.06 | 0.02x | 0 |
| 1 | 160.18 | 8.34 | 0.05x | 0 | 11.60 | 0.07x | 0 |
| 2 | 69.01 | 12.99 | 0.19x | 0 | 14.15 | 0.21x | 0 |
| 3 | 122.76 | 12.71 | 0.10x | 0 | 14.06 | 0.11x | 0 |
| 4 | 38.98 | 14.35 | 0.37x | 0 | 16.28 | 0.42x | 0 |
| 5 | 38.29 | 14.23 | 0.37x | 0 | 17.67 | 0.46x | 0 |
| 6 | 43.57 | 12.67 | 0.29x | 0 | 15.01 | 0.34x | 0 |
| 7 | 43.34 | 14.61 | 0.34x | 1 | 17.25 | 0.40x | 0 |
| 8 | 40.77 | 14.66 | 0.36x | 0 | 17.14 | 0.42x | 0 |
| 9 | 40.25 | 14.88 | 0.37x | 1 | 18.29 | 0.45x | 0 |
| 10 | 41.19 | 14.51 | 0.35x | 0 | 16.88 | 0.41x | 1 |
| 11 | 41.38 | 16.79 | 0.41x | 0 | 19.31 | 0.47x | 0 |
| 12 | 42.82 | 14.47 | 0.34x | 1 | 15.71 | 0.37x | 0 |
| 13 | 71.40 | 13.57 | 0.19x | 0 | 14.50 | 0.20x | 0 |
| 14 | 41.86 | 13.64 | 0.33x | 0 | 15.66 | 0.37x | 0 |
| 15 | 46.07 | 13.10 | 0.28x | 0 | 14.22 | 0.31x | 0 |
| 16 | 43.55 | 10.62 | 0.24x | 0 | 10.97 | 0.25x | 0 |
| 17 | 38.86 | 10.24 | 0.26x | 0 | 11.55 | 0.30x | 0 |
| 18 | 40.67 | 15.80 | 0.39x | 0 | 15.68 | 0.39x | 0 |
| 19 | 64.87 | 10.93 | 0.17x | 0 | 10.88 | 0.17x | 0 |
| 20 | 36.81 | 14.46 | 0.39x | 2 | 12.82 | 0.35x | 1 |
| 21 | 42.56 | 18.72 | 0.44x | 1 | 22.88 | 0.54x | 0 |
| 22 | 40.57 | 16.56 | 0.41x | 1 | 19.66 | 0.48x | 1 |
| 23 | 40.87 | 18.65 | 0.46x | 2 | 19.47 | 0.48x | 2 |
| 24 | 38.70 | 19.07 | 0.49x | 4 | 20.62 | 0.53x | 4 |
| 25 | 38.37 | 17.51 | 0.46x | 3 | 19.38 | 0.51x | 4 |
| 26 | 35.49 | 17.62 | 0.50x | 2 | 19.31 | 0.54x | 4 |
| 27 | 922.05 | 15.35 | 0.02x | 0 | 18.38 | 0.02x | 0 |

## Per-layer V norms

| layer | prompt_med | N256_med | N256/prompt | N256_outl>p99 | N64_med | N64/prompt | N64_outl>p99 |
|------:|-----------:|---------:|------------:|--------------:|--------:|-----------:|-------------:|
| 0 | 5.96 | 8.34 | 1.40x | 171 | 9.47 | 1.59x | 43 |
| 1 | 3.86 | 9.70 | 2.51x | 255 | 12.91 | 3.34x | 64 |
| 2 | 6.60 | 13.91 | 2.11x | 254 | 15.04 | 2.28x | 64 |
| 3 | 7.99 | 12.95 | 1.62x | 213 | 14.25 | 1.78x | 58 |
| 4 | 10.43 | 11.67 | 1.12x | 83 | 13.58 | 1.30x | 38 |
| 5 | 12.06 | 11.98 | 0.99x | 19 | 14.46 | 1.20x | 25 |
| 6 | 10.77 | 10.94 | 1.02x | 43 | 13.27 | 1.23x | 31 |
| 7 | 13.65 | 12.86 | 0.94x | 16 | 15.51 | 1.14x | 21 |
| 8 | 13.57 | 12.28 | 0.90x | 20 | 15.36 | 1.13x | 23 |
| 9 | 15.12 | 13.31 | 0.88x | 16 | 15.47 | 1.02x | 23 |
| 10 | 13.11 | 12.82 | 0.98x | 19 | 14.76 | 1.13x | 21 |
| 11 | 14.38 | 14.40 | 1.00x | 43 | 16.97 | 1.18x | 34 |
| 12 | 15.45 | 13.94 | 0.90x | 32 | 15.97 | 1.03x | 17 |
| 13 | 15.74 | 14.46 | 0.92x | 29 | 15.50 | 0.98x | 13 |
| 14 | 17.05 | 12.58 | 0.74x | 10 | 15.04 | 0.88x | 15 |
| 15 | 16.10 | 12.37 | 0.77x | 12 | 15.20 | 0.94x | 16 |
| 16 | 18.02 | 11.28 | 0.63x | 7 | 12.03 | 0.67x | 6 |
| 17 | 18.81 | 11.03 | 0.59x | 6 | 12.81 | 0.68x | 5 |
| 18 | 20.26 | 14.42 | 0.71x | 8 | 14.97 | 0.74x | 8 |
| 19 | 20.41 | 11.15 | 0.55x | 6 | 10.81 | 0.53x | 2 |
| 20 | 21.35 | 14.56 | 0.68x | 15 | 13.54 | 0.63x | 9 |
| 21 | 23.61 | 15.95 | 0.68x | 7 | 18.79 | 0.80x | 5 |
| 22 | 26.09 | 15.10 | 0.58x | 4 | 17.63 | 0.68x | 2 |
| 23 | 32.19 | 16.50 | 0.51x | 0 | 18.07 | 0.56x | 1 |
| 24 | 35.67 | 16.83 | 0.47x | 0 | 17.81 | 0.50x | 1 |
| 25 | 46.82 | 15.16 | 0.32x | 0 | 17.58 | 0.38x | 0 |
| 26 | 68.75 | 16.85 | 0.25x | 0 | 19.30 | 0.28x | 0 |
| 27 | 81.87 | 17.42 | 0.21x | 0 | 20.96 | 0.26x | 0 |
