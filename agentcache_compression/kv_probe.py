"""
Experiment C — KV-norm probe.

Question: is the exported centroid KV in the same numerical regime as the *real*
system-prompt KV it replaces? If the centroid is wildly off-scale (or has outlier
tokens), the N=256 turn-7/9 degeneration is likely a fixable export/scale bug. If the
centroid is in-distribution, the degeneration is a dynamic multi-turn context effect
(points to the train/serve mismatch, not a static bug).

Method: run the 7B forward on the system prompt, grab past_key_values, compute per-token
L2 norms per layer; compare to the centroid .npy norms. L2 norm is RoPE-invariant, so
comparing the pre-RoPE centroid against the post-RoPE prompt cache by norm is valid.

Usage:
  python agentcache_compression/kv_probe.py --model models/Qwen2.5-7B-Instruct
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent


def extract_layers(past_key_values):
    """Return list of (K, V) tensors per layer, each [num_kv_heads, seq, head_dim].
    Robust to transformers 5.x DynamicCache and legacy tuple formats."""
    layers = []
    # transformers 5.x: DynamicCache with .layers[i].keys/.values
    if hasattr(past_key_values, "layers"):
        for lyr in past_key_values.layers:
            layers.append((lyr.keys[0], lyr.values[0]))
        return layers
    # older: .key_cache / .value_cache lists
    if hasattr(past_key_values, "key_cache"):
        for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
            layers.append((k[0], v[0]))
        return layers
    # legacy tuple of (K, V)
    for k, v in past_key_values:
        layers.append((k[0], v[0]))
    return layers


def per_token_norms_from_cache(t):
    """t: [num_kv_heads, seq, head_dim] -> per-token L2 norm over flattened (heads,dim).
    Flatten to [seq, heads*head_dim] matching the centroid layout (transpose_tensors.py:80)."""
    h, s, d = t.shape
    flat = t.permute(1, 0, 2).reshape(s, h * d).float().cpu().numpy()
    return np.linalg.norm(flat, axis=1)  # [seq]


def per_token_norms_from_centroid(arr_layer):
    """arr_layer: [N, token_dim] -> per-token L2 norm. [N]"""
    return np.linalg.norm(arr_layer.astype(np.float32), axis=1)


def stats(x):
    return dict(mean=float(np.mean(x)), med=float(np.median(x)),
                p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)),
                mx=float(np.max(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--system-prompt", default=str(_HERE / "prompts" / "2000_python_agent_system.txt"))
    ap.add_argument("--centroid-dir", default=str(_HERE / "centroids_qwen7b"))
    ap.add_argument("--out", default=str(_HERE / "results" / "kv_probe.md"))
    args = ap.parse_args()

    sys_prompt = Path(args.system_prompt).read_text().strip()
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = tok.apply_chat_template(
        [{"role": "system", "content": sys_prompt}],
        tokenize=True, add_generation_prompt=False, return_tensors="pt", return_dict=True,
    )
    input_ids = enc["input_ids"]
    n_sys_tokens = input_ids.shape[1]
    print(f"system prompt tokens: {n_sys_tokens}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids.to(model.device), use_cache=True)
    layers = extract_layers(out.past_key_values)
    L = len(layers)
    print(f"layers: {L}  | per-layer cache K shape: {tuple(layers[0][0].shape)}")

    # real prompt per-token norms per layer
    promptK = [per_token_norms_from_cache(k) for k, v in layers]
    promptV = [per_token_norms_from_cache(v) for k, v in layers]

    cents = {}
    for N in (256, 64):
        K = np.load(f"{args.centroid_dir}/N{N}_2000_K.npy")  # [L, N, 512]
        V = np.load(f"{args.centroid_dir}/N{N}_2000_V.npy")
        cents[N] = (K, V)

    def build_table(which):  # which: 0=K, 1=V
        prompt = promptK if which == 0 else promptV
        lines = []
        lines.append("| layer | prompt_med | N256_med | N256/prompt | N256_outl>p99 | N64_med | N64/prompt | N64_outl>p99 |")
        lines.append("|------:|-----------:|---------:|------------:|--------------:|--------:|-----------:|-------------:|")
        agg = {256: [], 64: []}
        outl = {256: 0, 64: 0}
        for i in range(L):
            ps = stats(prompt[i]); p99 = ps["p99"]
            row = [i, ps["med"]]
            cellvals = {}
            for N in (256, 64):
                arr = cents[N][which][i]
                cn = per_token_norms_from_centroid(arr)
                cs = stats(cn)
                ratio = cs["med"] / ps["med"] if ps["med"] else float("nan")
                n_out = int(np.sum(cn > p99))
                agg[N].append(ratio); outl[N] += n_out
                cellvals[N] = (cs["med"], ratio, n_out)
            lines.append(
                f"| {i} | {ps['med']:.2f} | {cellvals[256][0]:.2f} | {cellvals[256][1]:.2f}x | "
                f"{cellvals[256][2]} | {cellvals[64][0]:.2f} | {cellvals[64][1]:.2f}x | {cellvals[64][2]} |"
            )
        summary = {N: (float(np.median(agg[N])), float(np.max(agg[N])), float(np.min(agg[N])), outl[N])
                   for N in (256, 64)}
        return "\n".join(lines), summary

    tblK, sumK = build_table(0)
    tblV, sumV = build_table(1)

    def verdict(sumX, total_tokens):
        # off-scale if median ratio far from 1 or many outlier tokens
        flags = []
        for N in (256, 64):
            med_ratio, mx, mn, outl = sumX[N]
            off = (med_ratio > 3 or med_ratio < 1/3 or mx > 5 or mn < 1/5)
            frac_out = outl / (total_tokens[N])
            flags.append((N, med_ratio, mx, mn, outl, frac_out, off))
        return flags

    totalsK = {256: 256 * L, 64: 64 * L}
    fK = verdict(sumK, totalsK)
    fV = verdict(sumV, totalsK)

    def fmt_flags(name, flags):
        s = [f"**{name}:**"]
        for (N, med, mx, mn, outl, frac, off) in flags:
            s.append(f"- N={N}: median ratio {med:.2f}x (range {mn:.2f}–{mx:.2f}x across layers), "
                     f"outlier tokens >prompt-p99: {outl} ({frac:.1%}) → "
                     f"{'OFF-SCALE' if off else 'in-distribution'}")
        return "\n".join(s)

    any_off = any(f[6] for f in fK) or any(f[6] for f in fV)
    overall = ("**OFF-SCALE detected** → the degeneration is likely a fixable export/scale "
               "(or RoPE/position) bug, not an intrinsic floor."
               if any_off else
               "**Centroid KV is in-distribution** (norms comparable to the real prompt) → the "
               "N=256 degeneration is NOT a static scale bug; it is a dynamic effect of the fixed "
               "centroid interacting with long multi-turn context → root cause points to the "
               "single-turn-train / multi-turn-serve mismatch, not the exporter.")

    md = f"""# Experiment C — KV-norm probe (centroid vs real system-prompt KV)

System prompt tokens analyzed: **{n_sys_tokens}**. Base: {args.model}.
Norm = per-token L2 over the flattened (heads×head_dim=512) vector, computed in fp32.
**Caveat:** L2 norm is RoPE-invariant, so comparing the pre-RoPE centroid to the
post-RoPE prompt cache by norm is valid; this probes *magnitude/scale*, not direction.

## Verdict

{overall}

{fmt_flags("Keys", fK)}

{fmt_flags("Values", fV)}

## Per-layer K norms

{tblK}

## Per-layer V norms

{tblV}
"""
    Path(args.out).write_text(md)
    print("\n" + overall)
    print(fmt_flags("Keys", fK))
    print(fmt_flags("Values", fV))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
