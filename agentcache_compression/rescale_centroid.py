"""
Build key-rescaled centroid variants to test the Experiment-C hypothesis that the
N=256 centroid's keys are under-scaled relative to the real system-prompt KV (which
weakens prefix attention and drives late-turn degeneration).

Loads the 7B once to get the real prompt's per-layer median key norm, then writes two
variants (keys rescaled, values untouched) to separate centroid dirs:
  - global   : one scalar factor = median over layers of (prompt_med / centroid_med)
  - perlayer : factor[L] = prompt_med[L] / centroid_med[L]

Usage:
  python agentcache_compression/rescale_centroid.py --model models/Qwen2.5-7B-Instruct
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent


def prompt_per_layer_median_knorm(model, tok, sys_prompt):
    enc = tok.apply_chat_template(
        [{"role": "system", "content": sys_prompt}],
        tokenize=True, add_generation_prompt=False, return_tensors="pt", return_dict=True,
    )
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"].to(model.device), use_cache=True)
    pkv = out.past_key_values
    layers = pkv.layers if hasattr(pkv, "layers") else None
    meds = []
    for i in range(len(layers)):
        k = layers[i].keys[0]  # [heads, seq, head_dim]
        h, s, d = k.shape
        flat = k.permute(1, 0, 2).reshape(s, h * d).float().cpu().numpy()
        meds.append(float(np.median(np.linalg.norm(flat, axis=1))))
    return np.array(meds)  # [L]


def centroid_per_layer_median_knorm(K):  # K: [L, N, 512]
    return np.array([float(np.median(np.linalg.norm(K[i].astype(np.float32), axis=1)))
                     for i in range(K.shape[0])])


def write_variant(out_dir, K_scaled, V, src_dir, N):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"N{N}_2000_K.npy", K_scaled.astype(np.float16))
    np.save(out / f"N{N}_2000_V.npy", V.astype(np.float16))
    sidecar = Path(src_dir) / "sys_prefix_num_tokens.txt"
    if sidecar.exists():
        shutil.copy(sidecar, out / "sys_prefix_num_tokens.txt")
    print(f"  wrote {out}/N{N}_2000_{{K,V}}.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--system-prompt", default=str(_HERE / "prompts" / "2000_python_agent_system.txt"))
    ap.add_argument("--src-dir", default=str(_HERE / "centroids_qwen7b"))
    ap.add_argument("--N", type=int, default=256)
    args = ap.parse_args()

    N = args.N
    K = np.load(f"{args.src_dir}/N{N}_2000_K.npy")  # [L, N, 512]
    V = np.load(f"{args.src_dir}/N{N}_2000_V.npy")
    L = K.shape[0]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    prompt_med = prompt_per_layer_median_knorm(model, tok, Path(args.system_prompt).read_text().strip())
    cent_med = centroid_per_layer_median_knorm(K)
    ratio = prompt_med / cent_med  # per-layer target multiplier
    global_factor = float(np.median(ratio))
    print(f"per-layer prompt/centroid K-norm ratio: median={global_factor:.2f} "
          f"min={ratio.min():.2f} max={ratio.max():.2f}")

    # Variant 1: global uniform scale on keys
    Kg = K.astype(np.float32) * global_factor
    write_variant(f"{_HERE}/centroids_qwen7b_kscale_global", Kg, V.astype(np.float32), args.src_dir, N)

    # Variant 2: per-layer median match on keys
    Kp = K.astype(np.float32) * ratio[:, None, None]
    write_variant(f"{_HERE}/centroids_qwen7b_kscale_perlayer", Kp, V.astype(np.float32), args.src_dir, N)

    print(f"global_factor={global_factor:.3f}; per-layer factors saved into the two variant dirs.")


if __name__ == "__main__":
    main()
