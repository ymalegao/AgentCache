"""
run_all.py — full AgentCache-on-Mac matrix: model x precision x system-prompt length.

For each model:
  1. Train the prefix adapter (bf16) if it does not already exist.
  2. For each precision (fp16 / int8 / int4): load the base ONCE, then
       - cold   : eval across every system-prompt length (prefill grows with length)
       - inject : eval ONCE (prompt is user-only, so it's length-independent by design)
Records are written to results/matrix/<model_tag>.jsonl, one JSON object per task.

Efficiency notes:
  - inject is run once per (model, precision), not per prompt length — the inject prompt
    has no system text, so its TTFT does not depend on prompt length. We compare that one
    inject value against each cold@length point.
  - The base model is loaded once per (model, precision); cold runs before the PeftModel
    wrap so the un-adapted base is used for the baseline.

Robustness: every (model, precision) is wrapped in try/except so a failure (e.g. int4 not
supported on this MPS build) is recorded and skipped rather than aborting the whole matrix.

ACKNOWLEDGMENT: HF/MPS numbers, not vLLM. See README.md — trust the shape, not the ratio.
"""

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import torch

# Reuse the eval primitives from hf_eval (single source of truth).
from hf_eval import (
    pick_device, load_base, build_cold_inputs, build_inject_inputs,
    measure_ttft_s, generate_output, check_coherent, check_task,
)

REPO = Path(__file__).resolve().parent.parent
MAC = REPO / "agentcache_mac"
CMP = REPO / "agentcache_compression"

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def model_tag(m: str) -> str:
    return Path(m.rstrip("/")).name


def ensure_adapter(model, tokens, adapter_dir, train_dtype, epochs, batch_size, system_prompt, train_data):
    """Train the prefix adapter via train_prefix_mac.py if it is not present."""
    import subprocess, sys
    adapter = Path(adapter_dir) / f"{model_tag(model)}_N{tokens}"
    if (adapter / "adapter_model.safetensors").exists():
        print(f"[{model_tag(model)}] adapter exists — skipping training.")
        return str(adapter)
    adapter.mkdir(parents=True, exist_ok=True)
    print(f"[{model_tag(model)}] training adapter -> {adapter}")
    subprocess.run([
        sys.executable, str(MAC / "train_prefix_mac.py"),
        "--model", model, "--data", train_data,
        "--system-prompt", system_prompt, "--output", str(adapter),
        "--num-virtual-tokens", str(tokens),
        "--epochs", str(epochs), "--batch-size", str(batch_size),
        "--dtype", train_dtype,
    ], check=True)
    return str(adapter)


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def eval_tasks(model_obj, tokenizer, tasks, device, mode, system_text, n_runs, max_tokens, build_fn):
    """Run cold or inject over the task list; yield one record per task."""
    for task in tasks:
        user_text = task["user"]
        must = task.get("checks", {}).get("must_include_any", [])
        if mode == "cold":
            inputs = build_fn(tokenizer, system_text, user_text, device)
        else:
            inputs = build_fn(tokenizer, user_text, device)
        phys = inputs["input_ids"].shape[1]
        ttft_mean, _ = measure_ttft_s(model_obj, inputs, device, n_runs)
        out = generate_output(model_obj, tokenizer, inputs, max_tokens)
        yield {
            "id": task["id"], "mode": mode,
            "physical_prompt_tokens": phys,
            "ttft_mean_s": round(ttft_mean, 5),
            "checks": {
                "coherent": check_coherent(out),
                "ends_with_goodbye": out.strip().endswith("GOODBYE"),
                "task_check_pass": check_task(out, must) if must else False,
            },
            "output": out,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--dtypes", default="fp16,int8,int4")
    ap.add_argument("--prompts", default="200,500,1000,2000")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-dtype", default="bf16", choices=["fp32", "bf16"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=10, help="tasks per cell (0 = all 25)")
    ap.add_argument("--n-ttft-runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--adapter-dir", default=str(MAC / "adapters"))
    ap.add_argument("--eval-data", default=str(CMP / "data" / "python_agent_eval.jsonl"))
    ap.add_argument("--train-data", default=str(CMP / "data" / "python_agent_train.jsonl"))
    ap.add_argument("--train-prompt", default=str(CMP / "prompts" / "2000_python_agent_system.txt"))
    ap.add_argument("--out-dir", default=str(MAC / "results" / "matrix"))
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    device = pick_device("auto")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    plabels = [p.strip() for p in args.prompts.split(",") if p.strip()]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    from peft import PeftModel

    tasks_all = [json.loads(l) for l in Path(args.eval_data).read_text().splitlines() if l.strip()]
    tasks = tasks_all[: args.limit] if args.limit else tasks_all
    prompt_text = {pl: (CMP / "prompts" / f"{pl}_python_agent_system.txt").read_text().strip()
                   for pl in plabels}
    prompt_tok = {pl: None for pl in plabels}  # filled per tokenizer

    started = time.time()
    print(f"Matrix: {len(models)} models x {len(dtypes)} dtypes x {len(plabels)} prompts | "
          f"{len(tasks)} tasks/cell | device={device}\n")

    for model in models:
        tag = model_tag(model)
        out_path = out_dir / f"{tag}.jsonl"
        # fresh per-model file
        out_path.unlink(missing_ok=True)

        # 1) adapter
        if args.skip_train:
            adapter = str(Path(args.adapter_dir) / f"{tag}_N{args.tokens}")
        else:
            adapter = ensure_adapter(model, args.tokens, args.adapter_dir, args.train_dtype,
                                     args.epochs, args.batch_size, args.train_prompt, args.train_data)

        tokenizer = AutoTokenizer.from_pretrained(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        for pl in plabels:
            prompt_tok[pl] = len(tokenizer(prompt_text[pl], add_special_tokens=False)["input_ids"])

        with open(out_path, "a") as fout:
            for dtype in dtypes:
                t0 = time.time()
                try:
                    print(f"\n=== {tag} | dtype={dtype} ===")
                    base = load_base(model, dtype, device)
                    base.eval()

                    # cold across prompt lengths
                    for pl in plabels:
                        for rec in eval_tasks(base, tokenizer, tasks, device, "cold",
                                              prompt_text[pl], args.n_ttft_runs, args.max_tokens,
                                              build_cold_inputs):
                            rec.update(model=tag, dtype=dtype, sys_prompt_label=pl,
                                       sys_prompt_tokens=prompt_tok[pl], N=args.tokens)
                            fout.write(json.dumps(rec) + "\n")
                        fout.flush()
                        print(f"  cold@{pl} done")

                    # inject once (length-independent)
                    peft = PeftModel.from_pretrained(base, adapter).to(device)
                    peft.eval()
                    for rec in eval_tasks(peft, tokenizer, tasks, device, "inject",
                                          None, args.n_ttft_runs, args.max_tokens,
                                          build_inject_inputs):
                        rec.update(model=tag, dtype=dtype, sys_prompt_label="-",
                                   sys_prompt_tokens=0, N=args.tokens, pad_tokens=args.tokens)
                        fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    print(f"  inject done  ({time.time()-t0:.0f}s for dtype={dtype})")
                    free(peft, base)
                except Exception as e:
                    print(f"  SKIP {tag}/{dtype}: {e}")
                    traceback.print_exc()
                    fout.write(json.dumps({
                        "model": tag, "dtype": dtype, "mode": "ERROR",
                        "error": str(e),
                    }) + "\n")
                    fout.flush()
                    free()

        print(f"[{tag}] written -> {out_path}")

    print(f"\nMATRIX COMPLETE in {(time.time()-started)/60:.1f} min. JSONL in {out_dir}")


if __name__ == "__main__":
    main()
