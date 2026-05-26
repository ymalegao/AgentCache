"""
hf_eval.py — single-turn cold-vs-inject evaluation on HuggingFace + MPS.

This is the Mac/Metal analog of agentcache_compression/test_compression.py. There is
NO vLLM and NO raw-KV surgery: the trained PEFT prefix is injected the way HuggingFace
natively supports it — PeftModel.generate() prepends the prefix as past_key_values, so
the model attends to it exactly as it would the original system prompt. RoPE/positions
are handled by the same PEFT code path used during training (no manual rotation).

Modes (one per invocation, like test_compression.py):
  cold     Base model, full system+user prompt. Prefill over system+user. Baseline.
  inject   PeftModel + adapter, user turn only (no system text). Prefix injected as
           past_key_values; the model prefills ONLY the user tokens. The system prompt's
           behavior is carried by the trained prefix.

Records are written in the SAME schema as test_compression.py so the shared
analyze_results.py can summarize them.

ACKNOWLEDGMENT: This deployment has none of vLLM's optimizations (paged attention,
continuous batching, fused CUDA kernels, APC, scheduler prefill-skip). Absolute TTFT is
higher and the speedup ratio differs from the vLLM 2.8x figure. What it demonstrates is
the *mechanism*: skipping system-prompt prefill lowers TTFT, and the trained prefix
preserves task behavior — fully reproducibly, on a laptop. See README.md.

Usage:
  python agentcache_mac/hf_eval.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --adapter agentcache_mac/adapters/Llama-3.2-1B_N64 \
    --system-prompt agentcache_compression/prompts/2000_python_agent_system.txt \
    --data agentcache_compression/data/python_agent_eval.jsonl \
    --mode inject --synthetic-len 64 --dtype fp16 \
    --out agentcache_mac/results/Llama-3.2-1B_N64_2000.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_REPO = Path(__file__).resolve().parent.parent
_CMP = _REPO / "agentcache_compression"


# ---------------------------------------------------------------------------
# Device / precision
# ---------------------------------------------------------------------------

def pick_device(choice: str) -> str:
    if choice != "auto":
        return choice
    return "mps" if torch.backends.mps.is_available() else "cpu"


def sync(device: str) -> None:
    """MPS analog of torch.cuda.synchronize() — required for honest wall-clock TTFT."""
    if device == "mps":
        torch.mps.synchronize()


def load_base(model_path: str, dtype: str, device: str):
    """Load the base causal LM at the requested precision.

    fp16/bf16: standard load + .to(device).
    int8/int4: torchao weight-only quantization (keeps the standard HF forward path,
               so PeftModel prefix injection via past_key_values still works — unlike
               MLX/llama.cpp, which cannot inject arbitrary trained K/V).
    """
    if dtype in ("fp16", "bf16"):
        tdtype = torch.float16 if dtype == "fp16" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=tdtype)
        return model.to(device)

    # Quantized inference path.
    from transformers import TorchAoConfig
    quant_type = "int8_weight_only" if dtype == "int8" else "int4_weight_only"
    try:
        qcfg = TorchAoConfig(quant_type)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=qcfg, torch_dtype=torch.bfloat16
        )
        return model.to(device)
    except Exception as e:
        raise RuntimeError(
            f"torchao {quant_type} load/move to {device} failed: {e}\n"
            "int4 in particular may be unsupported on MPS in your torch build. "
            "Record this in the README coverage table and fall back to fp16/int8."
        ) from e


# ---------------------------------------------------------------------------
# Prompt construction (mirrors test_compression.py)
# ---------------------------------------------------------------------------

def build_cold_inputs(tokenizer, system_text: str, user_text: str, device: str):
    """Full system+user chat prompt → model inputs (cold mode)."""
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text},
         {"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}


def build_inject_inputs(tokenizer, user_text: str, device: str):
    """User-only chat prompt → model inputs (inject mode; NO system text).

    PeftModel.generate prepends the trained prefix as past_key_values internally and
    extends the attention mask by num_virtual_tokens — we only supply the user tokens.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}


# ---------------------------------------------------------------------------
# TTFT + generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_ttft_s(model, inputs, device: str, n_runs: int, warmup: int = 2):
    """Mean/median TTFT via generate(max_new_tokens=1), MPS-synced. Returns (mean, runs)."""
    gen_kwargs = dict(max_new_tokens=1, do_sample=False, use_cache=True)
    for _ in range(warmup):
        model.generate(**inputs, **gen_kwargs)
    sync(device)

    times = []
    for _ in range(n_runs):
        sync(device)
        t0 = time.perf_counter()
        model.generate(**inputs, **gen_kwargs)
        sync(device)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), times


@torch.no_grad()
def generate_output(model, tokenizer, inputs, max_tokens: int) -> str:
    out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True)
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = out[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Quality checks (copied verbatim from test_compression.py)
# ---------------------------------------------------------------------------

def check_coherent(text: str) -> bool:
    """At least 20 words and no obvious repeated-token degeneration."""
    words = text.split()
    if len(words) < 20:
        return False
    tail = words[-30:] if len(words) >= 30 else words
    for w in set(tail):
        if tail.count(w) / len(tail) > 0.4:
            return False
    return True


def check_task(text: str, must_include_any: list[list[str]]) -> bool:
    lower = text.lower()
    for group in must_include_any:
        if any(kw.lower() in lower for kw in group):
            return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None,
                   help="PEFT adapter dir (required for --mode inject).")
    p.add_argument("--data", default=str(_CMP / "data" / "python_agent_eval.jsonl"))
    p.add_argument("--system-prompt", default=str(_CMP / "prompts" / "2000_python_agent_system.txt"))
    p.add_argument("--mode", choices=["cold", "inject"], required=True)
    p.add_argument("--synthetic-len", type=int, default=64,
                   help="N virtual tokens (recorded as pad_tokens for inject mode).")
    p.add_argument("--dtype", choices=["fp16", "bf16", "int8", "int4"], default="fp16")
    p.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "mac_eval.jsonl"))
    p.add_argument("--n-ttft-runs", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=0, help="Eval only the first K tasks (0 = all).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    if args.mode == "inject" and not args.adapter:
        raise SystemExit("--adapter is required for --mode inject")

    system_text = Path(args.system_prompt).read_text().strip()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model ({args.dtype}) on {device}...")
    model = load_base(args.model, args.dtype, device)

    if args.mode == "inject":
        from peft import PeftModel
        print(f"Wrapping with PEFT adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.to(device)
    model.eval()

    tasks = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if args.limit:
        tasks = tasks[: args.limit]

    sys_tok_len = len(tokenizer(system_text, add_special_tokens=False)["input_ids"])
    print(f"\n=== mode={args.mode}  N={args.synthetic_len}  dtype={args.dtype}  "
          f"sys_prompt_tokens={sys_tok_len}  tasks={len(tasks)} ===\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a") as fout:
        for task in tasks:
            user_text = task["user"]
            must_include_any = task.get("checks", {}).get("must_include_any", [])

            if args.mode == "inject":
                inputs = build_inject_inputs(tokenizer, user_text, device)
                user_tokens = inputs["input_ids"].shape[1]
                physical_tokens = user_tokens   # HF prefills ONLY user tokens; prefix is injected
                pad_tokens = args.synthetic_len  # injected (saved), not physically prefilled
            else:
                inputs = build_cold_inputs(tokenizer, system_text, user_text, device)
                physical_tokens = inputs["input_ids"].shape[1]
                user_tokens = None
                pad_tokens = 0

            ttft_mean, ttft_runs = measure_ttft_s(model, inputs, device, args.n_ttft_runs)
            output_text = generate_output(model, tokenizer, inputs, args.max_tokens)

            coherent = check_coherent(output_text)
            ends_with_goodbye = output_text.strip().endswith("GOODBYE")
            task_check_pass = check_task(output_text, must_include_any) if must_include_any else False

            record = {
                "id": task["id"],
                "mode": args.mode,
                "N": args.synthetic_len,
                "dtype": args.dtype,
                "sys_prompt_tokens": sys_tok_len,
                "physical_prompt_tokens": physical_tokens,
                "pad_tokens": pad_tokens,
                "user_tokens": user_tokens,
                "ttft_mean_s": round(ttft_mean, 5),
                "ttft_runs_s": [round(t, 5) for t in ttft_runs],
                "checks": {
                    "coherent": coherent,
                    "ends_with_goodbye": ends_with_goodbye,
                    "task_check_pass": task_check_pass,
                },
                "output": output_text,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            status = "ok " if coherent else "BAD"
            print(f"  [{status}] {task['id']:<10} ttft={ttft_mean:.4f}s  "
                  f"phys_tokens={physical_tokens}  coherent={coherent}  "
                  f"task_check={task_check_pass}")

    print(f"\nResults appended to {out_path}")


if __name__ == "__main__":
    main()
