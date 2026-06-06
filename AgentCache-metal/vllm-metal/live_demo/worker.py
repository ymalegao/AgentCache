#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Line-oriented worker for the AgentCache live demo.

Each worker owns one vLLM engine. The parent process talks to it over JSON lines
on stdin/stdout so baseline and centroid modes can run with different startup
environments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

METAL_ROOT = Path(__file__).resolve().parents[1]
if str(METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(METAL_ROOT))


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)


def debug(mode: str, message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[worker {mode} {timestamp}] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgentCache live demo worker")
    p.add_argument("--mode", choices=["baseline", "centroid"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--system-prompt", required=True)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=1)
    p.add_argument("--mock", action="store_true")
    return p.parse_args()


class DemoWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.history: list[tuple[str, str]] = []
        debug(args.mode, f"reading system prompt from {args.system_prompt}")
        self.system_text = Path(args.system_prompt).read_text().strip()
        debug(args.mode, f"system prompt loaded ({len(self.system_text)} chars)")
        self.centroid_armed = os.environ.get("VLLM_CENTROID_SCHEDULER") == "1"
        self.tokenizer = None
        self.llm = None
        self.sampling_params_cls = None
        self.tokens_prompt_cls = None
        self.pad_id = 0

    @staticmethod
    def normalize_history(raw: Any) -> list[tuple[str, str]]:
        history: list[tuple[str, str]] = []
        if not isinstance(raw, list):
            return history
        for item in raw:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
            ):
                history.append((str(item[0]), str(item[1])))
            elif isinstance(item, dict):
                history.append((
                    str(item.get("user", "")),
                    str(item.get("assistant", "")),
                ))
        return history

    def load(self) -> None:
        debug(
            self.args.mode,
            f"load requested model={self.args.model} max_model_len={self.args.max_model_len} "
            f"max_num_seqs={self.args.max_num_seqs} n={self.args.n} mock={self.args.mock}",
        )
        debug(
            self.args.mode,
            "vLLM worker multiprocessing method="
            f"{os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', '<default>')}",
        )
        debug(
            self.args.mode,
            f"PYTHONPATH includes vllm-metal root={str(METAL_ROOT) in sys.path}",
        )
        if self.centroid_armed:
            debug(
                self.args.mode,
                "centroid env armed "
                f"k={os.environ.get('VLLM_CENTROID_K_PATH')} "
                f"v={os.environ.get('VLLM_CENTROID_V_PATH')} "
                f"sys_tokens={os.environ.get('VLLM_CENTROID_SYS_TOKENS')} "
                f"layout={os.environ.get('VLLM_CENTROID_LAYOUT')}",
            )
        if self.args.mock:
            self.pad_id = 0
            debug(self.args.mode, "mock mode active; skipping tokenizer/model load")
            return

        debug(self.args.mode, "importing transformers and vLLM")
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams, TokensPrompt

        debug(self.args.mode, "loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        debug(
            self.args.mode,
            f"tokenizer ready pad_token_id={self.tokenizer.pad_token_id} "
            f"eos_token_id={self.tokenizer.eos_token_id} effective_pad_id={self.pad_id}",
        )
        self.sampling_params_cls = SamplingParams
        self.tokens_prompt_cls = TokensPrompt
        debug(self.args.mode, "loading vLLM engine")
        self.llm = LLM(
            model=self.args.model,
            max_model_len=self.args.max_model_len,
            max_num_seqs=self.args.max_num_seqs,
        )
        debug(self.args.mode, "vLLM engine ready")

    def build_messages(self, user_text: str, *, include_system: bool) -> list[dict]:
        messages: list[dict] = []
        if include_system:
            messages.append({"role": "system", "content": self.system_text})
        for user, assistant in self.history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": user_text})
        return messages

    def tokenize_messages(self, messages: list[dict]) -> list[int]:
        if self.args.mock:
            return [1] * self.mock_prompt_tokens(messages)
        assert self.tokenizer is not None
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def prompt_ids(self, user_text: str) -> tuple[list[int], dict[str, bool | int]]:
        if self.args.mode == "centroid":
            messages = self.build_messages(user_text, include_system=False)
            ids = [self.pad_id] * self.args.n + self.tokenize_messages(messages)
            return ids, {
                "system_prompt_sent": False,
                "prefix_injected": bool(self.centroid_armed),
                "synthetic_tokens": self.args.n,
            }

        messages = self.build_messages(user_text, include_system=True)
        ids = self.tokenize_messages(messages)
        return ids, {
            "system_prompt_sent": True,
            "prefix_injected": False,
            "synthetic_tokens": 0,
        }

    def mock_prompt_tokens(self, messages: list[dict]) -> int:
        words = sum(len(str(m.get("content", "")).split()) for m in messages)
        overhead = 26 * len(messages)
        if any(m.get("role") == "system" for m in messages):
            overhead += 1700
        return max(16, words + overhead)

    def mock_generate(self, user_text: str, prompt_tokens: int, max_tokens: int) -> str:
        time.sleep(0.05 if self.args.mode == "centroid" else 0.08)
        task = user_text.split(".")[0].strip()
        return (
            f"{self.args.mode} response for turn {len(self.history) + 1}: "
            f"{task}. This is mock output for dashboard validation. "
            f"It used {prompt_tokens} prompt tokens and max_tokens={max_tokens}."
        )

    def generate(self, ids: list[int], max_tokens: int) -> tuple[str, float]:
        if self.args.mock:
            raise RuntimeError("mock_generate should be called directly")

        assert self.llm is not None
        assert self.sampling_params_cls is not None
        assert self.tokens_prompt_cls is not None
        effective_max = min(max_tokens, max(1, self.args.max_model_len - len(ids) - 32))
        params = self.sampling_params_cls(max_tokens=effective_max, temperature=0.0)
        t0 = time.perf_counter()
        out = self.llm.generate(
            [self.tokens_prompt_cls(prompt_token_ids=ids)],
            params,
            use_tqdm=False,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        return out[0].outputs[0].text or "", total_ms

    def warmup_measurement_path(self) -> float:
        """Run one unmeasured request to remove first-use kernel/injector overhead."""
        messages = [{"role": "user", "content": "Warm up the Python agent. Reply with ok."}]
        ids = self.tokenize_messages(messages)
        if self.args.mode == "centroid":
            ids = [self.pad_id] * self.args.n + ids
        debug(
            self.args.mode,
            f"measurement warmup start prompt_tokens={len(ids)}",
        )
        if self.args.mock:
            warmup_ms = 8.0 if self.args.mode == "centroid" else 12.0
            time.sleep(warmup_ms / 1000.0)
        else:
            warmup_ms = self.measure_ttft(ids)
        debug(self.args.mode, f"measurement warmup complete {warmup_ms:.1f} ms")
        return warmup_ms

    def measure_ttft(self, ids: list[int]) -> float:
        if self.args.mock:
            base = 26.0 if self.args.mode == "baseline" else 17.0
            growth = len(self.history) * (4.0 if self.args.mode == "baseline" else 2.0)
            time.sleep((base + growth) / 1000.0)
            return base + growth

        assert self.llm is not None
        assert self.sampling_params_cls is not None
        assert self.tokens_prompt_cls is not None
        params = self.sampling_params_cls(max_tokens=1, temperature=0.0)
        t0 = time.perf_counter()
        self.llm.generate(
            [self.tokens_prompt_cls(prompt_token_ids=ids)],
            params,
            use_tqdm=False,
        )
        return (time.perf_counter() - t0) * 1000.0

    def run_prompt(
        self,
        user_text: str,
        *,
        max_tokens: int,
        update_history: bool,
        warmup: bool = False,
    ) -> dict[str, Any]:
        ids, flags = self.prompt_ids(user_text)
        warmup_ms = self.warmup_measurement_path() if warmup else None
        debug(
            self.args.mode,
            f"measuring request prompt_tokens={len(ids)} max_tokens={max_tokens}",
        )
        ttft_ms = self.measure_ttft(ids)

        if self.args.mock:
            t0 = time.perf_counter()
            output = self.mock_generate(user_text, len(ids), max_tokens)
            total_ms = (time.perf_counter() - t0) * 1000.0
        else:
            output, total_ms = self.generate(ids, max_tokens)

        if update_history:
            self.history.append((user_text, output))

        return {
            "status": "ok",
            "mode": self.args.mode,
            "ttft_ms": round(ttft_ms, 3),
            "total_ms": round(total_ms, 3),
            "prompt_tokens": len(ids),
            "history_turns": len(self.history),
            "output": output,
            "warmup_ms": round(warmup_ms, 3) if warmup_ms is not None else None,
            "measurement_warmed": warmup,
            **flags,
        }

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if "history" in request:
            self.history = self.normalize_history(request["history"])
            debug(self.args.mode, f"restored {len(self.history)} history turns")
        if command == "health":
            return {
                "status": "ok",
                "mode": self.args.mode,
                "loaded": True,
                "mock": self.args.mock,
                "centroid_armed": bool(self.centroid_armed),
                "history_turns": len(self.history),
            }
        if command == "reset":
            self.history.clear()
            return {"status": "ok", "history_turns": 0}
        if command == "prepare":
            result = self.run_prompt(
                "Warm up the model. Reply with one short sentence.",
                max_tokens=int(request.get("max_tokens", 16)),
                update_history=False,
            )
            result["prepared"] = True
            return result
        if command == "run_turn":
            return self.run_prompt(
                str(request["user"]),
                max_tokens=int(request.get("max_tokens", 128)),
                update_history=True,
                warmup=bool(request.get("warmup", False)),
            )
        if command == "run_single":
            return self.run_prompt(
                str(request["user"]),
                max_tokens=int(request.get("max_tokens", 128)),
                update_history=False,
                warmup=bool(request.get("warmup", False)),
            )
        raise ValueError(f"unknown command: {command!r}")


def main() -> int:
    args = parse_args()
    try:
        debug(args.mode, "worker process starting")
        worker = DemoWorker(args)
        worker.load()
        debug(args.mode, "worker ready")
        emit({
            "event": "ready",
            "mode": args.mode,
            "mock": args.mock,
            "centroid_armed": worker.centroid_armed,
        })
    except Exception as exc:
        debug(args.mode, "startup failed")
        traceback.print_exc(file=sys.stderr)
        emit({"event": "startup_error", "mode": args.mode, "error": str(exc)})
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = worker.handle(request)
            response["id"] = request.get("id")
            emit(response)
        except Exception as exc:
            emit({
                "id": request.get("id") if "request" in locals() else None,
                "status": "error",
                "error": str(exc),
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
