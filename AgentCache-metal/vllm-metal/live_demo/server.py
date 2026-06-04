#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local dashboard server for the AgentCache live presentation demo."""

from __future__ import annotations

import argparse
import errno
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


DEMO_ROOT = Path(__file__).resolve().parent
METAL_ROOT = DEMO_ROOT.parent
STATIC_ROOT = DEMO_ROOT / "static"
DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-bf16"
DEFAULT_MAX_TOKENS = 192
DEFAULT_MAX_NUM_SEQS = 1
DEFAULT_EXECUTION_MODE = "parallel"
DEFAULT_METAL_MEMORY_FRACTION = "0.10"
CANNED_PROMPTS = [
    "Write a Python function that reverses a string.",
    "Create a CLI with argparse that reads a CSV and prints column summaries.",
    "Explain how to add retries with exponential backoff to an HTTP request.",
]


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[live_demo {timestamp}] {message}", file=sys.stderr, flush=True)


def find_agentcache_root() -> Path:
    env_root = os.environ.get("AGENTCACHE_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "agentcache_compression").is_dir():
            return root
        raise FileNotFoundError(
            f"AGENTCACHE_ROOT={root} does not contain agentcache_compression/"
        )
    for candidate in (METAL_ROOT, *METAL_ROOT.parents):
        if (candidate / "agentcache_compression").is_dir():
            return candidate
    raise FileNotFoundError("Could not find agentcache_compression/")


AGENTCACHE_ROOT = find_agentcache_root()
COMPRESSION_ROOT = AGENTCACHE_ROOT / "agentcache_compression"
PROMPT_PATH = COMPRESSION_ROOT / "prompts" / "2000_python_agent_system.txt"
CONVERSATIONS_ROOT = COMPRESSION_ROOT / "conversations"
CENTROID_K = COMPRESSION_ROOT / "centroids" / "N128_2000_K.npy"
CENTROID_V = COMPRESSION_ROOT / "centroids" / "N128_2000_V.npy"
VENV_PY = METAL_ROOT / ".venv-vllm-metal" / "bin" / "python"


def default_python() -> str:
    return str(VENV_PY if VENV_PY.exists() else Path(sys.executable))


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def load_conversations() -> dict[str, list[str]]:
    conversations: dict[str, list[str]] = {}
    for path in sorted(CONVERSATIONS_ROOT.glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            conversations[path.stem] = data
    if "csv_cli" not in conversations:
        raise FileNotFoundError(CONVERSATIONS_ROOT / "csv_cli.json")
    return conversations


class WorkerClient:
    def __init__(
        self,
        *,
        name: str,
        mode: str,
        python: str,
        model: str,
        n: int,
        max_model_len: int,
        max_num_seqs: int,
        metal_memory_fraction: str | None,
        mock: bool,
        env_extra: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.mode = mode
        self.ready = False
        self.startup_error: str | None = None
        self.last_error: str | None = None
        self.last_event: dict[str, Any] | None = None
        self.stdout_log: deque[str] = deque(maxlen=80)
        self.stderr_log: deque[str] = deque(maxlen=120)
        self.responses: dict[str, dict[str, Any]] = {}
        self.condition = threading.Condition()
        self.stopping = False

        env = os.environ.copy()
        if VENV_PY.exists():
            env["PATH"] = f"{VENV_PY.parent}{os.pathsep}{env.get('PATH', '')}"
        env["PYTHONPATH"] = (
            f"{METAL_ROOT}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(METAL_ROOT)
        )
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        env.setdefault("VLLM_LOGGING_STREAM", "ext://sys.stderr")
        if metal_memory_fraction and metal_memory_fraction.lower() != "auto":
            env["VLLM_METAL_MEMORY_FRACTION"] = metal_memory_fraction
        else:
            env.pop("VLLM_METAL_MEMORY_FRACTION", None)
        for key in list(env):
            if mode == "baseline" and key.startswith("VLLM_CENTROID_"):
                env.pop(key, None)
        if env_extra:
            env.update(env_extra)

        cmd = [
            python,
            str(DEMO_ROOT / "worker.py"),
            "--mode",
            mode,
            "--model",
            model,
            "--system-prompt",
            str(PROMPT_PATH),
            "--n",
            str(n),
            "--max-model-len",
            str(max_model_len),
            "--max-num-seqs",
            str(max_num_seqs),
        ]
        if mock:
            cmd.append("--mock")

        log(
            f"starting {self.name} worker "
            f"(mode={mode}, model={model}, n={n}, max_model_len={max_model_len}, "
            f"max_num_seqs={max_num_seqs}, "
            f"metal_memory_fraction={env.get('VLLM_METAL_MEMORY_FRACTION', 'auto')}, "
            f"mock={mock})"
        )
        self.proc = subprocess.Popen(
            cmd,
            cwd=METAL_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        log(f"{self.name} worker pid={self.proc.pid}")
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self.stdout_log.append(line)
                continue
            with self.condition:
                if payload.get("event") == "ready":
                    self.ready = True
                    self.last_event = payload
                    log(
                        f"{self.name} ready "
                        f"(mode={payload.get('mode')}, mock={payload.get('mock')}, "
                        f"centroid_armed={payload.get('centroid_armed')})"
                    )
                elif payload.get("event") == "startup_error":
                    self.startup_error = str(payload.get("error", "startup error"))
                    self.last_error = self.startup_error
                    self.last_event = payload
                    log(f"{self.name} startup error: {self.startup_error}")
                elif payload.get("id"):
                    self.responses[str(payload["id"])] = payload
                    if payload.get("status") == "error":
                        self.last_error = str(payload.get("error", "worker error"))
                        log(f"{self.name} request error: {self.last_error}")
                else:
                    self.last_event = payload
                    if payload.get("event"):
                        log(f"{self.name} event: {payload}")
                self.condition.notify_all()
        with self.condition:
            if self.proc.poll() is not None and not self.startup_error and not self.stopping:
                self.last_error = f"worker exited with code {self.proc.returncode}"
                log(f"{self.name} {self.last_error}")
            self.condition.notify_all()

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            line = line.rstrip()
            if line:
                self.stderr_log.append(line)
                log(f"{self.name}: {line}")

    def wait_until_ready(self, timeout: float = 900.0) -> None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.ready:
                if self.startup_error:
                    raise RuntimeError(f"{self.name} startup failed: {self.startup_error}")
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"{self.name} worker exited before ready with code {self.proc.returncode}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timeout waiting for {self.name} worker to load")
                self.condition.wait(timeout=min(remaining, 1.0))

    def request(self, command: str, payload: dict[str, Any] | None = None, timeout: float = 600.0) -> dict[str, Any]:
        if self.proc.poll() is not None:
            raise RuntimeError(f"{self.name} worker is not running")
        if not self.ready and command != "health":
            raise RuntimeError(f"{self.name} worker is not ready")
        req_id = uuid.uuid4().hex
        request = {"id": req_id, "command": command}
        if payload:
            request.update(payload)
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()

        deadline = time.monotonic() + timeout
        with self.condition:
            while req_id not in self.responses:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timeout waiting for {self.name} {command}")
                self.condition.wait(timeout=remaining)
            return self.responses.pop(req_id)

    def health(self, *, ping_worker: bool = True) -> dict[str, Any]:
        alive = self.proc.poll() is None
        payload = {
            "name": self.name,
            "mode": self.mode,
            "alive": alive,
            "ready": self.ready,
            "startup_error": self.startup_error,
            "last_error": self.last_error,
            "last_event": self.last_event,
            "stderr_tail": list(self.stderr_log)[-8:],
        }
        if ping_worker and alive and self.ready:
            try:
                payload["worker"] = self.request("health", timeout=10.0)
            except Exception as exc:
                payload["last_error"] = str(exc)
        return payload

    def stop(self) -> None:
        if self.proc.poll() is None:
            log(f"stopping {self.name} worker pid={self.proc.pid}")
            self.stopping = True
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log(f"{self.name} did not terminate; killing pid={self.proc.pid}")
                self.proc.kill()
                self.proc.wait(timeout=10)
            log(f"{self.name} stopped with code {self.proc.returncode}")
        if self.stopping and self.proc.returncode in (-15, 0):
            self.last_error = None


class DemoState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.conversations = load_conversations()
        self.conversation_id = "csv_cli"
        self.turn_index = 0
        self.max_tokens = args.default_max_tokens
        self.turn_has_run = False
        self.prepared = False
        self.timeline: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] | None = None
        self.histories: dict[str, list[tuple[str, str]]] = {
            "baseline": [],
            "centroid": [],
        }
        self.workers: dict[str, WorkerClient] = {}
        self.active_worker: WorkerClient | None = None
        self.last_worker_health: dict[str, dict[str, Any]] = {
            "baseline": self._stopped_worker_health("baseline"),
            "centroid": self._stopped_worker_health("centroid"),
        }
        self.lock = threading.Lock()

    @staticmethod
    def _stopped_worker_health(mode: str) -> dict[str, Any]:
        return {
            "name": mode,
            "mode": mode,
            "alive": False,
            "ready": False,
            "startup_error": None,
            "last_error": None,
            "last_event": {"event": "stopped", "mode": mode},
            "stderr_tail": [],
        }

    def _centroid_env(self) -> dict[str, str]:
        return {
            "VLLM_CENTROID_SCHEDULER": "1",
            "VLLM_CENTROID_K_PATH": str(CENTROID_K),
            "VLLM_CENTROID_V_PATH": str(CENTROID_V),
            "VLLM_CENTROID_SYS_TOKENS": "0",
            "VLLM_CENTROID_LAYOUT": "compression",
        }

    def _new_worker(self, mode: str) -> WorkerClient:
        return WorkerClient(
            name=mode,
            mode=mode,
            python=self.args.python,
            model=self.args.model,
            n=self.args.n,
            max_model_len=self.args.max_model_len,
            max_num_seqs=self.args.max_num_seqs,
            metal_memory_fraction=self.args.metal_memory_fraction,
            mock=self.args.mock,
            env_extra=self._centroid_env() if mode == "centroid" else None,
        )

    def _record_worker_health(self, worker: WorkerClient, *, stopped: bool) -> None:
        health = worker.health(ping_worker=False)
        if stopped:
            health["alive"] = False
            health["ready"] = False
        self.last_worker_health[worker.name] = health

    def _run_mode(
        self,
        mode: str,
        command: str,
        payload: dict[str, Any],
        *,
        load_timeout: float = 900.0,
        request_timeout: float = 900.0,
    ) -> dict[str, Any]:
        log(f"sequential mode: loading {mode} worker for {command}")
        worker = self._new_worker(mode)
        self.active_worker = worker
        self.last_worker_health[mode] = worker.health(ping_worker=False)
        try:
            worker.wait_until_ready(timeout=load_timeout)
            result = worker.request(command, payload, timeout=request_timeout)
            if result.get("status") == "error":
                raise RuntimeError(f"{mode} {command} failed: {result.get('error')}")
            return result
        finally:
            self._record_worker_health(worker, stopped=False)
            worker.stop()
            self._record_worker_health(worker, stopped=True)
            if self.active_worker is worker:
                self.active_worker = None
            log(f"sequential mode: {mode} worker finished {command}")

    def _ensure_persistent_worker(self, mode: str) -> WorkerClient:
        worker = self.workers.get(mode)
        if worker is not None and worker.proc.poll() is None and worker.ready:
            return worker
        if worker is not None:
            self._record_worker_health(worker, stopped=worker.proc.poll() is not None)
            worker.stop()
            self.workers.pop(mode, None)

        log(f"parallel mode: loading persistent {mode} worker")
        worker = self._new_worker(mode)
        self.active_worker = worker
        self.last_worker_health[mode] = worker.health(ping_worker=False)
        try:
            worker.wait_until_ready(timeout=900.0)
            self.workers[mode] = worker
            self._record_worker_health(worker, stopped=False)
            return worker
        finally:
            if self.active_worker is worker:
                self.active_worker = None

    def _request_mode(
        self,
        mode: str,
        command: str,
        payload: dict[str, Any],
        *,
        request_timeout: float = 900.0,
    ) -> dict[str, Any]:
        if self.args.execution_mode == "sequential":
            return self._run_mode(
                mode,
                command,
                payload,
                request_timeout=request_timeout,
            )

        worker = self._ensure_persistent_worker(mode)
        result = worker.request(command, payload, timeout=request_timeout)
        if result.get("status") == "error":
            raise RuntimeError(f"{mode} {command} failed: {result.get('error')}")
        self._record_worker_health(worker, stopped=False)
        return result

    def current_prompt(self) -> str:
        return self.conversations[self.conversation_id][self.turn_index]

    def current_payload(self) -> dict[str, Any]:
        prompts = self.conversations[self.conversation_id]
        return {
            "conversation": self.conversation_id,
            "turn": self.turn_index + 1,
            "total_turns": len(prompts),
            "user": prompts[self.turn_index],
            "previous": prompts[self.turn_index - 1] if self.turn_index > 0 else None,
            "next": prompts[self.turn_index + 1] if self.turn_index + 1 < len(prompts) else None,
            "turn_has_run": self.turn_has_run,
            "prepared": self.prepared,
            "timeline": self.timeline,
            "execution_mode": self.args.execution_mode,
        }

    def reset(self, conversation_id: str, max_tokens: int | None) -> dict[str, Any]:
        if conversation_id not in self.conversations:
            raise ValueError(f"unknown conversation: {conversation_id}")
        with self.lock:
            self.conversation_id = conversation_id
            self.turn_index = 0
            self.max_tokens = max_tokens or self.args.default_max_tokens
            self.turn_has_run = False
            self.prepared = False
            self.timeline = []
            self.last_result = None
            self.histories = {"baseline": [], "centroid": []}
            return self.current_payload()

    def prepare(self) -> dict[str, Any]:
        with self.lock:
            results = {}
            for mode in ("baseline", "centroid"):
                results[mode] = self._request_mode(
                    mode,
                    "prepare",
                    {"max_tokens": 16, "history": self.histories[mode]},
                    request_timeout=900.0,
                )
            self.prepared = all(r.get("status") == "ok" for r in results.values())
            return {"prepared": self.prepared, "workers": results, **self.current_payload()}

    def run_turn(self, max_tokens: int | None = None) -> dict[str, Any]:
        with self.lock:
            user = self.current_prompt()
            tokens = max_tokens or self.max_tokens
            baseline = self._request_mode(
                "baseline",
                "run_turn",
                {
                    "user": user,
                    "max_tokens": tokens,
                    "turn": self.turn_index + 1,
                    "history": self.histories["baseline"],
                    "warmup": self.args.measurement_warmup,
                },
                request_timeout=900.0,
            )
            centroid = self._request_mode(
                "centroid",
                "run_turn",
                {
                    "user": user,
                    "max_tokens": tokens,
                    "turn": self.turn_index + 1,
                    "history": self.histories["centroid"],
                    "warmup": self.args.measurement_warmup,
                },
                request_timeout=900.0,
            )
            speedup = compute_speedup(baseline, centroid)
            result = {
                "conversation": self.conversation_id,
                "turn": self.turn_index + 1,
                "total_turns": len(self.conversations[self.conversation_id]),
                "user": user,
                "baseline": baseline,
                "centroid": centroid,
                "speedup": speedup,
            }
            self.turn_has_run = True
            self.histories["baseline"].append((user, str(baseline.get("output") or "")))
            self.histories["centroid"].append((user, str(centroid.get("output") or "")))
            self.last_result = result
            self.timeline.append({
                "turn": self.turn_index + 1,
                "baseline_ttft_ms": baseline.get("ttft_ms"),
                "centroid_ttft_ms": centroid.get("ttft_ms"),
                "baseline_prompt_tokens": baseline.get("prompt_tokens"),
                "centroid_prompt_tokens": centroid.get("prompt_tokens"),
                "ttft_ratio": speedup.get("ttft_ratio"),
            })
            result["timeline"] = self.timeline
            return result

    def advance(self) -> dict[str, Any]:
        with self.lock:
            if not self.turn_has_run:
                raise RuntimeError("run the current turn before advancing")
            total = len(self.conversations[self.conversation_id])
            if self.turn_index + 1 >= total:
                return {"at_end": True, **self.current_payload()}
            self.turn_index += 1
            self.turn_has_run = False
            return {"at_end": False, **self.current_payload()}

    def run_single(self, prompt: str, max_tokens: int | None) -> dict[str, Any]:
        tokens = max_tokens or self.max_tokens
        with self.lock:
            baseline = self._request_mode(
                "baseline",
                "run_single",
                {
                    "user": prompt,
                    "max_tokens": tokens,
                    "history": self.histories["baseline"],
                    "warmup": self.args.measurement_warmup,
                },
                request_timeout=900.0,
            )
            centroid = self._request_mode(
                "centroid",
                "run_single",
                {
                    "user": prompt,
                    "max_tokens": tokens,
                    "history": self.histories["centroid"],
                    "warmup": self.args.measurement_warmup,
                },
                request_timeout=900.0,
            )
            return {
                "conversation": "single",
                "turn": None,
                "user": prompt,
                "baseline": baseline,
                "centroid": centroid,
                "speedup": compute_speedup(baseline, centroid),
            }

    def health(self) -> dict[str, Any]:
        workers = dict(self.last_worker_health)
        for mode, worker in self.workers.items():
            workers[mode] = worker.health(ping_worker=False)
        if self.active_worker is not None:
            workers[self.active_worker.name] = self.active_worker.health(ping_worker=False)
        return {
            "server": "ok",
            "mock": self.args.mock,
            "execution_mode": self.args.execution_mode,
            "active_worker": self.active_worker.name if self.active_worker else None,
            "workers": workers,
        }

    def stop(self) -> None:
        if self.active_worker is not None:
            self.active_worker.stop()
        for worker in list(self.workers.values()):
            worker.stop()
        self.workers.clear()


def compute_speedup(baseline: dict[str, Any], centroid: dict[str, Any]) -> dict[str, Any]:
    b_ttft = float(baseline.get("ttft_ms") or 0.0)
    c_ttft = float(centroid.get("ttft_ms") or 0.0)
    b_tokens = int(baseline.get("prompt_tokens") or 0)
    c_tokens = int(centroid.get("prompt_tokens") or 0)
    return {
        "ttft_ratio": round(b_ttft / c_ttft, 3) if c_ttft > 0 else None,
        "ttft_saved_ms": round(b_ttft - c_ttft, 3),
        "prompt_tokens_saved": b_tokens - c_tokens,
    }


class Handler(BaseHTTPRequestHandler):
    server: "DemoHTTPServer"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[live_demo] " + fmt % args + "\n")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            return json_response(self, 200, self.server.api_health())
        if parsed.path == "/api/config":
            return json_response(self, 200, self.server.api_config())
        if parsed.path == "/api/turn/current":
            return json_response(self, 200, self.server.state.current_payload())
        return self.serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = read_json(self)
            if parsed.path == "/api/session/reset":
                payload = self.server.state.reset(
                    str(body.get("conversation") or "csv_cli"),
                    int(body["max_tokens"]) if body.get("max_tokens") else None,
                )
                return json_response(self, 200, payload)
            if parsed.path == "/api/prefix/prepare":
                return json_response(self, 200, self.server.state.prepare())
            if parsed.path == "/api/turn/run":
                max_tokens = int(body["max_tokens"]) if body.get("max_tokens") else None
                return json_response(self, 200, self.server.state.run_turn(max_tokens))
            if parsed.path == "/api/turn/advance":
                return json_response(self, 200, self.server.state.advance())
            if parsed.path == "/api/run_single":
                prompt = str(body.get("prompt") or "").strip()
                if not prompt:
                    return json_response(self, 400, {"error": "prompt is required"})
                max_tokens = int(body["max_tokens"]) if body.get("max_tokens") else None
                return json_response(self, 200, self.server.state.run_single(prompt, max_tokens))
            return json_response(self, 404, {"error": "not found"})
        except Exception as exc:
            return json_response(self, 500, {"error": str(exc)})

    def serve_static(self, raw_path: str) -> None:
        path = "/" if raw_path == "" else raw_path
        if path == "/":
            file_path = DEMO_ROOT / "index.html"
        elif path == "/v2":
            file_path = DEMO_ROOT / "index_v2.html"
        else:
            rel = Path(unquote(path.lstrip("/")))
            if rel.parts and rel.parts[0] in ("static", "static_v2"):
                file_path = DEMO_ROOT / rel
            else:
                return json_response(self, 404, {"error": "not found"})

        try:
            file_path.resolve().relative_to(DEMO_ROOT.resolve())
        except ValueError:
            return json_response(self, 403, {"error": "forbidden"})
        if not file_path.exists() or not file_path.is_file():
            return json_response(self, 404, {"error": "not found"})

        body = file_path.read_bytes()
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class DemoHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], state: DemoState | None) -> None:
        super().__init__(server_address, Handler)
        self.state = state

    def api_health(self) -> dict[str, Any]:
        assert self.state is not None
        return self.state.health()

    def api_config(self) -> dict[str, Any]:
        assert self.state is not None
        return {
            "model": self.state.args.model,
            "max_model_len": self.state.args.max_model_len,
            "max_num_seqs": self.state.args.max_num_seqs,
            "execution_mode": self.state.args.execution_mode,
            "default_max_tokens": self.state.args.default_max_tokens,
            "measurement_warmup": self.state.args.measurement_warmup,
            "metal_memory_fraction": self.state.args.metal_memory_fraction,
            "system_prompt": str(PROMPT_PATH),
            "centroid_k": str(CENTROID_K),
            "centroid_v": str(CENTROID_V),
            "n": self.state.args.n,
            "conversations": {
                name: {"turns": len(turns), "prompts": turns}
                for name, turns in self.state.conversations.items()
            },
            "default_conversation": "csv_cli",
            "canned_prompts": CANNED_PROMPTS,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgentCache live demo dashboard server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    p.add_argument(
        "--execution-mode",
        choices=["parallel", "sequential"],
        default=DEFAULT_EXECUTION_MODE,
        help=(
            "parallel keeps baseline and centroid workers loaded after Prepare; "
            "sequential loads and stops one worker per request."
        ),
    )
    p.add_argument(
        "--metal-memory-fraction",
        default=os.environ.get("VLLM_METAL_MEMORY_FRACTION", DEFAULT_METAL_MEMORY_FRACTION),
        help=(
            "Value for VLLM_METAL_MEMORY_FRACTION in demo workers. "
            "Use 'auto' to leave vLLM-Metal's default allocation behavior enabled."
        ),
    )
    p.add_argument(
        "--no-measurement-warmup",
        action="store_false",
        dest="measurement_warmup",
        help="Disable the hidden one-token warmup before each reported TTFT measurement.",
    )
    p.add_argument("--default-max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--python", default=default_python())
    p.add_argument("--mock", action="store_true", help="Run without loading vLLM/model")
    p.set_defaults(measurement_warmup=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log(
        f"starting server host={args.host} port={args.port} model={args.model} "
        f"n={args.n} max_model_len={args.max_model_len} "
        f"max_num_seqs={args.max_num_seqs} execution_mode={args.execution_mode} "
        f"metal_memory_fraction={args.metal_memory_fraction} "
        f"measurement_warmup={args.measurement_warmup} mock={args.mock}"
    )
    for path in (PROMPT_PATH, CENTROID_K, CENTROID_V):
        if not path.exists():
            raise FileNotFoundError(path)
    log(f"system prompt: {PROMPT_PATH}")
    log(f"centroid K: {CENTROID_K}")
    log(f"centroid V: {CENTROID_V}")
    state: DemoState | None = None
    try:
        server = DemoHTTPServer((args.host, args.port), None)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                f"Port {args.host}:{args.port} is already in use. Stop the existing demo "
                f"or run with --port {args.port + 1}. To find it: "
                f"lsof -nP -iTCP:{args.port} -sTCP:LISTEN",
                file=sys.stderr,
            )
            return 1
        raise

    state = DemoState(args)
    server.state = state
    url = f"http://{args.host}:{args.port}"
    print(f"AgentCache live demo: {url}")
    if args.mock:
        print("mock mode is enabled; no model will be loaded")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        if state is not None:
            state.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
