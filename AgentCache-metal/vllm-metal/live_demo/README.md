# AgentCache Live Demo

Local browser dashboard for the presentation-day AgentCache demo.

## Mock Smoke Test

Use mock mode to validate the dashboard without loading vLLM or the model:

```bash
python live_demo/server.py --mock
```

Open:

```text
http://127.0.0.1:8765
```

## Real Demo

Run from the vLLM-Metal repo root:

```bash
python live_demo/server.py
```

The demo defaults to `--max-num-seqs 1` because the presentation UI sends one
request at a time and running two Metal engines with vLLM's larger default batch
capacity can over-allocate KV/cache memory. If port `8765` is already occupied,
stop the old process or run with `--port 8766`.

The demo also defaults to `--metal-memory-fraction 0.10`. vLLM-Metal's paged
attention path otherwise allocates a very large KV cache by default, which is
useful for throughput testing but noisy for a one-request live presentation. Use
`--metal-memory-fraction auto` if you intentionally want vLLM-Metal's default
allocation behavior.

The server defaults to `--execution-mode parallel`: **Prepare System Prefix**
loads baseline and centroid once, then keeps both workers alive for the live
demo. That makes the first prepare step slow, but later turns avoid the
load/shutdown cost and show the TTFT comparison quickly. If memory pressure is a
problem, run with `--execution-mode sequential`; that mode loads baseline, runs
the prompt, stops baseline, then does the same for centroid on every request.
Conversation history is kept by the dashboard server and restored into each
worker.

Each reported TTFT is warmed by default. The worker runs one hidden one-token
request after engine startup and before the measured prompt so first-use Metal
kernel setup and centroid injector setup do not dominate the presentation
number. Disable this with `--no-measurement-warmup` only when you explicitly
want cold-request behavior.

Demo workers also default to `VLLM_WORKER_MULTIPROC_METHOD=spawn`. This avoids
forking a Metal worker from a Python process that has already imported
tokenizer/MLX/vLLM dependencies, which is fragile on macOS.
The server also adds the vLLM-Metal repo root to `PYTHONPATH` so spawned vLLM
worker processes can import the local `vllm_metal` plugin instead of falling
back to CPU.

The server starts one worker subprocess at a time:

- `baseline`: full Python-agent system prompt.
- `centroid`: `N=128` centroid injection using `N128_2000_K/V.npy`.

The terminal shows labeled load/debug logs for both workers, including tokenizer
load, vLLM engine load, centroid env settings, ready state, startup failures,
and worker request errors.

The default scripted conversation is:

```text
agentcache_compression/conversations/csv_cli.json
```

Before presenting, run the RoPE/layout gate for the selected centroid:

```bash
python tools/centroid_rope_parity.py \
  --model mlx-community/Llama-3.2-1B-Instruct-bf16 \
  --centroid-k /abs/path/to/agentcache_compression/centroids/N128_2000_K.npy \
  --centroid-v /abs/path/to/agentcache_compression/centroids/N128_2000_V.npy \
  --sys-tokens 0
```

## Controls

- **Reset Conversation** clears both worker histories and returns to turn 1.
- **Prepare System Prefix** runs a warmup request and verifies both modes.
- **Run Current Turn** runs the active prompt through baseline then centroid.
- **Continue to Next Turn** advances only after a turn has been run.
- **Single Prompt Fallback** runs an ad hoc prompt without changing conversation
  history.
