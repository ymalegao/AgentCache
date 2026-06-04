# Live Presentation Demo Plan

## Summary

Build a local web dashboard that demonstrates AgentCache live on presentation
day using a scripted multi-turn agent conversation. The demo compares two
pre-warmed local vLLM-Metal workers:

- `baseline`: full Python-agent system prompt, no centroid injection.
- `centroid`: AgentCache centroid injection, no physical system prompt.

The presenter selects the `csv_cli` conversation, arms the system-prefix modes,
then advances through turns one at a time. For each turn, the dashboard shows
status, prompt-token counts, TTFT, total latency, speedup, cache/prefix state,
and generated outputs side by side.

Primary live-demo target:

| Item | Value |
|---|---|
| Machine | M3 Max, 96 GB RAM |
| Model | `mlx-community/Llama-3.2-1B-Instruct-bf16` |
| System prompt | `agentcache_compression/prompts/2000_python_agent_system.txt` |
| Centroid | `agentcache_compression/centroids/N128_2000_K.npy` and `_V.npy` |
| Centroid length | `N=128` |
| Primary conversation | `agentcache_compression/conversations/csv_cli.json` |
| LMCache | Not used |

## Architecture

Create a new `live_demo/` folder under the vLLM-Metal repo root with three
components:

| File | Role |
|---|---|
| `live_demo/server.py` | Local HTTP server, frontend host, worker manager, JSON API. |
| `live_demo/worker.py` | Long-lived vLLM worker process for one mode. |
| `live_demo/index.html` | Simple browser dashboard for the presentation. |

Run two local worker processes:

| Worker | Runtime setup | Prompt shape |
|---|---|---|
| `baseline` | No centroid environment variables. | `chat_template([system, user])` |
| `centroid` | `VLLM_CENTROID_SCHEDULER=1`, centroid K/V paths, `VLLM_CENTROID_SYS_TOKENS=0`, `VLLM_CENTROID_LAYOUT=compression`. | `[pad] * 128 + chat_template([user])` |

The centroid worker is configured at process startup, but the learned KV prefix
is seeded into the KV cache on each centroid-mode prefill request. The UI should
therefore distinguish between:

- **Centroid armed**: worker env and centroid files are loaded.
- **Prefix injected this turn**: the current request used `[pad] * N` and the
  injector seeded the centroid for that prefill.

The frontend presents the two modes side by side, but the backend should time
them sequentially during each run. This avoids both workers competing for the
same Metal GPU during TTFT measurement while still making the comparison easy
to understand.

The server owns demo session state:

- selected conversation id
- current turn index
- baseline conversation history
- centroid conversation history
- last successful per-turn metrics
- whether the prefix modes have been warmed/verified

## Backend API

Expose these local-only endpoints from `server.py`:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | `GET` | Return worker status, model loaded state, centroid enabled state, and last error. |
| `/api/config` | `GET` | Return model, prompt file, centroid paths, centroid length, and available conversations. |
| `/api/session/reset` | `POST` | Select a conversation, clear both histories, reset to turn 1, and clear metrics. |
| `/api/prefix/prepare` | `POST` | Warm both workers and verify baseline system prompt / centroid prefix mode. |
| `/api/turn/current` | `GET` | Return current turn text, turn number, and history summary. |
| `/api/turn/run` | `POST` | Run the current turn through both workers and return metrics and outputs. |
| `/api/turn/advance` | `POST` | Advance to the next scripted prompt after the current turn has run. |
| `/api/run_single` | `POST` | Optional fallback: run one ad hoc prompt outside the scripted conversation. |

`POST /api/session/reset` request body:

```json
{
  "conversation": "csv_cli",
  "max_tokens": 256
}
```

`POST /api/turn/run` request body:

```json
{
  "turn": 1,
  "max_tokens": 128
}
```

`POST /api/turn/run` response shape:

```json
{
  "conversation": "csv_cli",
  "turn": 1,
  "user": "Write a Python CLI script using argparse...",
  "baseline": {
    "status": "ok",
    "ttft_ms": 29.4,
    "total_ms": 820.1,
    "prompt_tokens": 2216,
    "synthetic_tokens": 0,
    "system_prompt_sent": true,
    "prefix_injected": false,
    "output": "..."
  },
  "centroid": {
    "status": "ok",
    "ttft_ms": 23.2,
    "total_ms": 735.6,
    "prompt_tokens": 173,
    "synthetic_tokens": 128,
    "system_prompt_sent": false,
    "prefix_injected": true,
    "output": "..."
  },
  "speedup": {
    "ttft_ratio": 1.27,
    "ttft_saved_ms": 6.2,
    "prompt_tokens_saved": 2043
  }
}
```

## Frontend Behavior

The dashboard should be intentionally simple and presentation-safe:

- Header with model, centroid, and machine metadata.
- Worker health indicators for `baseline` and `centroid`.
- Conversation selector:
  - `csv_cli` as the default and recommended path.
  - `single prompt` as a fallback mode.
- Turn controller:
  - current turn number, total turns, and current user prompt.
  - previous/next scripted prompt preview.
  - disabled state until the session is reset and workers are ready.
- Buttons:
  - **Reset Conversation**: clears both histories and returns to turn 1.
  - **Prepare System Prefix**: runs worker warmup and verifies both prefix modes.
  - **Run Current Turn**: runs the current scripted prompt through both workers.
  - **Continue to Next Turn**: advances only after both outputs are captured.
  - **Auto Run Remaining**: optional rehearsal-only control; keep hidden or
    secondary during the actual presentation.
- Metrics cards:
  - TTFT
  - total latency
  - prompt tokens
  - synthetic tokens
  - tokens saved
  - TTFT speedup
  - prefix state: `system prompt sent` vs `centroid injected`
- Side-by-side generated outputs.
- Per-turn timeline chart or table with TTFT and prompt tokens by turn.
- Event log showing:
  - worker loaded
  - prefix prepared
  - baseline sent full system prompt
  - centroid injected learned prefix
  - turn completed
- Hidden fallback panel for cached last-good results if a worker fails.

Primary scripted conversation:

```text
agentcache_compression/conversations/csv_cli.json
```

This is a 10-turn coding-agent workflow. It starts with a CSV CLI request and
then incrementally adds filtering, output, refactoring, validation, sampling,
verbose logging, tests, schema validation, and packaging.

Fallback single prompts:

1. `Write a Python function that reverses a string.`
2. `Create a CLI with argparse that reads a CSV and prints column summaries.`
3. `Explain how to add retries with exponential backoff to an HTTP request.`

## Demo Flow

Before the presentation:

1. Confirm the model and centroid files are already downloaded locally.
2. Run the RoPE/layout gate:

   ```bash
   python tools/centroid_rope_parity.py \
     --model mlx-community/Llama-3.2-1B-Instruct-bf16 \
     --centroid-k /abs/path/to/agentcache_compression/centroids/N128_2000_K.npy \
     --centroid-v /abs/path/to/agentcache_compression/centroids/N128_2000_V.npy \
     --sys-tokens 0
   ```

3. Start the local demo server from the vLLM-Metal root.
4. Open the dashboard in the browser.
5. Select `csv_cli`.
6. Click **Reset Conversation**.
7. Click **Prepare System Prefix** and confirm:
   - baseline is ready to send the full system prompt.
   - centroid worker is armed with `N=128`.
   - a warmup request succeeds for both workers.

During the presentation:

1. Explain the baseline path: every request physically prefills the long system
   prompt.
2. Explain the centroid path: the system prompt is replaced with a fixed learned
   KV prefix. The worker is armed at startup, and the prefix is injected when
   each centroid turn runs.
3. Show turn 1 of `csv_cli` and click **Run Current Turn**.
4. Point out:
   - baseline prompt-token count is much larger.
   - centroid prompt-token count is roughly `128 + user tokens`.
   - the UI marks `system prompt sent` for baseline.
   - the UI marks `centroid injected` for centroid.
   - TTFT is lower in the centroid path.
   - outputs are generated live by the local model.
5. Click **Continue to Next Turn**.
6. Run turns 2 and 3 to show prompt/history growth.
7. Point out that both modes maintain their own conversation history, while only
   the baseline keeps paying for the physical system prompt.
8. If time is short, stop after 3 turns; if time allows, run more turns or show
   the per-turn TTFT chart.

## Test Plan

Preflight checks:

- `GET /api/health` returns both workers as ready.
- Baseline worker reports centroid disabled.
- Centroid worker reports centroid enabled.
- `POST /api/session/reset` loads `csv_cli` and returns turn 1.
- `POST /api/prefix/prepare` succeeds for both workers.
- `POST /api/turn/run` returns non-empty outputs for both modes.
- `POST /api/turn/advance` moves to the next scripted prompt only after a turn
  has been run.

Metrics checks:

- On turn 1, baseline prompt tokens are much larger than centroid prompt tokens.
- On later turns, both modes include growing conversation history.
- `ttft_ms`, `total_ms`, and speedup values are numeric.
- `prompt_tokens_saved` is positive.
- Baseline result has `system_prompt_sent=true`.
- Centroid result has `prefix_injected=true`.
- The UI does not crash if one worker returns an error.

Presentation rehearsal:

- Run the exact multi-turn demo sequence twice before presentation day.
- Rehearse stopping after turn 3 and after turn 5, so there is a short and long
  version depending on time.
- Keep terminal logs available but minimized.
- Keep a saved last-good result available for the fallback panel.
- Avoid changing model, centroid, prompt file, or environment on presentation
  day.

## Assumptions

- The live demo runs on the M3 Max 96 GB Mac.
- The primary demo model is `mlx-community/Llama-3.2-1B-Instruct-bf16`.
- The primary demo centroid is `N128_2000_K/V.npy`.
- The primary demo conversation is `csv_cli.json`.
- The demo does not use LMCache.
- The dashboard is local-only and does not need authentication.
- The first implementation should optimize for reliability and clarity, not
  visual polish.
