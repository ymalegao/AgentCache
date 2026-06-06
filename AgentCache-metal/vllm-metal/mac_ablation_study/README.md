# Mac Ablation Study

This folder contains the runnable Mac/vLLM-Metal ablation study for AgentCache.

The notebook is the report and orchestration layer. The scripts are the source
of benchmark execution and should be run as fresh subprocesses so vLLM-Metal,
Metal memory state, and centroid environment variables are isolated per mode.

## Layout

| Path | Purpose |
|---|---|
| `Mac_Ablation_Study.ipynb` | Notebook report/orchestrator for the study. |
| `scripts/` | Runner scripts that invoke the existing vLLM-Metal tools. |
| `results/` | JSON/JSONL outputs written by the runners. |

## Main Scripts

| Script | Purpose |
|---|---|
| `scripts/train_centroid_cuda.py` | Runs CUDA prefix training and exports `centroid_K.npy` / `centroid_V.npy`. |
| `scripts/run_rope_parity.py` | Validates centroid RoPE/layout for a model before benchmark runs. |
| `scripts/run_cold_inject.py` | Runs cold full-prompt vs centroid-injected TTFT/quality. |
| `scripts/run_longctx.py` | Runs synthetic long-context cold TTFT sweep and saves parsed JSON. |
| `scripts/run_model_sweep.py` | Optional TTFT-only model-size sweep using real or dummy centroids. |
| `scripts/run_prefix_cache_multiturn.py` | Optional multi-turn cold/native-prefix-cache/synthetic comparison. |

## Recommended Flow

1. Open `Mac_Ablation_Study.ipynb`.
2. Keep `RUN_LIVE = False` to inspect cached outputs and commands.
3. Set `RUN_LIVE = True` when ready to regenerate measurements.
4. Run real-centroid quality experiments only for models with matching trained
   centroids. Use dummy centroids for TTFT-only model-size timing.

## CUDA Centroid Training

On a CUDA machine, train and export a real centroid with:

```bash
python mac_ablation_study/scripts/train_centroid_cuda.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --tokens 128 \
  --batch-size 1
```

The script writes a self-contained directory under
`agentcache_compression/centroids/` containing `centroid_K.npy`,
`centroid_V.npy`, `sys_prefix_num_tokens.txt`, and `metadata.json`. Copy or sync
that directory back to the Mac and pass the two `.npy` paths to
`run_cold_inject.py`.
