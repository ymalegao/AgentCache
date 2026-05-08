# SPDX-License-Identifier: Apache-2.0
"""Optional centroid KV warm-start hooks shared by V1 and V2 GPU model runners."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)

# ---------- Scheduler-side centroid gap (Path B) ----------
# These are module-level caches so the gate + sys-count resolution runs once
# per process, not on every schedule() call.
_centroid_sched_enabled: bool | None = None
_centroid_sched_sys_count: int = 0


def _centroid_sched_check_once() -> tuple[bool, int]:
    """Return (enabled, sys_token_count). Reads env / sidecar files once and caches.

    Scheduler-safe: no torch tensors created, no numpy arrays loaded — just file
    I/O and env-var reads.  Import of CentroidInjector helpers (pure Python +
    pathlib) is deferred to avoid circular-import issues at module load time.
    """
    global _centroid_sched_enabled, _centroid_sched_sys_count
    if _centroid_sched_enabled is not None:
        return _centroid_sched_enabled, _centroid_sched_sys_count

    if os.environ.get("VLLM_CENTROID_SCHEDULER", "0") != "1":
        _centroid_sched_enabled = False
        _centroid_sched_sys_count = 0
        return False, 0

    k, _ = centroid_paths()
    if not os.path.exists(k):
        logger.info(
            "[CENTROID] VLLM_CENTROID_SCHEDULER=1 but centroid K file not found: %s"
            " — scheduler mode disabled.",
            k,
        )
        _centroid_sched_enabled = False
        _centroid_sched_sys_count = 0
        return False, 0

    try:
        from vllm.centroid_injector import load_sys_prefix_token_count

        n = load_sys_prefix_token_count(k)
        
        # Get centroid sequence length safely
        import numpy as np
        k_np = np.load(k, mmap_mode='r')
        centroid_len = k_np.shape[1] if k_np.ndim == 3 else 1
        
        total_len = n + centroid_len
    except Exception:
        logger.exception(
            "[CENTROID] Failed to load sys_token_count or centroid shape; scheduler mode disabled."
        )
        _centroid_sched_enabled = False
        _centroid_sched_sys_count = 0
        return False, 0

    _centroid_sched_enabled = True
    _centroid_sched_sys_count = total_len
    logger.info(
        "[CENTROID] Scheduler mode enabled: total_synthetic_len=%d (sys=%d, centroid=%d, K=%s)", 
        total_len, n, centroid_len, k
    )
    return True, total_len


def centroid_scheduler_mode() -> bool:
    """Return True when ``VLLM_CENTROID_SCHEDULER=1`` and the K-file exists.

    Thin wrapper over ``_centroid_sched_check_once`` so callers (model runners)
    can express the mutual-exclusivity check without importing ``os`` themselves.
    """
    enabled, _ = _centroid_sched_check_once()
    return enabled


def centroid_sched_gap(num_prompt_tokens: int, base_computed: int) -> int:
    """Extra external-computed-token count to inject so the centroid prefix
    appears already computed in the scheduler.

    The gap is ``max(0, min(sys_token_count, num_prompt_tokens - 1) - base_computed)``.
    ``num_prompt_tokens - 1`` mirrors vLLM's prefix-cache convention of always
    recomputing the last prompt token for logits.

    Returns 0 when:
    - ``VLLM_CENTROID_SCHEDULER`` is not ``"1"``;
    - the centroid K-path file does not exist;
    - ``base_computed`` already covers the whole centroid prefix; or
    - ``num_prompt_tokens <= 1`` (degenerate prompt, centroid cannot help).
    """
    enabled, sys_n = _centroid_sched_check_once()
    if not enabled or sys_n <= 0:
        return 0
    effective_cap = min(sys_n, num_prompt_tokens - 1)
    gap = max(0, effective_cap - base_computed)
    if gap > 0:
        logger.debug(
            "[CENTROID] sched gap=%d (sys=%d, prompt_tokens=%d, base_computed=%d)",
            gap,
            sys_n,
            num_prompt_tokens,
            base_computed,
        )
    return gap


def centroid_paths() -> tuple[str, str]:
    k = os.environ.get(
        "VLLM_CENTROID_K_PATH",
        "/home/yash/agentcache/centroid_output/centroid_K_B.npy",
    )
    v = os.environ.get(
        "VLLM_CENTROID_V_PATH",
        "/home/yash/agentcache/centroid_output/centroid_V_B.npy",
    )
    return k, v


def try_load_centroid_injector(
    device: torch.device, *, log_missing_file: bool = True
) -> Any | None:
    """Return a CentroidInjector if centroid .npy files exist; else None."""
    from vllm.centroid_injector import CentroidInjector

    k, v = centroid_paths()
    if not os.path.exists(k):
        if log_missing_file:
            logger.warning("CentroidInjector not loaded, file not found: %s", k)
        return None
    try:
        inj = CentroidInjector(k, v, device=device)
        logger.info("CentroidInjector loaded: %s", k)
        return inj
    except Exception:
        logger.exception("CentroidInjector failed to load from %s", k)
        return None


def ensure_centroid_injector_lazy(runner: Any) -> None:
    """Lazy-load centroid weights if runner has no injector yet (quiet if still missing)."""
    if getattr(runner, "_centroid_injector", None) is not None:
        return
    inj = try_load_centroid_injector(runner.device, log_missing_file=False)
    runner._centroid_injector = inj
    if inj is not None:
        logger.info("CentroidInjector lazily loaded: %s", centroid_paths()[0])


def centroid_override_num_computed(
    num_computed: int, injector: Any | None
) -> int:
    """Treat scheduler-reported 0 computed tokens as warm-started prefix length."""
    if injector is None or num_computed != 0:
        return num_computed
    n = int(getattr(injector, "sys_token_count", 128)) + int(getattr(injector, "centroid_len", 1))
    logger.info("[CENTROID] Overriding num_computed_tokens to %s", n)
    return n


def _unwrap_model(runner: Any) -> Any:
    m = runner.model
    if hasattr(m, "unwrap"):
        return m.unwrap()
    return m


def try_get_rotary_emb(runner: Any) -> Any | None:
    """First decoder layer's RoPE module (Qwen/Llama-style); None if not present."""
    try:
        m = _unwrap_model(runner)
        layers = m.model.layers
        attn = layers[0].self_attn
        return getattr(attn, "rotary_emb", None)
    except Exception:
        return None


def _prompt_lens_np(
    runner: Any, num_reqs: int, input_batch: Any | None
) -> np.ndarray:
    """Per-row prompt lengths in **batch order** (matches block_table rows)."""
    if input_batch is not None:
        im = getattr(input_batch, "idx_mapping_np", None)
        if im is not None:
            pl = np.zeros(num_reqs, dtype=np.int32)
            rs = runner.req_states
            for b in range(num_reqs):
                rsi = int(im[b])
                pl[b] = int(rs.prompt_len.np[rsi])
            return pl
        npt = getattr(input_batch, "num_prompt_tokens", None)
        if npt is not None:
            return np.asarray(npt[:num_reqs], dtype=np.int32)
    return np.asarray(runner.input_batch.num_prompt_tokens[:num_reqs], dtype=np.int32)


def apply_centroid_block_table(
    runner: Any,
    block_table: torch.Tensor,
    num_reqs: int,
    input_batch: Any | None = None,
) -> torch.Tensor:
    """Seed centroid K/V into KV cache for the warm-start prefix. Leaves block_table unchanged."""
    inj = getattr(runner, "_centroid_injector", None)
    if inj is None:
        return block_table
    if getattr(runner, "is_pooling_model", False):
        return block_table
    kvc = getattr(runner, "kv_caches", None)
    if not kvc:
        return block_table
    kv_cfg = getattr(runner, "kv_cache_config", None)
    if kv_cfg is None or not kv_cfg.kv_cache_groups:
        return block_table

    from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec

    try:
        spec0 = kv_cfg.kv_cache_groups[0].kv_cache_spec
        if isinstance(spec0, EncoderOnlyAttentionSpec):
            return block_table
        block_size = spec0.block_size
        num_kv = runner.model_config.get_num_kv_heads(runner.parallel_config)
        head_dim = runner.model_config.get_head_size()
    except Exception:
        logger.exception("centroid: could not read KV layout; skipping inject")
        return block_table

    # Profiling / CUDA-graph warmup often runs with an all-null block table before
    # any requests allocate KV blocks — skip quietly (avoids a misleading warning).
    if num_reqs <= 0:
        return block_table
    bt_live = block_table[:num_reqs]
    if bt_live.numel() > 0 and int(bt_live.max().item()) == NULL_BLOCK_ID:
        return block_table

    prompt_lens_np = _prompt_lens_np(runner, num_reqs, input_batch)

    req_id_list: list[str] | None = None
    if input_batch is not None:
        rids = getattr(input_batch, "req_ids", None)
        if rids is not None:
            im = getattr(input_batch, "idx_mapping_np", None)
            if im is not None and len(im) >= num_reqs:
                req_id_list = [str(rids[int(im[i])]) for i in range(int(num_reqs))]
            else:
                req_id_list = [str(rids[i]) for i in range(int(num_reqs))]

    inj.seed_prefix_into_kv_cache(
        kvc,
        block_table,
        num_reqs=num_reqs,
        prompt_lens_np=prompt_lens_np,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        block_size=block_size,
        null_block_id=NULL_BLOCK_ID,
        rotary_emb=try_get_rotary_emb(runner),
        num_query_heads=int(getattr(runner, "num_query_heads", num_kv)),
        target_dtype=getattr(runner, "dtype", None),
        device=getattr(runner, "device", None),
        req_ids=req_id_list,
    )
    return block_table
