# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side centroid gap (version-independent port of the CUDA path).

vLLM-Metal reuses vLLM *core*'s V1 scheduler. To make the engine skip prefill
for the synthetic prefix we inflate a request's externally-computed token count
by ``gap`` so the scheduler believes the first ``gap`` positions are already in
cache. The runner-side injector then actually seeds those positions.

This module is pure Python + numpy (no torch / no mlx): safe to import inside the
scheduler. The single insertion point in core ``scheduler.py`` is:

    from vllm_metal.centroid.sched_gap import centroid_sched_gap
    num_external_computed_tokens += centroid_sched_gap(
        request.num_prompt_tokens,
        num_new_local_computed_tokens + num_external_computed_tokens,
    )

(placed where ``num_external_computed_tokens`` is finalized, before
``num_computed_tokens`` is computed — skip when ``load_kv_async`` so a real KV
connector keeps ownership).

Long-term, a ``KVConnectorBase`` returning ``gap`` from
``get_num_new_matched_tokens`` is the fork-free equivalent.
"""

from __future__ import annotations

import os

import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)

_enabled: bool | None = None
_synthetic_len: int = 0


def _check_once() -> tuple[bool, int]:
    """Resolve (enabled, total_synthetic_len) once; cache for the process.

    ``total_synthetic_len = sys_token_count + centroid_len`` for pure-PEFT
    compression, or ``sys_token_count`` when an exact-system KV is provided
    (``VLLM_EXACT_SYS_K_PATH``), matching the CUDA accounting.
    """
    global _enabled, _synthetic_len
    if _enabled is not None:
        return _enabled, _synthetic_len

    if os.environ.get("VLLM_CENTROID_SCHEDULER", "0") != "1":
        _enabled, _synthetic_len = False, 0
        return _enabled, _synthetic_len

    k = os.environ.get("VLLM_CENTROID_K_PATH")
    if not k or not os.path.exists(k):
        logger.info("[CENTROID] scheduler gap disabled: K path missing (%s)", k)
        _enabled, _synthetic_len = False, 0
        return _enabled, _synthetic_len

    try:
        k_np = np.load(k, mmap_mode="r")
        centroid_len = int(k_np.shape[1]) if k_np.ndim == 3 else 1

        env = os.environ.get("VLLM_CENTROID_SYS_TOKENS")
        if env is not None:
            sys_count = int(env)
        else:
            sidecar = os.path.join(
                os.path.dirname(k), "sys_prefix_num_tokens.txt"
            )
            sys_count = (
                int(open(sidecar).read().strip()) if os.path.isfile(sidecar) else 0
            )

        if os.environ.get("VLLM_EXACT_SYS_K_PATH"):
            total = sys_count
        else:
            total = sys_count + centroid_len
    except Exception:
        logger.exception("[CENTROID] failed to resolve synthetic len; gap disabled")
        _enabled, _synthetic_len = False, 0
        return _enabled, _synthetic_len

    _enabled, _synthetic_len = True, total
    logger.info("[CENTROID] scheduler gap enabled: total_synthetic_len=%d", total)
    return _enabled, _synthetic_len


def centroid_sched_gap(num_prompt_tokens: int, base_computed: int) -> int:
    """Extra externally-computed tokens to inject for this request.

    ``gap = max(0, min(synthetic_len, num_prompt_tokens - 1) - base_computed)``.
    The ``num_prompt_tokens - 1`` cap mirrors vLLM's prefix-cache convention of
    always recomputing the last prompt token for logits. Returns 0 when disabled
    or already covered.
    """
    enabled, syn = _check_once()
    if not enabled or syn <= 0 or num_prompt_tokens <= 1:
        return 0
    return max(0, min(syn, num_prompt_tokens - 1) - base_computed)
