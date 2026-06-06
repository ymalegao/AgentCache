# SPDX-License-Identifier: Apache-2.0
"""Glue between the Metal centroid injector and ``MetalModelRunner``.

Two integration points (mirrors the CUDA centroid_integration.py):

  1. ``try_load_metal_centroid_injector()`` — called once at runner init; returns
     a ``MetalCentroidInjector`` when ``VLLM_CENTROID_SCHEDULER=1`` and the
     centroid K-file exists, else None.

  2. ``apply_metal_centroid(runner, prefill_reqs)`` — called inside
     ``MetalModelRunner._start_paged_forward`` right after ``prepare_unified``
     and before the model forward, seeding the synthetic prefix for each
     not-yet-seeded prefill request.

The scheduler-side "gap" that marks the prefix already-computed lives in vLLM
*core* (``centroid_sched_gap`` in the patched core scheduler) and is reused as-is;
nothing here touches scheduling.

Env vars (same as CUDA):
    VLLM_CENTROID_SCHEDULER   "1" to enable
    VLLM_CENTROID_K_PATH      abs path to centroid_K.npy
    VLLM_CENTROID_V_PATH      abs path to centroid_V.npy
    VLLM_CENTROID_SYS_TOKENS  exact-sys token count (0 = pure-PEFT compression)
"""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger

from vllm_metal.paged_attention_common import find_attn_attr, find_layers

logger = init_logger(__name__)

_enabled: bool | None = None
_warned_no_rope = False


def centroid_enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = os.environ.get("VLLM_CENTROID_SCHEDULER", "0") == "1"
    return _enabled


def centroid_paths() -> tuple[str | None, str | None]:
    return (
        os.environ.get("VLLM_CENTROID_K_PATH"),
        os.environ.get("VLLM_CENTROID_V_PATH"),
    )


def try_load_metal_centroid_injector() -> Any | None:
    """Load the injector when centroid mode is on and the .npy files exist."""
    if not centroid_enabled():
        return None
    k, v = centroid_paths()
    if not k or not os.path.exists(k):
        logger.warning(
            "[CENTROID] VLLM_CENTROID_SCHEDULER=1 but K path missing: %s — "
            "centroid injection disabled.",
            k,
        )
        return None
    if not v or not os.path.exists(v):
        logger.warning("[CENTROID] V path missing: %s — disabled.", v)
        return None
    from vllm_metal.centroid.injector_mlx import MetalCentroidInjector

    try:
        inj = MetalCentroidInjector(k, v)
        logger.info("[CENTROID] injector loaded from %s", k)
        return inj
    except Exception:
        logger.exception("[CENTROID] failed to load injector from %s", k)
        return None


def get_layer_ropes(runner: Any) -> list | None:
    """Resolve each decoder layer's ``attn.rope`` once, cached on the runner.

    Returns None if any layer's rope can't be found (we refuse to inject
    unrotated K rather than silently corrupt the cache).
    """
    cached = getattr(runner, "_centroid_layer_ropes", None)
    if cached is not None:
        return cached

    model = getattr(runner, "model", None)
    if model is None:
        return None
    try:
        layers = find_layers(model)
    except Exception:
        logger.exception("[CENTROID] could not locate transformer layers")
        return None

    ropes: list = []
    for layer in layers:
        attr = find_attn_attr(layer)
        attn = getattr(layer, attr) if attr else None
        # vLLM-Metal wraps the real mlx_lm attention in a paged-attention
        # wrapper (MetalKernelPagedAttentionWrapper); the rope lives on the
        # inner module. Fall back to the module itself for unwrapped models.
        inner = getattr(attn, "_inner", attn) if attn is not None else None
        rope = getattr(inner, "rope", None) if inner is not None else None
        ropes.append(rope)

    if any(r is None for r in ropes):
        return None

    runner._centroid_layer_ropes = ropes
    logger.info("[CENTROID] resolved %d layer ropes", len(ropes))
    return ropes


def apply_metal_centroid(runner: Any, prefill_reqs: list) -> None:
    """Seed centroid K/V for each not-yet-seeded prefill request.

    Safe no-op when injection is disabled, no prefill work, no paged backend,
    a turboquant cache (unsupported in v1), or ropes can't be resolved.
    """
    global _warned_no_rope
    inj = getattr(runner, "_centroid_injector", None)
    if inj is None or not prefill_reqs:
        return

    backend = getattr(runner, "_paged_attention_backend", None)
    if backend is None:
        return
    try:
        kv_cache = backend.kv_cache
    except Exception:
        return

    if getattr(kv_cache, "turboquant", False):
        if not _warned_no_rope:
            logger.warning(
                "[CENTROID] turboquant cache is unsupported by the centroid "
                "injector (v1); skipping injection."
            )
            _warned_no_rope = True
        return

    ropes = get_layer_ropes(runner)
    if ropes is None:
        if not _warned_no_rope:
            logger.warning(
                "[CENTROID] could not resolve per-layer rope; refusing to "
                "inject unrotated K. Centroid injection skipped."
            )
            _warned_no_rope = True
        return

    for pr in prefill_reqs:
        req_id = getattr(pr, "req_id", None)
        if inj.already_seeded(req_id):
            continue
        inj.seed_request(
            kv_cache,
            pr.block_ids,
            pr.start_pos,
            ropes,
            req_id=req_id,
        )
