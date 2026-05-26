# SPDX-License-Identifier: Apache-2.0
"""Optional centroid KV warm-start hooks shared by V1 and V2 GPU model runners."""

from __future__ import annotations

import os
import re
import time
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)

# Throttle runner-side RoPE logs (same env as centroid_injector.CENTROID_DEBUG_ROPE).
_runner_rope_dbg_calls = 0
_centroid_timing_apply_calls = [0]  # list avoids `global` in apply_centroid_block_table
_centroid_perf_dbg_apply_calls = [0]

_ROTARY_CACHE_UNSET = object()


def _maybe_log_runner_rope_context(
    runner: Any,
    num_reqs: int,
    input_batch: Any | None,
    prompt_lens_np: np.ndarray,
) -> None:
    """Compare scheduler/forward positions to what the injector assumes (0..sys-1, sys..)."""
    global _runner_rope_dbg_calls
    if os.environ.get("CENTROID_DEBUG_ROPE", "0") != "1":
        return
    if _runner_rope_dbg_calls >= 24:
        return
    call = _runner_rope_dbg_calls
    _runner_rope_dbg_calls += 1
    if call == 0:
        logger.warning(
            "[CENTROID ROPE] CENTROID_DEBUG_ROPE=1 — verbose logs and CPU tensor reads "
            "run on every attention step; **unset for TTFT / throughput benchmarks**."
        )

    try:
        ib = getattr(runner, "input_batch", None) or input_batch
        uses_mrope = bool(getattr(runner, "uses_mrope", False))
        xdrope = int(getattr(runner, "uses_xdrope_dim", 0) or 0)
        if uses_mrope or xdrope > 0:
            logger.warning(
                "[CENTROID ROPE] runner_ctx#%s M-RoPE/XD-RoPE active (uses_mrope=%s "
                "xdrope_dim=%s) — flat position IDs in centroid_injector likely **wrong**.",
                call,
                uses_mrope,
                xdrope,
            )

        nc = getattr(ib, "num_computed_tokens_cpu", None)
        npt = getattr(ib, "num_prompt_tokens", None)
        if nc is not None and num_reqs > 0:
            nc_l = nc[:num_reqs].tolist() if hasattr(nc, "tolist") else list(nc[:num_reqs])
        else:
            nc_l = "n/a"
        if npt is not None and num_reqs > 0:
            npt_l = npt[:num_reqs].tolist() if hasattr(npt, "tolist") else list(npt[:num_reqs])
        else:
            npt_l = "n/a"

        pl = (
            prompt_lens_np[:num_reqs].tolist()
            if hasattr(prompt_lens_np, "tolist")
            else list(prompt_lens_np[:num_reqs])
        )

        qsl_gpu = getattr(runner, "query_start_loc", None)
        positions = getattr(runner, "positions", None)
        n_tok = None
        pos_sample = None
        if qsl_gpu is not None and positions is not None and num_reqs > 0:
            try:
                n_tok = int(qsl_gpu.gpu[num_reqs].item())
                pos_flat = positions[:n_tok].detach().cpu().numpy()
                pos_sample = pos_flat[: min(32, pos_flat.shape[0])].tolist()
                per_req = []
                for i in range(num_reqs):
                    s = int(qsl_gpu.gpu[i].item())
                    e = int(qsl_gpu.gpu[i + 1].item())
                    seg = pos_flat[s:e].tolist() if e > s else []
                    per_req.append(
                        {"req": i, "num_scheduled": e - s, "positions_minmax": [min(seg), max(seg)] if seg else None}
                    )
            except Exception as ex:
                per_req = [f"(positions parse failed: {ex})"]
        else:
            per_req = ["(no runner.query_start_loc.gpu or runner.positions)"]

        sm = None
        try:
            bt0 = ib.block_table[0] if ib is not None else None
            if bt0 is not None and hasattr(bt0, "slot_mapping") and n_tok:
                sm_gpu = bt0.slot_mapping.gpu[:n_tok]
                sm = sm_gpu.detach().cpu().numpy()[: min(16, n_tok)].tolist()
        except Exception:
            sm = None

        logger.info(
            "[CENTROID ROPE] runner_ctx#%s num_computed_tokens_cpu[:n_req]=%s "
            "num_prompt_tokens[:n_req]=%s prompt_lens_np(inject)=%s",
            call,
            nc_l,
            npt_l,
            pl,
        )
        logger.info(
            "[CENTROID ROPE] runner_ctx#%s this-step positions (first up to 32)=%s "
            "n_tokens=%s per_req_summary=%s",
            call,
            pos_sample,
            n_tok,
            per_req,
        )
        if sm is not None:
            logger.info(
                "[CENTROID ROPE] runner_ctx#%s slot_mapping gid0 (first up to 16)=%s "
                "(injector maps logical_pos -> block_table[row,col] + intra; must agree)",
                call,
                sm,
            )
    except Exception:
        logger.exception("[CENTROID ROPE] runner_ctx#%s failed to collect runner context", call)

# ---------- Layout mode ----------
_centroid_layout_cached: str | None = None


def centroid_layout() -> str:
    """Return the active centroid layout mode (marker only — no vLLM behaviour changes).

    ``VLLM_CENTROID_LAYOUT=compression``: the test harness prepends ``centroid_len``
    pad tokens to the user prompt so the gap mechanism allocates N+M KV blocks and
    schedules only the M real user tokens.  The scheduler gap is NOT suppressed.

    Default is ``"replacement"`` (original behaviour, system+user prompt).
    """
    global _centroid_layout_cached
    if _centroid_layout_cached is None:
        _centroid_layout_cached = os.environ.get("VLLM_CENTROID_LAYOUT", "replacement")
        logger.info("[CENTROID] layout=%s", _centroid_layout_cached)
    return _centroid_layout_cached


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

        k_np = np.load(k, mmap_mode='r')
        raw_len = k_np.shape[1] if k_np.ndim == 3 else 1
        cap = int(os.environ.get("VLLM_CENTROID_LEN", raw_len))
        centroid_len = min(cap, raw_len)
        sys_count = load_sys_prefix_token_count(k)

        has_exact_sys = bool(os.environ.get("VLLM_EXACT_SYS_K_PATH"))
        if has_exact_sys:
            # Exact sys KV covers 0..sys_count-1; centroid fills sys_count..
            # but gets overwritten when the model processes user-query tokens.
            # Gap = sys_count only so the model still computes the user query.
            total_len = sys_count
        else:
            # Pure PEFT / no exact sys KV: centroid fills 0..sys_count+centroid_len-1.
            # Including centroid_len is safe here because sys_count is typically 0 or 1
            # and sys_count+centroid_len << prompt_len.
            total_len = sys_count + centroid_len
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
        "[CENTROID] Scheduler mode enabled: total_synthetic_len=%d (K=%s)",
        total_len, k,
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
        if os.environ.get("CENTROID_DEBUG_ROPE", "0") == "1":
            logger.info(
                "[CENTROID] sched gap=%d (sys=%d, prompt_tokens=%d, base_computed=%d)",
                gap,
                sys_n,
                num_prompt_tokens,
                base_computed,
            )
        else:
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
        "/home/yash/agentcache/centroid_K.npy",
    )
    v = os.environ.get(
        "VLLM_CENTROID_V_PATH",
        "/home/yash/agentcache/centroid_V.npy",
    )
    return k, v


def try_load_centroid_injector(
    device: torch.device, *, log_missing_file: bool = True
) -> Any | None:
    """Return a CentroidInjector when centroid mode is on and .npy files exist."""
    if not centroid_scheduler_mode():
        return None
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
    if not centroid_scheduler_mode():
        return
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
    sys_n = int(getattr(injector, "sys_token_count", 0))
    cent_n = int(getattr(injector, "centroid_len", 0))
    has_exact_sys = bool(os.environ.get("VLLM_EXACT_SYS_K_PATH"))
    n = sys_n if has_exact_sys else (sys_n + cent_n)
    logger.info("[CENTROID] Overriding num_computed_tokens to %s", n)
    return n


def _unwrap_model(runner: Any) -> Any:
    m = runner.model
    if hasattr(m, "unwrap"):
        return m.unwrap()
    return m


def try_get_rotary_emb(runner: Any) -> Any | None:
    """First decoder layer's RoPE module; None if not present.

    Different model families hang the decoder attention module off different
    attributes. Qwen/Llama commonly use ``self_attn`` while GPT-OSS uses
    ``attn``.
    """
    try:
        m = _unwrap_model(runner)
        layers = m.model.layers
        layer0 = layers[0]

        for attn_attr in ("self_attn", "attn"):
            attn = getattr(layer0, attn_attr, None)
            rotary_emb = getattr(attn, "rotary_emb", None)
            if rotary_emb is not None:
                return rotary_emb

        return getattr(layer0, "rotary_emb", None)
    except Exception:
        return None


def try_get_rotary_emb_cached(runner: Any) -> Any | None:
    """Same as ``try_get_rotary_emb`` but resolved once per runner (``apply_centroid`` is hot)."""
    cached = getattr(runner, "_centroid_cached_rotary_emb", _ROTARY_CACHE_UNSET)
    if cached is not _ROTARY_CACHE_UNSET:
        return cached  # type: ignore[return-value]
    emb = try_get_rotary_emb(runner)
    setattr(runner, "_centroid_cached_rotary_emb", emb)
    return emb


_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")


def _parse_transformer_layer_idx(layer_name: str) -> int | None:
    match = _LAYER_INDEX_RE.search(layer_name)
    if match is None:
        return None
    return int(match.group(1))


def _centroid_layer_kv_cache_groups(runner: Any, num_layers: int) -> list[int] | None:
    attn_groups = getattr(runner, "attn_groups", None)
    kv_cfg = getattr(runner, "kv_cache_config", None)
    if not attn_groups or kv_cfg is None or num_layers <= 0:
        return None

    group_ids = [-1] * num_layers
    for gid, groups in enumerate(attn_groups):
        for attn_group in groups:
            for layer_name in getattr(attn_group, "layer_names", ()):
                layer_idx = _parse_transformer_layer_idx(layer_name)
                if layer_idx is None or layer_idx < 0 or layer_idx >= num_layers:
                    continue
                prev_gid = group_ids[layer_idx]
                if prev_gid not in (-1, gid):
                    return None
                group_ids[layer_idx] = gid

    if any(gid < 0 for gid in group_ids):
        return None
    return group_ids


def _centroid_kv_group_block_sizes(runner: Any) -> list[int] | None:
    kv_cfg = getattr(runner, "kv_cache_config", None)
    if kv_cfg is None or not kv_cfg.kv_cache_groups:
        return None
    try:
        return [int(group.kv_cache_spec.block_size) for group in kv_cfg.kv_cache_groups]
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
    all_block_tables: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """Seed centroid K/V into KV cache for the warm-start prefix. Leaves block_table unchanged."""
    if not centroid_scheduler_mode():
        return block_table
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

    _maybe_log_runner_rope_context(runner, num_reqs, input_batch, prompt_lens_np)

    req_id_list: list[str] | None = None
    if input_batch is not None:
        rids = getattr(input_batch, "req_ids", None)
        if rids is not None:
            im = getattr(input_batch, "idx_mapping_np", None)
            if im is not None and len(im) >= num_reqs:
                req_id_list = [str(rids[int(im[i])]) for i in range(int(num_reqs))]
            else:
                req_id_list = [str(rids[i]) for i in range(int(num_reqs))]

    if os.environ.get("CENTROID_PERF_DEBUG", "0") == "1":
        idx = _centroid_perf_dbg_apply_calls[0]
        _centroid_perf_dbg_apply_calls[0] = idx + 1
        if idx < 200:
            n_sched_pf = None
            pos_range = None
            expected_start = None
            try:
                qsl = getattr(runner, "query_start_loc", None)
                if qsl is not None and num_reqs > 0:
                    n_sched_pf = int(qsl.gpu[num_reqs].item())
                    positions = getattr(runner, "positions", None)
                    if positions is not None and n_sched_pf > 0:
                        pos_flat = positions[:n_sched_pf]
                        pos_min = int(pos_flat.min().item())
                        pos_max = int(pos_flat.max().item())
                        pos_range = (pos_min, pos_max)
                        # For prompt prefill this should match:
                        # expected_start = prompt_len - n_scheduled_tokens.
                        # With centroid gap=256 and prompt_len=1005 -> 256.
                        expected_start = int(prompt_lens_np[0]) - int(n_sched_pf)
            except Exception:
                pass
            seeded = getattr(inj, "_centroid_seeded_req_ids", set())
            pre_skip = False
            if req_id_list is not None and num_reqs > 0:
                pre_skip = all(str(req_id_list[i]) in seeded for i in range(num_reqs))
            pl0 = int(prompt_lens_np[0]) if num_reqs > 0 else -1
            logger.info(
                "[CENTROID PERF] apply_pre call=%s n_scheduled_tokens=%s num_reqs=%s "
                "req_ids=%s pre_seed_skip_all_seeded=%s prompt_lens[0]=%s",
                idx,
                n_sched_pf,
                num_reqs,
                req_id_list[:4] if req_id_list else None,
                pre_skip,
                pl0,
            )
            if idx < 64:
                logger.info(
                    "[CENTROID PERF] apply_pos call=%s positions_minmax=%s expected_start=%s "
                    "start_matches=%s",
                    idx,
                    pos_range,
                    expected_start,
                    (
                        bool(pos_range is not None and expected_start is not None
                             and pos_range[0] == expected_start)
                    ),
                )

    timing = os.environ.get("CENTROID_TIMING", "0")
    t0 = None
    if timing in ("1", "cuda"):
        if timing == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

    layer_kv_cache_groups = None
    kv_group_block_sizes = None
    if all_block_tables is not None and len(all_block_tables) > 1:
        layer_kv_cache_groups = _centroid_layer_kv_cache_groups(
            runner,
            int(getattr(inj, "num_layers", 0)),
        )
        kv_group_block_sizes = _centroid_kv_group_block_sizes(runner)

    hf_cfg = getattr(getattr(runner, "model_config", None), "hf_config", None)
    disable_centroid_rope = bool(getattr(hf_cfg, "model_type", None) == "gpt_oss")

    inj.seed_prefix_into_kv_cache(
        kvc,
        block_table,
        num_reqs=num_reqs,
        prompt_lens_np=prompt_lens_np,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        block_size=block_size,
        null_block_id=NULL_BLOCK_ID,
        rotary_emb=try_get_rotary_emb_cached(runner),
        num_query_heads=int(getattr(runner, "num_query_heads", num_kv)),
        target_dtype=getattr(runner, "dtype", None),
        device=getattr(runner, "device", None),
        req_ids=req_id_list,
        kv_block_tables=all_block_tables,
        layer_kv_cache_groups=layer_kv_cache_groups,
        kv_group_block_sizes=kv_group_block_sizes,
        disable_rope=disable_centroid_rope,
    )

    if t0 is not None:
        if timing == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        n_sched = None
        try:
            qsl = getattr(runner, "query_start_loc", None)
            if qsl is not None and num_reqs > 0:
                n_sched = int(qsl.gpu[num_reqs].item())
        except Exception:
            pass
        idx = _centroid_timing_apply_calls[0]
        if idx < 48:
            logger.info(
                "[CENTROID TIMING] apply_centroid mode=%s wall_ms=%.3f "
                "n_scheduled_tokens=%s num_reqs=%s call_idx=%s req_ids=%s",
                timing,
                wall_ms,
                n_sched,
                num_reqs,
                idx,
                req_id_list[:2] if req_id_list else None,
            )
        _centroid_timing_apply_calls[0] = idx + 1

    return block_table


# ---------------------------------------------------------------------------
# Startup-time APC pre-registration for centroid prefix blocks
# ---------------------------------------------------------------------------

def centroid_preregister_prefix_blocks(
    block_pool: Any,
    block_size: int,
    kv_cache_group_ids: list[int],
) -> int:
    """Pre-register centroid prefix blocks in vLLM's APC at scheduler startup.

    Populates the prefix cache hash table with the centroid pad-token blocks so
    that the very first synthetic request sees local APC hits (prefix_cache_hits
    increments) rather than relying solely on the external-computed-token gap path.

    Requires VLLM_CENTROID_PAD_TOKEN_ID to be set; otherwise this is a no-op.
    KV data is uninitialized at registration time — the GPU worker writes the
    correct centroid tensors on the first forward pass (same as today).
    Subsequent requests get true APC hits and the GPU worker re-writes the same
    idempotent data.

    Returns the number of blocks registered (0 if skipped).
    """
    enabled, total_len = _centroid_sched_check_once()
    if not enabled or total_len <= 0:
        return 0

    if not block_pool.enable_caching:
        logger.info("[CENTROID] APC disabled — skipping prefix pre-registration")
        return 0

    pad_token_id_str = os.environ.get("VLLM_CENTROID_PAD_TOKEN_ID")
    if pad_token_id_str is None:
        logger.info(
            "[CENTROID] VLLM_CENTROID_PAD_TOKEN_ID not set — "
            "skipping APC pre-registration (first request uses gap path)"
        )
        return 0

    pad_token_id = int(pad_token_id_str)
    num_prefix_blocks = (total_len + block_size - 1) // block_size

    if block_pool.get_num_free_blocks() < num_prefix_blocks:
        logger.warning(
            "[CENTROID] Not enough free blocks (%d needed, %d available) "
            "for APC pre-registration — skipping",
            num_prefix_blocks,
            block_pool.get_num_free_blocks(),
        )
        return 0

    try:
        from vllm.v1.core.kv_cache_utils import hash_block_tokens
        from vllm.v1.core.block_pool import make_block_hash_with_group_id
        from vllm.utils.hashing import get_hash_fn_by_name
    except ImportError:
        logger.warning("[CENTROID] Could not import hashing utilities — skipping APC pre-registration")
        return 0

    hash_algo = os.environ.get("VLLM_PREFIX_CACHING_HASH_ALGO", "sha256")
    try:
        caching_hash_fn = get_hash_fn_by_name(hash_algo)
    except Exception:
        logger.warning(
            "[CENTROID] Could not get hash function '%s' — skipping APC pre-registration",
            hash_algo,
        )
        return 0

    parent_hash = None
    registered = 0

    for blk_idx in range(num_prefix_blocks):
        # Pad the last partial block to block_size with the same pad token so
        # the hash matches what the request block hasher will compute.
        token_ids = tuple([pad_token_id] * block_size)
        block_hash = hash_block_tokens(caching_hash_fn, parent_hash, token_ids)
        parent_hash = block_hash

        for group_id in kv_cache_group_ids:
            hash_with_gid = make_block_hash_with_group_id(block_hash, group_id)

            # Idempotent: skip if already registered
            if block_pool.cached_block_hash_to_block.get_one_block(hash_with_gid) is not None:
                continue

            blk_list = block_pool.get_new_blocks(1)
            if not blk_list:
                logger.warning(
                    "[CENTROID] get_new_blocks returned empty — "
                    "stopping pre-registration at block %d",
                    blk_idx,
                )
                return registered

            blk = blk_list[0]
            blk.block_hash = hash_with_gid
            block_pool.cached_block_hash_to_block.insert(hash_with_gid, blk)

            # Decrement ref so the block sits in the evictable pool — the
            # standard state for a cached-but-unreferenced APC block.
            blk.ref_cnt -= 1
            block_pool.free_block_queue.append(blk)
            registered += 1

    logger.info(
        "[CENTROID] Pre-registered %d APC block(s) for pad_token_id=%d "
        "prefix_len=%d block_size=%d groups=%s",
        registered,
        pad_token_id,
        total_len,
        block_size,
        kv_cache_group_ids,
    )
    return registered
