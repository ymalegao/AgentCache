# SPDX-License-Identifier: Apache-2.0
"""MLX centroid injector for vLLM-Metal.

Loads offline-trained PEFT prefix K/V tensors (``centroid_{K,V}.npy``) and seeds
them directly into the Metal paged KV cache (``MetalPagedKVCache``) at the
positions the scheduler has marked as already-computed (the "gap").  This is the
faithful Apple-Silicon analog of the CUDA ``centroid_injector.py``:

  - K is RoPE-rotated offline at positions ``sys_token_count .. sys+N-1`` using
    the model's OWN ``attn.rope`` (never a hand-rolled rotation) so the stored
    keys are positionally consistent with the user tokens that follow.
  - V is stored RAW (RoPE applies to Q/K only).
  - The write mirrors ``attention_sdpa.sdpa_forward`` exactly: flatten the
    per-layer cache to ``(num_blocks*block_size, kv_heads, head_dim)``, scatter
    by absolute slot index, reshape back, and rebind the cache array.

Centroid layout (same artifact as the CUDA pipeline / transpose_tensors.py):
    centroid_K.npy / centroid_V.npy : float, shape [num_layers, N, kv_heads*head_dim]
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)


def load_sys_prefix_token_count(centroid_k_path: str) -> int:
    """Resolve the exact-system-prompt token count (0 = pure-PEFT compression).

    Order: ``VLLM_CENTROID_SYS_TOKENS`` env → ``sys_prefix_num_tokens.txt``
    sidecar next to the centroid → 0 (compression-mode default).
    """
    env = os.environ.get("VLLM_CENTROID_SYS_TOKENS")
    if env is not None:
        return int(env)
    sidecar = Path(centroid_k_path).with_name("sys_prefix_num_tokens.txt")
    if sidecar.is_file():
        return int(sidecar.read_text().strip())
    logger.warning(
        "[CENTROID] No VLLM_CENTROID_SYS_TOKENS and no %s — assuming 0 "
        "(pure-PEFT compression mode).",
        sidecar,
    )
    return 0


class MetalCentroidInjector:
    """Seeds centroid K/V into a ``MetalPagedKVCache`` for the synthetic prefix."""

    def __init__(self, centroid_k_path: str, centroid_v_path: str) -> None:
        K = np.load(centroid_k_path)
        V = np.load(centroid_v_path)
        # Accept [layers, kv_dim] (single virtual token) or [layers, N, kv_dim].
        if K.ndim == 2:
            K = K[:, None, :]
            V = V[:, None, :]
        if K.shape != V.shape:
            raise ValueError(
                f"[CENTROID] K/V shape mismatch: {K.shape} vs {V.shape}"
            )

        self.num_layers = int(K.shape[0])
        self.centroid_len = int(K.shape[1])
        self.kv_dim = int(K.shape[2])

        # Master copies in fp32; cast to the cache dtype at seed time so the
        # offline RoPE runs in the same precision as the live forward pass.
        self._K_np = K.astype(np.float32)
        self._V_np = V.astype(np.float32)
        self.K = mx.array(self._K_np)  # [layers, N, kv_dim]
        self.V = mx.array(self._V_np)

        self.sys_token_count = load_sys_prefix_token_count(centroid_k_path)
        self._seeded_req_ids: set[str] = set()

        # Cache of RoPE-rotated K, keyed by (fill, dtype-str). Position-stable
        # because the rotation offset (sys_token_count) is fixed.
        self._rope_cache_key: tuple | None = None
        self._rope_cache: list[mx.array] | None = None

        logger.info(
            "[CENTROID] MetalCentroidInjector loaded: layers=%d N=%d kv_dim=%d "
            "sys_tokens=%d",
            self.num_layers,
            self.centroid_len,
            self.kv_dim,
            self.sys_token_count,
        )

    # -- bookkeeping ---------------------------------------------------------

    def already_seeded(self, req_id: str | None) -> bool:
        return req_id is not None and req_id in self._seeded_req_ids

    def reset(self) -> None:
        """Drop per-request dedup state (e.g. between benchmark runs)."""
        self._seeded_req_ids.clear()

    # -- RoPE ----------------------------------------------------------------

    def _rotate_k(
        self,
        layer_ropes: list,
        num_kv_heads: int,
        head_dim: int,
        fill: int,
        dtype: mx.Dtype,
    ) -> list[mx.array]:
        """Return per-layer RoPE-rotated K, each shaped ``[fill, kv_heads, head_dim]``.

        Uses each layer's own ``rope(x, offset=)`` exactly as
        ``apply_packed_rope`` does on the live path: input ``[1, kv_heads, N,
        head_dim]`` (B, H, S, D), rotated at positions ``sys .. sys+fill-1``.
        """
        key = (fill, str(dtype))
        if self._rope_cache_key == key and self._rope_cache is not None:
            return self._rope_cache

        rotated: list[mx.array] = []
        for layer_idx in range(self.num_layers):
            rope = layer_ropes[layer_idx]
            k = self.K[layer_idx, :fill, :].reshape(fill, num_kv_heads, head_dim)
            k = k.astype(dtype)
            # [fill, H, D] -> [1, H, fill, D]
            k_bhsd = mx.contiguous(k.transpose(1, 0, 2)[None])
            k_rot = rope(k_bhsd, offset=self.sys_token_count)  # [1, H, fill, D]
            # back to cache layout [fill, H, D]
            rotated.append(mx.contiguous(k_rot[0].transpose(1, 0, 2)))
        mx.eval(*rotated)

        self._rope_cache_key = key
        self._rope_cache = rotated
        return rotated

    # -- seeding -------------------------------------------------------------

    def seed_request(
        self,
        kv_cache,
        block_ids: list[int],
        start_pos: int,
        layer_ropes: list,
        *,
        req_id: str | None = None,
    ) -> bool:
        """Seed centroid K/V into the gap region for one prefill request.

        ``start_pos`` is the request's ``num_computed_tokens`` (the scheduler
        gap). Centroid fills logical positions ``sys .. sys+fill-1`` where
        ``fill = min(centroid_len, start_pos - sys_token_count)``.

        Returns True if anything was written.
        """
        if self.already_seeded(req_id):
            return False

        fill = min(self.centroid_len, max(0, start_pos - self.sys_token_count))
        if fill <= 0:
            return False

        block_size = kv_cache.block_size
        dtype = kv_cache.dtype

        # Heads/dims must be uniform across layers for a single centroid file.
        num_kv_heads = kv_cache.num_kv_heads
        head_dim = kv_cache.head_dim
        if self.kv_dim != num_kv_heads * head_dim:
            logger.error(
                "[CENTROID] kv_dim mismatch: centroid kv_dim=%d but cache "
                "num_kv_heads*head_dim=%d*%d=%d. Retrain/export for this model; "
                "skipping seed.",
                self.kv_dim,
                num_kv_heads,
                head_dim,
                num_kv_heads * head_dim,
            )
            return False

        # Absolute slot index for each seeded position (same formula as
        # paged_attention_common.prepare_unified / attention_sdpa).
        positions = self.sys_token_count + np.arange(fill, dtype=np.int64)
        block_for_pos = np.asarray(
            [block_ids[int(p) // block_size] for p in positions], dtype=np.int64
        )
        slots = block_for_pos * block_size + (positions % block_size)
        slot_mapping = mx.array(slots, dtype=mx.int64)

        k_rotated = self._rotate_k(layer_ropes, num_kv_heads, head_dim, fill, dtype)

        for layer_idx in range(self.num_layers):
            orig_shape = kv_cache.key_caches[layer_idx].shape
            k_seed = k_rotated[layer_idx].astype(dtype)  # [fill, H, D]
            v_seed = (
                self.V[layer_idx, :fill, :]
                .reshape(fill, num_kv_heads, head_dim)
                .astype(dtype)
            )

            flat_k = kv_cache.key_caches[layer_idx].reshape(-1, num_kv_heads, head_dim)
            flat_k[slot_mapping] = k_seed
            kv_cache.key_caches[layer_idx] = flat_k.reshape(orig_shape)

            flat_v = kv_cache.value_caches[layer_idx].reshape(
                -1, num_kv_heads, head_dim
            )
            flat_v[slot_mapping] = v_seed
            kv_cache.value_caches[layer_idx] = flat_v.reshape(orig_shape)

        # Materialize before the forward reads these slots (the live write path
        # is lazy + fenced via graph provenance; our seed precedes the graph, so
        # eval explicitly to make the buffers concrete).
        mx.eval(*kv_cache.key_caches, *kv_cache.value_caches)

        if req_id is not None:
            self._seeded_req_ids.add(req_id)
        logger.info(
            "[CENTROID] seeded req=%s fill=%d start_pos=%d (positions %d..%d)",
            req_id,
            fill,
            start_pos,
            self.sys_token_count,
            self.sys_token_count + fill - 1,
        )
        return True
