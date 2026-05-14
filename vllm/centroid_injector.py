# SPDX-License-Identifier: Apache-2.0
"""Inject offline centroid K/V into vLLM KV cache for synthetic system-prefix warmup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def load_sys_prefix_token_count(centroid_k_path: str) -> int:
    env = os.environ.get("VLLM_CENTROID_SYS_TOKENS")
    if env is not None:
        return int(env)
    sidecar = Path(centroid_k_path).with_name("sys_prefix_num_tokens.txt")
    if sidecar.is_file():
        return int(sidecar.read_text().strip())
    logger.warning(
        "No VLLM_CENTROID_SYS_TOKENS and no %s — using 128 (likely wrong; set env "
        "or add sys_prefix_num_tokens.txt next to centroids)",
        sidecar,
    )
    return 128


class CentroidInjector:
    def __init__(self, centroid_K_path, centroid_V_path, device="cuda"):
        K = np.load(centroid_K_path)
        V = np.load(centroid_V_path)
        self.num_layers = K.shape[0]
        
        if K.ndim == 2:
            self.centroid_len = 1
            self.K = torch.tensor(K, dtype=torch.float16, device=device).unsqueeze(1)
            self.V = torch.tensor(V, dtype=torch.float16, device=device).unsqueeze(1)
        else:
            self.centroid_len = K.shape[1]
            self.K = torch.tensor(K, dtype=torch.float16, device=device)
            self.V = torch.tensor(V, dtype=torch.float16, device=device)

        self._centroid_k_path = centroid_K_path
        self.sys_token_count = load_sys_prefix_token_count(centroid_K_path)
        self.use_lmcache = os.environ.get("VLLM_CENTROID_USE_LMCACHE", "0") == "1"
        self.sink_blend = float(os.environ.get("VLLM_CENTROID_SINK_BLEND", "0.35"))
        
        self.sys_K = None
        self.sys_V = None
        if not self.use_lmcache:
            sys_k_path = os.environ.get("VLLM_EXACT_SYS_K_PATH")
            sys_v_path = os.environ.get("VLLM_EXACT_SYS_V_PATH")
            if sys_k_path and os.path.exists(sys_k_path):
                sys_k_np = np.load(sys_k_path)
                sys_v_np = np.load(sys_v_path)
                # Ensure 3D shape [layers, M, kv_dim]
                if sys_k_np.ndim == 2:
                    sys_k_np = np.expand_dims(sys_k_np, 1)
                    sys_v_np = np.expand_dims(sys_v_np, 1)
                self.sys_K = torch.tensor(sys_k_np, dtype=torch.float16, device=device)
                self.sys_V = torch.tensor(sys_v_np, dtype=torch.float16, device=device)
                logger.info("Loaded exact system prefix from %s", sys_k_path)

        self._centroid_seeded_req_ids: set[str] = set()
        self._rope_k_cache_key: tuple[Any, ...] | None = None
        self._rope_k_cache_tensor: torch.Tensor | None = None
        self._rope_sys_k_cache_tensor: torch.Tensor | None = None
        self._sink_k_template: torch.Tensor | None = None
        self._sink_v_template: torch.Tensor | None = None
        self._rope_sink_k_cache_tensor: torch.Tensor | None = None
        logger.info(
            "CentroidInjector: sys_token_count=%s, centroid_len=%s, use_lmcache=%s, sink_blend=%.2f",
            self.sys_token_count, self.centroid_len, self.use_lmcache, self.sink_blend
        )
        if self.sys_K is not None and self.sys_V is not None and self.sys_K.shape[1] > 0:
            # A structural sink template avoids displacing the model's native sink
            # behavior when the synthetic block is injected before user tokens.
            sink_window = min(4, self.sys_K.shape[1])
            self._sink_k_template = self.sys_K[:, :sink_window, :].mean(dim=1)
            self._sink_v_template = self.sys_V[:, :sink_window, :].mean(dim=1)

    def _batch_rope(
        self,
        rotary_emb: Any,
        K: torch.Tensor,
        positions: torch.Tensor,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        seq_len = positions.shape[0]
        n_layers = K.shape[0]

        K_flat = K.reshape(n_layers * seq_len, num_kv_heads * head_dim)
        pos_expanded = positions.repeat(n_layers)

        q_dummy = torch.zeros(
            n_layers * seq_len,
            num_query_heads * head_dim,
            device=device,
            dtype=dtype,
        )

        fn = getattr(rotary_emb, "forward_native", None) or rotary_emb
        _, k_out = fn(pos_expanded, q_dummy, K_flat.to(dtype))

        return k_out.view(n_layers, seq_len, num_kv_heads, head_dim)

    def seed_prefix_into_kv_cache(
        self,
        kv_caches: list,
        block_table: torch.Tensor,
        *,
        num_reqs: int,
        prompt_lens_np,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        null_block_id: int = 0,
        rotary_emb: Any | None = None,
        num_query_heads: int | None = None,
        target_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        req_ids: list[str] | None = None,
    ) -> None:
        if rotary_emb is None:
            logger.warning("CentroidInjector: rotary_emb is None — injecting unrotated K")

        n_layers = self.num_layers
        dev = device or self.K.device
        tgt_dtype = target_dtype or self.K.dtype
        n_q_heads = int(num_query_heads) if num_query_heads is not None else num_kv_heads

        max_centroid_fill = 0
        for seq in range(num_reqs):
            fl = min(self.centroid_len, max(0, int(prompt_lens_np[seq]) - self.sys_token_count))
            if fl > max_centroid_fill:
                max_centroid_fill = fl

        k_rotated_max: torch.Tensor | None = None
        if max_centroid_fill > 0 and rotary_emb is not None:
            cache_key = (max_centroid_fill, dev, tgt_dtype, n_q_heads, num_kv_heads, head_dim, id(rotary_emb))
            if self._rope_k_cache_key == cache_key and self._rope_k_cache_tensor is not None:
                k_rotated_max = self._rope_k_cache_tensor
            else:
                # RoPE positions for centroid are sys_token_count .. sys_token_count + max_centroid_fill - 1
                all_pos = self.sys_token_count + torch.arange(max_centroid_fill, device=dev, dtype=torch.long)
                # self.K has shape [n_layers, N, kv_dim]. We take the slice we need.
                K_slice = self.K[:, :max_centroid_fill, :].contiguous()
                k_rotated_max = self._batch_rope(
                    rotary_emb, K_slice, all_pos, n_q_heads, num_kv_heads, head_dim, dev, tgt_dtype
                )
                self._rope_k_cache_key = cache_key
                self._rope_k_cache_tensor = k_rotated_max
        elif max_centroid_fill > 0:
            k_rotated_max = self.K[:, :max_centroid_fill, :].view(n_layers, max_centroid_fill, num_kv_heads, head_dim)

        sys_rotated: torch.Tensor | None = None
        if not self.use_lmcache and self.sys_K is not None:
            actual_sys_len = min(self.sys_token_count, self.sys_K.shape[1])
            if rotary_emb is not None:
                if self._rope_sys_k_cache_tensor is not None:
                    sys_rotated = self._rope_sys_k_cache_tensor
                else:
                    all_sys_pos = torch.arange(actual_sys_len, device=dev, dtype=torch.long)
                    sys_K_slice = self.sys_K[:, :actual_sys_len, :].contiguous()
                    sys_rotated = self._batch_rope(
                        rotary_emb, sys_K_slice, all_sys_pos, n_q_heads, num_kv_heads, head_dim, dev, tgt_dtype
                    )
                    self._rope_sys_k_cache_tensor = sys_rotated
            else:
                sys_rotated = self.sys_K[:, :actual_sys_len, :].view(n_layers, actual_sys_len, num_kv_heads, head_dim)

        sink_rotated: torch.Tensor | None = None
        if (
            self._sink_k_template is not None
            and self._sink_v_template is not None
            and self.sink_blend > 0.0
            and max_centroid_fill > 0
        ):
            if rotary_emb is not None:
                if self._rope_sink_k_cache_tensor is not None:
                    sink_rotated = self._rope_sink_k_cache_tensor
                else:
                    sink_pos = torch.tensor([self.sys_token_count], device=dev, dtype=torch.long)
                    sink_k = self._sink_k_template.unsqueeze(1).contiguous()
                    sink_rotated = self._batch_rope(
                        rotary_emb, sink_k, sink_pos, n_q_heads, num_kv_heads, head_dim, dev, tgt_dtype
                    )
                    self._rope_sink_k_cache_tensor = sink_rotated
            else:
                sink_rotated = self._sink_k_template.view(n_layers, 1, num_kv_heads, head_dim)

        debug = os.environ.get("CENTROID_DEBUG", "0") == "1"
        if debug and not getattr(self, "_debug_printed", False):
            self._debug_printed = True
            kv0 = kv_caches[0]
            logger.info(
                "[CENTROID DEBUG] kv_caches[0] type=%s shape=%s dtype=%s  "
                "block_table shape=%s  block_size=%s  num_kv_heads=%s  head_dim=%s  "
                "n_layers=%s  num_reqs=%s  prompt_lens=%s",
                type(kv0).__name__,
                getattr(kv0, "shape", "N/A"),
                getattr(kv0, "dtype", "N/A"),
                tuple(block_table.shape),
                block_size, num_kv_heads, head_dim, n_layers, num_reqs,
                prompt_lens_np[:num_reqs].tolist() if hasattr(prompt_lens_np, "tolist") else list(prompt_lens_np[:num_reqs]),
            )
            logger.info("[CENTROID DEBUG] block_table[0,:8] = %s", block_table[0, :8].tolist())

        wrote_any = False

        for seq in range(num_reqs):
            req_id = req_ids[seq] if req_ids is not None else None
            if req_id is not None and req_id in self._centroid_seeded_req_ids:
                continue

            prompt_len = int(prompt_lens_np[seq])

            # --- 1. Inject Exact System Prompt KV (if needed and available) ---
            if not self.use_lmcache and self.sys_K is not None:
                actual_sys_len = min(self.sys_token_count, self.sys_K.shape[1])
                sys_fill_len = min(actual_sys_len, prompt_len)
                if sys_fill_len > 0:
                    pos_idx_sys = torch.arange(sys_fill_len, device=dev)
                    blk_cols_sys = pos_idx_sys // block_size
                    intras_sys = pos_idx_sys % block_size
                    phys_blocks_sys = block_table[seq, blk_cols_sys]

                    if not int((phys_blocks_sys == null_block_id).any().item()):
                        for layer_idx in range(n_layers):
                            kv_tensor = kv_caches[layer_idx]
                            slot_dtype = kv_tensor.dtype

                            k_sys_row = sys_rotated[layer_idx, :sys_fill_len, :, :].to(slot_dtype)
                            v_sys_row = self.sys_V[layer_idx, :sys_fill_len, :].view(sys_fill_len, num_kv_heads, head_dim).to(slot_dtype)
                            kv_tensor[0, phys_blocks_sys, intras_sys, :, :] = k_sys_row
                            kv_tensor[1, phys_blocks_sys, intras_sys, :, :] = v_sys_row
                        if debug:
                            kv0 = kv_caches[0]
                            readback = kv0[0, int(phys_blocks_sys[0]), int(intras_sys[0]), :, :]
                            expected = sys_rotated[0, 0, :, :].to(kv0.dtype)
                            match = torch.allclose(readback, expected, atol=1e-2)
                            logger.info(
                                "[CENTROID DEBUG] sys_K write readback match=%s  "
                                "phys_block=%s  intra=%s  written_norm=%.4f  readback_norm=%.4f",
                                match, int(phys_blocks_sys[0]), int(intras_sys[0]),
                                expected.float().norm().item(),
                                readback.float().norm().item(),
                            )
                        wrote_any = True

            # --- 2. Inject Domain Centroid KV ---
            centroid_fill_len = min(self.centroid_len, max(0, prompt_len - self.sys_token_count))
            if centroid_fill_len > 0 and k_rotated_max is None:
                continue

            if centroid_fill_len > 0:
                k_rotated_all = k_rotated_max[:, :centroid_fill_len, :, :]
                v_all = self.V[:, :centroid_fill_len, :].view(n_layers, centroid_fill_len, num_kv_heads, head_dim)
                if sink_rotated is not None:
                    alpha = min(max(self.sink_blend, 0.0), 1.0)
                    # Blend only the first injected token with a sink template.
                    # This preserves domain-specific centroid structure for the rest.
                    k_rotated_all = k_rotated_all.clone()
                    v_all = v_all.clone()
                    k_rotated_all[:, 0, :, :] = (1.0 - alpha) * k_rotated_all[:, 0, :, :] + alpha * sink_rotated[:, 0, :, :]
                    sink_v = self._sink_v_template.view(n_layers, num_kv_heads, head_dim)
                    v_all[:, 0, :, :] = (1.0 - alpha) * v_all[:, 0, :, :] + alpha * sink_v
                
                pos_idx_cent = self.sys_token_count + torch.arange(centroid_fill_len, device=dev)
                blk_cols_cent = pos_idx_cent // block_size
                intras_cent = pos_idx_cent % block_size
                phys_blocks_cent = block_table[seq, blk_cols_cent]

                if not int((phys_blocks_cent == null_block_id).any().item()):
                    for layer_idx in range(n_layers):
                        kv_tensor = kv_caches[layer_idx]
                        slot_dtype = kv_tensor.dtype
                        kv_tensor[0, phys_blocks_cent, intras_cent, :, :] = k_rotated_all[layer_idx].to(slot_dtype)
                        kv_tensor[1, phys_blocks_cent, intras_cent, :, :] = v_all[layer_idx].to(slot_dtype)

                    if debug:
                        kv0 = kv_caches[0]
                        readback = kv0[0, int(phys_blocks_cent[0]), int(intras_cent[0]), :, :]
                        expected = k_rotated_all[0, 0, :, :].to(kv0.dtype)
                        match = torch.allclose(readback, expected, atol=1e-2)
                        logger.info(
                            "[CENTROID DEBUG] centroid write readback match=%s  "
                            "pos=%d  phys_block=%s  intra=%s  phys_blocks_cent[:5]=%s  intras_cent[:5]=%s  "
                            "written_norm=%.4f  readback_norm=%.4f",
                            match, int(pos_idx_cent[0]),
                            int(phys_blocks_cent[0]), int(intras_cent[0]),
                            phys_blocks_cent[:5].tolist(), intras_cent[:5].tolist(),
                            expected.float().norm().item(),
                            readback.float().norm().item(),
                        )

                    wrote_any = True

            if wrote_any and req_id is not None:
                self._centroid_seeded_req_ids.add(req_id)

        if wrote_any:
            logger.info(
                "[CENTROID] Seeded centroid K/V (sys_len=%s, centroid_len=%s)",
                self.sys_token_count, self.centroid_len
            )
        elif not getattr(self, "_warned_no_write", False):
            self._warned_no_write = True
            logger.warning("[CENTROID] Did not write any centroid KV")

    def inject(self, kv_caches, block_tables, num_kv_heads, head_dim, block_size):
        return block_tables
