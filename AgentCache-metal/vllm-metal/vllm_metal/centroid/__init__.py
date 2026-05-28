# SPDX-License-Identifier: Apache-2.0
"""AgentCache centroid injection for vLLM-Metal (Apple Silicon / MLX).

Faithful MLX port of the CUDA centroid-injection mechanism: seed offline-trained
PEFT prefix K/V ("centroids") directly into the Metal paged KV cache so the
scheduler can skip prefill for the synthetic prefix.

Activated by the same env vars as the CUDA path (see integration_mlx.py).
"""
