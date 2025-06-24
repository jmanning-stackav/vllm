# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV Cache operations."""

import torch

from vllm.attention.backends.abstract import is_quantized_kv_cache
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform


@CustomOp.register("reshape_and_cache_flash")
class ReshapeAndCacheFlash(CustomOp):
    """Reshape_and_cache, Flash Attn KV Cache format."""

    def __init__(
        self,
        kv_cache_dtype: str,
    ) -> None:
        super().__init__()
        self.kv_cache_dtype = kv_cache_dtype

    @staticmethod
    def forward_static(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        """PyTorch implementation of reshape_and_cache_flash, to be compiled.

        Args:
            key: New key vectors, shape:
                 (num_tokens, num_kv_heads, head_size).
            value: New value vectors, shape:
                   (num_tokens, num_kv_heads, head_size).
            key_cache: Key cache, shape:
                       (num_pages, cache_block_size, num_kv_heads, head_size).
            value_cache: Value cache, shape:
                         (num_pages, cache_block_size, num_kv_heads, head_size).
            slot_mapping: Tensor listing what slots in the cache each key/value
                          vector should be placed in, shape: (num_tokens,).
            kv_cache_dtype: String datatype of kv cache elements.
            k_scale: Fp8 scaling factor for k.
            v_scale: Fp8 scaling factor for v.
        """
        num_tokens = slot_mapping.size(0)
        _, block_size, _, _ = key_cache.shape

        if is_quantized_kv_cache(kv_cache_dtype):
            k_scale = 1.0 / k_scale
            v_scale = 1.0 / v_scale
            key = (key * k_scale).to(current_platform.fp8_dtype()).view(
                key_cache.dtype)
            value = (value * v_scale).to(current_platform.fp8_dtype()).view(
                value_cache.dtype)

        block_indicies = torch.div(slot_mapping,
                                   block_size,
                                   rounding_mode="floor")
        block_offsets = slot_mapping % block_size
        key_cache[block_indicies, block_offsets, :, :] = key[:num_tokens]
        value_cache[block_indicies, block_offsets, :, :] = value[:num_tokens]

    def forward_native(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        self.forward_static(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            k_scale,
            v_scale,
        )

    def forward_cuda(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        if torch.compiler.is_compiling():
            return self.forward_native(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                k_scale,
                v_scale,
            )

        if not getattr(self, "_is_compiled", False):
            self.forward_static = torch.compile(  # type: ignore
                self.forward_static)
            self._is_compiled = True

        return self.forward_native(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            k_scale,
            v_scale,
        )
