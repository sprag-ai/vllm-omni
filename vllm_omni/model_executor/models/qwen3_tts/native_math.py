# SPDX-License-Identifier: Apache-2.0
"""Opt-in Qwen3-TTS arithmetic matching the audited Torch 2.7 native path.

This correctness baseline uses serial, unchunked eager inference. It is not a
paged attention backend. Positions reset each layer's cache on every prefill,
including replay after preemption. Unsupported engine settings fail at startup.
"""

import os
import types

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def enabled():
    return os.environ.get("SPRAG_QWEN_NATIVE_MATH") == "1"


def legacy_mean(x):
    """Reproduce CUDA Reduce.cuh in Torch 2.7, including ascending warp offsets.

    Vectorization and block width depend on row count. Serial serving preserves
    the native prefill/decode shapes; batching would change this addition tree.
    Keep each FP32 add explicit and keep this function outside torch.compile.
    """
    shape = x.shape
    cols = shape[-1]
    rows = x.numel() // cols
    assert cols in (64, 128, 1024, 2048) and x.dtype == torch.float32
    vector = 4 if cols > 128 else 1
    dim0 = cols // vector

    def power(n):
        return 1 << (n.bit_length() - 1)

    height = min(power(rows), 16)
    width = min(power(dim0), 512 // height)
    data = x.reshape(rows, -1, width, vector)
    if vector == 4:
        accum = data[:, 0]
        for i in range(1, data.shape[1]):
            accum = accum + data[:, i]
        value = accum[..., 0] + accum[..., 1]
        value = value + accum[..., 2]
        value = value + accum[..., 3]
    else:
        accum = [torch.zeros_like(data[:, 0, :, 0]) for _ in range(4)]
        for i in range(data.shape[1]):
            accum[i % 4] = accum[i % 4] + data[:, i, :, 0]
        value = accum[0] + accum[1]
        value = value + accum[2]
        value = value + accum[3]
    offset = width // 2
    while offset >= 32:
        value = value[:, :offset] + value[:, offset : offset * 2]
        offset //= 2
    for offset in [1, 2, 4, 8, 16]:
        value = value[:, : value.shape[-1] - offset] + value[:, offset:]
    return (value / cols).reshape(*shape[:-1], 1)


def norm_forward(self, x, residual=None):
    if residual is not None:
        x = x + residual
        residual = x
    fp32 = x.float()
    variance = legacy_mean(fp32.square())
    normalized = (fp32 * torch.rsqrt(variance + self.variance_epsilon)).to(x.dtype)
    result = normalized * self.weight
    return (result, residual) if residual is not None else result


class SerialFlashCache:
    """Full KV cache for one active request, keyed by absolute token position."""

    def __init__(self):
        self.key = None
        self.value = None

    def attend(self, q, k, v, positions, scale, *, profiling=False):
        tokens = q.shape[2]
        start = int(positions.reshape(-1)[0])
        if profiling or start == 0:
            self.key, self.value = k.clone(), v.clone()
        else:
            if tokens != 1 or self.key is None or self.key.shape[2] != start:
                raise RuntimeError("Native Qwen attention requires a complete prefill followed by serial decode")
            self.key = torch.cat((self.key, k), dim=2)
            self.value = torch.cat((self.value, v), dim=2)
        repeat = q.shape[1] // k.shape[1]
        key = self.key.repeat_interleave(repeat, dim=1)
        value = self.value.repeat_interleave(repeat, dim=1)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, key, value, scale=scale, is_causal=tokens > 1)


def attention_forward(self, positions, hidden_states):
    from vllm.forward_context import get_forward_context

    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).reshape(q.shape)
    k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).reshape(k.shape)
    q, k = self.rotary_emb(positions, q, k)
    tokens = q.shape[0]
    q = q.reshape(1, tokens, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.reshape(1, tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
    v = v.reshape(1, tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
    output = (
        self._native_cache.attend(
            q,
            k,
            v,
            positions,
            self.scaling,
            profiling=get_forward_context().attn_metadata is None,
        )
        .transpose(1, 2)
        .reshape(tokens, -1)
    )
    return self.o_proj(output)[0]


def validate_config(config):
    model, scheduler, cache = config.model_config, config.scheduler_config, config.cache_config
    parallel = config.parallel_config
    checks = {
        "enforce_eager=True": model.enforce_eager,
        "max_num_seqs=1": scheduler.max_num_seqs == 1,
        "enable_chunked_prefill=False": not scheduler.enable_chunked_prefill,
        "enable_prefix_caching=False": not cache.enable_prefix_caching,
        "tensor_parallel_size=1": parallel.tensor_parallel_size == 1,
        "pipeline_parallel_size=1": parallel.pipeline_parallel_size == 1,
        "dtype=bfloat16": model.dtype == torch.bfloat16,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError("SPRAG_QWEN_NATIVE_MATH requires " + ", ".join(failed))
    if not torch.cuda.is_available():
        raise ValueError("SPRAG_QWEN_NATIVE_MATH requires CUDA flash SDPA")


def install(talker, config):
    from vllm.model_executor.layers.layernorm import RMSNorm

    validate_config(config)
    for module in talker.model.modules():
        if isinstance(module, RMSNorm):
            module.forward = types.MethodType(norm_forward, module)
    for layer in talker.model.layers:
        attention = layer.self_attn
        rotary = attention.rotary_emb
        if not rotary.is_neox_style or rotary.rotary_dim != rotary.head_size:
            raise ValueError("Native Qwen mode requires full Neox-style RoPE")
        rotary.forward = rotary.forward_native
        attention._native_cache = SerialFlashCache()
        attention.forward = types.MethodType(attention_forward, attention)
    install_predictor(talker.code_predictor)
    talker.talker_mtp_graph_safe = False


def install_predictor(predictor):
    from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorAttention, _RMSNorm

    predictor._native_math = True
    for module in predictor.modules():
        if isinstance(module, _RMSNorm):
            module.forward = types.MethodType(norm_forward, module)
        elif isinstance(module, CodePredictorAttention):
            module._native_math = True
