# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint projection, rotary and grouped-expert arithmetic."""

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextExperts,
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
    Qwen3OmniMoeThinkerTextTopKRouter,
    apply_rotary_pos_emb,
)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

from .branch_attention import REQUEST
from .isolated_attention import LAYOUTS
from .token_operations import PackingCache, grouped_forward


class CheckpointDecoderLayer(Qwen3MoeDecoderLayer):
    """Complete the decoder residual before adding DeepStack visual features."""

    def __init__(self, source):
        nn.Module.__init__(self)
        self.hidden_size = source.hidden_size
        for name in ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm"):
            setattr(self, name, getattr(source, name))

    def forward(self, positions, hidden_states, residual):
        hidden_states, residual = super().forward(positions, hidden_states, residual)
        return hidden_states + residual, torch.zeros_like(residual)


def canonical_projection(value, operation):
    requests = LAYOUTS.get()
    if requests is None:
        return operation(value)
    outputs = []
    for start, end, tree, cache in requests:
        if not hasattr(cache, "packing"):
            cache.packing = PackingCache()
        token = REQUEST.set((tree, cache))
        try:
            forward = grouped_forward(operation, lambda: tree, cache.packing)
            outputs.append(forward(None, value[start:end]))
        finally:
            REQUEST.reset(token)
    return torch.cat(outputs)


class CheckpointLinear(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight = source.weight
        self.bias = source.bias

    def forward(self, value):
        return canonical_projection(value, lambda x: nn.functional.linear(x, self.weight, self.bias)), None


class CheckpointQKV(CheckpointLinear):
    def __init__(self, source, query_size, kv_size):
        super().__init__(source)
        self.query_size = query_size
        self.kv_size = kv_size
        self._adapters = None
        self._load_projection = source.load_weights

    def load_weights(self, weights):
        return self._load_projection(weights)

    def bind_adapters(self, qa, qb, va, vb, scale):
        self._adapters = ((qa, qb), None, (va, vb))
        self.scale = scale

    def forward(self, value):
        if self._adapters is None:
            raise RuntimeError("SPEV adapters have not been loaded")

        def operation(x):
            sizes = (self.query_size, self.kv_size, self.kv_size)
            weights = self.weight.split(sizes)
            biases = self.bias.split(sizes) if self.bias is not None else (None, None, None)
            outputs = []
            for weight, bias, adapter in zip(weights, biases, self._adapters):
                output = nn.functional.linear(x, weight, bias)
                if adapter is not None:
                    a, b = adapter
                    residual = nn.functional.linear(nn.functional.linear(x.float(), a), b) * self.scale
                    output = output + residual.to(x.dtype)
                outputs.append(output)
            return torch.cat(outputs, dim=-1)

        return canonical_projection(value, operation), None


class CheckpointRotary(nn.Module):
    def __init__(self, config, device, head_dim):
        super().__init__()
        with torch.device("cpu"):
            self.rotary = Qwen3OmniMoeThinkerTextRotaryEmbedding(config)
        self.rotary.to(device)
        self.head_dim = head_dim

    def forward(self, positions, query, key):
        length = len(query)
        q = query.view(1, length, -1, self.head_dim).transpose(1, 2)
        k = key.view(1, length, -1, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(q, positions[:, None, :])
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q.transpose(1, 2).reshape(length, -1), k.transpose(1, 2).reshape(length, -1)


class CheckpointExperts(nn.Module):
    def __init__(self, source, config):
        super().__init__()
        self.gate = source.gate
        self.experts = source.experts
        self.config = type(config).from_dict(config.to_dict())
        self.config._experts_implementation = "grouped_mm"
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.act_fn = ACT2FN[config.hidden_act]

    @property
    def weight(self):
        return self.gate.weight

    @property
    def gate_up_proj(self):
        return self.experts.routed_experts.w13_weight

    @property
    def down_proj(self):
        return self.experts.routed_experts.w2_weight

    def _apply_gate(self, value):
        return Qwen3OmniMoeThinkerTextExperts._apply_gate(self, value)

    def _forward_request(self, value):
        _, weights, indices = Qwen3OmniMoeThinkerTextTopKRouter.forward(self, value)
        return Qwen3OmniMoeThinkerTextExperts.forward(self, value, indices, weights)

    def forward(self, value):
        requests = LAYOUTS.get()
        if requests is None:
            return self._forward_request(value)
        return torch.cat([self._forward_request(value[start:end]) for start, end, _, _ in requests])
