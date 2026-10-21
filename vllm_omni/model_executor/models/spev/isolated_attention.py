# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched complete-prefill attention for independent decision branches."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn

from .branch_attention import REQUEST, RequestMaskCache, branch_attention

LAYOUTS = ContextVar("spev_batch_layouts", default=None)


@contextmanager
def isolated_batch(layouts):
    offset = 0
    requests = []
    for layout in layouts:
        requests.append((offset, offset + layout.tree.length, layout.tree, RequestMaskCache()))
        offset += layout.tree.length
    token = LAYOUTS.set(requests)
    try:
        yield
    finally:
        LAYOUTS.reset(token)


class IsolatedAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        for name in ("qkv_proj", "o_proj", "q_norm", "k_norm", "rotary_emb", "attn"):
            setattr(self, name, getattr(attention, name))
        for name in ("head_dim", "q_size", "kv_size", "num_heads", "num_kv_heads", "scaling"):
            setattr(self, name, getattr(attention, name))

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        requests = LAYOUTS.get()
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query = self.q_norm(query.view(-1, self.num_heads, self.head_dim)).view(query.shape)
        key = self.k_norm(key.view(-1, self.num_kv_heads, self.head_dim)).view(key.shape)
        query, key = self.rotary_emb(positions, query, key)
        if requests is None:
            output = self.attn(query, key, value)
        else:
            if not requests or requests[-1][1] != len(query):
                raise ValueError("Decision attention requires complete request prefills")
            output = torch.empty_like(query)
            for start, end, tree, cache in requests:
                q = query[start:end].view(1, end - start, self.num_heads, self.head_dim).transpose(1, 2)
                k = key[start:end].view(1, end - start, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v = value[start:end].view(1, end - start, self.num_kv_heads, self.head_dim).transpose(1, 2)
                token = REQUEST.set((tree, cache))
                try:
                    result, _ = branch_attention(self, q, k, v, scaling=self.scaling)
                finally:
                    REQUEST.reset(token)
                output[start:end] = result.reshape(end - start, self.q_size)
        result, _ = self.o_proj(output)
        return result
