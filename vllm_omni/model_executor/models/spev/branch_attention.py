# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared-prefix branch attention with request-local immutable mask metadata."""

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch.nn import functional as F

REQUEST = ContextVar("spev_attention_request", default=None)


class RequestMaskCache:
    def __init__(self):
        self.tree = None
        self.device = None
        self.plan = None
        self.builds = 0
        self.hits = 0

    def __call__(self, tree, device):
        if tree is self.tree and device == self.device:
            self.hits += 1
            return self.plan
        buckets = defaultdict(list)
        start = tree.prefix_length
        for length in tree.branch_lengths:
            buckets[((length + 31) // 32) * 32].append((start, length))
            start += length
        assert start == tree.length
        plan = []
        p = tree.prefix_length
        for width, branches in sorted(buckets.items()):
            for offset in range(0, len(branches), 8):
                block = branches[offset : offset + 8]
                allowed = torch.zeros(8, 1, width, p + width, device=device, dtype=torch.bool)
                allowed[:, :, :, 0] = True
                for slot, (_, length) in enumerate(block):
                    allowed[slot, 0, :length, :p] = True
                    allowed[slot, 0, :length, p : p + length] = torch.ones(
                        length, length, device=device, dtype=torch.bool
                    ).tril()
                plan.append((width, block, allowed))
        self.tree = tree
        self.device = device
        self.plan = plan
        self.builds += 1
        return plan

    def stats(self):
        return dict(
            builds=self.builds,
            hits=self.hits,
            blocks=len(self.plan or []),
            branches=sum(len(block) for _, block, _ in self.plan or []),
        )


def branch_attention(
    module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, sliding_window=None, **kwargs
):
    request = REQUEST.get()
    tree = request[0] if request else None
    if tree is None or query.shape[0] != 1 or key.shape[-2] != tree.length or query.shape[-2] != tree.length:
        raise ValueError("Branch attention requires an explicit single-request tree layout")
    if sliding_window is not None:
        raise ValueError("Sliding-window branch attention is not implemented")
    if dropout:
        raise ValueError("Branch attention reference requires zero attention dropout")
    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    p = tree.prefix_length
    output = torch.zeros_like(query)
    output[:, :, :p] = F.scaled_dot_product_attention(
        query[:, :, :p], key[:, :, :p], value[:, :, :p], is_causal=True, scale=scaling
    )
    cache = request[1]
    heads, dim = query.shape[1], query.shape[-1]
    batch_slots = 8
    for width, block, allowed in cache(tree, query.device):
        q = query.new_zeros(batch_slots, heads, width, dim)
        k = key.new_zeros(batch_slots, heads, p + width, dim)
        v = value.new_zeros(batch_slots, heads, p + width, dim)
        for slot, (start, length) in enumerate(block):
            q[slot, :, :length] = query[0, :, start : start + length]
            k[slot, :, :p] = key[0, :, :p]
            v[slot, :, :p] = value[0, :, :p]
            k[slot, :, p : p + length] = key[0, :, start : start + length]
            v[slot, :, p : p + length] = value[0, :, start : start + length]
        values = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=scaling)
        for slot, (start, length) in enumerate(block):
            output[0, :, start : start + length] = values[slot, :, :length]
    return output.transpose(1, 2).contiguous(), None


def install(model):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    implementation = f"{__name__}.isolated_branches"
    ALL_ATTENTION_FUNCTIONS.register(implementation, branch_attention)
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = implementation


@contextmanager
def request_context(tree):
    token = REQUEST.set((tree, RequestMaskCache()))
    try:
        yield
    finally:
        REQUEST.reset(token)
