# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared-prefix branch attention with request-local immutable mask metadata."""

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch
from torch.nn import functional as F

REQUEST = ContextVar("spev_attention_request", default=None)


@dataclass(frozen=True)
class AttentionBlock:
    width: int
    branches: tuple[tuple[int, int], ...]
    allowed: torch.Tensor
    query_indices: torch.Tensor
    key_indices: torch.Tensor
    query_padding: torch.Tensor
    key_padding: torch.Tensor
    output_indices: torch.Tensor
    valid_indices: torch.Tensor

    def pack(self, value, *, keys=False):
        indices = self.key_indices if keys else self.query_indices
        padding = self.key_padding if keys else self.query_padding
        heads, _, dim = value.shape
        packed = value.index_select(1, indices).reshape(heads, 8, -1, dim).transpose(0, 1).contiguous()
        return packed.masked_fill(padding[:, None, :, None], 0)


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
                positions = torch.arange(width, device=device)[None, :]
                starts = torch.tensor([s for s, _ in block] + [0] * (8 - len(block)), device=device)[:, None]
                lengths = torch.tensor([n for _, n in block] + [0] * (8 - len(block)), device=device)[:, None]
                query_padding = positions >= lengths
                query_indices = (starts + positions).masked_fill(query_padding, 0)
                prefix = torch.arange(p, device=device)[None, :].expand(8, -1)
                prefix_padding = (lengths == 0).expand(-1, p)
                key_padding = torch.cat((prefix_padding, query_padding), dim=1)
                key_indices = torch.cat((prefix, query_indices), dim=1).masked_fill(key_padding, 0)
                valid_indices = (~query_padding).flatten().nonzero().flatten()
                plan.append(
                    AttentionBlock(
                        width,
                        tuple(block),
                        allowed,
                        query_indices.flatten(),
                        key_indices.flatten(),
                        query_padding,
                        key_padding,
                        query_indices.flatten()[valid_indices],
                        valid_indices,
                    )
                )
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
            branches=sum(len(block.branches) for block in self.plan or []),
        )


class VarlenAttentionPlan:
    """Shared-prefix branch attention as per-sequence FlashAttention calls whose rows ignore the rest of the batch."""

    def __init__(self, requests, device, version=2):
        self.version = version
        request_bounds, prefix_bounds, request_branches, branch_bounds = [0], [0], [0], [0]
        prefix_tokens, branch_tokens = [], []
        for start, end, tree, _ in requests:
            p = tree.prefix_length
            request_bounds.append(end)
            prefix_bounds.append(prefix_bounds[-1] + p)
            prefix_tokens.append(torch.arange(start, start + p))
            branch_tokens.append(torch.arange(start + p, end))
            request_branches.append(request_branches[-1] + end - start - p)
            for length in tree.branch_lengths:
                branch_bounds.append(branch_bounds[-1] + length)
        lengths = [b - a for a, b in zip(request_bounds, request_bounds[1:])]

        def tensor(values):
            return torch.tensor(values, dtype=torch.int32, device=device)

        self.requests = tensor(request_bounds)
        self.prefix_bounds = tensor(prefix_bounds)
        self.prefix_tokens = torch.cat(prefix_tokens).to(device)
        self.request_branches = tensor(request_branches)
        self.branch_bounds = tensor(branch_bounds)
        self.branch_tokens = torch.cat(branch_tokens).to(device)
        self.longest_request = max(lengths)
        self.longest_prefix = max(b - a for a, b in zip(prefix_bounds, prefix_bounds[1:]))
        self.most_branch_tokens = max(b - a for a, b in zip(request_branches, request_branches[1:]))
        self.longest_branch = max(b - a for a, b in zip(branch_bounds, branch_bounds[1:]))

    def forward(self, query, key, value, scaling):
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        def attend(q, k, v, **kwargs):
            return flash_attn_varlen_func(
                q, k, v, softmax_scale=scaling, num_splits=1, fa_version=self.version, **kwargs
            )

        output = attend(
            query,
            key,
            value,
            max_seqlen_q=self.longest_request,
            cu_seqlens_q=self.requests,
            max_seqlen_k=self.longest_request,
            cu_seqlens_k=self.requests,
            causal=True,
        )
        branch_query = query.index_select(0, self.branch_tokens)
        shared, shared_lse = attend(
            branch_query,
            key.index_select(0, self.prefix_tokens),
            value.index_select(0, self.prefix_tokens),
            max_seqlen_q=self.most_branch_tokens,
            cu_seqlens_q=self.request_branches,
            max_seqlen_k=self.longest_prefix,
            cu_seqlens_k=self.prefix_bounds,
            return_softmax_lse=True,
        )
        local, local_lse = attend(
            branch_query,
            key.index_select(0, self.branch_tokens),
            value.index_select(0, self.branch_tokens),
            max_seqlen_q=self.longest_branch,
            cu_seqlens_q=self.branch_bounds,
            max_seqlen_k=self.longest_branch,
            cu_seqlens_k=self.branch_bounds,
            causal=True,
            return_softmax_lse=True,
        )
        total = torch.logaddexp(shared_lse, local_lse)
        merged = shared.float() * (shared_lse - total).exp().t()[..., None]
        merged += local.float() * (local_lse - total).exp().t()[..., None]
        return output.index_copy_(0, self.branch_tokens, merged.to(output.dtype))


def attention_plan(requests, device, dtype):
    """Build the fastest branch attention plan that supports the device and dtype."""
    if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

        version = get_flash_attn_version()
        if version is not None:
            return VarlenAttentionPlan(requests, device, version)
    return BatchAttentionPlan(requests, device)


class BatchAttentionPlan:
    """Combine equally shaped attention blocks with independent batch masks."""

    max_packed_key_tokens = 65536

    def __init__(self, requests, device):
        prefixes, branches = defaultdict(list), defaultdict(list)
        for start, _, tree, cache in requests:
            p = tree.prefix_length
            prefixes[p].append(torch.arange(start, start + p, device=device))
            for block in cache(tree, device):
                branches[(p, block.width)].append((start, block))
        self.prefixes = [(p, len(items), torch.cat(items)) for p, items in prefixes.items()]
        self.branches = []
        for (prefix, width), items in branches.items():
            limit = max(1, self.max_packed_key_tokens // (8 * (prefix + width)))
            bucket = items
            for begin in range(0, len(bucket), limit):
                items = bucket[begin : begin + limit]
                count = 8 * len(items)
                self.branches.append(
                    (
                        width,
                        prefix + width,
                        count,
                        torch.cat([block.query_indices + start for start, block in items]),
                        torch.cat([block.key_indices + start for start, block in items]),
                        torch.cat([block.query_padding for _, block in items]),
                        torch.cat([block.key_padding for _, block in items]),
                        torch.cat([block.allowed for _, block in items]),
                        torch.cat([block.output_indices + start for start, block in items]),
                        torch.cat([block.valid_indices + i * 8 * width for i, (_, block) in enumerate(items)]),
                    )
                )

    def forward(self, query, key, value, scaling):
        heads, dim = query.shape[1:]
        groups = heads // key.shape[1]
        if groups > 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        output = torch.empty_like(query)

        def pack(source, indices, count, width, padding=None):
            result = source.index_select(0, indices).view(count, width, heads, dim).transpose(1, 2).contiguous()
            return result if padding is None else result.masked_fill(padding[:, None, :, None], 0)

        for width, count, indices in self.prefixes:
            q, k, v = (pack(source, indices, count, width) for source in (query, key, value))
            # Causal SDPA kernels can change arithmetic with batch size; retain the checkpoint's prefix shape.
            result = torch.cat(
                [
                    F.scaled_dot_product_attention(
                        q[i : i + 1], k[i : i + 1], v[i : i + 1], is_causal=True, scale=scaling
                    )
                    for i in range(count)
                ]
            )
            output.index_copy_(0, indices, result.transpose(1, 2).reshape(-1, heads, dim))
        for width, key_width, count, qi, ki, qp, kp, allowed, oi, valid in self.branches:
            result = F.scaled_dot_product_attention(
                pack(query, qi, count, width, qp),
                pack(key, ki, count, key_width, kp),
                pack(value, ki, count, key_width, kp),
                attn_mask=allowed,
                scale=scaling,
            )
            selected = result.transpose(1, 2).reshape(-1, heads, dim).index_select(0, valid)
            output.index_copy_(0, oi, selected)
        return output


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
    for block in cache(tree, query.device):
        q = block.pack(query[0])
        k = block.pack(key[0], keys=True)
        v = block.pack(value[0], keys=True)
        values = F.scaled_dot_product_attention(q, k, v, attn_mask=block.allowed, scale=scaling)
        selected = values.transpose(1, 2).reshape(8 * block.width, heads, dim).index_select(0, block.valid_indices)
        output[0].transpose(0, 1).index_copy_(0, block.output_indices, selected)
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
