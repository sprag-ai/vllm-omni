# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reuse request-local gather indices for fixed-shape token-local operations."""

from collections import defaultdict
from contextvars import ContextVar

import torch

DEPTH = ContextVar("spev_vectorized_token_ops_depth", default=0)


class PackingCache:
    def __init__(self):
        self.tree = None
        self.device = None
        self.plan = None

    def __call__(self, tree, device):
        if tree is self.tree and device == self.device:
            return self.plan
        buckets = defaultdict(list)
        start = tree.prefix_length
        for index, length in enumerate(tree.branch_lengths):
            buckets[((length + 31) // 32) * 32].append((index, start, length))
            start += length
        assert start == tree.length
        blocks = []
        restore = [None] * len(tree.branch_lengths)
        offset = 0
        for width, branches in sorted(buckets.items()):
            for begin in range(0, len(branches), 8):
                block = branches[begin : begin + 8]
                indices = [0] * (8 * width)
                valid = [False] * (8 * width)
                for slot, (index, start, length) in enumerate(block):
                    a = slot * width
                    indices[a : a + length] = range(start, start + length)
                    valid[a : a + length] = [True] * length
                    restore[index] = torch.arange(length, device=device) + offset + a
                blocks.append(
                    (
                        width,
                        torch.tensor(indices, device=device, dtype=torch.long),
                        torch.tensor(valid, device=device, dtype=torch.bool),
                    )
                )
                offset += 8 * width
        order = torch.cat(restore) if restore else torch.empty(0, device=device, dtype=torch.long)
        self.tree = tree
        self.device = device
        self.plan = (blocks, order)
        return self.plan


def grouped_forward(original, tree_getter, plan_getter=None):
    cache = PackingCache() if plan_getter is None else plan_getter

    def forward(module, value, *args, **kwargs):
        if DEPTH.get():
            return original(value, *args, **kwargs)
        tree = tree_getter()
        if tree is None:
            raise ValueError("Grouped token operations require a tree layout")
        if value.ndim not in (2, 3) or value.shape[-2] != tree.length:
            raise ValueError("Grouped token operations require the packed token sequence")
        if value.ndim == 3 and value.shape[0] != 1:
            raise ValueError("Grouped token operations support one packed request")
        x = value[0] if value.ndim == 3 else value
        blocks, order = cache(tree, x.device)
        token = DEPTH.set(DEPTH.get() + 1)
        try:
            prefix = original(x[: tree.prefix_length].contiguous(), *args, **kwargs)
            outputs = []
            for width, indices, valid in blocks:
                packed = x.index_select(0, indices).masked_fill(~valid[:, None], 0).reshape(8, width, x.shape[-1])
                output = original(packed, *args, **kwargs)
                assert isinstance(output, torch.Tensor) and output.shape[:2] == packed.shape[:2]
                outputs.append(output.reshape(8 * width, output.shape[-1]))
            branch = (
                (outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)).index_select(0, order)
                if outputs
                else prefix[:0]
            )
            output = torch.cat([prefix, branch], dim=0)
            return output[None] if value.ndim == 3 else output
        finally:
            DEPTH.reset(token)

    return forward
