# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared evidence prefix and isolated candidate branch layout."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CandidateTree:
    prefix_length: int
    branch_lengths: tuple[int, ...]

    def __post_init__(self):
        if self.prefix_length <= 0 or not self.branch_lengths or min(self.branch_lengths) <= 0:
            raise ValueError("A nonempty prefix and nonempty branches are required")

    @property
    def length(self):
        return self.prefix_length + sum(self.branch_lengths)

    def allowed(self, *, device=None):
        groups = [-1] * self.prefix_length
        for index, length in enumerate(self.branch_lengths):
            groups.extend([index] * length)
        groups = torch.tensor(groups, device=device)
        physical = torch.arange(self.length, device=device)
        causal = physical[None, :] <= physical[:, None]
        return causal & ((groups[None, :] == -1) | (groups[:, None] == groups[None, :]))

    def additive_mask(self, *, dtype=torch.float32, device=None):
        allowed = self.allowed(device=device)
        return torch.zeros_like(allowed, dtype=dtype).masked_fill(~allowed, float("-inf"))[None, None]

    def position_ids(self, prefix_positions):
        if prefix_positions.ndim != 3 or prefix_positions.shape[-1] != self.prefix_length:
            raise ValueError("Expected native prefix positions with shape (axes, batch, prefix)")
        if prefix_positions.shape[1] != 1:
            raise ValueError("This reference packs one request")
        origin = int(prefix_positions.max()) + 1
        parts = [prefix_positions]
        for length in self.branch_lengths:
            branch = torch.arange(origin, origin + length, device=prefix_positions.device)
            parts.append(branch.view(1, 1, -1).expand(prefix_positions.shape[0], 1, -1))
        return torch.cat(parts, dim=-1)

    def attention_pairs(self):
        prefix = self.prefix_length
        return prefix * (prefix + 1) // 2 + sum(
            prefix * length + length * (length + 1) // 2 for length in self.branch_lengths
        )
