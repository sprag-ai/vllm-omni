# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""RMS normalization with the checkpoint's BF16 rounding order."""

import torch
from torch import nn


class CheckpointRMSNorm(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight = source.weight
        self.variance_epsilon = source.variance_epsilon

    def forward(self, value: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is not None:
            value = value + residual
        original = value
        value = value.float()
        variance = value.square().mean(-1, keepdim=True)
        value = value * torch.rsqrt(variance + self.variance_epsilon)
        result = self.weight * value.to(original.dtype)
        return result if residual is None else (result, original)
