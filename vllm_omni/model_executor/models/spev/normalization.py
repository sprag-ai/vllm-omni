# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""RMS normalization with the checkpoint's BF16 rounding order."""

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _rms_norm_kernel(
    value_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    combined_ptr,
    inner,
    value_outer,
    value_inner,
    residual_outer,
    residual_inner,
    width,
    epsilon,
    has_residual: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    outer = row // inner
    position = row % inner
    columns = tl.arange(0, block)
    mask = columns < width
    value = tl.load(value_ptr + outer * value_outer + position * value_inner + columns, mask=mask, other=0.0)
    if has_residual:
        residual = tl.load(
            residual_ptr + outer * residual_outer + position * residual_inner + columns, mask=mask, other=0.0
        )
        value = (value.to(tl.float32) + residual.to(tl.float32)).to(value.dtype)
        tl.store(combined_ptr + row * width + columns, value, mask=mask)
    scaled = value.to(tl.float32)
    variance = tl.sum(scaled * scaled, axis=0) / width
    scaled = (scaled * tl.rsqrt(variance + epsilon)).to(value.dtype)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0)
    output = (weight.to(tl.float32) * scaled.to(tl.float32)).to(value.dtype)
    tl.store(output_ptr + row * width + columns, output, mask=mask)


def _rows(value):
    if value.dim() == 2:
        return value.shape[0], 1, value.stride(0), 0
    return value.shape[0], value.shape[1], value.stride(0), value.stride(1)


def _supports_kernel(value, weight, residual):
    tensors = (value, residual) if residual is not None else (value,)
    return (
        value.is_cuda
        and value.dtype in (torch.bfloat16, torch.float16)
        and weight.dtype == value.dtype
        and all(t.dim() in (2, 3) and t.stride(-1) == 1 and t.shape == value.shape for t in tensors)
    )


def rms_norm(value, weight, epsilon, residual=None):
    """Normalize each row independently of the other rows in the batch.

    Returns:
        The normalized rows, and with a residual also the rounded sum of value and residual.
    """
    outer, inner, value_outer, value_inner = _rows(value)
    residual_outer, residual_inner = _rows(residual)[2:] if residual is not None else (0, 0)
    width = value.shape[-1]
    output = torch.empty(value.shape, device=value.device, dtype=value.dtype)
    combined = torch.empty_like(output) if residual is not None else output
    _rms_norm_kernel[(outer * inner,)](
        value,
        residual if residual is not None else value,
        weight,
        output,
        combined,
        inner,
        value_outer,
        value_inner,
        residual_outer,
        residual_inner,
        width,
        epsilon,
        has_residual=residual is not None,
        block=triton.next_power_of_2(width),
    )
    return output if residual is None else (output, combined)


class CheckpointRMSNorm(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight = source.weight
        self.variance_epsilon = source.variance_epsilon

    def forward(self, value: torch.Tensor, residual: torch.Tensor | None = None):
        if _supports_kernel(value, self.weight, residual):
            return rms_norm(value, self.weight, self.variance_epsilon, residual)
        if residual is not None:
            value = value + residual
        original = value
        value = value.float()
        variance = value.square().mean(-1, keepdim=True)
        value = value * torch.rsqrt(variance + self.variance_epsilon)
        result = self.weight * value.to(original.dtype)
        return result if residual is None else (result, original)
