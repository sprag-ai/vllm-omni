# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-shape matrix products for checkpoint arithmetic."""

import torch
from torch.nn import functional as F


def batched_linear(value, weight, bias=None):
    if value.ndim == 3 and value.is_cuda and value.dtype == torch.bfloat16:
        outputs = []
        for start in range(0, len(value), 8):
            part = value[start : start + 8]
            packed = F.pad(part, (0, 0, 0, 0, 0, 8 - len(part)))
            matrices = weight.T.expand(8, -1, -1)
            result = (
                torch.bmm(packed, matrices)
                if bias is None
                else torch.baddbmm(bias.expand(8, value.shape[1], -1), packed, matrices)
            )
            outputs.append(result[: len(part)])
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    if value.ndim != 3 or value.shape[0] == 1:
        return F.linear(value, weight, bias)
    if value.dtype == torch.float32:
        return torch.stack([F.linear(matrix, weight, bias) for matrix in value])
    matrices = weight.T.expand(value.shape[0], -1, -1)
    return (
        torch.bmm(value, matrices)
        if bias is None
        else torch.baddbmm(bias.expand(value.shape[0], value.shape[1], -1), value, matrices)
    )


def grouped_product(value, weight, offsets):
    if not value.is_cuda:
        return F.grouped_mm(value, weight, offs=offsets)
    starts = torch.cat((offsets.new_zeros(1), offsets[:-1])).long()
    counts = offsets.long() - starts
    rows = 32
    blocks = (int(counts.max().item()) + rows - 1) // rows
    output = value.new_zeros((len(value), weight.shape[-1]))
    columns = torch.arange(rows, device=value.device)
    for block in range(blocks):
        local = columns + block * rows
        valid = local[None, :] < counts[:, None]
        indices = (starts[:, None] + local[None, :]).masked_fill(~valid, 0)
        packed = value.index_select(0, indices.flatten()).view(len(offsets), rows, -1)
        packed = packed.masked_fill(~valid[:, :, None], 0)
        products = F.grouped_mm(packed, weight)
        output.index_copy_(0, indices[valid], products[valid])
    return output


def expert_forward(module, value, indices, weights):
    top_k = indices.shape[-1]
    expert_ids, permutation = indices.flatten().sort()
    selected = value[permutation // top_k]
    counts = torch.bincount(expert_ids, minlength=module.num_experts)
    offsets = counts.cumsum(0, dtype=torch.int32)
    projected = grouped_product(selected, module.gate_up_proj.transpose(-2, -1), offsets)
    activated = module._apply_gate(projected)
    projected = grouped_product(activated, module.down_proj.transpose(-2, -1), offsets)
    weighted = projected * weights.flatten()[permutation, None]
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation), device=value.device)
    return weighted[inverse].view(len(value), top_k, -1).sum(1)
