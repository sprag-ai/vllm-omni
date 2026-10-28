# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Row-invariant matrix products whose per-token results ignore the rest of the batch."""

import torch
import triton
import triton.language as tl
from vllm.model_executor.determinism.batch_invariant import matmul_persistent
from vllm.model_executor.layers.fused_moe import fused_experts, override_config

EXPERT_CONFIG = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "SPLIT_K": 1}


def supports_invariant(value):
    return value.is_cuda and value.dtype in (torch.bfloat16, torch.float32)


@triton.jit
def _ieee_matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        ks = k * BLOCK_K + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + ks[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (ks[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + ks[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(ks[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _ieee_matmul(a, b):
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))
    _ieee_matmul_kernel[grid](
        a, b, c, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
    )
    return c


def invariant_linear(value, weight, bias=None):
    rows = value.reshape(-1, value.shape[-1])
    if value.dtype == torch.float32:
        output = _ieee_matmul(rows, weight.t())
        if bias is not None:
            output = output + bias
    else:
        output = matmul_persistent(rows, weight.t(), bias)
    return output.view(*value.shape[:-1], weight.shape[0])


def invariant_experts(value, gate_up, down, weights, indices):
    with override_config(EXPERT_CONFIG):
        return fused_experts(value, gate_up, down, weights.float(), indices.to(torch.int32))
