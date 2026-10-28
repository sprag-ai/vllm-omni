# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-row results of SPEV's invariant kernels across batch sizes and positions."""

import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def rows_in_batches(operation, value):
    alone = operation(value[:7])
    small = operation(value[:33])[:7]
    shifted = operation(torch.cat([value[40:], value[:7]]))[-7:]
    return alone, small, shifted


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_invariant_linear_ignores_batch(dtype):
    from vllm_omni.model_executor.models.spev.invariant_ops import invariant_linear

    torch.manual_seed(5)
    value = torch.randn(300, 2048, device="cuda", dtype=dtype)
    weight = torch.randn(1024, 2048, device="cuda", dtype=dtype) / 45
    bias = torch.randn(1024, device="cuda", dtype=dtype)
    alone, small, shifted = rows_in_batches(lambda x: invariant_linear(x, weight, bias), value)
    assert torch.equal(alone, small)
    assert torch.equal(alone, shifted)


def test_invariant_linear_fp32_matches_ieee_reference():
    from vllm_omni.model_executor.models.spev.invariant_ops import invariant_linear

    torch.manual_seed(6)
    value = torch.randn(65, 2048, device="cuda", dtype=torch.float64)
    weight = torch.randn(96, 2048, device="cuda", dtype=torch.float64) / 45
    output = invariant_linear(value.float(), weight.float())
    torch.testing.assert_close(output.double(), value @ weight.t(), atol=1e-4, rtol=0)


def test_invariant_experts_ignore_batch():
    from vllm_omni.model_executor.models.spev.invariant_ops import invariant_experts

    torch.manual_seed(7)
    experts, hidden, intermediate, top_k = 16, 2048, 768, 8
    value = torch.randn(300, hidden, device="cuda", dtype=torch.bfloat16)
    gate_up = torch.randn(experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16) / 45
    down = torch.randn(experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16) / 28
    logits = torch.randn(300, experts, device="cuda")
    weights, indices = logits.softmax(-1).topk(top_k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)

    def run(selection):
        return invariant_experts(value[selection], gate_up, down, weights[selection], indices[selection])

    alone = run(torch.arange(7, device="cuda"))
    small = run(torch.arange(33, device="cuda"))[:7]
    shifted = run(torch.cat([torch.arange(40, 300, device="cuda"), torch.arange(7, device="cuda")]))[-7:]
    assert torch.equal(alone, small)
    assert torch.equal(alone, shifted)
