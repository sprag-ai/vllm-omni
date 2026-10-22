# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch.nn import functional as F

from vllm_omni.model_executor.models.spev.branch_attention import RequestMaskCache, branch_attention, request_context
from vllm_omni.model_executor.models.spev.candidate_tree import CandidateTree

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def reference(tree, query, key, value):
    groups = query.shape[1] // key.shape[1]
    key, value = (x.repeat_interleave(groups, dim=1) for x in (key, value))
    p, heads, dim = tree.prefix_length, query.shape[1], query.shape[-1]
    output = torch.zeros_like(query)
    output[:, :, :p] = F.scaled_dot_product_attention(
        query[:, :, :p], key[:, :, :p], value[:, :, :p], is_causal=True, scale=0.25
    )
    for block in RequestMaskCache()(tree, query.device):
        q = query.new_zeros(8, heads, block.width, dim)
        k = key.new_zeros(8, heads, p + block.width, dim)
        v = value.new_zeros(8, heads, p + block.width, dim)
        for slot, (start, length) in enumerate(block.branches):
            q[slot, :, :length] = query[0, :, start : start + length]
            for dest, source in ((k, key), (v, value)):
                dest[slot, :, :p] = source[0, :, :p]
                dest[slot, :, p : p + length] = source[0, :, start : start + length]
        for packed, source, keys in ((q, query, False), (k, key, True), (v, value, True)):
            actual = block.pack(source[0], keys=keys)
            assert actual.stride() == packed.stride()
            torch.testing.assert_close(actual, packed, atol=0, rtol=0)
        values = F.scaled_dot_product_attention(q, k, v, attn_mask=block.allowed, scale=0.25)
        for slot, (start, length) in enumerate(block.branches):
            output[0, :, start : start + length] = values[slot, :, :length]
    return output.transpose(1, 2).contiguous()


@pytest.mark.parametrize("branches", [(1,), (3, 3), (7,) * 17, (1, 31, 32, 33, 65, 2, 32, 5, 33)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packing_scatter_matches_reference_outputs_and_gradients(branches, dtype):
    torch.manual_seed(78)
    tree = CandidateTree(5, branches)
    tensors = [torch.randn(1, heads, tree.length, 8, dtype=dtype, requires_grad=True) for heads in (4, 2, 2)]
    expected = reference(tree, *tensors)
    with request_context(tree):
        actual, _ = branch_attention(None, *tensors, scaling=0.25)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    weights = torch.randn_like(actual)
    expected_grad = torch.autograd.grad((expected * weights).sum(), tensors)
    actual_grad = torch.autograd.grad((actual * weights).sum(), tensors)
    for a, b in zip(actual_grad, expected_grad):
        if dtype == torch.float32:
            torch.testing.assert_close(a, b, atol=2e-6, rtol=0)
        else:
            assert torch.isfinite(a).all() and torch.isfinite(b).all()


def test_cache_reuses_only_its_own_layout():
    cache = RequestMaskCache()
    tree = CandidateTree(2, (3, 33, 4))
    first = cache(tree, torch.device("cpu"))
    assert cache(tree, torch.device("cpu")) is first
    assert cache.stats() == dict(builds=1, hits=1, blocks=2, branches=3)
    assert cache(CandidateTree(2, (4, 33, 3)), torch.device("cpu")) is not first
    assert cache.stats()["builds"] == 2


def test_branch_and_future_tokens_have_no_gradient_into_other_answers():
    tree = CandidateTree(3, (5, 5))
    tensors = [torch.randn(1, 2, tree.length, 8, requires_grad=True) for _ in range(3)]
    with request_context(tree):
        output, _ = branch_attention(None, *tensors)
    grads = torch.autograd.grad(output[:, 5].sum(), tensors)
    for grad in grads:
        assert torch.count_nonzero(grad[:, :, 6:]) == 0
    assert grads[1][:, :, :3].abs().sum() > 0
