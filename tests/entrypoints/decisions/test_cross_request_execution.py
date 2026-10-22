# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.spev.branch_attention import (
    BatchAttentionPlan,
    RequestMaskCache,
    branch_attention,
    request_context,
)
from vllm_omni.model_executor.models.spev.candidate_tree import CandidateTree
from vllm_omni.model_executor.models.spev.checkpoint_ops import CheckpointLinear, canonical_projection
from vllm_omni.model_executor.models.spev.isolated_attention import isolated_batch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def trees():
    return [CandidateTree(3, (33, 1, 7)), CandidateTree(5, (2,) * 17), CandidateTree(3, (7, 33))]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_projection_ragged_order_gradients_and_grouping(dtype, bias, monkeypatch):
    torch.manual_seed(78)
    items = [SimpleNamespace(tree=tree) for tree in trees()]
    source = nn.Linear(16, 12, bias=bias).to(dtype)
    model = CheckpointLinear(source)
    value = torch.randn(sum(item.tree.length for item in items), 16, dtype=dtype, requires_grad=True)
    with isolated_batch(items):
        expected = canonical_projection(value, source)
        actual, _ = model(value)
    torch.testing.assert_close(actual, expected, atol=1e-6 if dtype == torch.float32 else 0, rtol=0)
    actual[:3].float().sum().backward()
    assert value.grad[:3].abs().sum() > 0
    assert torch.count_nonzero(value.grad[3:]) == 0
    calls = []
    original = torch.baddbmm if bias else torch.bmm

    def product(*args, **kwargs):
        calls.append(args[-2].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "baddbmm" if bias else "bmm", product)
    with torch.no_grad(), isolated_batch(items) as _:
        repeated, _ = model(value.detach())
    torch.testing.assert_close(repeated, actual.detach(), atol=0, rtol=0)
    if dtype == torch.bfloat16:
        assert calls and max(shape[0] for shape in calls) > 1
        assert all(shape[1] % 256 == 0 for shape in calls)
    else:
        assert not calls


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("key_token_limit", [65536, 128])
def test_attention_ragged_reference_and_request_causal_isolation(dtype, key_token_limit, monkeypatch):
    monkeypatch.setattr(BatchAttentionPlan, "max_packed_key_tokens", key_token_limit)
    torch.manual_seed(78)
    layouts = trees()
    requests, offset = [], 0
    for tree in layouts:
        requests.append((offset, offset + tree.length, tree, RequestMaskCache()))
        offset += tree.length
    q = torch.randn(offset, 4, 8, dtype=dtype, requires_grad=True)
    k = torch.randn(offset, 2, 8, dtype=dtype, requires_grad=True)
    v = torch.randn(offset, 2, 8, dtype=dtype, requires_grad=True)
    expected = []
    for start, end, tree, _ in requests:
        with request_context(tree):
            result, _ = branch_attention(
                None,
                q[start:end].transpose(0, 1)[None],
                k[start:end].transpose(0, 1)[None],
                v[start:end].transpose(0, 1)[None],
                scaling=0.3,
            )
            expected.append(result[0])
    plan = BatchAttentionPlan(requests, q.device)
    actual = plan.forward(q, k, v, 0.3)
    torch.testing.assert_close(actual, torch.cat(expected), atol=1e-6 if dtype == torch.float32 else 0, rtol=0)
    # First branch's first position may see only its own prefix and first token.
    actual[3].float().sum().backward()
    assert v.grad[:4].abs().sum() > 0
    for tensor in (q, k, v):
        assert torch.count_nonzero(tensor.grad[4:]) == 0
    changed = v.detach().clone()
    changed[layouts[0].length :] += 100
    again = plan.forward(q.detach(), k.detach(), changed, 0.3)
    torch.testing.assert_close(again[: layouts[0].length], actual[: layouts[0].length], atol=0, rtol=0)


def test_batch_metadata_expires_with_context_and_restores_nested_context():
    from vllm_omni.model_executor.models.spev.isolated_attention import LAYOUTS

    item = SimpleNamespace(tree=CandidateTree(3, (2,)))
    model = CheckpointLinear(nn.Linear(4, 4))
    with isolated_batch([item]):
        outer = LAYOUTS.get()
        model(torch.ones(5, 4))
        plan = outer.projection
        model(torch.ones(5, 4))
        assert outer.projection is plan
        with isolated_batch([item, item]):
            assert LAYOUTS.get() is not outer
            model(torch.ones(10, 4))
        assert LAYOUTS.get() is outer and outer.projection is plan
        with pytest.raises(ValueError, match="complete request"):
            model(torch.ones(6, 4))
    assert LAYOUTS.get() is None
